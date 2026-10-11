//! ¿Cuánto cuesta un paso de entrenamiento LoRA en esta máquina?
//!
//! Entrenar es la mitad del objetivo del proyecto y hoy no existe nada. Antes
//! de escribir un entrenador entero conviene saber si el paso cabe en el
//! presupuesto, porque de eso depende todo lo demás.
//!
//! ## Lo que decide el diseño
//!
//! `QMatMul` de candle **no implementa `bwd`**, así que el gradiente no
//! atraviesa una multiplicación cuantizada. La matemática del backward pide
//! `dL/dx = dL/dy · W`, o sea multiplicar por `W` sin transponer, y candle
//! solo sabe hacer `x · Wᵀ`.
//!
//! O sea que entrenar sobre el modelo cuantizado pide un kernel nuevo. La
//! alternativa, que es la que se mide acá, es descuantizar la base a f32 y
//! dejar que el autograd de candle haga su trabajo. Cuesta memoria: el 0,5B en
//! f32 son ~2 GB y entra; el 3B serían ~12 GB y no.
//!
//! ## Qué mide y qué no
//!
//! Una **sola capa** con sus siete multiplicaciones —q, k, v, o, gate, up,
//! down—, con adaptadores LoRA sobre q y v, y de ahí se proyecta al modelo
//! entero. Deja fuera la atención, RoPE y las normalizaciones: son baratas al
//! lado de las matrices, pero por eso la proyección es un **piso**, no una
//! estimación exacta.
//!
//!     cargo run --release --features candle --example measure_lora

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{AdamW, Linear, Optimizer, ParamsAdamW, VarMap};

/// Un adaptador LoRA: dos matrices chicas que se suman a una grande congelada.
///
/// `y = W·x + (B·A·x) · (alfa/r)`. Solo `A` y `B` tienen gradiente; `W` no.
struct Lora {
    a: Var,
    b: Var,
    escala: f64,
}

impl Lora {
    fn nuevo(entrada: usize, salida: usize, r: usize, dev: &Device) -> candle_core::Result<Self> {
        // `A` arranca con ruido y `B` en cero, que es lo habitual: así el
        // adaptador no altera el modelo en el primer paso y la pérdida
        // empieza donde estaba.
        let a = Var::from_tensor(&Tensor::randn(0f32, 0.02f32, (r, entrada), dev)?)?;
        let b = Var::from_tensor(&Tensor::zeros((salida, r), DType::F32, dev)?)?;
        Ok(Self {
            a,
            b,
            escala: 16.0 / r as f64,
        })
    }

    fn aplicar(&self, x: &Tensor, base: &Tensor) -> candle_core::Result<Tensor> {
        let h = x.broadcast_matmul(&self.a.t()?)?;
        let d = h.broadcast_matmul(&self.b.t()?)?;
        base + (d * self.escala)?
    }
}

fn lineal(entrada: usize, salida: usize, dev: &Device) -> candle_core::Result<Linear> {
    // Pesos congelados: `Tensor`, no `Var`. Da igual que sean aleatorios, lo
    // que se mide es el coste de las operaciones y ese no depende del valor.
    let w = Tensor::randn(0f32, 0.02f32, (salida, entrada), dev)?;
    Ok(Linear::new(w, None))
}

fn main() -> candle_core::Result<()> {
    let dev = Device::Cpu;

    // Config del Qwen2.5-0,5B. Son las cifras del gguf, no inventadas.
    let (oculto, intermedio, capas) = (896usize, 4864usize, 24usize);
    let (cabezas, cabezas_kv, dim_cabeza) = (14usize, 2usize, 64usize);
    let r = 8usize;
    let tokens = 256usize;

    println!();
    println!("  Qwen2.5-0,5B en f32, una capa, LoRA r={r} sobre q y v");
    println!("  secuencia de {tokens} tokens, {capas} capas en el modelo completo");
    println!();

    let q = lineal(oculto, cabezas * dim_cabeza, &dev)?;
    let k = lineal(oculto, cabezas_kv * dim_cabeza, &dev)?;
    let v = lineal(oculto, cabezas_kv * dim_cabeza, &dev)?;
    let o = lineal(cabezas * dim_cabeza, oculto, &dev)?;
    let gate = lineal(oculto, intermedio, &dev)?;
    let up = lineal(oculto, intermedio, &dev)?;
    let down = lineal(intermedio, oculto, &dev)?;

    let lora_q = Lora::nuevo(oculto, cabezas * dim_cabeza, r, &dev)?;
    let lora_v = Lora::nuevo(oculto, cabezas_kv * dim_cabeza, r, &dev)?;

    let mapa = VarMap::new();
    let entrenables = vec![
        lora_q.a.clone(),
        lora_q.b.clone(),
        lora_v.a.clone(),
        lora_v.b.clone(),
    ];
    let n_entrenables: usize = entrenables.iter().map(|v| v.elem_count()).sum();
    let mut opt = AdamW::new(entrenables, ParamsAdamW::default())?;
    drop(mapa);

    let x = Tensor::randn(0f32, 1f32, (tokens, oculto), &dev)?;

    // La capa, simplificada a sus multiplicaciones. La atención, RoPE y las
    // normalizaciones quedan fuera a propósito: son baratas al lado de esto,
    // y dejarlas fuera convierte el resultado en un piso honesto.
    let paso = |x: &Tensor| -> candle_core::Result<Tensor> {
        let hq = lora_q.aplicar(x, &q.forward(x)?)?;
        let _hk = k.forward(x)?;
        let hv = lora_v.aplicar(x, &v.forward(x)?)?;
        // En lugar de la atención completa, se usa hq y hv para que los dos
        // adaptadores queden en el camino del gradiente, que es lo que importa
        // para medir el backward.
        let mezcla = (hq.narrow(1, 0, cabezas_kv * dim_cabeza)? + hv)?;
        let relleno = hq.narrow(
            1,
            cabezas_kv * dim_cabeza,
            cabezas * dim_cabeza - cabezas_kv * dim_cabeza,
        )?;
        let junto = Tensor::cat(&[&mezcla, &relleno], 1)?;
        let y = o.forward(&junto)?;
        let g = gate.forward(&y)?.silu()?;
        let u = up.forward(&y)?;
        down.forward(&(g * u)?)
    };

    // Una pasada de calentamiento, para no medir la primera asignación.
    let salida = paso(&x)?;
    let perdida = salida.sqr()?.mean_all()?;
    let grads = perdida.backward()?;
    opt.step(&grads)?;

    let medir = |f: &mut dyn FnMut() -> candle_core::Result<f64>| -> candle_core::Result<f64> {
        let mut v = Vec::new();
        for _ in 0..3 {
            v.push(f()?);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(v[1])
    };

    let ms_fwd = medir(&mut || {
        let t0 = Instant::now();
        let s = paso(&x)?;
        let _ = s.sqr()?.mean_all()?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    })?;

    let ms_total = medir(&mut || {
        let t0 = Instant::now();
        let s = paso(&x)?;
        let p = s.sqr()?.mean_all()?;
        let g = p.backward()?;
        opt.step(&g)?;
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    })?;

    println!(
        "  {:<34}{:>12}",
        "parametros entrenables (1 capa)", n_entrenables
    );
    println!("  {:<34}{:>12}", "forward", format!("{ms_fwd:.1} ms"));
    println!(
        "  {:<34}{:>12}",
        "backward + paso del optimizador",
        format!("{:.1} ms", ms_total - ms_fwd)
    );
    println!(
        "  {:<34}{:>12}",
        "paso completo",
        format!("{ms_total:.1} ms")
    );
    println!(
        "  {:<34}{:>12}",
        "backward / forward",
        format!("{:.2}x", (ms_total - ms_fwd) / ms_fwd)
    );

    println!();
    println!("  Proyectado al modelo entero ({capas} capas), como piso:");
    let paso_modelo = ms_total * capas as f64 / 1000.0;
    println!("  {:<34}{:>12}", "un paso", format!("{paso_modelo:.2} s"));
    for n in [100usize, 1000] {
        println!(
            "  {:<34}{:>12}",
            format!("{n} pasos"),
            format!("{:.1} min", paso_modelo * n as f64 / 60.0)
        );
    }
    println!(
        "  {:<34}{:>12}",
        "parametros entrenables, modelo",
        n_entrenables * capas
    );

    // -- Cuanto del backward es trabajo tirado --
    //
    // `backprop.rs` calcula los dos gradientes de cada multiplicacion sin
    // preguntar si hacen falta:
    //
    //     lhs_grad = grad.matmul(rhs.t())   // dL/dx, se necesita
    //     rhs_grad = lhs.t().matmul(grad)   // dL/dW, con W congelada NO
    //
    // Los dos cuestan `tokens * in * out`, asi que en una capa congelada la
    // mitad del backward sobra. Se mide con las siete formas reales.
    println!();
    println!("  Cuanto del backward sobra, por capa");
    println!("  {:<22}{:>12}{:>12}", "matriz", "dL/dx ms", "dL/dW ms");
    println!("  {:-<46}", "");

    let formas = [
        ("q    896 -> 896", oculto, cabezas * dim_cabeza),
        ("k    896 -> 128", oculto, cabezas_kv * dim_cabeza),
        ("v    896 -> 128", oculto, cabezas_kv * dim_cabeza),
        ("o    896 -> 896", cabezas * dim_cabeza, oculto),
        ("gate 896 -> 4864", oculto, intermedio),
        ("up   896 -> 4864", oculto, intermedio),
        ("down 4864 -> 896", intermedio, oculto),
    ];

    let mut total_dx = 0.0f64;
    let mut total_dw = 0.0f64;
    for (etq, dentro, fuera) in formas {
        let w = Tensor::randn(0f32, 0.02f32, (fuera, dentro), &dev)?;
        let xs = Tensor::randn(0f32, 1f32, (tokens, dentro), &dev)?;
        let g = Tensor::randn(0f32, 1f32, (tokens, fuera), &dev)?;

        let cronometrar =
            |f: &dyn Fn() -> candle_core::Result<Tensor>| -> candle_core::Result<f64> {
                let _ = f()?;
                let mut v = Vec::new();
                for _ in 0..3 {
                    let t0 = Instant::now();
                    for _ in 0..5 {
                        let _ = f()?;
                    }
                    v.push(t0.elapsed().as_secs_f64() * 1000.0 / 5.0);
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(v[1])
            };

        let ms_dx = cronometrar(&|| g.matmul(&w))?;
        let ms_dw = cronometrar(&|| xs.t()?.matmul(&g))?;
        total_dx += ms_dx;
        total_dw += ms_dw;
        println!(
            "  {:<22}{:>12}{:>12}",
            etq,
            format!("{ms_dx:.2}"),
            format!("{ms_dw:.2}")
        );
    }
    println!("  {:-<46}", "");
    println!(
        "  {:<22}{:>12}{:>12}",
        "suma",
        format!("{total_dx:.2}"),
        format!("{total_dw:.2}")
    );
    println!();
    println!(
        "  Con la base congelada, esos {:.1} ms por capa son trabajo tirado:",
        total_dw
    );
    println!(
        "  {:.1} s por paso del modelo entero, de los {:.2} s medidos.",
        total_dw * capas as f64 / 1000.0,
        paso_modelo
    );

    println!();
    println!("  Deja fuera atencion, RoPE y normalizaciones, asi que el modelo");
    println!("  entero va a costar mas. Sirve para saber si estamos en minutos");
    println!("  o en dias, que es la pregunta que bloquea el diseno.");

    Ok(())
}
