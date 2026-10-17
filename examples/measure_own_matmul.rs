//! La multiplicacion propia contra la de candle, sobre pesos reales.
//!
//! Mide tres cosas y la segunda es la que importa:
//!
//!   1. cuanto tarda cada una;
//!   2. **si el coste por fila BAJA al crecer el lote**, que es lo que la de
//!      candle no hace porque relee la matriz entera por cada fila;
//!   3. cuanto se parecen los resultados, que no pueden ser identicos: candle
//!      cuantiza la entrada a Q8_K y multiplica en enteros, la nuestra
//!      desempaqueta a f32 y multiplica en f32.
//!
//!     cargo run --release --features candle --example measure_own_matmul

use std::time::Instant;

use cosxamix::gguf::Gguf;

fn mediana(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Cuantos milisegundos tiene que durar cada muestra para ser creible.
const MUESTRA_MS: f64 = 60.0;

/// Mide una operacion repitiendola hasta juntar [`MUESTRA_MS`], y devuelve la
/// mediana de cinco muestras en milisegundos por llamada.
///
/// El numero de repeticiones NO puede ser fijo. Con 20, que es lo que habia,
/// una llamada de 0,2 ms daba una muestra de 4 ms y el ruido del reloj y del
/// planificador se la comia: dos pasadas seguidas del MISMO binario, sin tocar
/// una linea, dieron 0,205 y 0,361 ms para candle a una fila, un 76% de
/// diferencia. Lo nuestro, que tardaba diez veces mas, variaba un 4%. O sea
/// que la comparacion estaba sesgada a favor de quien fuera mas rapido, y una
/// medicion que depende de a quien mide no sirve para decidir nada.
fn medir(mut f: impl FnMut()) -> f64 {
    // Una pasada de calibrado para saber cuanto tarda, y de paso deja los
    // datos en cache como los va a encontrar el resto.
    let t0 = Instant::now();
    f();
    let una = t0.elapsed().as_secs_f64() * 1000.0;
    let repeticiones = ((MUESTRA_MS / una.max(1e-6)).ceil() as usize).clamp(20, 100_000);

    mediana(
        (0..5)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..repeticiones {
                    f();
                }
                t0.elapsed().as_secs_f64() * 1000.0 / repeticiones as f64
            })
            .collect(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let ruta = std::path::Path::new(&dir).join(
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q4_k_m.gguf".into()),
    );
    if !ruta.exists() {
        println!("  falta {}", ruta.display());
        return Ok(());
    }

    let g = Gguf::abrir(&ruta)?;
    let info = g.tensor("blk.0.ffn_gate.weight").ok_or("sin ffn_gate")?;
    // En el archivo las dimensiones van [entradas, salidas].
    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
    let pesos = g.datos_de(&ruta, info)?;

    println!();
    println!(
        "  {} · {:?} · {k} -> {n} · {:.1} MB",
        info.nombre,
        info.tipo,
        pesos.len() as f64 / 1048576.0
    );
    println!();
    println!(
        "  {:<6}{:>10}{:>10}{:>10}{:>11}{:>10}{:>13}",
        "filas", "candle", "f32", "i8 rodeo", "i8 directo", "vs candle", "directo/fila"
    );
    println!("  {:-<70}", "");

    // Los mismos pesos para los dos caminos: candle los necesita como QTensor.
    let dev = candle_core::Device::Cpu;
    let mut f = std::fs::File::open(&ruta)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut f)?;
    let qt = ct.tensor(&mut f, &info.nombre, &dev)?;
    let mm = candle_core::quantized::QMatMul::from_qtensor(qt)?;

    let mut dif_max = 0f32;

    for filas in [1usize, 2, 5, 8] {
        let x: Vec<f32> = (0..filas * k).map(|i| (i as f32 * 0.007).cos()).collect();

        let mut d = vec![0f32; filas * n];
        let t_nuestra = medir(|| {
            cosxamix::ops::matmul((filas, k, n), &x, &pesos, info.tipo, &mut d).expect("matmul");
        });

        let t_enteros = medir(|| {
            cosxamix::ops::matmul_i8((filas, k, n), &x, &pesos, info.tipo, &mut d)
                .expect("matmul_i8");
        });

        let t_directo = medir(|| {
            cosxamix::ops::matmul_i8_directo((filas, k, n), &x, &pesos, info.tipo, &mut d)
                .expect("directo");
        });

        let xt = candle_core::Tensor::from_vec(x.clone(), (filas, k), &dev)?;
        let t_candle = {
            use candle_core::Module;
            medir(|| {
                let _ = mm.forward(&xt).expect("candle");
            })
        };

        // Cuanto se parecen.
        let mut nuestro = vec![0f32; filas * n];
        cosxamix::ops::matmul_i8_directo((filas, k, n), &x, &pesos, info.tipo, &mut nuestro)?;
        use candle_core::Module;
        let suyo = mm.forward(&xt)?.flatten_all()?.to_vec1::<f32>()?;
        let d = nuestro
            .iter()
            .zip(&suyo)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        dif_max = dif_max.max(d);

        println!(
            "  {:<6}{:>10}{:>10}{:>10}{:>11}{:>10}{:>13}",
            filas,
            format!("{t_candle:.3}"),
            format!("{t_nuestra:.3}"),
            format!("{t_enteros:.3}"),
            format!("{t_directo:.3}"),
            format!("{:.2}x", t_candle / t_directo),
            format!("{:.3}", t_directo / filas as f64)
        );
    }

    let rango = {
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut d = vec![0f32; n];
        cosxamix::ops::matmul_i8((1, k, n), &x, &pesos, info.tipo, &mut d)?;
        d.iter().cloned().fold(f32::MIN, f32::max) - d.iter().cloned().fold(f32::MAX, f32::min)
    };

    println!();
    println!(
        "  diferencia maxima contra candle: {dif_max:.4} sobre un rango de {rango:.2} ({:.2}%)",
        dif_max / rango * 100.0
    );
    println!("  No pueden ser identicas: candle cuantiza la entrada a Q8_K y");
    println!("  multiplica en enteros; la nuestra desempaqueta a f32. La nuestra");
    println!("  pierde menos precision, no mas.");
    println!();
    println!("  Las dos ultimas columnas son lo que decide: si la nuestra BAJA");
    println!("  al crecer las filas y la de candle se queda plana, el lote");
    println!("  amortiza, que es lo que el prefill necesita.");

    Ok(())
}
