//! ¿Cuánto cuesta un paso de entrenamiento LoRA en esta máquina?
//!
//! Una **capa entera** del Qwen2.5-0,5B —atención con RoPE y cabezas agrupadas,
//! normas, SwiGLU—, con pesos al azar de la forma real, LoRA r=8 sobre q y v y
//! una secuencia de 256 tokens; de ahí se proyecta a las 24 capas. El
//! vocabulario queda chico a propósito: la proyección final es una sola y se
//! mide aparte en `train_lora`.
//!
//! Mide también lo que ahorra el backward propio con la base congelada: no
//! calcula `dL/dW` de las siete matrices, que es la mitad del trabajo de la
//! vuelta. Con candle eso no se podía evitar.
//!
//!     cargo run --release --features training --example measure_lora

use std::time::Instant;

use cosxamix::daiss::Azar;
use cosxamix::training::transformer::{
    AdamW, Capa, Forma, Lineal, Lora, Params, Tablas, Transformer,
};

const TOKENS: usize = 256;

fn al_azar(a: &mut Azar, n: usize) -> Vec<f32> {
    (0..n).map(|_| a.normales(0.02).0).collect()
}

/// Una capa del 0,5B; la base entrena o no según `base_entrena`.
fn capa(base_entrena: bool) -> (Transformer, Vec<f32>) {
    let f = Forma {
        dim: 896,
        ffn: 4864,
        capas: 1,
        cabezas: 14,
        cabezas_kv: 2,
        vocab: 512,
        eps: 1e-6,
        rope: 1e6,
    };
    let mut a = Azar::con_semilla(1);
    let mut p = Params::default();
    let mut lin = |p: &mut Params, n: &str, ent: usize, sal: usize, sesgo: bool, lora: bool| {
        let w = p.agregar(
            format!("{n}.weight"),
            vec![sal, ent],
            al_azar(&mut a, sal * ent),
            base_entrena,
        );
        let b =
            sesgo.then(|| p.agregar(format!("{n}.bias"), vec![sal], vec![0.0; sal], base_entrena));
        let lora = lora.then(|| Lora {
            a: p.agregar(
                format!("{n}.lora_a"),
                vec![8, ent],
                al_azar(&mut a, 8 * ent),
                true,
            ),
            b: p.agregar(
                format!("{n}.lora_b"),
                vec![sal, 8],
                al_azar(&mut a, sal * 8),
                true,
            ),
            r: 8,
            escala: 2.0,
        });
        Lineal {
            w,
            b,
            lora,
            ent,
            sal,
        }
    };
    let q = lin(&mut p, "q", f.dim, f.dim, true, true);
    let k = lin(&mut p, "k", f.dim, f.dim_kv(), true, false);
    let v = lin(&mut p, "v", f.dim, f.dim_kv(), true, true);
    let o = lin(&mut p, "o", f.dim, f.dim, false, false);
    let gate = lin(&mut p, "gate", f.dim, f.ffn, false, false);
    let up = lin(&mut p, "up", f.dim, f.ffn, false, false);
    let down = lin(&mut p, "down", f.ffn, f.dim, false, false);
    let norma_atencion = p.agregar("na", vec![f.dim], vec![1.0; f.dim], base_entrena);
    let norma_ffn = p.agregar("nf", vec![f.dim], vec![1.0; f.dim], base_entrena);
    let norma_final = p.agregar("nfin", vec![f.dim], vec![1.0; f.dim], base_entrena);
    let capas = vec![Capa {
        norma_atencion,
        q,
        k,
        v,
        o,
        norma_ffn,
        gate,
        up,
        down,
    }];
    let mut az = Azar::con_semilla(2);
    let tabla = al_azar(&mut az, f.vocab * f.dim);
    (
        Transformer::nuevo(f, p, capas, norma_final, TOKENS + 1),
        tabla,
    )
}

/// La mediana de cinco, en ms.
fn medir(mut f: impl FnMut()) -> f64 {
    f();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[2]
}

fn main() {
    cosxamix::machine_load::antes_de_medir();
    let lote = vec![(0..=TOKENS as u32)
        .map(|i| i * 7 % 512)
        .collect::<Vec<u32>>()];
    println!();
    println!("  Qwen2.5-0,5B, una capa, LoRA r=8 sobre q y v, {TOKENS} tokens");
    println!();
    let mut filas = Vec::new();
    for (etiqueta, base_entrena) in [("base congelada (LoRA)", false), ("base entrenable", true)] {
        let (mut m, tabla) = capa(base_entrena);
        let tablas = Tablas {
            entrada: &tabla,
            salida: &tabla,
            gradientes: false,
        };
        let ida = medir(|| {
            m.perdida(&lote, &tablas);
        });
        let vuelta = medir(|| {
            m.perdida_y_gradientes(&lote, &tablas);
        });
        let mut opt = AdamW::nuevo(1e-4);
        let (_, g) = m.perdida_y_gradientes(&lote, &tablas);
        let adam = medir(|| opt.paso(&mut m.params.valores, &g.params));
        let paso = vuelta + adam;
        println!(
            "  {etiqueta}: {} valores entrenables",
            m.params.entrenables()
        );
        println!("    {:<30}{:>10.1} ms", "forward", ida);
        println!("    {:<30}{:>10.1} ms", "forward + backward", vuelta);
        println!("    {:<30}{:>10.1} ms", "AdamW", adam);
        println!(
            "    {:<30}{:>10.2} s",
            "paso, 24 capas (proyectado)",
            paso * 24.0 / 1e3
        );
        println!();
        filas.push(paso);
    }
    println!(
        "  Congelar la base ahorra el {:.0}% del paso: la vuelta no calcula dL/dW.",
        (1.0 - filas[0] / filas[1]) * 100.0
    );
}
