//! El kernel fusionado de Q5_K en cada lenguaje, contra el camino de antes:
//! desempaquetar la fila a un buffer y después multiplicar.
//!
//! Con una fila de 11008 valores, la de `ffn_down` del Qwen 3B. `COSXAMIX_ancho`
//! la cambia.
//!
//!     cargo run --release --features zig,c --example measure_q5k

use std::hint::black_box;
use std::time::Instant;

use cosxamix::gguf::TipoGgml;
use cosxamix::kernels::q5k::{disponibles, dot_con, elegida};
use cosxamix::quant::{cuantizar_bloques, desempaquetar_i8};

/// La mediana de varias tandas, en ns por llamada.
fn medir(mut f: impl FnMut() -> f32) -> f64 {
    const VECES: u32 = 20_000;
    let mut v: Vec<f64> = (0..11)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..VECES {
                black_box(f());
            }
            t.elapsed().as_secs_f64() * 1e9 / f64::from(VECES)
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let ancho: usize = std::env::var("COSXAMIX_ancho")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11008);
    let mut azar = cosxamix::daiss::Azar::con_semilla(5);
    let w: Vec<f32> = (0..ancho).map(|_| azar.normales(0.02).0).collect();
    let mut pesos = Vec::new();
    cuantizar_bloques(TipoGgml::Q5K, &w, &mut pesos).unwrap();
    let xq: Vec<i8> = (0..ancho).map(|_| azar.siguiente() as i8).collect();
    let xe: Vec<f32> = (0..ancho / 32)
        .map(|_| azar.normales(0.1).0.abs())
        .collect();
    let xs: Vec<f32> = xq
        .chunks(32)
        .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
        .collect();

    let (mut q, mut e, mut d) = (
        vec![0i8; ancho],
        vec![0f32; ancho / 32],
        vec![0f32; ancho / 32],
    );
    let ns = medir(|| {
        desempaquetar_i8(TipoGgml::Q5K, &pesos, &mut q, &mut e, &mut d).unwrap();
        let mut total = cosxamix::kernels::dot_i8_escalado(&q, &xq, &e, &xe, 32);
        for ((dd, xee), xss) in d.iter().zip(&xe).zip(&xs) {
            total += dd * xee * xss;
        }
        total
    });
    let gbs = |ns: f64| pesos.len() as f64 / ns;
    println!(
        "  una fila de {ancho} valores en Q5_K, {} bytes",
        pesos.len()
    );
    println!("  elegida: {:?}", elegida());
    println!("  con buffer (antes)  {ns:>8.0} ns  {:>6.1} GB/s", gbs(ns));
    for copia in disponibles() {
        let n = medir(|| dot_con(copia, &pesos, &xq, &xe, &xs));
        println!(
            "  {:<19} {n:>8.0} ns  {:>6.1} GB/s  {:.2}x",
            format!("{copia:?}"),
            gbs(n),
            ns / n
        );
    }
}
