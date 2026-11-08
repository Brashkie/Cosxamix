//! El kernel fusionado de Q2_K en cada lenguaje, contra el camino de antes:
//! desempaquetar la fila a un buffer y después multiplicar.
//!
//! Con una fila de 6144 valores, el ancho de Chavín.
//!
//!     cargo run --release --features zig,c --example measure_q2k

use std::hint::black_box;
use std::time::Instant;

use cosxamix::gguf::TipoGgml;
use cosxamix::kernels::q2k::{disponibles, dot_con, elegida};
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
    const ANCHO: usize = 6144;
    let mut azar = cosxamix::daiss::Azar::con_semilla(5);
    let w: Vec<f32> = (0..ANCHO).map(|_| azar.normales(0.02).0).collect();
    let mut pesos = Vec::new();
    cuantizar_bloques(TipoGgml::Q2K, &w, &mut pesos).unwrap();
    let xq: Vec<i8> = (0..ANCHO).map(|_| azar.siguiente() as i8).collect();
    let xe: Vec<f32> = (0..ANCHO / 16)
        .map(|_| azar.normales(0.1).0.abs())
        .collect();
    let xs: Vec<f32> = xq
        .chunks(16)
        .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
        .collect();

    let (mut q, mut e, mut d) = (
        vec![0i8; ANCHO],
        vec![0f32; ANCHO / 16],
        vec![0f32; ANCHO / 16],
    );
    let ns = medir(|| {
        desempaquetar_i8(TipoGgml::Q2K, &pesos, &mut q, &mut e, &mut d).unwrap();
        let mut total = cosxamix::kernels::dot_i8_escalado(&q, &xq, &e, &xe, 16);
        for ((dd, xee), xss) in d.iter().zip(&xe).zip(&xs) {
            total += dd * xee * xss;
        }
        total
    });
    let gbs = |ns: f64| pesos.len() as f64 / ns;
    println!(
        "  una fila de {ANCHO} valores en Q2_K, {} bytes",
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
