//! Cuántos GFLOPS da `kernels::sgemm` con los tamaños del UNet de SD.
//!
//!     cargo run --release --example measure_sgemm
//!
//! Una convolución 3×3 a 64×64 con 320 canales es un GEMM de 320 × 4096 × 2880;
//! una capa lineal de la atención, 4096 × 320 × 320. Los pesos van en f16 como
//! en el modelo, y la entrada en f32.

use std::time::Instant;

use cosxamix::kernels::sgemm::{sgemm, Mat, MatF16, MatF16T};

fn medir(nombre: &str, (m, n, k): (usize, usize, usize), f16_en_a: bool) {
    let mut a = cosxamix::daiss::Azar::con_semilla(1);
    let pesos: Vec<u16> = (0..m.max(n) * k)
        .map(|_| 0x3c00 ^ (a.siguiente() as u16 & 0x03ff))
        .collect();
    let x: Vec<f32> = (0..m.max(n) * k).map(|_| a.normales(1.0).0).collect();
    let mut c = vec![0f32; m * n];
    let mut tiempos = Vec::new();
    for _ in 0..7 {
        c.fill(0.0);
        let t = Instant::now();
        if f16_en_a {
            sgemm(
                (m, n, k),
                &MatF16 {
                    datos: &pesos,
                    fila: k,
                },
                &Mat {
                    datos: &x[..k * n],
                    fila: n,
                    col: 1,
                },
                &mut c,
                n,
            );
        } else {
            sgemm(
                (m, n, k),
                &Mat {
                    datos: &x[..m * k],
                    fila: k,
                    col: 1,
                },
                &MatF16T {
                    datos: &pesos,
                    fila: k,
                },
                &mut c,
                n,
            );
        }
        tiempos.push(t.elapsed().as_secs_f64());
    }
    tiempos.sort_by(f64::total_cmp);
    let s = tiempos[3];
    let gflop = 2.0 * (m * n * k) as f64 / 1e9;
    println!(
        "  {nombre:<34} {m:>5} x {n:>5} x {k:>5}  {:>7.2} ms  {:>6.1} GFLOPS",
        s * 1e3,
        gflop / s
    );
}

fn main() {
    cosxamix::machine_load::antes_de_medir();
    println!("  hilos: {}", rayon::current_num_threads());
    medir("conv 3x3, 320 canales a 64x64", (320, 4096, 2880), true);
    medir("conv 3x3, 1280 canales a 8x8", (1280, 64, 11520), true);
    medir("lineal de atencion, 4096 x 320", (4096, 320, 320), false);
    medir("feed-forward GEGLU, 4096 x 2560", (4096, 2560, 320), false);
}
