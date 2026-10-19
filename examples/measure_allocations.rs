//! ¿Cuánto cuestan las asignaciones de memoria de cada `matmul`?
//!
//! `matmul_auto` reserva cuatro vectores por llamada —la entrada cuantizada,
//! sus escalas, sus sumas y la salida transpuesta— y un forward del 3B hace
//! **253 multiplicaciones**. La propuesta de tener un buffer reutilizable que
//! el motor conserve entre llamadas es razonable, y antes de escribirla
//! conviene saber cuánto hay para ganar.
//!
//! Se mide sobre las formas reales del 3B a una fila, que es el decode:
//!
//! | matriz | por capa | forma |
//! |---|---:|---|
//! | `attn_q`, `attn_output` | 2 | 2048 → 2048 |
//! | `attn_k`, `attn_v` | 2 | 2048 → 256 |
//! | `ffn_gate`, `ffn_up` | 2 | 2048 → 11008 |
//! | `ffn_down` | 1 | 11008 → 2048 |
//!
//! Por 36 capas, más `output` de 2048 → 151936.
//!
//! Los pesos son bytes al azar: el tiempo de una multiplicación no depende de
//! los valores, y así esto corre sin tener ningún modelo bajado.
//!
//!     cargo run --release --example measure_allocations

use std::time::Instant;

use cosxamix::gguf::TipoGgml;

/// Las formas de un forward del 3B, con cuántas veces aparece cada una.
const FORMAS: &[(usize, usize, usize, &str)] = &[
    (2048, 2048, 72, "attn_q + attn_output"),
    (2048, 256, 72, "attn_k + attn_v"),
    (2048, 11008, 72, "ffn_gate + ffn_up"),
    (11008, 2048, 36, "ffn_down"),
    (2048, 151936, 1, "output"),
];

fn mediana(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn medir(repeticiones: usize, mut f: impl FnMut()) -> f64 {
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

fn main() {
    let tipo = TipoGgml::Q4K;
    let m = 1usize;

    println!();
    println!("  Asignaciones contra multiplicación, formas del 3B, una fila");
    println!();
    println!(
        "  {:<24}{:>8}{:>12}{:>12}{:>10}",
        "matriz", "veces", "asignar", "multiplicar", "asignar %"
    );
    println!("  {:-<66}", "");

    let mut total_asignar = 0f64;
    let mut total_matmul = 0f64;

    for (k, n, veces, nombre) in FORMAS {
        let (k, n, veces) = (*k, *n, *veces);
        let n_bloques = k / 32;

        // Exactamente lo que reserva `matmul_i8_fusionado`, y nada más.
        let t_asignar = medir(200, || {
            let a = vec![0i8; m * k];
            let b = vec![0f32; m * n_bloques];
            let c = vec![0f32; m * n_bloques];
            let d = vec![0f32; n * m];
            std::hint::black_box((&a, &b, &c, &d));
        });

        let bytes_fila = k / 256 * 144;
        let pesos: Vec<u8> = (0..n * bytes_fila)
            .map(|i| ((i * 193 + 11) % 251) as u8)
            .collect();
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.017).cos()).collect();
        let mut dst = vec![0f32; m * n];

        // Pocas repeticiones en las matrices grandes: una sola de `output`
        // tarda decenas de milisegundos y no hace falta promediar tanto.
        let reps = if n > 50_000 { 3 } else { 20 };
        let t_matmul = medir(reps, || {
            cosxamix::ops::matmul_auto((m, k, n), &x, &pesos, tipo, &mut dst).expect("matmul");
        });

        total_asignar += t_asignar * veces as f64;
        total_matmul += t_matmul * veces as f64;

        println!(
            "  {:<24}{:>8}{:>12}{:>12}{:>10}",
            nombre,
            veces,
            format!("{:.4}", t_asignar),
            format!("{:.3}", t_matmul),
            format!("{:.2}%", t_asignar / t_matmul * 100.0),
        );
    }

    println!("  {:-<66}", "");
    println!(
        "  {:<24}{:>8}{:>12}{:>12}{:>10}",
        "un forward entero",
        253,
        format!("{total_asignar:.3}"),
        format!("{total_matmul:.1}"),
        format!("{:.2}%", total_asignar / total_matmul * 100.0),
    );

    println!();
    println!("  Como leerlo. Si la ultima columna es un porcentaje chico, el");
    println!("  buffer reutilizable no se escribe: seria complejidad en la API");
    println!("  —cada `matmul` pasaria a recibir un `&mut Espacio`— a cambio de");
    println!("  nada medible, y por la regla 3 se borraria igual. Si es grande,");
    println!("  se escribe.");
    println!();
}
