//! Nuestro cuantizador de k-quants contra el de los archivos oficiales.
//!
//! Los pesos de referencia salen del Qwen2.5 3B oficial en Q4_K, que es lo más
//! fino que hay en `modelos/`. Se cuantizan con nuestro código a Q3_K y Q2_K, y
//! se comparan con los Q3_K_M y Q2_K oficiales **del mismo modelo**: si el error
//! contra los mismos pesos queda igual o mejor, nuestro cuantizador sirve.
//!
//! Solo se comparan los tensores que el archivo oficial guarda en ese formato:
//! un `Q3_K_M` deja algunos en Q4_K a propósito.
//!
//!     cargo run --release --example measure_kquants

use std::path::Path;
use std::time::Instant;

use cosxamix::gguf::{Gguf, TipoGgml};
use cosxamix::quant::{cuantizar_bloques, descuantizar};

fn leer(g: &Gguf, ruta: &Path, nombre: &str) -> Result<(TipoGgml, Vec<f32>), String> {
    let t = g
        .tensores
        .iter()
        .find(|t| t.nombre == nombre)
        .ok_or_else(|| format!("no está {nombre}"))?;
    let datos = g.datos_de(ruta, t).map_err(|e| e.to_string())?;
    let mut y = vec![0f32; t.elementos()];
    descuantizar(t.tipo, &datos, &mut y)?;
    Ok((t.tipo, y))
}

fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let s: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum();
    (s / a.len() as f64).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = Path::new("modelos/qwen2.5-3b-instruct-q4_k_m.gguf");
    let otros = [
        (
            TipoGgml::Q3K,
            Path::new("modelos/qwen2.5-3b-instruct-q3_k_m.gguf"),
        ),
        (
            TipoGgml::Q2K,
            Path::new("modelos/qwen2.5-3b-instruct-q2_k.gguf"),
        ),
    ];
    let g = Gguf::abrir(base)?;
    let tensores = [
        "blk.0.attn_q.weight",
        "blk.7.attn_k.weight",
        "blk.7.attn_v.weight",
        "blk.15.attn_output.weight",
        "blk.15.ffn_gate.weight",
        "blk.20.ffn_up.weight",
        "blk.30.ffn_down.weight",
        "token_embd.weight",
    ];

    println!();
    println!("  error contra los pesos de Q4_K (RMSE ×1000), y ms por millón de valores");
    println!();
    println!(
        "  {:<28}{:>6}{:>11}{:>11}{:>9}{:>9}",
        "tensor", "tipo", "oficial", "nuestro", "razón", "ms/M"
    );
    for (tipo, ruta) in otros {
        let g2 = Gguf::abrir(ruta)?;
        let (mut sum_o, mut sum_n, mut n) = (0f64, 0f64, 0usize);
        for nombre in tensores {
            let (tb, w) = leer(&g, base, nombre)?;
            let (to, oficial) = leer(&g2, ruta, nombre)?;
            if to != tipo || tb != TipoGgml::Q4K {
                println!(
                    "  {nombre:<28}{to:>6?}   (el oficial no está en {tipo:?}; {tb:?} de base)"
                );
                continue;
            }
            // Las filas de a bloques: el largo es múltiplo de 256 en estos.
            let m = w.len().min(4 << 20);
            let m = m - m % 256;
            let t = Instant::now();
            let mut b = Vec::new();
            cuantizar_bloques(tipo, &w[..m], &mut b)?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / (m as f64 / 1e6);
            let mut nuestro = vec![0f32; m];
            descuantizar(tipo, &b, &mut nuestro)?;
            let (eo, en) = (rmse(&w[..m], &oficial[..m]), rmse(&w[..m], &nuestro));
            println!(
                "  {nombre:<28}{to:>6?}{:>11.3}{:>11.3}{:>9.3}{ms:>9.0}",
                eo * 1e3,
                en * 1e3,
                en / eo
            );
            sum_o += eo * eo * m as f64;
            sum_n += en * en * m as f64;
            n += m;
        }
        if n > 0 {
            println!(
                "  {:<28}{:>6?}{:>11.3}{:>11.3}{:>9.3}",
                "total",
                tipo,
                (sum_o / n as f64).sqrt() * 1e3,
                (sum_n / n as f64).sqrt() * 1e3,
                (sum_n / sum_o).sqrt()
            );
        }
        println!();
    }
    Ok(())
}
