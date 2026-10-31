//! La matmul por bloques contra lo que elegía `matmul_auto` antes de ella
//! —el fusionado en Q4_K, el fusionado o el de buffer en Q3_K, el de buffer
//! en los demás—, en el prefill.
//!
//! Sobre los tensores de verdad del 3B —las tres formas que hay: 2048 → 2048,
//! 2048 → 11008 y 11008 → 2048— y con los lotes del bot: 18 filas (un
//! mensaje con el *system* en caché) y 333 (el primer mensaje entero).
//!
//! Las dos hacen las mismas cuentas en otro orden, así que tienen que dar lo
//! mismo salvo el redondeo de sumar en otro orden: se imprime la diferencia
//! relativa más grande.
//!
//!     cargo run --release --example measure_blocked_matmul

use std::time::Instant;

use cosxamix::gguf::{Gguf, Pesos, TipoGgml};

fn medir(mut f: impl FnMut()) -> f64 {
    f();
    let mut v: Vec<f64> = (0..5)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[2]
}

/// La elección de `matmul_auto` antes de la matmul por bloques.
fn antes(
    (m, k, n): (usize, usize, usize),
    x: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    d: &mut [f32],
) -> Result<(), String> {
    let fusionable = matches!(tipo, TipoGgml::Q4K | TipoGgml::Q3K) && k % 256 == 0;
    let tope = if tipo == TipoGgml::Q4K { usize::MAX } else { 4 };
    if fusionable && m <= tope {
        cosxamix::ops::matmul_i8_fusionado((m, k, n), x, pesos, tipo, d)
    } else {
        cosxamix::ops::matmul_i8_directo((m, k, n), x, pesos, tipo, d)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo =
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q3_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    if !ruta.exists() {
        println!("  falta {}", ruta.display());
        return Ok(());
    }
    let g = Gguf::abrir(&ruta)?;
    let mapa = Pesos::abrir(&g, &ruta)?;

    println!();
    println!("  {archivo}");
    println!();
    println!(
        "  {:<22}{:>7}{:>6}{:>12}{:>12}{:>9}{:>12}",
        "tensor", "tipo", "filas", "antes ms", "bloques ms", "mejora", "dif máx"
    );
    println!("  {:-<80}", "");

    // Las matrices de las dos primeras capas: en el Q3_K_M no todas van en
    // Q3_K —algunas van en Q4_K, Q5_K o Q6_K—, y cada formato tiene hoy su
    // camino en `matmul_auto`.
    let nombres: Vec<String> = g
        .tensores
        .iter()
        .filter(|t| {
            (t.nombre.starts_with("blk.0.") || t.nombre.starts_with("blk.1."))
                && t.dims.len() == 2
                && t.dims[1] > 1
        })
        .map(|t| t.nombre.clone())
        .collect();
    // `COSXAMIX_FILAS=8,32,64` mide otros lotes, para elegir el cruce.
    let lotes: Vec<usize> = std::env::var("COSXAMIX_FILAS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![18, 333]);
    let detalle = std::env::var("COSXAMIX_FILAS").is_err();
    let (mut total_auto, mut total_blq) = (vec![0f64; lotes.len()], vec![0f64; lotes.len()]);
    for nombre in &nombres {
        let info = g.tensor(nombre).ok_or(nombre.as_str())?;
        let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
        let pesos = mapa.de(info)?;
        for (i, &filas) in lotes.iter().enumerate() {
            let x: Vec<f32> = (0..filas * k)
                .map(|i| ((i as f32 * 0.0137).sin() + (i % 7) as f32 * 0.03) * 0.8)
                .collect();
            let mut a = vec![0f32; filas * n];
            let mut b = vec![0f32; filas * n];
            let t_dir =
                medir(|| antes((filas, k, n), &x, pesos, info.tipo, &mut a).expect("antes"));
            let t_blq = medir(|| {
                cosxamix::ops::matmul_i8_bloques((filas, k, n), &x, pesos, info.tipo, &mut b)
                    .expect("bloques")
            });
            total_auto[i] += t_dir;
            total_blq[i] += t_blq;
            let escala = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
            let dif = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (p, q)| m.max((p - q).abs()))
                / escala;
            if !detalle {
                continue;
            }
            println!(
                "  {:<22}{:>7}{:>6}{:>12.2}{:>12.2}{:>8.2}x{:>12.1e}",
                format!("{} {k}→{n}", nombre.trim_end_matches(".weight")),
                format!("{:?}", info.tipo),
                filas,
                t_dir,
                t_blq,
                t_dir / t_blq,
                dif
            );
        }
    }
    println!("  {:-<80}", "");
    for (i, filas) in lotes.iter().enumerate() {
        println!(
            "  {:<35}{:>6}{:>12.2}{:>12.2}{:>8.2}x",
            "dos capas enteras",
            filas,
            total_auto[i],
            total_blq[i],
            total_auto[i] / total_blq[i]
        );
    }
    Ok(())
}
