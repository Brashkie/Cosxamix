//! Dónde se va el tiempo de un token del decode, por formato de pesos.
//!
//! El decode multiplica un solo vector por cada matriz del modelo, así que lo
//! que lo limita es traer los pesos de la RAM. Acá se recorren todas las
//! matrices en el orden de un forward —para que vengan de la RAM y no de la
//! caché, como en un token de verdad— con `matmul_auto` y una fila, y se
//! suma el tiempo y los bytes por formato y forma. Un formato que va a menos GB/s que
//! los demás tiene el kernel flojo; la diferencia entre la suma y lo que
//! tarda un forward de verdad es lo que no es multiplicación.
//!
//! **Sirve para encontrar candidatos, no para confirmarlos.** Suelta, una
//! matmul no compite por la RAM igual que dentro del forward: la de salida por
//! el camino con buffer daba 19,7 ms acá y 11,3 dentro del forward. Lo que se
//! cambie se confirma con la línea "un forward de un token", alternando con
//! y sin el cambio.
//!
//!     cargo run --release --features candle --example measure_decode

use std::collections::BTreeMap;
use std::time::Instant;

use cosxamix::forward::{Cache, Modelo};
use cosxamix::gguf::{Gguf, Pesos};

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

    // Las matrices en el orden del archivo, que es el de las capas. La de
    // embeddings no se multiplica: se leen solo las filas de los tokens.
    let matrices: Vec<_> = g
        .tensores
        .iter()
        .filter(|t| t.dims.len() == 2 && t.dims[1] > 1 && t.nombre != "token_embd.weight")
        .collect();

    let vueltas = 5;
    let mut por_tipo: BTreeMap<String, (f64, usize, usize)> = BTreeMap::new();
    let mut total_vuelta = Vec::new();
    for v in 0..vueltas {
        let t_vuelta = Instant::now();
        for info in &matrices {
            let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
            let pesos = mapa.de(info)?;
            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut d = vec![0f32; n];
            let t0 = Instant::now();
            cosxamix::ops::matmul_auto((1, k, n), &x, pesos, info.tipo, &mut d)?;
            let t = t0.elapsed().as_secs_f64();
            // La primera vuelta calienta el mapeo (trae el archivo del disco).
            if v > 0 {
                let e = por_tipo
                    .entry(if info.nombre == "output.weight" {
                        format!("{:?} salida", info.tipo)
                    } else {
                        format!("{:?} {k}→{n}", info.tipo)
                    })
                    .or_insert((0.0, 0, 0));
                e.0 += t;
                e.1 += pesos.len();
                e.2 += 1;
            }
        }
        if v > 0 {
            total_vuelta.push(t_vuelta.elapsed().as_secs_f64());
        }
    }
    let n = (vueltas - 1) as f64;

    // Un forward de verdad de un token, para comparar.
    let modelo = Modelo::abrir(&ruta)?;
    let mut cache = Cache::nuevo(&modelo.cfg);
    modelo.forward(&[9707, 11, 1246], &mut cache)?;
    let mut ts = Vec::new();
    for i in 0..20 {
        let t0 = Instant::now();
        modelo.forward(&[(100 + i) as u32], &mut cache)?;
        ts.push(t0.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_forward = ts[ts.len() / 2];

    println!();
    println!("  {archivo}");
    println!();
    println!(
        "  {:<20}{:>8}{:>10}{:>11}{:>9}{:>8}",
        "formato", "matrices", "MB", "ms/token", "GB/s", "% token"
    );
    println!("  {:-<62}", "");
    let mut suma = 0.0;
    for (tipo, (t, bytes, cuantas)) in &por_tipo {
        let ms = t / n * 1e3;
        suma += ms;
        println!(
            "  {:<20}{:>8}{:>10.0}{:>11.2}{:>9.1}{:>7.0}%",
            tipo,
            cuantas / (vueltas - 1),
            *bytes as f64 / n / 1e6,
            ms,
            *bytes as f64 / n / (t / n) / 1e9,
            ms / (t_forward * 1e3) * 100.0
        );
    }
    println!("  {:-<62}", "");
    let vuelta = total_vuelta.iter().sum::<f64>() / n * 1e3;
    println!("  matmuls sumadas            {suma:>8.2} ms");
    println!("  la vuelta entera, medida   {vuelta:>8.2} ms");
    println!(
        "  un forward de un token     {:>8.2} ms  ({:.1} tok/s)",
        t_forward * 1e3,
        1.0 / t_forward
    );
    println!(
        "  lo que no es matmul        {:>8.2} ms  ({:.0}%)",
        t_forward * 1e3 - suma,
        (1.0 - suma / (t_forward * 1e3)) * 100.0
    );
    Ok(())
}
