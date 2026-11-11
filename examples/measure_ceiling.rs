//! ¿Dónde se pierde la distancia entre el decode y el techo de la memoria?
//!
//! Generar un token es leer todos los pesos una vez, así que el techo es lo que
//! la RAM entrega. Acá se mide en cuatro escalones, cada uno sobre el anterior,
//! y lo que se pierde entre dos escalones tiene un solo culpable:
//!
//! 1. **la RAM**: un búfer de 1 GB leído en paralelo, sin nada más;
//! 2. **los pesos de un tirón**: todos los bytes del modelo, mapeados, en una
//!    sola pasada paralela. Contra 1, lo que cuesta leerlos del mapeo;
//! 3. **los pesos por grupos**: lo mismo, pero con una pasada paralela por cada
//!    llamada que hace el forward (q·k·v juntas, la salida de la atención, gate·up
//!    juntas, down y los logits). Contra 2, **lo que cuesta sincronizar los
//!    hilos**, porque los bytes son los mismos;
//! 4. **el forward de verdad**. Contra 3, el cálculo de los kernels y todo lo que
//!    no es multiplicar.
//!
//! Cada escalón se mide con 1, 2, 4, 6 y 12 hilos, en un pool propio: así se ve
//! también cuál deja de escalar.
//!
//!     COSXAMIX_GGUF=qwen2.5-3b-instruct-q4_k_m.gguf cargo run --release --features candle --example measure_ceiling

use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;

use cosxamix::forward::{Cache, Modelo};
use cosxamix::gguf::{Gguf, Pesos};

/// Suma los bytes de a `u64`, para que leerlos no se pueda quitar. Cuatro
/// acumuladores: lo que limita es la RAM, no la suma.
fn sumar(b: &[u8]) -> u64 {
    // SAFETY: `u64` acepta cualquier patrón de bits; `align_to` deja afuera lo
    // que no está alineado, y eso se suma aparte.
    let (antes, medio, despues) = unsafe { b.align_to::<u64>() };
    let (grupos, resto) = medio.as_chunks::<4>();
    let mut acc = [0u64; 4];
    for g in grupos {
        for (a, v) in acc.iter_mut().zip(g) {
            *a = a.wrapping_add(*v);
        }
    }
    let sueltos = antes
        .iter()
        .chain(despues)
        .fold(0u64, |s, x| s.wrapping_add(u64::from(*x)));
    resto
        .iter()
        .fold(sueltos, |s, x| s.wrapping_add(*x))
        .wrapping_add(acc.iter().fold(0, |s, x| s.wrapping_add(*x)))
}

/// Lee varias regiones en **una** pasada paralela, en trozos de a lo sumo 256 KB.
fn leer_juntas(regiones: &[&[u8]]) -> u64 {
    let trozos: Vec<&[u8]> = regiones.iter().flat_map(|r| r.chunks(256 << 10)).collect();
    trozos
        .par_iter()
        .map(|t| sumar(t))
        .reduce(|| 0, u64::wrapping_add)
}

/// El mejor de `n`: lo que se quiere es lo que el hardware puede, no las pausas.
fn mejor_de<T>(n: usize, mut f: impl FnMut() -> T) -> f64 {
    (0..n)
        .map(|_| {
            let t = Instant::now();
            std::hint::black_box(f());
            t.elapsed().as_secs_f64()
        })
        .fold(f64::MAX, f64::min)
}

/// A qué llamada del forward pertenece una matriz: las que el forward junta en
/// una pasada comparten grupo.
fn grupo(nombre: &str) -> String {
    let Some(resto) = nombre.strip_prefix("blk.") else {
        return nombre.to_string();
    };
    let (capa, peso) = resto.split_once('.').unwrap_or((resto, ""));
    let llamada = match peso.split('.').next().unwrap_or("") {
        "attn_q" | "attn_k" | "attn_v" => "qkv",
        "ffn_gate" | "ffn_up" => "gate_up",
        otro => otro,
    };
    // El orden del forward dentro de la capa.
    let orden = match llamada {
        "qkv" => 0,
        "attn_output" => 1,
        "gate_up" => 2,
        _ => 3,
    };
    format!("{capa:0>4}.{orden}.{llamada}")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo =
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q4_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    let g = Gguf::abrir(&ruta)?;
    let mapa = Pesos::abrir(&g, &ruta)?;

    // Las matrices que el decode lee enteras: todo lo de dos dimensiones menos los
    // embeddings, de los que se lee una fila por token.
    let mut grupos: BTreeMap<String, Vec<&[u8]>> = BTreeMap::new();
    for info in &g.tensores {
        if info.dims.len() == 2 && info.dims[1] > 1 && info.nombre != "token_embd.weight" {
            grupos
                .entry(grupo(&info.nombre))
                .or_default()
                .push(mapa.de(info)?);
        }
    }
    let todas: Vec<&[u8]> = grupos.values().flatten().copied().collect();
    let bytes: usize = todas.iter().map(|r| r.len()).sum();
    // Traer el archivo del disco antes de medir nada.
    std::hint::black_box(leer_juntas(&todas));

    let ram: Vec<u8> = (0..1usize << 30).map(|i| (i * 31) as u8).collect();

    let modelo = Modelo::abrir(&ruta)?;
    println!();
    println!(
        "  {archivo}: {:.2} GB en {} matrices, {} llamadas paralelas por token",
        bytes as f64 / 1e9,
        todas.len(),
        grupos.len()
    );
    println!();
    println!(
        "  {:>5}{:>12}{:>14}{:>14}{:>12}{:>10}",
        "hilos", "RAM GB/s", "un tirón", "por grupos", "forward", "tok/s"
    );

    // `COSXAMIX_HILOS=6` mide solo con esos, para alternar muchas veces rápido.
    let lista: Vec<usize> = std::env::var("COSXAMIX_HILOS")
        .ok()
        .map(|v| v.split(',').filter_map(|h| h.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 6, 12]);
    for hilos in lista {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(hilos).build()?;
        let (r, t1, t2, tf) = pool.install(|| -> Result<_, String> {
            let r = mejor_de(3, || leer_juntas(&[&ram]));
            let t1 = mejor_de(5, || leer_juntas(&todas));
            let t2 = mejor_de(5, || {
                grupos
                    .values()
                    .map(|regiones| leer_juntas(regiones))
                    .fold(0u64, u64::wrapping_add)
            });
            // El forward: un prompt corto y después tokens de a uno, la mediana.
            let mut cache = Cache::nuevo(&modelo.cfg);
            modelo.forward(&[9707, 11, 1246], &mut cache)?;
            let mut ts = Vec::new();
            for i in 0..16u32 {
                let t = Instant::now();
                modelo.forward(&[100 + i], &mut cache)?;
                ts.push(t.elapsed().as_secs_f64());
            }
            ts.sort_by(f64::total_cmp);
            Ok((r, t1, t2, ts[ts.len() / 2]))
        })?;
        let gbs = |t: f64, b: usize| b as f64 / t / 1e9;
        println!(
            "  {hilos:>5}{:>12.1}{:>14.1}{:>14.1}{:>12.1}{:>10.1}",
            gbs(r, ram.len()),
            gbs(t1, bytes),
            gbs(t2, bytes),
            gbs(tf, bytes),
            1.0 / tf
        );
    }
    // `COSXAMIX_EN_CACHE=1`: el techo de cálculo de cada kernel. ~1 MB de filas
    // de una matriz de cada forma —entra en la L3 de 12 MB, y con 64 filas el
    // costo fijo de cada llamada tapaba al kernel—, con un hilo, muchas veces: lo
    // que el kernel procesa cuando la RAM no lo frena. Si se acerca a lo que un
    // hilo lee de la RAM, ese formato está limitado por cálculo.
    if std::env::var_os("COSXAMIX_EN_CACHE").is_some() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
        let mut vistos = std::collections::BTreeSet::new();
        println!();
        for info in &g.tensores {
            if info.dims.len() != 2 || info.dims[1] < 64 || info.nombre == "token_embd.weight" {
                continue;
            }
            let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
            if !vistos.insert((format!("{:?}", info.tipo), k)) {
                continue;
            }
            let todo = mapa.de(info)?;
            let filas = ((1 << 20) / (todo.len() / n)).clamp(64, n);
            let pesos = &todo[..todo.len() / n * filas];
            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.37).sin()).collect();
            let mut d = vec![0f32; filas];
            let vueltas = 40;
            let t = pool.install(|| {
                mejor_de(3, || {
                    for _ in 0..vueltas {
                        cosxamix::ops::matmul_auto((1, k, filas), &x, pesos, info.tipo, &mut d)
                            .expect("matmul");
                    }
                })
            });
            println!(
                "  en caché, 1 hilo: {:?} con k = {k}: {:.1} GB/s",
                info.tipo,
                (pesos.len() * vueltas) as f64 / t / 1e9
            );
        }
    }

    // `COSXAMIX_PREFILL=256`: además, leer un prompt de ese largo, el mejor de 3.
    if let Some(n) = std::env::var("COSXAMIX_PREFILL")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        let prompt: Vec<u32> = (0..n).map(|i| 100 + i * 7 % 5000).collect();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(6).build()?;
        let t = pool.install(|| {
            mejor_de(3, || {
                let mut cache = Cache::nuevo(&modelo.cfg);
                modelo.forward(&prompt, &mut cache).map(|l| l.len())
            })
        });
        println!("  prefill de {n} tokens, 6 hilos: {:.0} ms", t * 1e3);
    }
    println!();
    println!("  Todo en GB/s de pesos leídos: el forward lee los mismos bytes por token.");
    println!("  un tirón contra RAM: el mapeo. por grupos contra un tirón: sincronizar.");
    println!("  forward contra por grupos: el cálculo y lo que no es multiplicar.");
    Ok(())
}
