//! Tres formas de llegar a los pesos, medidas antes de elegir una.
//!
//! El camino de hoy, `Gguf::datos_de`, abre el archivo **de nuevo por cada
//! tensor** y copia sus bytes a un `Vec`. En el daiss-7b son 339 aperturas y
//! 7,5 GB copiados al montículo. Hay dos arreglos posibles y conviene saber
//! cuánto aporta cada uno antes de quedarse con el que suma una dependencia:
//!
//!   1. **reabriendo**, que es lo que hace el código hoy;
//!   2. **un solo descriptor**, reusando el archivo abierto — gratis, sin
//!      dependencias;
//!   3. **mapeado**, con `Pesos`, que además no copia nada.
//!
//! Lo que se mide es llegar a todos los bytes de todos los tensores y sumarlos,
//! porque sumarlos obliga a tocarlos de verdad: sin eso el mapeo "ganaría" por
//! no haber leído nada.
//!
//! Hay que decir qué NO mide esto: el disco de esta máquina hace 41 MB/s y la
//! primera lectura de un archivo lo paga igual por los tres caminos. Lo que se
//! compara es lo que pasa cuando el archivo ya está en la caché del sistema
//! operativo, que es la situación de un servidor que ya sirvió una petición —o
//! sea, la normal.
//!
//!     cargo run --release --example measure_mmap

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use cosxamix::gguf::{Gguf, InfoTensor, Pesos};

fn mediana(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Suma los bytes, que es la forma más barata de obligar a tocarlos todos.
fn sumar(b: &[u8]) -> u64 {
    b.iter().map(|v| *v as u64).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo =
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q4_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    if !ruta.exists() {
        println!("  falta {}", ruta.display());
        return Ok(());
    }

    let g = Gguf::abrir(&ruta)?;
    let total: usize = g.tensores.iter().map(|t| t.bytes()).sum();
    println!();
    println!(
        "  {archivo} — {} tensores, {:.2} GB",
        g.tensores.len(),
        total as f64 / 1_073_741_824.0
    );
    println!();

    // Una pasada previa para que el archivo esté en la caché del sistema: se
    // compara el camino caliente, y sin esto el primero en correr pagaría el
    // disco y los otros dos no.
    {
        let p = Pesos::abrir(&g, &ruta)?;
        let mut s = 0u64;
        for t in &g.tensores {
            s += sumar(p.de(t)?);
        }
        std::hint::black_box(s);
    }

    let medir = |n: usize, mut f: Box<dyn FnMut() -> Result<u64, String> + '_>| -> (f64, u64) {
        let mut suma = 0u64;
        let t = mediana(
            (0..n)
                .map(|_| {
                    let t0 = Instant::now();
                    suma = f().expect("camino");
                    t0.elapsed().as_secs_f64() * 1000.0
                })
                .collect(),
        );
        (t, suma)
    };

    // ── 1. reabriendo, que es lo que hace el codigo hoy ─────────────────────
    let (t_reabrir, s1) = medir(
        3,
        Box::new(|| {
            let mut s = 0u64;
            for t in &g.tensores {
                s += sumar(&g.datos_de(&ruta, t)?);
            }
            Ok(s)
        }),
    );

    // ── 2. un solo descriptor abierto ───────────────────────────────────────
    let leer_una_vez = |f: &mut File, info: &InfoTensor, buf: &mut Vec<u8>| -> Result<(), String> {
        f.seek(SeekFrom::Start(g.inicio_datos + info.offset))
            .map_err(|e| e.to_string())?;
        buf.clear();
        buf.resize(info.bytes(), 0);
        f.read_exact(buf).map_err(|e| e.to_string())
    };
    let (t_abierto, s2) = medir(
        3,
        Box::new(|| {
            let mut f = File::open(&ruta).map_err(|e| e.to_string())?;
            let mut buf = Vec::new();
            let mut s = 0u64;
            for t in &g.tensores {
                leer_una_vez(&mut f, t, &mut buf)?;
                s += sumar(&buf);
            }
            Ok(s)
        }),
    );

    // ── 3. mapeado, sin copias ──────────────────────────────────────────────
    let (t_mapeado, s3) = medir(
        3,
        Box::new(|| {
            let p = Pesos::abrir(&g, &ruta).map_err(|e| e.to_string())?;
            let mut s = 0u64;
            for t in &g.tensores {
                s += sumar(p.de(t)?);
            }
            Ok(s)
        }),
    );

    // Los tres tienen que haber leido LO MISMO. Sin esto, un camino podria
    // "ganar" por leer de menos y el numero no diria nada.
    assert_eq!(s1, s2, "el descriptor unico leyo otra cosa");
    assert_eq!(s1, s3, "el mapeo leyo otra cosa");

    println!(
        "  {:<24}{:>12}{:>12}{:>14}",
        "camino", "ms", "GB/s", "vs hoy"
    );
    println!("  {:-<62}", "");
    for (nombre, t) in [
        ("reabriendo (hoy)", t_reabrir),
        ("un solo descriptor", t_abierto),
        ("mapeado, sin copias", t_mapeado),
    ] {
        println!(
            "  {:<24}{:>12}{:>12}{:>14}",
            nombre,
            format!("{t:.1}"),
            format!("{:.2}", total as f64 / 1_073_741_824.0 / (t / 1000.0)),
            format!("{:.2}x", t_reabrir / t),
        );
    }

    println!();
    // ── pasadas repetidas sobre UN tensor, que es lo que hace el motor ──────
    //
    // Lo de arriba es una pasada por todo. El camino caliente es otro: el
    // mismo tensor recorrido miles de veces. Ahi un `Vec` residente podria
    // ganarle al mapeo, porque no hay que recorrer 3.072 paginas por vuelta.
    //
    // Se mide EN EL MISMO PROCESO, que es la unica forma que vale en esta
    // maquina: entre corridas distintas el trabajo que recorre memoria varia
    // hasta 1,8x, asi que una comparacion entre procesos no dice nada.
    if let Some(info) = g.tensor("blk.0.ffn_gate.weight") {
        let p = Pesos::abrir(&g, &ruta)?;
        let mapeado = p.de(info)?;
        let residente = mapeado.to_vec();
        let vueltas = (256 << 20) / info.bytes().max(1);

        println!();
        println!(
            "  Pasadas repetidas sobre blk.0.ffn_gate.weight ({:.1} MB, {vueltas} vueltas)",
            info.bytes() as f64 / 1_048_576.0
        );
        println!();
        println!("  {:<24}{:>12}{:>12}", "camino", "ms/vuelta", "vs mapeado");

        let repetir = |b: &[u8]| -> f64 {
            mediana(
                (0..5)
                    .map(|_| {
                        let t0 = Instant::now();
                        let mut s = 0u64;
                        for _ in 0..vueltas {
                            s += sumar(b);
                        }
                        std::hint::black_box(s);
                        t0.elapsed().as_secs_f64() * 1000.0 / vueltas as f64
                    })
                    .collect(),
            )
        };
        // Una vuelta de calentamiento a cada uno, para que ninguno pague los
        // fallos de pagina del primer toque dentro de la medicion.
        std::hint::black_box(sumar(mapeado) + sumar(&residente));
        let t_map = repetir(mapeado);
        let t_res = repetir(&residente);
        println!("  {:-<48}", "");
        println!(
            "  {:<24}{:>12}{:>12}",
            "mapeado",
            format!("{t_map:.2}"),
            "1.00x"
        );
        println!(
            "  {:<24}{:>12}{:>12}",
            "copiado a monticulo",
            format!("{t_res:.2}"),
            format!("{:.2}x", t_map / t_res)
        );
        println!();
    }

    println!("  Las tres leyeron los mismos bytes: la suma coincide.");
    println!();
    println!(
        "  Lo que el mapeo ahorra y no aparece en el reloj: {:.2} GB de",
        total as f64 / 1_073_741_824.0
    );
    println!("  montículo. El camino de hoy copia cada tensor a un Vec propio,");
    println!("  y el mapeo devuelve una porcion que apunta al archivo — que es");
    println!("  lo que importa en una maquina con 7,3 GB libres.");
    println!();

    Ok(())
}
