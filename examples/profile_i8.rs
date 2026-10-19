//! Donde se van los milisegundos de `matmul_i8_directo`, por etapas.
//!
//! Hace falta porque la ventaja medida sobre Q4_K **no se sostuvo** en Q3_K:
//! 2,10x a favor sobre uno y 0,88x en contra sobre el otro, con una matriz mas
//! chica. Hay dos diferencias entre los dos casos y adivinar cual pesa seria
//! justo el error que este proyecto no comete:
//!
//!   1. **el desempaquetado**: Q3_K guarda 3 bits en dos piezas, una mascara de
//!      bits alta y dos bits bajos corridos, y rearma sus escalas de 6 bits
//!      desde cuatro palabras de 32; Q4_K es medio byte por valor y nada mas;
//!   2. **el sub-bloque**: 16 en Q3_K contra 32 en Q4_K, o sea el doble de
//!      escalas, y el camino de 16 del kernel arma su vector de escalas con un
//!      `setr_ps` de dos valores sueltos en cada paso.
//!
//! Se mide cada etapa por separado sobre los dos formatos, con el MISMO numero
//! de valores, que es lo unico que hace comparables los dos numeros.
//!
//!     cargo run --release --example profile_i8

use std::time::Instant;

use cosxamix::gguf::{Gguf, Pesos};
use cosxamix::quant::{desempaquetar_i8, sub_bloque};

fn mediana(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Mide una etapa varias veces y devuelve la mediana en milisegundos.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());

    println!();
    println!(
        "  {:<8}{:>8}{:>10}{:>12}{:>12}{:>12}",
        "formato", "sub", "MB", "desempaq.", "kernel", "total m=1"
    );
    println!("  {:-<64}", "");

    for archivo in [
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "qwen2.5-3b-instruct-q3_k_m.gguf",
        "qwen2.5-3b-instruct-q2_k.gguf",
    ] {
        let ruta = std::path::Path::new(&dir).join(archivo);
        if !ruta.exists() {
            println!("  falta {}", ruta.display());
            continue;
        }
        let g = Gguf::abrir(&ruta)?;
        let info = g.tensor("blk.0.ffn_gate.weight").ok_or("sin ffn_gate")?;
        let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
        // Mapeado y sin copias. `datos_de` reabria el archivo por cada tensor y
        // copiaba sus 12 MB al monticulo; medido, para releer el mismo tensor
        // miles de veces el mapeo y la copia dan lo mismo dentro del ruido
        // —1,17x, 0,93x y 1,04x en tres corridas—, asi que la copia no se
        // justifica. Ver DECISIONES 18.
        let mapa = Pesos::abrir(&g, &ruta)?;
        let pesos: &[u8] = mapa.de(info)?;
        let tipo = info.tipo;
        let sub = sub_bloque(tipo).ok_or("sin sub-bloque")?;
        let bytes_fila = k / tipo.valores_por_bloque() * tipo.bytes_por_bloque();

        // ── etapa 1: desempaquetar las n columnas, en un solo hilo ──────────
        let mut q = vec![0i8; k];
        let mut esc = vec![0f32; k / sub];
        let mut des = vec![0f32; k / sub];
        let t_desempaq = medir(3, || {
            for col in 0..n {
                let d = col * bytes_fila;
                desempaquetar_i8(tipo, &pesos[d..d + bytes_fila], &mut q, &mut esc, &mut des)
                    .expect("desempaquetar");
            }
        });

        // ── etapa 2: solo el kernel, n veces, sobre datos ya desempaquetados ─
        let xq: Vec<i8> = (0..k).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let xe: Vec<f32> = (0..k / sub).map(|i| 0.01 + i as f32 * 1e-5).collect();
        let t_kernel = medir(3, || {
            let mut acc = 0f32;
            for _ in 0..n {
                acc += cosxamix::kernels::dot_i8_escalado(&q, &xq, &esc, &xe, sub);
            }
            std::hint::black_box(acc);
        });

        // ── el total de verdad, con hilos ───────────────────────────────────
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.007).cos()).collect();
        let mut dst = vec![0f32; n];
        let t_total = medir(20, || {
            cosxamix::ops::matmul_i8_directo((1, k, n), &x, pesos, tipo, &mut dst)
                .expect("directo");
        });

        println!(
            "  {:<8}{:>8}{:>10}{:>12}{:>12}{:>12}",
            format!("{tipo:?}"),
            sub,
            format!("{:.1}", pesos.len() as f64 / 1048576.0),
            format!("{t_desempaq:.3}"),
            format!("{t_kernel:.3}"),
            format!("{t_total:.3}"),
        );
    }

    println!();
    println!("  Las dos columnas del medio son de UN hilo y la ultima de doce,");
    println!("  asi que no suman: sirven para comparar formato contra formato.");
    println!("  Si el desempaquetado de Q3_K dobla al de Q4_K, el problema esta");
    println!("  en los bits; si lo que dobla es el kernel, esta en el sub-bloque");
    println!("  de 16 — y son dos arreglos distintos.");
    println!();

    Ok(())
}
