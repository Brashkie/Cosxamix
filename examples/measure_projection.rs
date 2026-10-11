//! ¿Por qué evaluar 5 posiciones cuesta 2,25x evaluar una?
//!
//! La decodificación especulativa parte de que la pasada del modelo está
//! dominada por leer los pesos de atención y MLP, y que calcular los logits de
//! una posición más es barato. Si eso no se cumple, la técnica no rinde.
//!
//! La sospecha: la matriz de salida es enorme porque el vocabulario lo es
//! —151936 tokens— y proyectar sobre ella N veces no es gratis.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use cosxamix::engine_candle::MotorCandle;

fn mb(b: usize) -> f64 {
    b as f64 / 1048576.0
}

fn main() {
    let ruta = "modelos/qwen2.5-3b-instruct-q3_k_m.gguf";

    // ── Cuánto del modelo es la capa de salida ─────────────────────────────
    let mut f = File::open(ruta).expect("abrir");
    let c = gguf_file::Content::read(&mut f).expect("leer");

    let mut total = 0usize;
    let mut salida = 0usize;
    for (nombre, info) in &c.tensor_infos {
        let elems: usize = info.shape.dims().iter().product();
        let bytes = elems * info.ggml_dtype.type_size() / info.ggml_dtype.block_size();
        total += bytes;
        if nombre == "output.weight" {
            salida = bytes;
        }
    }

    println!("\n  modelo completo    {:>8.1} MB", mb(total));
    println!(
        "  capa de salida     {:>8.1} MB   ({:.0}% del total)",
        mb(salida),
        salida as f64 / total as f64 * 100.0
    );
    println!("  resto (atencion+mlp) {:>6.1} MB", mb(total - salida));

    // ── Cuánto cuesta evaluar N posiciones ─────────────────────────────────
    let m = MotorCandle::nuevo(
        "3b",
        Path::new(ruta),
        Path::new("modelos/tokenizer-3b.json"),
    )
    .expect("construir");
    m.precalentar().expect("precalentar");

    let base: Vec<u32> = (100u32..140).collect();
    let _ = m.evaluar(&base, 0);

    println!("\n  posiciones | ms por pasada | vs 1 posicion");
    println!("  -----------|---------------|---------------");

    let mut ref_ms = 0.0f64;
    for n in [1usize, 2, 4, 5, 8] {
        let bloque: Vec<u32> = (200u32..200 + n as u32).collect();

        // Tres repeticiones, la mediana. Se reinicia para que el cache no
        // acumule entre medidas y cambie el largo de la atencion.
        let mut t = Vec::new();
        for _ in 0..3 {
            m.reiniciar().expect("reiniciar");
            let _ = m.evaluar(&base, 0).expect("prefill");
            let t0 = Instant::now();
            m.evaluar(&bloque, base.len()).expect("evaluar");
            t.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = t[1];
        if n == 1 {
            ref_ms = ms;
        }
        println!("  {n:>10} | {ms:>13.0} | {:>13.2}x", ms / ref_ms);
    }

    println!();
    println!("  Si proyectar sobre el vocabulario fuera gratis, la columna de la");
    println!("  derecha seria ~1,0x en todas: la atencion y el MLP leen los mismos");
    println!("  pesos una sola vez sea cual sea el largo del bloque.");
}
