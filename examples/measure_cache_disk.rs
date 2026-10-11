//! ¿Vale la pena guardar el caché de atención en disco?
//!
//! La idea suena bien: si al reiniciar el servidor el caché siguiera ahí, la
//! primera respuesta de cada conversación no tendría que reprocesar el
//! *system prompt*. Pero el proyecto acaba de borrar dos kernels por no
//! superar a la referencia, así que saltarse la medición acá sería
//! incoherente.
//!
//! Son tres números y una pregunta:
//!
//!   1. cuánto pesa el caché de verdad;
//!   2. cuánto cuesta escribirlo y volver a leerlo;
//!   3. cuánto cuesta recalcularlo, que es la alternativa de no hacer nada.
//!
//! Si (2) no es mucho menor que (3), no hay nada que ganar y hay un formato en
//! disco que mantener, con el riesgo de restaurar un caché contra otro modelo
//! — que no da error, da basura.
//!
//!     cargo run --release --features candle --example measure_cache_disk

use std::path::Path;
use std::time::Instant;

use cosxamix::engine_candle::MotorCandle;

fn mb(b: usize) -> f64 {
    b as f64 / 1_048_576.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let casos = [
        (
            "0,5B",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "tokenizer.json",
        ),
        (
            "3B Q3",
            "qwen2.5-3b-instruct-q3_k_m.gguf",
            "tokenizer-3b.json",
        ),
    ];

    println!();
    println!("  Lo que pesa un caché de atención, y lo que cuesta no tenerlo");
    println!();
    println!(
        "  {:<8}{:>8}{:>12}{:>14}{:>14}{:>12}",
        "modelo", "tokens", "cache MB", "KB por token", "prefill ms", "16 sesiones"
    );
    println!("  {:-<70}", "");

    for (etiqueta, gguf, tok) in casos {
        let rg = Path::new(&dir).join(gguf);
        let rt = Path::new(&dir).join(tok);
        if !rg.exists() {
            println!("  {etiqueta:<8}  (no está {gguf})");
            continue;
        }

        let m = MotorCandle::nuevo(etiqueta, &rg, &rt)?;
        m.precalentar()?;

        // Un prompt del tamaño real de un system prompt de WinsiBot: 131
        // tokens, que es lo que se midio contra el bot de verdad.
        let texto = "Eres un asistente. ".repeat(40);
        let ids = m.tokenizar(&texto)?;
        let n = ids.len();

        m.reiniciar()?;
        let t0 = Instant::now();
        m.evaluar(&ids, 0)?;
        let prefill = t0.elapsed().as_secs_f64() * 1000.0;

        let bytes = m.bytes_de_cache();

        println!(
            "  {:<8}{:>8}{:>12}{:>14}{:>14}{:>12}",
            etiqueta,
            n,
            format!("{:.1}", mb(bytes)),
            format!("{:.1}", bytes as f64 / n as f64 / 1024.0),
            format!("{prefill:.0}"),
            format!("{:.1} GB", mb(bytes) * 16.0 * 1000.0 / n as f64 / 1024.0)
        );
    }

    println!();
    println!("  La última columna proyecta 16 sesiones de 1000 tokens, que es");
    println!("  el tope que el motor permite hoy.");
    println!();
    println!("  Para que persistir valga la pena, escribir y leer ese caché");
    println!("  tiene que costar bastante menos que el prefill. Un SSD hace");
    println!("  del orden de 1-2 GB/s, asi que la cuenta se puede hacer con");
    println!("  los MB de la tabla antes de escribir una linea de codigo.");

    Ok(())
}
