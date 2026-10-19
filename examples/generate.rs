//! Genera texto con el motor propio y lo compara con candle.
//!
//! Es la primera vez que el motor propio produce texto legible y no solo
//! logits: tokeniza un prompt de chat con el mismo formato que usa el bot,
//! genera con el muestreador propio, y decodifica.
//!
//! Se compara contra candle **en el mismo proceso** y en modo voraz, que es el
//! único donde las dos secuencias tienen que coincidir: con temperatura cada
//! una usa su propio generador de números y divergen desde el primer token,
//! sin que eso diga nada sobre si alguna está mal.
//!
//! El tokenizador sigue siendo el de Qwen —el crate `tokenizers`, que no es
//! candle aunque viaje con la misma feature— porque estos modelos se
//! entrenaron con él. El de un daiss con vocabulario propio es `Bpe`.
//!
//!     cargo run --release --features candle --example generate

use std::time::Instant;

use cosxamix::forward::{Cache, Modelo};
use cosxamix::sampler::Muestreador;

/// El formato de chat de Qwen2.5, el mismo que arma el motor de candle.
fn formatear(sistema: &str, usuario: &str) -> String {
    format!(
        "<|im_start|>system\n{sistema}<|im_end|>\n\
         <|im_start|>user\n{usuario}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo = std::env::var("COSXAMIX_GGUF")
        .unwrap_or_else(|_| "qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    let ruta_tok = std::path::Path::new(&dir)
        .join(std::env::var("COSXAMIX_TOKENIZER").unwrap_or_else(|_| "tokenizer.json".into()));
    for r in [&ruta, &ruta_tok] {
        if !r.exists() {
            println!("  falta {}", r.display());
            return Ok(());
        }
    }
    let max: usize = std::env::var("COSXAMIX_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);

    let tok = tokenizers::Tokenizer::from_file(&ruta_tok).map_err(|e| e.to_string())?;
    let prompt = formatear(
        "Sos un asistente que responde en español, breve y claro.",
        "¿Qué es un número primo? Dame un ejemplo.",
    );
    let ids: Vec<u32> = tok
        .encode(prompt, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();

    println!();
    println!(
        "  {archivo} — prompt de {} tokens, hasta {max} nuevos",
        ids.len()
    );

    // ── el motor propio ─────────────────────────────────────────────────────
    let modelo = Modelo::abrir(&ruta)?;
    let mut cache = Cache::nuevo(&modelo.cfg);
    let mut m = Muestreador::voraz();

    // Calentar ANTES de medir. El mapeo trae los pesos del disco en el primer
    // toque, asi que sin esto la primera llamada paga leer el modelo entero
    // —0,45 GB a los 41 MB/s del disco F:, unos 11 s— mientras candle lo paga
    // en `from_gguf`, fuera de su cronometro. Medido sin calentar daba un
    // prefill 9x peor que candle, y era el disco: los dos lo pagan, en sitios
    // distintos.
    let t0 = Instant::now();
    modelo.generar(&ids, 1, &mut m, &mut cache, |_| true)?;
    let t_frio = t0.elapsed().as_secs_f64();
    cache.limpiar();
    println!(
        "  arranque en frio del motor propio, con el disco: {:.1} s",
        t_frio
    );

    let t0 = Instant::now();
    let mut primer = None;
    let nuestros = modelo.generar(&ids, max, &mut m, &mut cache, |_| {
        primer.get_or_insert_with(|| t0.elapsed().as_secs_f64());
        true
    })?;
    let t_nuestro = t0.elapsed().as_secs_f64();
    let t_prefill = primer.unwrap_or(t_nuestro);

    // ── candle, el mismo prompt, voraz ──────────────────────────────────────
    let dev = candle_core::Device::Cpu;
    let mut f = std::fs::File::open(&ruta)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut f)?;
    let mut suyo = cosxamix::model::qwen2::ModelWeights::from_gguf(ct, &mut f, &dev)?;
    let fin = modelo.fin();

    // El mismo calentamiento para candle: su primera llamada tambien paga
    // cosas que no se repiten, y comparar uno caliente contra uno frio es el
    // error que DECISIONES §15 prohibe.
    {
        let e = candle_core::Tensor::from_vec(ids.clone(), (1, ids.len()), &dev)?;
        suyo.forward(&e, 0)?;
        suyo.limpiar_cache();
    }

    let t0 = Instant::now();
    let mut suyos = Vec::new();
    let mut entrada = ids.clone();
    let mut pos = 0usize;
    let mut t_prefill_suyo = 0f64;
    for _ in 0..max {
        let e = candle_core::Tensor::from_vec(entrada.clone(), (1, entrada.len()), &dev)?;
        let logits = suyo.forward(&e, pos)?.flatten_all()?.to_vec1::<f32>()?;
        if pos == 0 {
            t_prefill_suyo = t0.elapsed().as_secs_f64();
        }
        pos += entrada.len();
        let t = cosxamix::sampler::argmax(&logits);
        if Some(t) == fin {
            break;
        }
        suyos.push(t);
        entrada = vec![t];
    }
    let t_suyo = t0.elapsed().as_secs_f64();

    // ── el texto ────────────────────────────────────────────────────────────
    let texto = tok.decode(&nuestros, true).map_err(|e| e.to_string())?;
    println!();
    println!("  Respuesta del motor propio:");
    println!();
    for linea in texto.lines() {
        println!("    {linea}");
    }

    // ── la comparacion ──────────────────────────────────────────────────────
    let decode = |total: f64, prefill: f64, n: usize| {
        if n > 1 {
            (n - 1) as f64 / (total - prefill)
        } else {
            0.0
        }
    };
    println!();
    println!(
        "  {:<14}{:>10}{:>12}{:>14}{:>12}",
        "motor", "tokens", "prefill", "decode", "total"
    );
    println!("  {:-<62}", "");
    for (nombre, n, total, prefill) in [
        ("propio", nuestros.len(), t_nuestro, t_prefill),
        ("candle", suyos.len(), t_suyo, t_prefill_suyo),
    ] {
        println!(
            "  {:<14}{:>10}{:>12}{:>14}{:>12}",
            nombre,
            n,
            format!("{:.0} ms", prefill * 1000.0),
            format!("{:.1} tok/s", decode(total, prefill, n)),
            format!("{:.2} s", total),
        );
    }

    let iguales = nuestros
        .iter()
        .zip(&suyos)
        .take_while(|(a, b)| a == b)
        .count();
    println!();
    if iguales == nuestros.len() && nuestros.len() == suyos.len() {
        println!("  Las dos secuencias son IDÉNTICAS: {} tokens.", iguales);
    } else {
        println!(
            "  Coinciden los primeros {iguales} tokens de {} y {}.",
            nuestros.len(),
            suyos.len()
        );
        println!("  En un modelo cuantizado eso es esperable: las dos cuantizan la");
        println!("  entrada distinto y un empate cercano cae de otro lado. Sobre el");
        println!("  modelo en F32 coinciden enteras — ver compare_forward.");
    }
    println!();

    Ok(())
}
