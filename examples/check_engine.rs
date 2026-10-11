//! Prueba el motor real contra el GGUF de `modelos/`, midiendo frío y caliente.
use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_candle::MotorCandle;
use std::path::Path;
use std::time::Instant;

fn main() {
    let gguf = Path::new("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let tok = Path::new("modelos/tokenizer.json");

    let m = MotorCandle::nuevo("qwen2.5-0.5b", gguf, tok).expect("construir");
    println!("  caliente al construir: {}", m.esta_caliente());

    let pet = |t: &str| Peticion {
        modelo: "qwen2.5-0.5b".into(),
        mensajes: vec![
            Mensaje {
                rol: "system".into(),
                contenido: "Eres un bot de WhatsApp. Responde en una frase corta, en espanol."
                    .into(),
            },
            Mensaje {
                rol: "user".into(),
                contenido: t.into(),
            },
        ],
        temperatura: 0.8,
        max_tokens: 48,
        sesion: "demo".into(),
    };

    println!("\n  === FRIO (modelo sin cargar) ===");
    let t0 = Instant::now();
    let r = m.generar(&pet("hola, como estas?")).expect("generar");
    let frio = t0.elapsed().as_millis();
    println!(
        "  {} ms  | caliente={} | {} tokens",
        frio, r.estaba_caliente, r.tokens_generados
    );
    println!("  respuesta: {}", r.texto);

    println!("\n  === CALIENTE ===");
    let mut tiempos = vec![];
    for q in [
        "que dia es hoy?",
        "contame un chiste corto",
        "cuanto es 2+2?",
    ] {
        let t0 = Instant::now();
        let r = m.generar(&pet(q)).expect("generar");
        let ms = t0.elapsed().as_millis();
        tiempos.push((ms, r.tokens_generados));
        println!(
            "  {} ms | {} tokens | {}",
            ms,
            r.tokens_generados,
            r.texto.replace('\n', " ")
        );
    }

    let caliente = tiempos.iter().map(|(ms, _)| *ms).sum::<u128>() / tiempos.len() as u128;
    let toks: u32 = tiempos.iter().map(|(_, t)| *t).sum();
    let ms_tot: u128 = tiempos.iter().map(|(ms, _)| *ms).sum();

    println!("\n  ─────────────────────────────────");
    println!("  frio            {:>6} ms", frio);
    println!("  caliente (med)  {:>6} ms", caliente);
    println!("  la carga cuesta {:>6} ms", frio.saturating_sub(caliente));
    println!(
        "  velocidad       {:>6.1} tokens/s",
        toks as f64 / (ms_tot as f64 / 1000.0)
    );
}
