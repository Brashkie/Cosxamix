//! Genera un número fijo de tokens y reporta la velocidad.
//!
//! Pensado para correrse varias veces variando `RAYON_NUM_THREADS`. Lo que
//! interesa no es el número absoluto sino **si cambia** al dar más hilos: si no
//! cambia, la multiplicación de matrices no está repartiendo trabajo entre los
//! núcleos y el motor está al 7% del techo de memoria por eso.
//!
//! El prompt y la cantidad de tokens son fijos, y la temperatura 0, para que
//! las corridas sean comparables entre sí.

use std::path::Path;
use std::time::Instant;

use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_candle::MotorCandle;

const TOKENS: u32 = 64;

fn main() {
    let hilos = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "(sin fijar)".into());

    let m = MotorCandle::nuevo(
        "qwen2.5-0.5b",
        Path::new("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path::new("modelos/tokenizer.json"),
    )
    .expect("construir");

    let t_carga = Instant::now();
    m.precalentar().expect("precalentar");
    let carga = t_carga.elapsed().as_millis();

    let p = Peticion {
        modelo: "qwen2.5-0.5b".into(),
        mensajes: vec![
            Mensaje {
                rol: "system".into(),
                contenido: "Eres un asistente. Responde de forma extensa y detallada.".into(),
            },
            Mensaje {
                rol: "user".into(),
                contenido: "Explica como funciona la fotosintesis.".into(),
            },
        ],
        temperatura: 0.0,
        max_tokens: TOKENS,
        // Sin sesión: cada corrida reprocesa el prompt entero, que es lo que se
        // quiere medir. Con caché, la segunda corrida mediría otra cosa.
        sesion: String::new(),
    };

    // Una de descarte: la primera toca páginas de memoria por primera vez y
    // mide el arranque además de la generación.
    let _ = m.generar(&p).expect("calentar");

    // Dos corridas con el MISMO prompt y distinta cantidad de tokens. La
    // diferencia aísla el decode: el prefill es idéntico en las dos y se
    // cancela al restar.
    //
    // Hace falta porque dividir tokens generados por el tiempo total NO da la
    // velocidad de generación: con respuestas cortas el prefill domina y el
    // cociente mide sobre todo eso. Así salió una cifra de 6,1 tokens/s que en
    // realidad eran casi 19.
    let medir = |n: u32| {
        let mut q = p.clone();
        q.max_tokens = n;
        let t = Instant::now();
        let r = m.generar(&q).expect("generar");
        (t.elapsed().as_millis() as f64, r.tokens_generados as f64)
    };

    let (ms_corto, tok_corto) = medir(16);
    let (ms_largo, tok_largo) = medir(128);

    let decode = (tok_largo - tok_corto) / ((ms_largo - ms_corto) / 1000.0);
    let prefill = ms_corto - tok_corto / decode * 1000.0;

    println!(
        "hilos={hilos} carga={carga:.0}ms 16tok={ms_corto:.0}ms({tok_corto:.0}) \
         128tok={ms_largo:.0}ms({tok_largo:.0}) prefill={prefill:.0}ms decode={decode:.2}tok/s"
    );
}
