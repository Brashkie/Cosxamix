//! ¿Cuánto cuesta que dos conversaciones se pisen el caché?
//!
//! El caché de atención es uno solo y vive dentro del modelo. Mientras una
//! conversación avanza sola, cada turno reaprovecha el anterior. Pero en
//! WhatsApp hay muchos chats a la vez, y si llegan intercalados cada uno
//! invalida el del otro.
//!
//! Esto mide ese coste en vez de suponerlo, porque la alternativa —tener un
//! caché por sesión— exige o varias copias del modelo (491 MB cada una) o
//! parchear candle, que no expone el caché. Ninguna de las dos se hace sin un
//! número que la justifique.

use std::path::Path;
use std::time::Instant;

use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_candle::MotorCandle;

const SIS: &str = "Eres WinsiBot, un bot de WhatsApp. Responde corto, en espanol.";

fn motor() -> MotorCandle {
    let m = MotorCandle::nuevo(
        "qwen2.5-0.5b",
        Path::new("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path::new("modelos/tokenizer.json"),
    )
    .expect("construir");
    m.precalentar().expect("precalentar");
    m
}

/// Un turno con su historial. El historial va fijo para que las dos corridas
/// vean exactamente los mismos prompts.
fn turno(sesion: &str, hist: &[(&str, &str)], usuario: &str) -> Peticion {
    let mut m = vec![Mensaje {
        rol: "system".into(),
        contenido: SIS.into(),
    }];
    for (r, c) in hist {
        m.push(Mensaje {
            rol: (*r).into(),
            contenido: (*c).into(),
        });
    }
    m.push(Mensaje {
        rol: "user".into(),
        contenido: usuario.into(),
    });

    Peticion {
        modelo: "qwen2.5-0.5b".into(),
        mensajes: m,
        temperatura: 0.0,
        max_tokens: 24,
        sesion: sesion.into(),
    }
}

/// Corre una conversación de tres turnos, encadenando las respuestas REALES.
///
/// Esto importa más de lo que parece. La primera versión de esta medición
/// armaba el historial con texto inventado, y daba **cero** reutilización
/// incluso en el caso secuencial: el caché contiene los tokens que el modelo
/// generó, y si el historial trae otros el prefijo no encaja. Medir con
/// historial falso no mide el caché, mide nada.
fn conversacion(m: &MotorCandle, etiqueta: &str, preguntas: &[&str]) -> (u128, u32) {
    let mut hist: Vec<(String, String)> = Vec::new();
    let mut ms = 0;
    let mut cacheados = 0;

    for (i, q) in preguntas.iter().enumerate() {
        let refs: Vec<(&str, &str)> = hist.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
        let p = turno(etiqueta, &refs, q);

        let t = Instant::now();
        let r = m.generar(&p).expect("generar");
        let d = t.elapsed().as_millis();
        ms += d;
        cacheados += r.tokens_cacheados;

        println!(
            "    {etiqueta} #{}: {d:>5} ms | prompt {:>3} | reaprovechados {:>3}",
            i + 1,
            r.tokens_prompt,
            r.tokens_cacheados,
        );

        hist.push(("user".into(), (*q).into()));
        hist.push(("assistant".into(), r.texto.clone()));
    }

    (ms, cacheados)
}
fn main() {
    const A: [&str; 3] = [
        "hola, como te llamas?",
        "y que sabes hacer?",
        "dame un ejemplo",
    ];
    const B: [&str; 3] = [
        "cuanto es dos mas dos?",
        "y tres por tres?",
        "y la raiz de 81?",
    ];

    // ── Secuencial: una conversación entera, después la otra ───────────────
    println!(
        "
  === SECUENCIAL (A completa, despues B completa) ==="
    );
    let m = motor();
    let (msa, ca) = conversacion(&m, "A", &A);
    let (msb, cb) = conversacion(&m, "B", &B);
    let (ms_sec, cache_sec) = (msa + msb, ca + cb);

    // ── Intercalado: A1 B1 A2 B2 A3 B3, como llegan en un grupo ────────────
    //
    // Se simula manteniendo dos historiales en paralelo y alternando. Cada
    // petición encuentra el caché con la conversación del otro.
    println!(
        "
  === INTERCALADO (A1 B1 A2 B2 A3 B3) ==="
    );
    let m2 = motor();
    let mut ha: Vec<(String, String)> = Vec::new();
    let mut hb: Vec<(String, String)> = Vec::new();
    let mut ms_int = 0;
    let mut cache_int = 0;

    for i in 0..3 {
        for (etiqueta, preguntas, hist) in [("A", &A, &mut ha), ("B", &B, &mut hb)] {
            let refs: Vec<(&str, &str)> =
                hist.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
            let p = turno(etiqueta, &refs, preguntas[i]);

            let t = Instant::now();
            let r = m2.generar(&p).expect("generar");
            let d = t.elapsed().as_millis();
            ms_int += d;
            cache_int += r.tokens_cacheados;

            println!(
                "    {etiqueta}{}: {d:>5} ms | prompt {:>3} | reaprovechados {:>3}",
                i + 1,
                r.tokens_prompt,
                r.tokens_cacheados,
            );

            hist.push(("user".into(), preguntas[i].into()));
            hist.push(("assistant".into(), r.texto.clone()));
        }
    }

    // ── Resultado ──────────────────────────────────────────────────────────
    println!(
        "
  ─────────────────────────────────────────"
    );
    println!("  secuencial   {ms_sec:>6} ms | {cache_sec:>4} tokens reaprovechados");
    println!("  intercalado  {ms_int:>6} ms | {cache_int:>4} tokens reaprovechados");

    if ms_int > ms_sec {
        let perdida = ms_int - ms_sec;
        println!(
            "  el intercalado cuesta {perdida} ms mas  ({:.0}% peor)",
            perdida as f64 / ms_sec as f64 * 100.0,
        );
    } else {
        println!("  sin diferencia apreciable");
    }

    println!();
    println!("  Las mismas 6 peticiones y el mismo trabajo; lo unico que cambia");
    println!("  es el orden. Con un cache por sesion los dos casos empatan.");
    println!();
    println!("  Antes de tener cache por sesion esto daba 9770 ms contra 5541 y");
    println!("  CERO tokens reaprovechados al intercalar: un 76% peor.");
}
