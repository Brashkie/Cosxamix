//! Cuánto ahorra que un chat nuevo arranque con el prefijo de otro.
//!
//! Los *system* de WinsiBot empiezan igual en todos los grupos —quién es el
//! bot, su personalidad, las reglas— y después cambian con el perfil de cada
//! grupo. Acá se manda el primer mensaje del grupo A y después el primero del
//! grupo B, y se mide el de B. Con `COSXAMIX_SIN_PRESTAMO` el grupo B lee su
//! *system* entero; sin ella, copia de A el tramo común.
//!
//!     cargo run --release --features candle --example measure_shared_prefix
//!     set COSXAMIX_SIN_PRESTAMO=1 & cargo run ... (para comparar)
//!
//! Imprime también la respuesta de B, que tiene que salir igual en las dos
//! corridas: el caché de un token depende solo de los anteriores.

use std::time::Instant;

use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_own::MotorPropio;

/// El comienzo fijo del *system* de WinsiBot (`ai_chat.rs`, `prompt_sistema`).
const FIJO: &str = "Eres Hepein, la IA del bot WinsiBot en este grupo de WhatsApp.\n\
Tu personalidad es: natural, cercana y con humor ligero.\n\
Responde siempre en español. Sé breve: máximo 2 frases salvo que pidan algo largo.\n";

/// Instrucciones que también son fijas. Hasta la 9.0 de WinsiBot iban al
/// final, después del perfil del grupo; desde la 9.1.0 van arriba, y entran
/// en el tramo que se comparte. `COSXAMIX_ORDEN_VIEJO` mide el orden de antes.
const INSTRUCCIONES: &str = "Responde de forma natural, como lo haría un miembro de este grupo \
específico.\nGenerá una respuesta ORIGINAL y propia al mensaje del usuario — nunca copies, \
parafrasees ni reutilices los ejemplos de mensajes que aparezcan abajo, son solo guía de tono.\n";

fn sistema(grupo: &str) -> String {
    if std::env::var_os("COSXAMIX_ORDEN_VIEJO").is_some() {
        format!("{FIJO}\n{grupo}{INSTRUCCIONES}")
    } else {
        format!("{FIJO}{INSTRUCCIONES}\n{grupo}")
    }
}

fn peticion(sesion: &str, sistema: String, usuario: &str) -> Peticion {
    Peticion {
        modelo: String::new(),
        mensajes: vec![
            Mensaje {
                rol: "system".into(),
                contenido: sistema,
            },
            Mensaje {
                rol: "user".into(),
                contenido: usuario.into(),
            },
        ],
        temperatura: 0.0,
        max_tokens: 24,
        sesion: sesion.into(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let gguf = std::path::Path::new(&dir).join("qwen2.5-3b-instruct-q3_k_m.gguf");
    let tok = std::path::Path::new(&dir).join("tokenizer-3b.json");
    if !gguf.exists() || !tok.exists() {
        println!("  faltan los archivos del 3B en {dir}");
        return Ok(());
    }
    let motor = MotorPropio::nuevo("3b", &gguf, &tok)?;
    motor.precalentar()?;

    let grupo_a = "Vocabulario típico del grupo: xd, jaja, bro, partido, profe\n\
El grupo usa mensajes muy cortos e informales.\n\
Ejemplos de mensajes del grupo (SOLO para que copies el TONO, jamás el contenido):\n  \
• \"quien juega hoy\"\n  • \"jaja no puede ser\"\n\n";
    let grupo_b = "Vocabulario típico del grupo: tarea, examen, mañana, grupo, entregar\n\
El grupo escribe mensajes más elaborados.\n\
Ejemplos de mensajes del grupo (SOLO para que copies el TONO, jamás el contenido):\n  \
• \"alguien hizo el ejercicio 4?\"\n  • \"el examen es el jueves a las 8\"\n\n";

    motor.generar(&peticion("grupo-a", sistema(grupo_a), "hola, ¿qué tal?"))?;

    let t0 = Instant::now();
    let r = motor.generar(&peticion(
        "grupo-b",
        sistema(grupo_b),
        "¿cuándo era el examen?",
    ))?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    println!();
    println!(
        "  primer mensaje del grupo B: {ms:.0} ms · {} tokens de prompt, {} sacados de otra sesión",
        r.tokens_prompt, r.tokens_cacheados
    );
    println!("  respuesta: {}", r.texto);
    Ok(())
}
