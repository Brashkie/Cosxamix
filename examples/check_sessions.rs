//! ¿Una conversación se contamina con otra al compartir el modelo?
//!
//! Es el fallo peligroso de un caché por sesión: si al intercalar se restaura
//! el caché equivocado, el modelo responde condicionado por una conversación
//! ajena. No lanza ningún error y la respuesta suena perfectamente razonable —
//! solo que viene del contexto de otro chat.
//!
//! La prueba es directa: se corre la conversación A sola, después se corre
//! intercalada con B, y se comparan las respuestas de A. Si el aislamiento es
//! correcto tienen que salir **idénticas**: A hace exactamente la misma
//! secuencia de llamadas al modelo en los dos casos.
//!
//! Esto SÍ admite comparación exacta de texto, a diferencia de `check_cache`:
//! allá se comparaban dos caminos de cálculo distintos (en bloque contra de a
//! un token) y la diferencia numérica era esperable. Acá el camino es el mismo.

use std::path::Path;

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

fn turno(sesion: &str, hist: &[(String, String)], usuario: &str) -> Peticion {
    let mut m = vec![Mensaje {
        rol: "system".into(),
        contenido: SIS.into(),
    }];
    for (r, c) in hist {
        m.push(Mensaje {
            rol: r.clone(),
            contenido: c.clone(),
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

/// Avanza una conversación un turno y guarda la respuesta en su historial.
fn avanzar(
    m: &MotorCandle,
    sesion: &str,
    hist: &mut Vec<(String, String)>,
    pregunta: &str,
) -> String {
    let p = turno(sesion, hist, pregunta);
    let r = m.generar(&p).expect("generar");
    hist.push(("user".into(), pregunta.into()));
    hist.push(("assistant".into(), r.texto.clone()));
    r.texto
}

fn main() {
    const A: [&str; 3] = ["como te llamas?", "y que sabes hacer?", "dame un ejemplo"];
    const B: [&str; 3] = [
        "cuanto es dos mas dos?",
        "y tres por tres?",
        "y la raiz de 81?",
    ];

    // ── A sola ──────────────────────────────────────────────────────────────
    println!("\n  === A sola ===");
    let m = motor();
    let mut ha = Vec::new();
    let solas: Vec<String> = A
        .iter()
        .map(|q| {
            let r = avanzar(&m, "A", &mut ha, q);
            println!("    {}", r.replace('\n', " "));
            r
        })
        .collect();

    // ── A intercalada con B ─────────────────────────────────────────────────
    println!("\n  === A intercalada con B ===");
    let m2 = motor();
    let (mut ha2, mut hb2) = (Vec::new(), Vec::new());
    let mut intercaladas = Vec::new();

    for i in 0..3 {
        let ra = avanzar(&m2, "A", &mut ha2, A[i]);
        println!("    A: {}", ra.replace('\n', " "));
        intercaladas.push(ra);

        let rb = avanzar(&m2, "B", &mut hb2, B[i]);
        println!("    B: {}", rb.replace('\n', " "));
    }

    // ── Veredicto ───────────────────────────────────────────────────────────
    println!("\n  ─────────────────────────────────────");
    let mut mal = 0;
    for (i, (sola, inter)) in solas.iter().zip(&intercaladas).enumerate() {
        let ok = sola == inter;
        println!(
            "  turno {}: {}",
            i + 1,
            if ok { "identico" } else { "DISTINTO" }
        );
        if !ok {
            println!("    sola:        {}", sola.replace('\n', " "));
            println!("    intercalada: {}", inter.replace('\n', " "));
            mal += 1;
        }
    }

    if mal > 0 {
        eprintln!("\n  {mal} de 3 turnos cambian al intercalar: las sesiones se contaminan.");
        std::process::exit(1);
    }

    println!("\n  Las sesiones estan aisladas: intercalar no cambia ni una respuesta.");
}
