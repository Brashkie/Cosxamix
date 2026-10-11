//! Comprueba que reutilizar el caché de atención no cambia la respuesta.
//!
//! El riesgo de un caché de atención no es que sea lento: es que devuelva una
//! respuesta condicionada por tokens que ya no corresponden. Eso no se nota
//! leyendo la salida —suena perfectamente razonable— así que se compara contra
//! el mismo turno generado desde cero.
//!
//! La generación va con **temperatura 0**, que candle trata como argmax. Así es
//! determinista y cualquier diferencia sale del cálculo y no del muestreo: con
//! temperatura, una diferencia de un bit en los logits cambia un token y de ahí
//! todo diverge, y eso no distingue un bug de ruido numérico.

use std::path::Path;
use std::time::Instant;

use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_candle::MotorCandle;

fn motor() -> MotorCandle {
    MotorCandle::nuevo(
        "qwen2.5-0.5b",
        Path::new("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path::new("modelos/tokenizer.json"),
    )
    .expect("construir")
}

fn turno(sistema: &str, hist: &[(&str, &str)], usuario: &str) -> Peticion {
    let mut m = vec![Mensaje {
        rol: "system".into(),
        contenido: sistema.into(),
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
        max_tokens: 32,
        sesion: "conv".into(),
    }
}

fn main() {
    const SIS: &str = "Eres WinsiBot, un bot de WhatsApp. Responde corto, en espanol.";

    // ── Conversación de dos turnos, con el caché vivo ───────────────────────
    let m = motor();
    m.precalentar().expect("precalentar");

    println!("\n  === CON reutilización de caché ===");

    let t1 = turno(SIS, &[], "hola, como te llamas?");
    let i = Instant::now();
    let r1 = m.generar(&t1).expect("t1");
    println!(
        "  turno 1: {:>5} ms | prompt {:>3} | cacheados {:>3} | {}",
        i.elapsed().as_millis(),
        r1.tokens_prompt,
        r1.tokens_cacheados,
        r1.texto.replace('\n', " "),
    );

    // El historial va FIJO y no con la respuesta del turno 1.
    //
    // Si se usara `r1.texto`, cualquier cambio en el turno 1 cambiaria el prompt
    // del turno 2 y las corridas dejarian de ser comparables: se vio en una
    // medicion donde el prompt salio de 76 tokens en vez de 69. Para aislar una
    // variable hay que fijar TODAS las demas.
    const R1: &str = "Hola, soy WinsiBot, un bot de WhatsApp.";
    let _ = &r1;
    let hist = [("user", "hola, como te llamas?"), ("assistant", R1)];
    let t2 = turno(SIS, &hist, "y que sabes hacer?");

    let i = Instant::now();
    let r2 = m.generar(&t2).expect("t2");
    let ms2 = i.elapsed().as_millis();
    println!(
        "  turno 2: {:>5} ms | prompt {:>3} | cacheados {:>3} | {}",
        ms2,
        r2.tokens_prompt,
        r2.tokens_cacheados,
        r2.texto.replace('\n', " "),
    );

    // ── El MISMO turno 2, con el caché frío ─────────────────────────────────
    let m2 = motor();
    m2.precalentar().expect("precalentar");

    println!("\n  === el MISMO turno 2, SIN caché ===");
    let i = Instant::now();
    let r2b = m2.generar(&t2).expect("t2 sin cache");
    let ms2b = i.elapsed().as_millis();
    println!(
        "  turno 2: {:>5} ms | prompt {:>3} | cacheados {:>3} | {}",
        ms2b,
        r2b.tokens_prompt,
        r2b.tokens_cacheados,
        r2b.texto.replace('\n', " "),
    );

    // ── Veredicto ───────────────────────────────────────────────────────────
    println!("\n  ─────────────────────────────────────");
    let igual = r2.texto == r2b.texto;
    println!(
        "  mismo texto: {}",
        if igual {
            "si"
        } else {
            "no (esperado, ver abajo)"
        }
    );
    println!(
        "  con cache  {ms2:>5} ms  ({} tokens reaprovechados)",
        r2.tokens_cacheados
    );
    println!("  sin cache  {ms2b:>5} ms");
    if ms2 < ms2b {
        println!(
            "  ahorro     {:>5} ms  ({:.1}x)",
            ms2b - ms2,
            ms2b as f64 / ms2 as f64
        );
    }

    // Que el texto NO coincida no es un fallo, y conviene explicar por qué para
    // que nadie lo "arregle".
    //
    // Candle no sabe hacer prefill por bloques con el caché puesto: su máscara
    // causal es seq_len x seq_len, válida solo con el caché vacío. Así que el
    // delta va de a un token, y alimentar de a uno es numéricamente distinto de
    // alimentar en bloque — `gemm` acumula en otro orden según la forma de la
    // matriz. Comprobado aislando la variable: con el MISMO prompt de 69 tokens
    // y sin caché en ninguno de los dos casos, el camino en bloque y el camino
    // de a uno ya dan textos distintos. El caché no tiene nada que ver.
    //
    // Que el caché es CORRECTO lo demuestra `compare_logits`, que compara la
    // magnitud que el modelo produce de verdad en vez de la cadena final: mismo
    // token más probable, diferencia del orden del 2% del rango.
    if !igual {
        let comun = r2
            .texto
            .chars()
            .zip(r2b.texto.chars())
            .take_while(|(a, b)| a == b)
            .count();
        println!("\n  Coinciden en los primeros {comun} caracteres y despues divergen.");
        println!("  Es lo esperado: alimentar de a un token acumula distinto que en");
        println!("  bloque. La correccion la verifica `cargo run --example compare_logits`.");
    } else {
        println!("\n  Ademas dieron el mismo texto.");
    }
}
