//! Cuánto tarda en cargar y generar un modelo cualquiera, con el motor propio.
//!
//!     cargo run --release --features candle --example measure_generation -- modelo.gguf tokenizer.json [tokens]
//!
//! Sirve también con un modelo sin entrenar: el texto es ruido, pero leer los
//! pesos cuesta lo mismo, y eso es lo que decide la velocidad.
//!
//! Un daiss entrenado acá trae el vocabulario en el GGUF: se pasa el mismo
//! archivo como tokenizador. `COSXAMIX_PROMPT` cambia el texto del final, que
//! para un modelo sin chat es lo que continúa.

use std::path::Path;
use std::time::Instant;

use cosxamix::engine::{Mensaje, Motor, Peticion};
use cosxamix::engine_own::MotorPropio;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 2 {
        println!("  uso: measure_generation modelo.gguf tokenizer.json [tokens]");
        return Ok(());
    }
    let tokens: u32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let m = MotorPropio::nuevo("medir", Path::new(&a[0]), Path::new(&a[1]))?;

    let t = Instant::now();
    m.precalentar()?;
    println!("  carga: {:.1} s", t.elapsed().as_secs_f64());

    let pedir = |max: u32| Peticion {
        modelo: "medir".into(),
        mensajes: vec![Mensaje {
            rol: "user".into(),
            contenido: "Hola.".into(),
        }],
        temperatura: 0.0,
        max_tokens: max,
        sesion: String::new(),
    };
    // La primera pasada trae al caché del sistema lo que `precalentar` no
    // tocó. Sin esto, si la memoria está justa, la medición es del disco.
    let t = Instant::now();
    m.generar(&pedir(tokens))?;
    println!("  calentamiento: {:.1} s", t.elapsed().as_secs_f64());

    // Un token solo es casi todo leer el prompt: restándolo queda la
    // generación. Tres veces, y la mediana.
    let mut tasas = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        m.generar(&pedir(1))?;
        let uno = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let r = m.generar(&pedir(tokens))?;
        let todo = t.elapsed().as_secs_f64();
        let tasa = f64::from(r.tokens_generados.saturating_sub(1)) / (todo - uno);
        println!(
            "  prompt + 1 token: {uno:.2} s; prompt + {} tokens: {todo:.2} s; {tasa:.2} tok/s",
            r.tokens_generados
        );
        tasas.push(tasa);
    }
    tasas.sort_by(f64::total_cmp);
    println!("  generación: {:.2} tok/s", tasas[1]);
    // El texto, para comparar corridas: con temperatura 0 tiene que dar lo mismo.
    let r = m.generar(&Peticion {
        mensajes: vec![Mensaje {
            rol: "user".into(),
            contenido: std::env::var("COSXAMIX_PROMPT")
                .unwrap_or_else(|_| "Cual es la capital de Francia? Explica en dos frases.".into()),
        }],
        ..pedir(48)
    })?;
    println!("  texto: {:?}", r.texto);
    Ok(())
}
