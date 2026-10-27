//! Compara dos modelos en lo que de verdad decide: velocidad, memoria y acierto.
//!
//! Las preguntas no son decorativas. Son las que el 0,5B contesta mal —dice que
//! 2+2 es 3 y que la fotosíntesis es "un método de análisis de imágenes"— y
//! tienen respuesta verificable, así que se puede contar aciertos en vez de
//! opinar sobre si "suena mejor".
//!
//!   cargo run --release --features candle --example compare_models
//!   cargo run --release --features candle --example compare_models -- --cuant
//!   cargo run --release --features candle --example compare_models -- --motores
//!
//! `--motores` corre las mismas preguntas sobre el MISMO modelo —el 3B en
//! Q3_K_M, el del bot— con candle y con el motor propio: es la batería que
//! decide si el propio puede ser el de por defecto (DECISIONES §23).

use std::path::Path;
use std::time::Instant;

use cosxamix::engine::{Mensaje, Peticion};
use cosxamix::engine_candle::MotorCandle;
use cosxamix::engine_manager::{MotorCargable, TipoMotor};
use cosxamix::engine_own::MotorPropio;

/// Pregunta, y qué tiene que aparecer en la respuesta para darla por buena.
///
/// Se comprueba por contenido y no por igualdad: el modelo puede decir "cuatro"
/// o "4" o "son 4", y las tres están bien. Lo que se mide es si acierta, no si
/// redacta igual.
const PRUEBAS: &[(&str, &[&str])] = &[
    // ── Aritmetica de varios pasos ──────────────────────────────────────
    // Una sola operacion la acierta cualquiera por memoria. Encadenar dos o
    // tres obliga a seguir el razonamiento.
    ("Cuanto es 17 + 25? Solo el numero.",                      &["42"]),
    ("Cuanto es 12 x 12? Solo el numero.",                      &["144"]),
    ("Si compro 3 cajas de 8 lapices y regalo 5, cuantos me quedan? Solo el numero.", &["19"]),
    ("Un tren sale a las 14:30 y tarda 2 horas y media. A que hora llega? Solo la hora.", &["17:00", "17.00", "5:00"]),
    ("Cuanto es 100 menos 37, y a eso dividido por 7? Solo el numero final.", &["9"]),

    // ── Seguir instrucciones de formato ─────────────────────────────────
    // Mide algo distinto del conocimiento: si obedece. Un modelo chico suele
    // saber la respuesta y aun asi ignorar el formato pedido.
    ("Nombra exactamente tres colores primarios, separados por comas, sin ninguna otra palabra.",
                                                                 &["rojo", "azul"]),
    ("Responde unicamente con la palabra SI o la palabra NO: el agua hierve a 100 grados al nivel del mar?",
                                                                 &["si", "sí"]),

    // ── Conocimiento verificable ────────────────────────────────────────
    ("Cual es la capital de Australia? Una palabra.",            &["canberra"]),
    ("En que anio llego el hombre a la Luna? Solo el anio.",     &["1969"]),
    ("Cuantos huesos tiene aproximadamente el cuerpo humano adulto? Solo el numero.",
                                                                 &["206"]),

    // ── Razonamiento con trampa ─────────────────────────────────────────
    // La respuesta intuitiva es la equivocada; distingue recitar de razonar.
    ("Que pesa mas: un kilo de plomo o un kilo de plumas? Responde corto.",
                                                                 &["igual", "lo mismo", "pesan lo mismo", "mismo peso"]),
    ("Si en una carrera adelantas al segundo, en que posicion quedas? Solo el numero u ordinal.",
                                                                 &["segundo", "2"]),

    // ── Comprension y sintesis ──────────────────────────────────────────
    ("En una frase: que hacen las plantas en la fotosintesis?",
                                                                 &["luz", "sol", "dioxido", "co2"]),
    ("Explica en una frase por que flota un barco de acero.",
                                                                 &["densidad", "empuje", "desplaza", "agua", "flotacion"]),

    // ── Negacion, que los modelos chicos suelen ignorar ─────────────────
    ("Nombra un animal que NO sea mamifero. Una palabra.",
                                                                 &["aguila", "pez", "rana", "serpiente", "pollo", "gallina", "loro", "tiburon", "pinguino", "lagarto", "sapo", "ave", "pajaro", "cocodrilo", "tortuga", "abeja"]),
];
fn medir(nombre: &str, gguf: &str, tok: &str) {
    medir_con(nombre, gguf, tok, TipoMotor::Candle);
}

fn medir_con(nombre: &str, gguf: &str, tok: &str, tipo: TipoMotor) {
    let g = Path::new(gguf);
    if !g.exists() {
        println!("\n  {nombre}: no esta {gguf}, lo salteo");
        return;
    }

    let construido: Result<Box<dyn MotorCargable>, _> = match tipo {
        TipoMotor::Candle => {
            MotorCandle::nuevo(nombre, g, Path::new(tok)).map(|m| Box::new(m) as _)
        }
        TipoMotor::Propio => {
            MotorPropio::nuevo(nombre, g, Path::new(tok)).map(|m| Box::new(m) as _)
        }
    };
    let m = match construido {
        Ok(m) => m,
        Err(e) => {
            println!("\n  {nombre}: no se pudo construir ({e})");
            return;
        }
    };

    let t = Instant::now();
    if let Err(e) = m.precalentar() {
        println!("\n  {nombre}: no se pudo cargar ({e})");
        return;
    }
    let carga = t.elapsed().as_millis();

    let disco = std::fs::metadata(g).map(|x| x.len() / 1048576).unwrap_or(0);
    println!("\n  ══ {nombre} ══  {disco} MB en disco · carga {carga} ms");

    let pedir = |texto: &str, max: u32| Peticion {
        modelo: nombre.into(),
        mensajes: vec![
            Mensaje {
                rol: "system".into(),
                contenido: "Eres un asistente preciso. Responde corto y en espanol.".into(),
            },
            Mensaje {
                rol: "user".into(),
                contenido: texto.into(),
            },
        ],
        temperatura: 0.0,
        max_tokens: max,
        sesion: String::new(),
    };

    // ── Velocidad: se aísla el decode restando dos corridas ────────────────
    let _ = m.generar(&pedir("Hola.", 8));
    let corto = {
        let t = Instant::now();
        let r = m
            .generar(&pedir("Explica la gravedad.", 16))
            .expect("corto");
        (t.elapsed().as_millis() as f64, r.tokens_generados as f64)
    };
    let largo = {
        let t = Instant::now();
        let r = m
            .generar(&pedir("Explica la gravedad.", 96))
            .expect("largo");
        (t.elapsed().as_millis() as f64, r.tokens_generados as f64)
    };
    let decode = (largo.1 - corto.1) / ((largo.0 - corto.0) / 1000.0);
    let prefill = corto.0 - corto.1 / decode * 1000.0;

    // Dos repeticiones mas: una sola corrida dio 21,6 tok/s y otra 32,3 para
    // el MISMO modelo, asi que un valor suelto no sirve para comparar nada.
    let mut tasas = vec![decode];
    for _ in 0..2 {
        let a = {
            let t = Instant::now();
            let r = m.generar(&pedir("Explica la gravedad.", 16)).expect("a");
            (t.elapsed().as_millis() as f64, r.tokens_generados as f64)
        };
        let b = {
            let t = Instant::now();
            let r = m.generar(&pedir("Explica la gravedad.", 96)).expect("b");
            (t.elapsed().as_millis() as f64, r.tokens_generados as f64)
        };
        tasas.push((b.1 - a.1) / ((b.0 - a.0) / 1000.0));
    }
    tasas.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mediana = tasas[tasas.len() / 2];

    println!(
        "  decode {mediana:.1} tok/s (de 3: {:.1} / {:.1} / {:.1}) · prefill {prefill:.0} ms",
        tasas[0], tasas[1], tasas[2],
    );
    let decode = mediana;
    println!(
        "  una respuesta de 40 tokens: ~{:.1} s",
        prefill / 1000.0 + 40.0 / decode
    );

    // ── Acierto ────────────────────────────────────────────────────────────
    let mut bien = 0;
    println!();
    for (pregunta, claves) in PRUEBAS {
        let r = match m.generar(&pedir(pregunta, 40)) {
            Ok(r) => r.texto.to_lowercase(),
            Err(e) => {
                println!("    ERROR  {e}");
                continue;
            }
        };
        let ok = claves.iter().any(|c| r.contains(c));
        if ok {
            bien += 1;
        }
        let corta: String = r.replace('\n', " ").chars().take(70).collect();
        println!("    {}  {corta}", if ok { "OK" } else { "--" });
    }

    println!("\n  aciertos: {bien}/{}", PRUEBAS.len());
}

fn main() {
    // Por defecto compara los tres tamaños; con `--cuant` compara las
    // cuantizaciones del 3B, que es otra pregunta: ahí el modelo es el mismo y
    // lo único que cambia es cuántos bytes pesa cada peso.
    let cuant = std::env::args().any(|a| a == "--cuant");

    if std::env::args().any(|a| a == "--motores") {
        for (nombre, tipo) in [
            ("3b-q3_k_m candle", TipoMotor::Candle),
            ("3b-q3_k_m propio", TipoMotor::Propio),
        ] {
            medir_con(
                nombre,
                "modelos/qwen2.5-3b-instruct-q3_k_m.gguf",
                "modelos/tokenizer-3b.json",
                tipo,
            );
        }
        return;
    }

    if cuant {
        // candle NO soporta los formatos IQ (IQ3_M, IQ2_XXS): su enumeración
        // llega hasta los k-quants clásicos. Q3K y Q2K son lo más cercano
        // disponible — misma idea de bajar bytes por peso, algo peor en
        // calidad por bit.
        for q in ["q2_k", "q3_k_m", "q4_k_m"] {
            medir(
                &format!("3b-{q}"),
                &format!("modelos/qwen2.5-3b-instruct-{q}.gguf"),
                "modelos/tokenizer-3b.json",
            );
        }
        return;
    }

    medir(
        "qwen2.5-0.5b",
        "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "modelos/tokenizer.json",
    );
    medir(
        "qwen2.5-1.5b",
        "modelos/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "modelos/tokenizer-1.5b.json",
    );
    medir(
        "qwen2.5-3b",
        "modelos/qwen2.5-3b-instruct-q4_k_m.gguf",
        "modelos/tokenizer-3b.json",
    );
}
