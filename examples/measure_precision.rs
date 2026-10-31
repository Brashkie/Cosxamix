//! Cuánto se aparta un camino de la matmul de la referencia, en los logits.
//!
//! Una batería de 15 preguntas no alcanza para decidir entre dos caminos que
//! difieren en el redondeo: un acierto de más o de menos puede ser una
//! pregunta que estaba en el límite. Acá se compara la distribución entera
//! del próximo token —los logits— contra la del camino que **no** cuantiza la
//! entrada (`COSXAMIX_MATMUL=f32`), en diez prompts de 32 a 320 tokens, todos
//! en el prefill, que es donde se usa la matmul por bloques.
//!
//! Se corre una vez por camino, porque los caminos se eligen con variables de
//! entorno que se leen al arrancar:
//!
//!     set COSXAMIX_MATMUL=f32 & cargo run --release --features candle --example measure_precision -- guardar ref.bin
//!     cargo run --release --features candle --example measure_precision -- comparar ref.bin
//!     set COSXAMIX_SIN_BLOQUES=1 & cargo run ... -- comparar ref.bin
//!
//! Imprime la divergencia KL media —cuánta información se pierde, en nats— y
//! en cuántos de los diez coincide el token más probable.

use cosxamix::forward::{Cache, Modelo};

const TEXTO: &str = "Sos WinsiBot, un bot de WhatsApp que conversa en grupos \
y en privado. Respondés en el idioma del mensaje, breve, natural y con buena \
onda, como una persona más del grupo. No inventás datos: si no sabés algo, lo \
decís. No usás listas largas ni títulos, porque WhatsApp los muestra mal. Usás \
emojis con moderación. Si alguien pide un comando, explicás cómo se usa con el \
prefijo #. La fotosíntesis es el proceso por el cual las plantas convierten la \
luz del sol, el agua y el dióxido de carbono en glucosa y oxígeno. Ocurre en los \
cloroplastos, gracias a la clorofila, que absorbe sobre todo la luz roja y la \
azul. El Everest mide 8.849 metros y está en la frontera entre Nepal y China. \
Un kilo de plomo y un kilo de plumas pesan lo mismo: un kilo. La Revolución \
Francesa empezó en 1789 con la toma de la Bastilla. El agua hierve a 100 grados \
al nivel del mar, pero a menos en la montaña, porque la presión es menor. Los \
pingüinos no vuelan, pero nadan muy rápido. Un número primo solo se divide por \
uno y por sí mismo: 2, 3, 5, 7, 11 y 13 son primos. En el grupo hoy se habló \
de fútbol, de la tarea de matemática y de quién trae la comida el sábado. \
Alguien preguntó cómo se hace un sticker de un video y otro respondió que con \
el comando #sticker respondiendo al video, que puede durar hasta diez segundos.";

fn softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let e: Vec<f64> = v.iter().map(|x| (*x as f64 - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map_or(0, |(i, _)| i)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (modo, archivo) = match (args.get(1), args.get(2)) {
        (Some(m), Some(a)) if m == "guardar" || m == "comparar" => (m.clone(), a.clone()),
        _ => {
            println!("  uso: measure_precision (guardar|comparar) <archivo>");
            return Ok(());
        }
    };
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let ruta = std::path::Path::new(&dir).join(
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q3_k_m.gguf".into()),
    );
    let ruta_tok = std::path::Path::new(&dir)
        .join(std::env::var("COSXAMIX_TOKENIZER").unwrap_or_else(|_| "tokenizer-3b.json".into()));
    let tok = tokenizers::Tokenizer::from_file(&ruta_tok).map_err(|e| e.to_string())?;
    let ids = tok
        .encode(TEXTO, false)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    let modelo = Modelo::abrir(&ruta)?;

    let largos: Vec<usize> = (1..=10)
        .map(|i| i * 32)
        .filter(|l| *l <= ids.len())
        .collect();
    let mut todos: Vec<Vec<f32>> = Vec::new();
    for &l in &largos {
        let mut cache = Cache::nuevo(&modelo.cfg);
        todos.push(modelo.forward(&ids[..l], &mut cache)?);
    }

    if modo == "guardar" {
        let bytes: Vec<u8> = todos
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&archivo, bytes)?;
        println!("  guardados {} prompts en {archivo}", largos.len());
        return Ok(());
    }

    let crudo = std::fs::read(&archivo)?;
    let vocab = todos[0].len();
    let mut kl_total = 0f64;
    let mut iguales = 0;
    for (i, nuestro) in todos.iter().enumerate() {
        let r: Vec<f32> = crudo[i * vocab * 4..(i + 1) * vocab * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let (p, q) = (softmax(&r), softmax(nuestro));
        let kl: f64 = p
            .iter()
            .zip(&q)
            .filter(|(a, _)| **a > 0.0)
            .map(|(a, b)| a * (a / b.max(1e-30)).ln())
            .sum();
        kl_total += kl;
        if argmax(&r) == argmax(nuestro) {
            iguales += 1;
        }
        println!("    {:>4} tokens   KL {:.5}", largos[i], kl);
    }
    println!(
        "  KL media {:.5} nats · token más probable igual en {iguales} de {}",
        kl_total / largos.len() as f64,
        largos.len()
    );
    Ok(())
}
