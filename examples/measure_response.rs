//! Dónde se va el tiempo de una respuesta del bot, y qué tan lejos está del
//! límite de esta máquina.
//!
//! Una respuesta tiene dos partes que se aceleran con cosas distintas:
//!
//! - **leer el prompt** (prefill): todos los tokens del prompt pasan juntos,
//!   así que cada peso que se trae de la RAM se usa muchas veces. Lo limita
//!   la cuenta —GFLOP/s—, y lo que lo acelera es una matmul por bloques;
//! - **generar** (decode): un token por vez, así que cada token obliga a
//!   recorrer los pesos enteros. Lo limita la RAM —GB/s—, y lo que lo acelera
//!   es leer menos bytes o leerlos mejor.
//!
//! Se mide como lo usa WinsiBot: un *system* largo, un primer mensaje sin
//! nada en caché, la respuesta, y un segundo mensaje con el *system* ya en
//! caché. Al final se compara el decode con lo que la RAM puede dar: recorrer
//! los mismos bytes con todos los núcleos, sin hacer ninguna cuenta.
//!
//!     cargo run --release --features candle --example measure_response
//!
//! `COSXAMIX_GGUF` y `COSXAMIX_TOKENIZER` cambian el modelo (por defecto, el
//! 3B Q3_K_M que baja WinsiBot).

use std::time::Instant;

use cosxamix::forward::{Cache, Modelo};
use cosxamix::sampler::Muestreador;

/// Un *system* del largo del de WinsiBot: personalidad, reglas y comandos.
const SISTEMA: &str = "Sos WinsiBot, un bot de WhatsApp que conversa en grupos \
y en privado. Respondés en el idioma del mensaje, breve, natural y con buena \
onda, como una persona más del grupo. No inventás datos: si no sabés algo, lo \
decís. No usás listas largas ni títulos, porque WhatsApp los muestra mal. Usás \
emojis con moderación. Si alguien pide un comando, explicás cómo se usa con el \
prefijo #. Comandos útiles: #menu muestra todos los comandos; #sticker convierte \
una imagen o video en sticker; #play busca y manda una canción; #rw tira una \
waifu al azar para coleccionar; #bank, #pay y #transfer manejan la economía del \
grupo; #coinflip y #roulette son apuestas; #clan arma clanes y guerras; #explorar \
abre la cueva del grupo; #perfil muestra la tarjeta del usuario; #tts lee un \
texto en voz alta; #recordar programa un recordatorio; #encuesta arma una \
votación; #resumen resume lo último que se habló. Las reglas del grupo las \
deciden los admins: no ayudás a saltarse un ban, no repetís insultos ni datos \
personales de nadie, y si alguien está mal, respondés con cuidado. Hoy es \
jueves y el grupo se llama Los del Barrio.";

fn formatear(usuario: &str) -> String {
    formatear_con(SISTEMA, usuario)
}

fn formatear_con(sistema: &str, usuario: &str) -> String {
    format!(
        "<|im_start|>system\n{sistema}<|im_end|>\n\
         <|im_start|>user\n{usuario}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// GB/s que da la RAM leyendo `bytes` con todos los núcleos y sin hacer
/// cuentas: el techo del decode. Se suma para que el compilador no lo borre.
fn ancho_de_banda(bytes: &[u8]) -> f64 {
    let hilos = std::thread::available_parallelism().map_or(1, |n| n.get());
    let trozo = bytes.len().div_ceil(hilos);
    let mut mejor = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let total: u64 = std::thread::scope(|s| {
            let hs: Vec<_> = bytes
                .chunks(trozo)
                .map(|c| {
                    s.spawn(move || {
                        let (pre, medio, post) = unsafe { c.align_to::<u64>() };
                        let mut a = [0u64; 4];
                        for b in medio.chunks_exact(4) {
                            a[0] = a[0].wrapping_add(b[0]);
                            a[1] = a[1].wrapping_add(b[1]);
                            a[2] = a[2].wrapping_add(b[2]);
                            a[3] = a[3].wrapping_add(b[3]);
                        }
                        let resto: u64 = pre.iter().chain(post).map(|&x| x as u64).sum();
                        a.iter().fold(resto, |x, y| x.wrapping_add(*y))
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .fold(0, u64::wrapping_add)
        });
        std::hint::black_box(total);
        mejor = mejor.min(t0.elapsed().as_secs_f64());
    }
    bytes.len() as f64 / mejor / 1e9
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo =
        std::env::var("COSXAMIX_GGUF").unwrap_or_else(|_| "qwen2.5-3b-instruct-q3_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    let ruta_tok = std::path::Path::new(&dir)
        .join(std::env::var("COSXAMIX_TOKENIZER").unwrap_or_else(|_| "tokenizer-3b.json".into()));
    for r in [&ruta, &ruta_tok] {
        if !r.exists() {
            println!("  falta {}", r.display());
            return Ok(());
        }
    }
    let tok = tokenizers::Tokenizer::from_file(&ruta_tok).map_err(|e| e.to_string())?;
    let ids = |s: &str| -> Result<Vec<u32>, String> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec())
    };

    let modelo = Modelo::abrir(&ruta)?;
    let pesos = modelo.bytes() as f64;
    // Los parámetros, para las cuentas: cada uno es una multiplicación y una
    // suma por token. Se cuentan desde los bytes y los bits por peso del
    // archivo, que alcanza para el orden de magnitud.
    let params = std::env::var("COSXAMIX_PARAMS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(3.09e9);

    // Calentar: el primer toque trae el modelo del disco (ver generate.rs).
    {
        let mut c = Cache::nuevo(&modelo.cfg);
        modelo.forward(&ids(&formatear("hola"))?, &mut c)?;
    }

    let primero = ids(&formatear("che, ¿cómo hago un sticker de un video?"))?;
    let segundo_txt = formatear("¿y cuánto puede durar el video como máximo?");
    let segundo = ids(&segundo_txt)?;
    let nuevos = 48;

    println!();
    println!(
        "  {archivo} — {:.0} MB de pesos, {} núcleos",
        pesos / 1e6,
        std::thread::available_parallelism().map_or(1, |n| n.get())
    );

    // ── primer mensaje: nada en caché ──────────────────────────────────────
    let mut cache = Cache::nuevo(&modelo.cfg);
    let t0 = Instant::now();
    let mut logits = modelo.forward(&primero, &mut cache)?;
    let t_prefill = t0.elapsed().as_secs_f64();

    let mut m = Muestreador::voraz();
    let mut por_token = Vec::with_capacity(nuevos);
    for _ in 0..nuevos {
        let t = m.elegir(&logits);
        let t0 = Instant::now();
        logits = modelo.forward(&[t], &mut cache)?;
        por_token.push(t0.elapsed().as_secs_f64());
    }
    por_token.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t_token = por_token[por_token.len() / 2];

    // ── segundo mensaje: el system ya está en caché ─────────────────────────
    let comun = primero
        .iter()
        .zip(&segundo)
        .take_while(|(a, b)| a == b)
        .count();
    cache.truncar(comun);
    let t0 = Instant::now();
    modelo.forward(&segundo[comun..], &mut cache)?;
    let t_tibio = t0.elapsed().as_secs_f64();

    // ── segundo mensaje con el system cambiado cerca del principio ──────────
    // Es lo que pasaba en WinsiBot antes de la 9.1.0: el system traía los
    // últimos mensajes del grupo como ejemplo, y cambiaba con cada mensaje
    // justo después de las primeras líneas fijas.
    let cambiado = ids(&formatear_con(
        &SISTEMA.replace("No inventás datos", "Nunca inventás datos"),
        "¿y cuánto puede durar el video como máximo?",
    ))?;
    let comun_c = segundo
        .iter()
        .zip(&cambiado)
        .take_while(|(a, b)| a == b)
        .count();
    cache.truncar(comun_c);
    let t0 = Instant::now();
    modelo.forward(&cambiado[comun_c..], &mut cache)?;
    let t_cambiado = t0.elapsed().as_secs_f64();

    let gbs = ancho_de_banda(modelo_bytes(&ruta)?.as_ref());

    let gflops = 2.0 * params * primero.len() as f64 / t_prefill / 1e9;
    let gbs_decode = pesos / t_token / 1e9;

    println!();
    println!("  primer mensaje, sin caché");
    println!(
        "    leer el prompt   {:>4} tokens  {:>7.0} ms   {:>5.1} tok/s   {:>5.1} GFLOP/s",
        primero.len(),
        t_prefill * 1e3,
        primero.len() as f64 / t_prefill,
        gflops
    );
    println!(
        "    generar          {:>4} tokens  {:>7.0} ms   {:>5.1} tok/s   {:>5.1} GB/s",
        nuevos,
        t_token * nuevos as f64 * 1e3,
        1.0 / t_token,
        gbs_decode
    );
    println!("  segundo mensaje, system en caché");
    println!(
        "    leer el prompt   {:>4} tokens  {:>7.0} ms",
        segundo.len() - comun,
        t_tibio * 1e3
    );
    println!("  segundo mensaje, system cambiado al principio (antes de la foto)");
    println!(
        "    leer el prompt   {:>4} tokens  {:>7.0} ms",
        cambiado.len() - comun_c,
        t_cambiado * 1e3
    );
    println!();
    println!("  la RAM da {gbs:.1} GB/s leyendo con todos los núcleos");
    println!(
        "  el decode usa {:.0}% de eso: techo de {:.1} tok/s",
        gbs_decode / gbs * 100.0,
        gbs * 1e9 / pesos
    );
    let total = t_prefill + t_token * nuevos as f64;
    println!(
        "  en la respuesta del primer mensaje: {:.0}% leer, {:.0}% generar",
        t_prefill / total * 100.0,
        t_token * nuevos as f64 / total * 100.0
    );
    Ok(())
}

/// El archivo mapeado entero: ya está en la caché del sistema tras calentar,
/// así que recorrerlo mide la RAM y no el disco.
fn modelo_bytes(ruta: &std::path::Path) -> Result<memmap2::Mmap, Box<dyn std::error::Error>> {
    let f = std::fs::File::open(ruta)?;
    Ok(unsafe { memmap2::Mmap::map(&f)? })
}
