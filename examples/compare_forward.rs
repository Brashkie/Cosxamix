//! El forward propio contra el de candle, sobre el mismo modelo.
//!
//! Es la prueba de la fase 3. Se comparan **los logits**, no el texto
//! generado: dos implementaciones que difieran en el último bit pueden elegir
//! tokens distintos en el primer paso y divergir del todo, y entonces comparar
//! texto no diría nada. Los logits se comparan número a número.
//!
//! Se mide en dos regímenes, que son los dos que existen:
//!
//!   1. **prefill**: varios tokens de una vez, con el caché vacío;
//!   2. **decode**: un token más sobre el caché caliente, que es donde la
//!      máscara causal tiene que usar la posición absoluta y no el índice
//!      dentro de la tanda.
//!
//!     cargo run --release --features candle --example compare_forward

use std::time::Instant;

use cosxamix::forward::{Cache, Modelo};

/// Qué tan parecidas son dos listas de logits, en las tres formas que
/// importan.
fn comparar(nuestro: &[f32], suyo: &[f32]) -> (f32, f32, usize, usize) {
    let peor = nuestro
        .iter()
        .zip(suyo)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let rango = suyo.iter().cloned().fold(f32::MIN, f32::max)
        - suyo.iter().cloned().fold(f32::MAX, f32::min);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    (peor, rango, argmax(nuestro), argmax(suyo))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo = std::env::var("COSXAMIX_GGUF")
        .unwrap_or_else(|_| "qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let ruta = std::path::Path::new(&dir).join(&archivo);
    if !ruta.exists() {
        println!("  falta {}", ruta.display());
        return Ok(());
    }

    let t0 = Instant::now();
    let modelo = Modelo::abrir(&ruta)?;
    let cfg = modelo.cfg;
    println!();
    println!(
        "  {archivo} — {} capas, {} de ancho, {}/{} cabezas, {} palabras",
        cfg.capas, cfg.dim, cfg.cabezas, cfg.cabezas_kv, cfg.vocab
    );
    println!(
        "  abierto en {:.2} s, {:.2} GB de pesos, salida {}",
        t0.elapsed().as_secs_f64(),
        modelo.bytes() as f64 / 1_073_741_824.0,
        if modelo.salida_atada() {
            "atada a los embeddings"
        } else {
            "propia"
        }
    );

    // Tokens cualquiera: la comparación no necesita que digan nada, solo que
    // los dos caminos vean lo mismo.
    let prompt: Vec<u32> = vec![9707, 11, 1246, 525, 498, 30];
    let siguiente: u32 = 358;

    // ── candle ──────────────────────────────────────────────────────────────
    let dev = candle_core::Device::Cpu;
    let mut f = std::fs::File::open(&ruta)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut f)?;
    let mut suyo = cosxamix::model::qwen2::ModelWeights::from_gguf(ct, &mut f, &dev)?;

    println!();
    println!(
        "  {:<12}{:>10}{:>13}{:>13}{:>20}{:>14}",
        "régimen", "tokens", "nuestro ms", "candle ms", "peor dif.", "argmax"
    );
    println!("  {:-<82}", "");

    let mut cache = Cache::nuevo(&cfg);

    // Cada régimen se mide VARIAS veces, con una pasada de calentamiento
    // descartada. La primera llamada de cada camino paga cosas que no se
    // repiten —fallos de página del mapeo en el nuestro, inicialización
    // perezosa en el suyo— y medir una sola vez daba 9x a nuestro favor, que
    // es justo el error que DECISIONES §15 prohíbe cometer otra vez.
    const PASADAS: usize = 4;
    let mediana = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    for (nombre, tokens, pos) in [
        ("prefill", prompt.clone(), 0usize),
        ("decode", vec![siguiente], prompt.len()),
    ] {
        let entrada = candle_core::Tensor::from_vec(tokens.clone(), (1, tokens.len()), &dev)?;

        // Preparar el estado: para el decode hace falta el prefijo en los dos
        // cachés, y para el prefill hace falta que estén vacíos.
        let preparar = |cache: &mut Cache,
                        suyo: &mut cosxamix::model::qwen2::ModelWeights|
         -> Result<(), Box<dyn std::error::Error>> {
            cache.limpiar();
            suyo.limpiar_cache();
            if pos > 0 {
                modelo.forward(&prompt, cache)?;
                let previo =
                    candle_core::Tensor::from_vec(prompt.clone(), (1, prompt.len()), &dev)?;
                suyo.forward(&previo, 0)?;
            }
            Ok(())
        };

        let mut nuestros = Vec::new();
        let mut tiempos = Vec::new();
        for _ in 0..=PASADAS {
            preparar(&mut cache, &mut suyo)?;
            let t0 = Instant::now();
            nuestros = modelo.forward(&tokens, &mut cache)?;
            tiempos.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let t_nuestro = mediana(tiempos[1..].to_vec());

        let mut suyos = Vec::new();
        let mut tiempos = Vec::new();
        for _ in 0..=PASADAS {
            preparar(&mut cache, &mut suyo)?;
            let t0 = Instant::now();
            suyos = suyo
                .forward(&entrada, pos)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            tiempos.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let t_suyo = mediana(tiempos[1..].to_vec());

        let (peor, rango, a, b) = comparar(&nuestros, &suyos);
        println!(
            "  {:<12}{:>10}{:>13}{:>13}{:>20}{:>14}",
            nombre,
            tokens.len(),
            format!("{t_nuestro:.0}"),
            format!("{t_suyo:.0}"),
            format!("{:.4} ({:.2}%)", peor, peor / rango * 100.0),
            if a == b {
                format!("= {a}")
            } else {
                format!("{a} vs {b}")
            },
        );
    }

    println!();
    println!(
        "  caché: {} tokens, {:.1} KB; mediana de {PASADAS} pasadas, sin contar",
        cache.largo(),
        cache.bytes() as f64 / 1024.0
    );
    println!("  la de calentamiento.");
    // ── la prueba de verdad: que generen LO MISMO, no un token ──────────────
    //
    // Coincidir en un argmax puede ser suerte. Lo que decide si el forward
    // propio sirve para generar es que la secuencia entera coincida: cada
    // token elegido realimenta al siguiente, asi que una sola discrepancia
    // hace divergir todo lo que viene.
    println!();
    print!("  Generación voraz, 24 tokens: ");
    cache.limpiar();
    suyo.limpiar_cache();
    let mut nuestra = prompt.clone();
    let mut suya = prompt.clone();
    let mut tokens_nuestros = Vec::new();
    let mut tokens_suyos = Vec::new();

    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    };

    let mut entrada_n: Vec<u32> = prompt.clone();
    let mut entrada_s: Vec<u32> = prompt.clone();
    let mut pos_s = 0usize;
    let mut primera_diferencia = None;
    for paso in 0..24 {
        let l = modelo.forward(&entrada_n, &mut cache)?;
        let tn = argmax(&l);
        tokens_nuestros.push(tn);
        nuestra.push(tn);
        entrada_n = vec![tn];

        let e = candle_core::Tensor::from_vec(entrada_s.clone(), (1, entrada_s.len()), &dev)?;
        let l = suyo.forward(&e, pos_s)?.flatten_all()?.to_vec1::<f32>()?;
        let ts = argmax(&l);
        pos_s += entrada_s.len();
        tokens_suyos.push(ts);
        suya.push(ts);
        entrada_s = vec![ts];

        if tn != ts && primera_diferencia.is_none() {
            primera_diferencia = Some((paso, tn, ts));
        }
    }

    match primera_diferencia {
        None => println!("IDÉNTICA en los 24"),
        Some((paso, a, b)) => println!("coinciden {paso} y en el {paso} divergen: {a} contra {b}"),
    }
    println!("    nuestra: {tokens_nuestros:?}");
    if primera_diferencia.is_some() {
        println!("    candle:  {tokens_suyos:?}");
    }

    println!();
    println!("  Como leerlo. El argmax es lo que decide: si los dos caminos");
    println!("  eligen el mismo token, el forward propio sirve para generar.");
    println!();
    println!("  Sobre un modelo CUANTIZADO la diferencia no puede ser cero: los");
    println!("  dos cuantizan, y distinto — nosotros la entrada a i8 con una");
    println!("  escala cada 32, candle a Q8_K con una cada 256. Sobre un modelo");
    println!("  en F32 si tiene que ser cero, porque ahi ninguno cuantiza nada,");
    println!("  y medido lo es: 0,0000 y 24 tokens identicos. Eso es lo que");
    println!("  prueba que el forward propio esta bien y que la diferencia del");
    println!("  modelo cuantizado era cuantizacion y no un error.");
    println!();
    println!("  Para fabricar el de F32: cargo run --release --example make_f32_model");
    println!();

    Ok(())
}
