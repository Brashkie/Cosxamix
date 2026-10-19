//! Entrena el tokenizador propio y lo compara con el de Qwen2.5.
//!
//! Contesta la pregunta que decide los tamaños chicos de daiss: **¿cuánto peor
//! comprime un vocabulario de 8.192 que uno de 151.936?** Si la respuesta es
//! "un 10% más de tokens", el canje vale, porque el vocabulario de Qwen hace
//! que un daiss chico sea 97% tabla de búsqueda. Si es "el doble", no.
//!
//! El corpus son los documentos del proyecto y los comentarios de su código:
//! español técnico, unos 400 KB. Es **de juguete** y hay que decirlo — un
//! tokenizador de verdad se entrena con gigabytes y con texto variado. Lo que
//! esta medición puede decir con honestidad es la forma de la curva, no el
//! número final.
//!
//! Se mide sobre el **20% final del corpus, que el entrenamiento no vio**. Sin
//! eso el número sería mentira: un BPE entrenado sobre su propio texto de
//! prueba comprime fantásticamente y no generaliza nada.
//!
//!     cargo run --release --example train_tokenizer
//!     cargo run --release --features candle --example train_tokenizer

use std::time::Instant;

use cosxamix::daiss::Config;
use cosxamix::gguf::TipoGgml;
use cosxamix::tokenizer::Bpe;

/// Junta el corpus: los `.md` del proyecto y los comentarios de su código.
fn corpus() -> String {
    let mut t = String::new();
    for entrada in std::fs::read_dir(".").into_iter().flatten().flatten() {
        let ruta = entrada.path();
        if ruta.extension().is_some_and(|e| e == "md") {
            if let Ok(s) = std::fs::read_to_string(&ruta) {
                t.push_str(&s);
                t.push('\n');
            }
        }
    }
    // Y los comentarios del codigo, que son otro tanto de español técnico.
    for dir in [
        "src",
        "examples",
        "src/kernels",
        "src/model",
        "src/training",
    ] {
        for entrada in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let ruta = entrada.path();
            if ruta.extension().is_some_and(|e| e == "rs") {
                if let Ok(s) = std::fs::read_to_string(&ruta) {
                    for linea in s.lines() {
                        let l = linea.trim_start();
                        if l.starts_with("//") {
                            t.push_str(l.trim_start_matches('/').trim_start());
                            t.push('\n');
                        }
                    }
                }
            }
        }
    }
    t
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let todo = corpus();
    if todo.len() < 10_000 {
        println!(
            "  El corpus salio de {} bytes: corré esto desde la raiz del",
            todo.len()
        );
        println!("  proyecto, que es donde estan los .md y src/.");
        return Ok(());
    }

    // Corte por un límite de carácter, no de byte: partir un carácter UTF-8 a
    // la mitad haría fallar el slice.
    let corte = todo
        .char_indices()
        .map(|(i, _)| i)
        .find(|i| *i >= todo.len() * 4 / 5)
        .unwrap_or(todo.len());
    let (entrenamiento, prueba) = todo.split_at(corte);

    println!();
    println!(
        "  Corpus: {:.0} KB, entrenamiento {:.0} KB, prueba {:.0} KB (no vista)",
        todo.len() as f64 / 1024.0,
        entrenamiento.len() as f64 / 1024.0,
        prueba.len() as f64 / 1024.0,
    );
    println!();
    println!(
        "  {:<10}{:>10}{:>12}{:>14}{:>12}{:>10}",
        "vocab", "fusiones", "tokens", "bytes/token", "vs bytes", "entrenar"
    );
    println!("  {:-<68}", "");

    let bytes_prueba = prueba.len() as f64;
    let solo_bytes = Bpe::solo_bytes().codificar(prueba).len();
    println!(
        "  {:<10}{:>10}{:>12}{:>14}{:>12}{:>10}",
        "256",
        0,
        solo_bytes,
        format!("{:.2}", bytes_prueba / solo_bytes as f64),
        "1.00x",
        "-"
    );

    let mut mejores: Vec<(usize, usize)> = Vec::new();
    for objetivo in [1024usize, 2048, 4096, 8192, 16384, 32768] {
        let t0 = Instant::now();
        let bpe = Bpe::entrenar(entrenamiento, objetivo);
        let segundos = t0.elapsed().as_secs_f64();
        let n = bpe.codificar(prueba).len();
        mejores.push((objetivo, n));
        println!(
            "  {:<10}{:>10}{:>12}{:>14}{:>12}{:>10}",
            objetivo,
            bpe.fusiones(),
            n,
            format!("{:.2}", bytes_prueba / n as f64),
            format!("{:.2}x", solo_bytes as f64 / n as f64),
            format!("{segundos:.1}s"),
        );

        if objetivo == 8192 {
            let ruta = std::path::Path::new("modelos").join("daiss-8192.bpe");
            std::fs::create_dir_all("modelos")?;
            bpe.guardar(&ruta)?;
            // La comprobación que importa: lo guardado tiene que codificar
            // igual que lo que está en memoria.
            let vuelta = Bpe::cargar(&ruta)?;
            assert_eq!(vuelta.codificar(prueba), bpe.codificar(prueba));
            println!(
                "             guardado en {} ({} bytes), y lo cargado codifica igual",
                ruta.display(),
                std::fs::metadata(&ruta)?.len()
            );
        }
    }

    // ── contra el de Qwen ───────────────────────────────────────────────────
    #[cfg(feature = "candle")]
    {
        let ruta = std::path::Path::new("modelos").join("tokenizer-3b.json");
        if ruta.exists() {
            let tok = tokenizers::Tokenizer::from_file(&ruta).map_err(|e| e.to_string())?;
            let n = tok
                .encode(prueba, false)
                .map_err(|e| e.to_string())?
                .get_ids()
                .len();
            println!();
            println!(
                "  Qwen2.5, vocabulario 151.936: {n} tokens, {:.2} bytes/token",
                bytes_prueba / n as f64
            );
            for (objetivo, nuestros) in &mejores {
                if *objetivo == 8192 || *objetivo == 32768 {
                    println!(
                        "    el nuestro de {objetivo} necesita {:.1}% mas tokens",
                        (*nuestros as f64 / n as f64 - 1.0) * 100.0
                    );
                }
            }
        }
    }

    // ── lo que el vocabulario le hace a un modelo chico ─────────────────────
    println!();
    println!("  Lo que esto le hace a daiss-mini (256 de ancho, 6 capas):");
    println!();
    println!(
        "  {:<12}{:>14}{:>16}{:>14}",
        "vocab", "parametros", "% embeddings", "Q8_0"
    );
    println!("  {:-<56}", "");
    let base = *Config::de_nombre("daiss-mini").unwrap();
    for vocab in [1024usize, 8192, 32768, 151_936] {
        let mut c = base;
        c.vocab = vocab;
        let p = c.parametros();
        // Embeddings y proyeccion de salida: las dos tablas de vocabulario.
        let tablas = 2 * vocab as u64 * c.dim as u64;
        println!(
            "  {:<12}{:>14}{:>16}{:>14}",
            vocab,
            format!("{:.1} M", p as f64 / 1e6),
            format!("{:.0}%", tablas as f64 / p as f64 * 100.0),
            format!("{:.0} MB", c.bytes_en(TipoGgml::Q8_0) as f64 / 1_048_576.0),
        );
    }

    println!();
    println!("  Como leerlo. La columna de bytes/token dice cuanto comprime: mas");
    println!("  es mejor. Y la ultima tabla dice el precio del vocabulario de");
    println!("  Qwen en un modelo chico. El canje se decide con las dos juntas.");
    println!();
    println!("  Y la advertencia: el corpus son 400 KB de español tecnico de este");
    println!("  proyecto. La forma de la curva es creible; el numero exacto no se");
    println!("  puede extrapolar a texto de verdad.");
    println!();

    Ok(())
}
