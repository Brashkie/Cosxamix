//! Cuánto cuesta el tokenizador: entrenar, pre-tokenizar, codificar y
//! decodificar, y con `--features candle` lo mismo con el crate `tokenizers`,
//! que es el que sirve hoy los modelos de Qwen.
//!
//! El corpus son los documentos del proyecto, como en `train_tokenizer`.
//!
//!     cargo run --release --example measure_tokenizer
//!     COSXAMIX_SOLO=propio cargo run --release --example measure_tokenizer

use std::time::Instant;

use cosxamix::tokenizer::{pretokenizar, Bpe};

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
    t
}

/// La mediana de `veces` corridas, en ms.
fn mediana(veces: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..veces)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn mb_s(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / 1e6 / (ms / 1e3)
}

/// Solo cargar y codificar un prompt, para medir la memoria desde afuera:
/// `COSXAMIX_SOLO=propio` o `crate`.
fn solo(cual: &str) {
    let texto = "<|im_start|>user\nHola, ¿cómo andás?<|im_end|>\n";
    let t = Instant::now();
    let n = match cual {
        "propio" => cosxamix::tokenizer_json::TokenizadorJson::desde_archivo(std::path::Path::new(
            "modelos/tokenizer.json",
        ))
        .unwrap()
        .codificar(texto)
        .len(),
        _ => tokenizers::Tokenizer::from_file("modelos/tokenizer.json")
            .unwrap()
            .encode(texto, false)
            .unwrap()
            .get_ids()
            .len(),
    };
    println!(
        "{cual}: {n} tokens, {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
}

fn main() {
    if let Ok(cual) = std::env::var("COSXAMIX_SOLO") {
        solo(&cual);
        return;
    }
    let texto = corpus();

    // El lector propio del `tokenizer.json`, que es lo que usa el motor.
    if std::path::Path::new("modelos/tokenizer.json").exists() {
        use cosxamix::tokenizer_json::TokenizadorJson;
        let ruta = std::path::Path::new("modelos/tokenizer.json");
        println!("  el propio, con modelos/tokenizer.json:");
        let mut tk = None;
        let ms = mediana(5, || {
            tk = Some(TokenizadorJson::desde_archivo(ruta).unwrap())
        });
        println!("  cargar: {ms:.1} ms");
        let tk = tk.unwrap();
        let mut ids = Vec::new();
        let ms = mediana(5, || ids = tk.codificar(&texto));
        println!(
            "  codificar: {ms:.2} ms, {:.1} MB/s, {} tokens",
            mb_s(texto.len(), ms),
            ids.len()
        );
        let ms = mediana(5, || {
            std::hint::black_box(tk.decodificar(&ids, false));
        });
        println!(
            "  decodificar: {ms:.2} ms, {:.1} MB/s",
            mb_s(texto.len(), ms)
        );
        println!();
    }
    println!("  corpus: {} KB", texto.len() / 1024);

    let t = Instant::now();
    let bpe = Bpe::entrenar(&texto, 8192);
    println!(
        "  entrenar 8192: {:.0} ms, {} fusiones",
        t.elapsed().as_secs_f64() * 1e3,
        bpe.fusiones()
    );

    let ms = mediana(9, || {
        std::hint::black_box(pretokenizar(&texto));
    });
    println!(
        "  pre-tokenizar: {ms:.2} ms, {:.0} MB/s",
        mb_s(texto.len(), ms)
    );

    let mut ids = Vec::new();
    let ms = mediana(9, || {
        ids = bpe.codificar(&texto);
    });
    println!(
        "  codificar: {ms:.2} ms, {:.0} MB/s, {} tokens",
        mb_s(texto.len(), ms),
        ids.len()
    );

    let ms = mediana(9, || {
        std::hint::black_box(bpe.decodificar_bytes(&ids).unwrap());
    });
    println!(
        "  decodificar: {ms:.2} ms, {:.0} MB/s",
        mb_s(texto.len(), ms)
    );

    // El kernel de clasificar, solo, en cada lenguaje, sobre 16 MB.
    use cosxamix::kernels::classify::{disponibles, elegida, Mascaras};
    let grande: Vec<u8> = texto
        .as_bytes()
        .iter()
        .cycle()
        .take(16 << 20)
        .copied()
        .collect();
    println!("  elegida: {:?}", elegida());
    for copia in disponibles() {
        let ms = mediana(15, || {
            std::hint::black_box(Mascaras::con(copia, &grande));
        });
        println!(
            "  clasificar {copia:?}: {ms:.2} ms, {:.0} MB/s",
            mb_s(grande.len(), ms)
        );
    }

    // Un corpus de 256 MB, que es donde se ve el paralelismo: lo que importa
    // para preparar datos de entrenamiento.
    let enorme: String = texto.repeat((256usize << 20) / texto.len());
    let t = Instant::now();
    let n = bpe.codificar(&enorme).len();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "  codificar {} MB: {ms:.0} ms, {:.0} MB/s, {n} tokens",
        enorme.len() >> 20,
        mb_s(enorme.len(), ms)
    );
    let t = Instant::now();
    let b2 = Bpe::entrenar(&enorme, 8192);
    println!(
        "  entrenar con {} MB: {:.0} ms, {} fusiones",
        enorme.len() >> 20,
        t.elapsed().as_secs_f64() * 1e3,
        b2.fusiones()
    );
    drop(enorme);

    let ruta = std::env::temp_dir().join("cosxamix-measure.bpe");
    bpe.guardar(&ruta).unwrap();
    let ms = mediana(9, || {
        std::hint::black_box(Bpe::cargar(&ruta).unwrap());
    });
    println!("  cargar: {ms:.2} ms");
    let _ = std::fs::remove_file(&ruta);

    crate_tokenizers(&texto);
}

fn crate_tokenizers(texto: &str) {
    use tokenizers::Tokenizer;
    let ruta = "modelos/tokenizer.json";
    println!();
    println!("  el crate tokenizers, con {ruta}:");
    let mut tk = None;
    let ms = mediana(5, || {
        tk = Some(Tokenizer::from_file(ruta).unwrap());
    });
    println!("  cargar: {ms:.1} ms");
    let tk = tk.unwrap();
    let mut ids = Vec::new();
    let ms = mediana(5, || {
        ids = tk.encode(texto, false).unwrap().get_ids().to_vec();
    });
    println!(
        "  codificar: {ms:.2} ms, {:.1} MB/s, {} tokens",
        mb_s(texto.len(), ms),
        ids.len()
    );
    let ms = mediana(5, || {
        std::hint::black_box(tk.decode(&ids, false).unwrap());
    });
    println!(
        "  decodificar: {ms:.2} ms, {:.1} MB/s",
        mb_s(texto.len(), ms)
    );
}
