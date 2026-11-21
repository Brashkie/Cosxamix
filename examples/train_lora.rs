//! Ajusta un modelo con LoRA en una máquina sin GPU.
//!
//! ```bash
//! cargo run --release --features training --example train_lora -- corpus.txt
//! ```
//!
//! Argumentos, todos opcionales menos el primero:
//!
//! ```text
//! 1  corpus.txt   texto plano con el que ajustar
//! 2  pasos        cuántos pasos de optimización (por defecto 50)
//! 3  gguf         modelo base (por defecto el 0,5B)
//! 4  tokenizador  (por defecto tokenizer.json)
//! ```
//!
//! Lo que sale es `adaptador.safetensors`, de ~2 MB, que es lo único que hay
//! que guardar: el modelo base no se toca.
//!
//! ## Qué esperar
//!
//! Cincuenta pasos sirven para ver que la pérdida baja; mil empiezan a servir
//! para algo. No es entrenar un modelo: es moverle el estilo a uno que ya sabe.
//! El tiempo por paso lo imprime el propio bucle, y `measure_lora` lo mide
//! aparte.

use std::path::Path;
use std::time::Instant;

use cosxamix::tokenizer_json::TokenizadorJson;
use cosxamix::training::{guardar, AdamW, ModeloEntrenable};

/// Tokens por secuencia. Más largo entrena mejor y cuesta más que lineal,
/// porque la atención es cuadrática en el largo.
const LARGO: usize = 256;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let corpus = args.next().unwrap_or_else(|| {
        eprintln!("falta el corpus: cargo run ... --example train_lora -- corpus.txt");
        std::process::exit(2)
    });
    let pasos: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let gguf = args
        .next()
        .unwrap_or_else(|| "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let tok_ruta = args
        .next()
        .unwrap_or_else(|| "modelos/tokenizer.json".into());

    // ── El corpus ────────────────────────────────────────────────────────
    let texto = std::fs::read_to_string(&corpus).map_err(|e| format!("{corpus}: {e}"))?;
    let tok = TokenizadorJson::para_modelo(Path::new(&gguf), Path::new(&tok_ruta))?;
    let ids = tok.codificar(&texto);

    // Hace falta un token mas que el largo: el ultimo es el objetivo del
    // anteultimo y no tiene entrada propia.
    if ids.len() < LARGO + 1 {
        eprintln!(
            "el corpus tiene {} tokens y hacen falta al menos {}",
            ids.len(),
            LARGO + 1
        );
        std::process::exit(2);
    }
    let ventanas = ids.len() - LARGO;

    println!();
    println!("  corpus      {corpus}");
    println!("  tokens      {}", ids.len());
    println!("  ventanas    {ventanas} de {LARGO} tokens");

    // ── El modelo ────────────────────────────────────────────────────────
    println!("  cargando el modelo y descuantizando a f32...");
    let t0 = Instant::now();
    let mut modelo = ModeloEntrenable::cargar(Path::new(&gguf), 8, 16.0)?;
    let (entrenables, matrices) = modelo.entrenables();
    println!("  listo en {:.1} s", t0.elapsed().as_secs_f64());
    println!("  entrenables {entrenables} parametros en {matrices} matrices");
    println!();

    let mut opt = AdamW::nuevo(1e-4);

    // ── El bucle ─────────────────────────────────────────────────────────
    println!(
        "  {:<8}{:>12}{:>12}{:>14}",
        "paso", "perdida", "ms", "restante"
    );
    println!("  {:-<46}", "");

    let mut primera = f32::NAN;
    let mut ultima = f32::NAN;
    let inicio = Instant::now();

    for paso in 1..=pasos {
        // Ventana distinta en cada paso, recorriendo el corpus de punta a
        // punta. Repetir siempre la misma haria bajar la perdida sin que el
        // modelo aprenda nada mas que esa ventana.
        let desde = (paso - 1) * (ventanas.max(1) / pasos.max(1)).max(1) % ventanas;
        // LARGO + 1 tokens: el último es el objetivo del anteúltimo.
        let lote = [ids[desde..desde + LARGO + 1].to_vec()];

        let t = Instant::now();
        let (p, grads) = modelo.perdida_y_gradientes(&lote);
        modelo.aplicar(&mut opt, &grads);
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        if paso == 1 {
            primera = p;
        }
        ultima = p;

        let restante = (pasos - paso) as f64 * inicio.elapsed().as_secs_f64() / paso as f64;
        if paso <= 3 || paso % 5 == 0 || paso == pasos {
            println!(
                "  {:<8}{:>12}{:>12}{:>14}",
                paso,
                format!("{p:.4}"),
                format!("{ms:.0}"),
                format!("{:.1} min", restante / 60.0)
            );
        }
    }

    // ── El resultado ─────────────────────────────────────────────────────
    let salida = Path::new("adaptador.safetensors");
    let n = guardar(modelo.params(), salida)?;
    let bytes = std::fs::metadata(salida).map(|m| m.len()).unwrap_or(0);

    println!();
    println!("  {:<24}{:>14}", "perdida inicial", format!("{primera:.4}"));
    println!("  {:<24}{:>14}", "perdida final", format!("{ultima:.4}"));
    println!(
        "  {:<24}{:>14}",
        "cambio",
        format!("{:+.1}%", (ultima - primera) / primera * 100.0)
    );
    println!(
        "  {:<24}{:>14}",
        "tiempo total",
        format!("{:.1} min", inicio.elapsed().as_secs_f64() / 60.0)
    );
    println!(
        "  {:<24}{:>14}",
        "adaptador",
        format!("{n} tensores, {:.1} MB", bytes as f64 / 1048576.0)
    );
    println!();
    println!("  El adaptador se guarda SOLO: no lleva el modelo base dentro.");
    println!("  Si la perdida no bajo, el corpus es muy chico o faltan pasos.");

    Ok(())
}
