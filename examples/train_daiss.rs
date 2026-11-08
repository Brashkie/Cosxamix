//! Entrena un daiss desde cero y mide cuánto aprendió, en **bits por byte** sobre
//! texto que no vio.
//!
//! Bits por byte es lo que permite comparar vocabularios distintos: la pérdida por
//! token no, porque un token de BPE lleva más texto que uno de bytes. Es también lo
//! que mide un compresor, así que hay contra qué pararse: `xz` en el mismo texto.
//!
//! Es el experimento del ROADMAP §8: ¿hace falta un tokenizador? El mismo bloque,
//! el mismo tiempo de CPU y el mismo corpus, con tres vocabularios.
//!
//!     COSXAMIX_VOCAB=bytes cargo run --release --features training,zig,c --example train_daiss -- corpus.txt
//!
//! `COSXAMIX_VOCAB` es `bytes` o un número (BPE de ese tamaño). `COSXAMIX_MINUTOS`,
//! `COSXAMIX_LOTE`, `COSXAMIX_LARGO`, `COSXAMIX_LR`, `COSXAMIX_PARTES` (en cuántos
//! tramos paralelos se reparte el lote), `COSXAMIX_SALIDA` (GGUF) y `COSXAMIX_TT`
//! (rango: las tablas de vocabulario en Tensor-Train desde el principio).
//!
//! Los minutos cuentan solo el entrenamiento: medir no le quita tiempo, así que un
//! modelo con ventanas largas, que tarda más en medirse, no sale perjudicado. La
//! prueba usa la misma ventana que el entrenamiento, y la cifra que vale es la del
//! final, sobre todo el texto de prueba.

use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rayon::prelude::*;

use cosxamix::daiss::{Azar, Config};
use cosxamix::gguf::TipoGgml;
use cosxamix::tokenizer::Bpe;
use cosxamix::training::full::ModeloCompleto;

fn env<T: std::str::FromStr>(k: &str, defecto: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(defecto)
}

/// El cuerpo de un libro de Project Gutenberg, sin la licencia del principio y
/// del final. Un texto sin esas marcas queda entero.
fn cuerpo(t: &str) -> &str {
    let desde = t
        .find("*** START OF")
        .and_then(|i| t[i..].find('\n').map(|j| i + j + 1))
        .unwrap_or(0);
    let hasta = t.find("*** END OF").unwrap_or(t.len());
    &t[desde..hasta]
}

/// Un lote de ventanas del corpus.
fn lote(
    ids: &[u32],
    b: usize,
    t: usize,
    azar: &mut Azar,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let mut v = Vec::with_capacity(b * t);
    for _ in 0..b {
        let i = (azar.siguiente() % (ids.len() - t) as u64) as usize;
        v.extend_from_slice(&ids[i..i + t]);
    }
    Tensor::from_vec(v, (b, t), dev)
}

/// Bits por byte sobre **todo** el texto de prueba, en ventanas de `t` sin solapar:
/// sin azar, y cada token predicho pesa lo mismo. Las primeras posiciones de cada
/// ventana tienen poco contexto, y por eso importa cuánto texto cubre una ventana.
fn bits_en_todo(
    m: &ModeloCompleto,
    ids: &[u32],
    t: usize,
    bytes_por_token: f64,
    dev: &Device,
) -> candle_core::Result<f64> {
    let ventanas: Vec<&[u32]> = ids.chunks_exact(t).collect();
    // Tandas chicas: candle guarda el grafo de cada una hasta terminarla, y por hilo
    // son cientos de MB. Con 16 ventanas de 484 no alcanzan 16 GB.
    let sumas = ventanas
        .par_chunks((512 / t).max(1))
        .map(|tanda| {
            let planos: Vec<u32> = tanda.iter().flat_map(|v| v.iter().copied()).collect();
            let l = Tensor::from_vec(planos, (tanda.len(), t), dev)?;
            Ok(f64::from(m.perdida(&l)?.to_scalar::<f32>()?) * tanda.len() as f64)
        })
        .collect::<candle_core::Result<Vec<f64>>>()?;
    let media = sumas.iter().sum::<f64>() / ventanas.len() as f64;
    Ok(media / std::f64::consts::LN_2 / bytes_por_token)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ruta = std::env::args().nth(1).expect("falta el corpus");
    let crudo = String::from_utf8_lossy(&std::fs::read(&ruta)?).replace("\r\n", "\n");
    let texto = cuerpo(&crudo);

    // El 10% final no se ve nunca: cortado en un salto de línea.
    let corte = texto[..texto.len() / 10 * 9].rfind('\n').unwrap_or(0) + 1;
    let (entreno, prueba) = texto.split_at(corte);

    let vocab = std::env::var("COSXAMIX_VOCAB").unwrap_or_else(|_| "bytes".into());
    let t0 = Instant::now();
    let (tok, ids, ids_prueba) = if vocab == "bytes" {
        let b = Bpe::solo_bytes();
        let a = b.codificar_bytes(entreno.as_bytes());
        let p = b.codificar_bytes(prueba.as_bytes());
        (b, a, p)
    } else {
        let b = Bpe::entrenar(entreno, vocab.parse()?);
        let a = b.codificar(entreno);
        let p = b.codificar(prueba);
        (b, a, p)
    };
    let bytes_por_token = prueba.len() as f64 / ids_prueba.len() as f64;
    println!(
        "  {ruta}: {} KB para entrenar, {} KB de prueba",
        entreno.len() >> 10,
        prueba.len() >> 10
    );
    println!(
        "  vocabulario {vocab}: {} tokens, {bytes_por_token:.2} bytes por token, {:.1} s",
        tok.tamano(),
        t0.elapsed().as_secs_f64()
    );

    // La forma de daiss-mini, con el vocabulario que toque.
    let base = Config::de_nombre("daiss-mini").expect("existe");
    let cfg = Config {
        vocab: tok.tamano(),
        ..*base
    };
    let (minutos, b, t, lr): (f64, usize, usize, f64) = (
        env("COSXAMIX_MINUTOS", 10.0),
        env("COSXAMIX_LOTE", 16),
        env("COSXAMIX_LARGO", 128),
        env("COSXAMIX_LR", 1e-3),
    );
    let partes: usize = env("COSXAMIX_PARTES", 6);
    let dev = Device::Cpu;
    let rango_tt: usize = env("COSXAMIX_TT", 0);
    let m = if rango_tt > 0 {
        ModeloCompleto::con_tensor_train(&cfg, 2026, t, rango_tt, &dev)?
    } else {
        ModeloCompleto::nuevo(&cfg, 2026, t, &dev)?
    };
    let sin_tablas =
        ModeloCompleto::nuevo(&cfg, 0, 1, &dev)?.parametros() - 2 * cfg.vocab * cfg.dim;
    let tablas = m.parametros() - sin_tablas;
    println!(
        "  {:.2} M parámetros, {:.1}% en las tablas de vocabulario{}; lote {b} x {t}",
        m.parametros() as f64 / 1e6,
        tablas as f64 * 100.0 / m.parametros() as f64,
        if rango_tt > 0 {
            format!(" (Tensor-Train de rango {rango_tt})")
        } else {
            String::new()
        }
    );

    let mut opt = AdamW::new(
        m.variables(),
        ParamsAdamW {
            lr,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;

    // Siempre las mismas ventanas de prueba, para que las medidas se comparen; unos
    // 16 K tokens, sea cual sea el largo.
    let lotes_prueba: Vec<Tensor> = {
        let mut azar = Azar::con_semilla(99);
        let por_lote = (2048 / t).max(1);
        (0..8)
            .map(|_| lote(&ids_prueba, por_lote, t, &mut azar, &dev))
            .collect::<candle_core::Result<_>>()?
    };
    let bits_por_byte = |m: &ModeloCompleto| -> candle_core::Result<f64> {
        let mut s = 0f64;
        for l in &lotes_prueba {
            s += f64::from(m.perdida(l)?.to_scalar::<f32>()?);
        }
        Ok(s / lotes_prueba.len() as f64 / std::f64::consts::LN_2 / bytes_por_token)
    };

    println!("  al empezar: {:.3} bits por byte", bits_por_byte(&m)?);
    println!();
    println!(
        "  {:>7}{:>8}{:>10}{:>11}{:>12}",
        "min", "pasos", "ms/paso", "pérdida", "bits/byte"
    );

    let mut azar = Azar::con_semilla(7);
    let total = minutos * 60.0;
    // Diez medidas en toda la corrida, no más de una por minuto.
    let cada = (total / 10.0).max(60.0);
    let (mut pasos, mut suma, mut desde_medida) = (0usize, 0f64, 0usize);
    let (mut entrenado, mut proxima, mut tiempo_medida) = (0f64, cada, 0f64);
    while entrenado < total {
        // Sube los primeros 30 s y baja en coseno hasta el final.
        let p = entrenado / total;
        let calentar = (entrenado / 30.0).min(1.0);
        opt.set_learning_rate(
            lr * calentar * (0.5 * (1.0 + (std::f64::consts::PI * p).cos())).max(0.02),
        );

        let paso = Instant::now();
        let l = lote(&ids, b, t, &mut azar, &dev)?;
        let (perdida, gradientes) = m.perdida_y_gradientes(&l, partes)?;
        opt.step(&gradientes)?;
        let duro = paso.elapsed().as_secs_f64();
        entrenado += duro;
        tiempo_medida += duro;
        suma += f64::from(perdida);
        pasos += 1;
        desde_medida += 1;

        if entrenado >= proxima {
            let ms = tiempo_medida * 1e3 / desde_medida as f64;
            println!(
                "  {:>7.1}{pasos:>8}{ms:>10.0}{:>11.3}{:>12.3}",
                entrenado / 60.0,
                suma / desde_medida as f64,
                bits_por_byte(&m)?
            );
            proxima += cada;
            tiempo_medida = 0.0;
            suma = 0.0;
            desde_medida = 0;
        }
    }
    let final_ = bits_por_byte(&m)?;
    println!();
    println!("  final: {final_:.3} bits por byte, en {pasos} pasos");
    println!("  por token: {:.3} bits", final_ * bytes_por_token);
    let todo = bits_en_todo(&m, &ids_prueba, t, bytes_por_token, &dev)?;
    println!(
        "  en todo el texto de prueba: {todo:.3} bits por byte, ventanas de {:.0} bytes",
        t as f64 * bytes_por_token
    );

    // Lo que escribe, para mirarlo: a temperatura baja, desde un comienzo.
    let semilla = "En un lugar de la Mancha";
    let mut sec = if vocab == "bytes" {
        tok.codificar_bytes(semilla.as_bytes())
    } else {
        tok.codificar(semilla)
    };
    let total_tokens = (400.0 / bytes_por_token) as usize;
    let mut azar = Azar::con_semilla(11);
    for _ in 0..total_tokens {
        let desde = sec.len().saturating_sub(t);
        let n = sec.len() - desde;
        let ids = Tensor::new(&sec[desde..], &dev)?.unsqueeze(0)?;
        let log = m
            .forward(&ids)?
            .narrow(1, n - 1, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let temp = 0.7f32;
        let maximo = log.iter().cloned().fold(f32::MIN, f32::max);
        let pr: Vec<f32> = log.iter().map(|v| ((v - maximo) / temp).exp()).collect();
        let suma: f32 = pr.iter().sum();
        let mut u = (azar.siguiente() >> 40) as f32 / 16_777_216.0 * suma;
        let mut elegido = 0;
        for (i, p) in pr.iter().enumerate() {
            u -= p;
            if u <= 0.0 {
                elegido = i;
                break;
            }
        }
        sec.push(elegido as u32);
    }
    println!();
    println!("  {}", tok.decodificar(&sec)?.replace('\n', "\n  "));

    if let Ok(salida) = std::env::var("COSXAMIX_SALIDA") {
        let n = m.exportar(TipoGgml::F32, Some(&tok), Path::new(&salida))?;
        println!();
        println!("  guardado en {salida}: {:.1} MB", n as f64 / 1_048_576.0);
    }
    Ok(())
}
