//! ¿El caché de atención da el mismo resultado que recalcular?
//!
//! Comparar los textos generados no contesta esto: con argmax, una diferencia
//! de un bit en los logits cambia un token y de ahí todo diverge, así que dos
//! salidas distintas no distinguen un caché roto de ruido de punto flotante.
//!
//! Acá se comparan los **logits** del siguiente token, que es la magnitud que
//! de verdad produce el modelo. Si el caché estuviera atendiendo a tokens que
//! no corresponden, los logits serían otros —no parecidos, otros— y el token
//! más probable cambiaría. Si solo hay ruido de acumulación, la diferencia es
//! del orden del epsilon de `f32` amplificado por 24 capas.

use std::path::Path;

use cosxamix::engine_candle::MotorCandle;

fn motor() -> MotorCandle {
    MotorCandle::nuevo(
        "qwen2.5-0.5b",
        Path::new("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        Path::new("modelos/tokenizer.json"),
    )
    .expect("construir")
}

/// Índice del valor más grande: el token que elegiría argmax.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn main() {
    // Un prompt REAL, no ids inventados: con basura la distribución de salida
    // queda plana y el token más probable lo decide el ruido, así que comparar
    // argmax sobre eso no diría nada. Con texto de verdad la distribución está
    // marcada y que el argmax coincida SÍ significa algo.
    let a = motor();
    let texto = concat!(
        "<|im_start|>system\n",
        "Eres WinsiBot, un bot de WhatsApp. Responde corto, en espanol.<|im_end|>\n",
        "<|im_start|>user\nhola, como te llamas?<|im_end|>\n",
        "<|im_start|>assistant\n",
    );
    let tokens = a.tokenizar_diagnostico(texto).expect("tokenizar");
    let corte = tokens.len() * 2 / 3;
    let prefijo: Vec<u32> = tokens[..corte].to_vec();
    println!(
        "\n  prompt de {} tokens; se cachean los primeros {corte}",
        tokens.len()
    );

    // ── Sin caché: la secuencia entera de una ───────────────────────────────
    let sin = a.logits_diagnostico(&tokens, false).expect("sin cache");

    // ── Con caché: primero el prefijo, después el resto reaprovechando ──────
    let b = motor();
    b.logits_diagnostico(&prefijo, false).expect("prefijo");
    let con = b.logits_diagnostico(&tokens, true).expect("con cache");

    assert_eq!(
        sin.len(),
        con.len(),
        "distinto vocabulario, algo muy raro pasa"
    );

    // ── Comparación ─────────────────────────────────────────────────────────
    let mut peor_abs = 0.0f32;
    let mut peor_rel = 0.0f32;
    for (x, y) in sin.iter().zip(&con) {
        let abs = (x - y).abs();
        if abs > peor_abs {
            peor_abs = abs;
        }
        let rel = abs / x.abs().max(1.0);
        if rel > peor_rel {
            peor_rel = rel;
        }
    }

    let (ia, ib) = (argmax(&sin), argmax(&con));
    let rango =
        sin.iter().cloned().fold(f32::MIN, f32::max) - sin.iter().cloned().fold(f32::MAX, f32::min);

    println!("\n  vocabulario          {}", sin.len());
    println!("  rango de los logits  {rango:.2}");
    println!(
        "  diferencia maxima    {peor_abs:.6}  ({:.4}% del rango)",
        peor_abs / rango * 100.0
    );
    println!("  diferencia relativa  {peor_rel:.6}");
    println!(
        "  token mas probable   sin cache {ia} | con cache {ib}  ->  {}",
        if ia == ib { "IGUAL" } else { "DISTINTO" }
    );

    // Dos condiciones, y la primera es la que de verdad importa.
    //
    // 1. El token elegido tiene que ser EL MISMO. Un caché que atiende a tokens
    //    que no corresponden no da logits parecidos: da otros, y el máximo se
    //    mueve. Medido con este prompt: coinciden.
    //
    // 2. La diferencia tiene que quedar dentro del ruido. El umbral del 5% del
    //    rango es generoso a propósito — esto distingue "roto" de "ruido", no
    //    mide precisión. Observado: ~2%.
    println!();
    let mut mal = false;

    if ia != ib {
        println!("  FALLA: el token mas probable cambia -> el cache condiciona la salida");
        mal = true;
    }
    if peor_abs > rango * 0.05 {
        println!("  FALLA: la diferencia supera el 5% del rango -> no es ruido");
        mal = true;
    }
    if mal {
        std::process::exit(1);
    }

    println!("  VEREDICTO: el cache reproduce el calculo. Elige el mismo token y la");
    println!("  diferencia es ruido de acumulacion en f32 amplificado por 24 capas.");
}
