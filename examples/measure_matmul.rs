//! ¿Por qué de 6 núcleos salen 2,6x y no 6x?
//!
//! Es la pregunta abierta del ROADMAP, y hasta ahora solo se midió el forward
//! completo, que mezcla todo: las multiplicaciones de matrices, las
//! normalizaciones, RoPE, el softmax, los residuales, y las barreras del pool
//! de hilos entre capa y capa.
//!
//! Este ejemplo aísla **una sola multiplicación**, que es donde está el 99% de
//! los parámetros, y la mide a distintos números de hilos. Separa dos
//! hipótesis que el forward completo no puede separar:
//!
//!   A. las multiplicaciones no escalan  -> el problema es el kernel, o el
//!      ancho de banda de la memoria;
//!   B. escalan bien                     -> el problema está FUERA de ellas:
//!      en las partes que corren en serie o en el coste de sincronizar 36
//!      veces por token.
//!
//! Se mide el caso de **decode**, una fila de entrada, que es el que domina la
//! latencia percibida y el que la tabla de hilos del README midió.
//!
//!     cargo run --release --features candle --example measure_matmul

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Module, Tensor};

/// Las matrices del 3B que vale la pena mirar, con su participación real.
///
/// `attn_k` y `attn_v` están a propósito pese a ser diminutas: son las que el
/// reparto de candle no puede partir en más de 2 trozos, porque tienen 256
/// columnas de salida y el `with_min_len(128)` pide 128 por trozo. Sirven de
/// control: si el tope de 2 hilos se ve acá y no en las grandes, se confirma
/// que ese detalle no explica la brecha global.
const MATRICES: &[(&str, &str)] = &[
    ("blk.0.ffn_gate.weight", "FFN gate   2048 -> 11008"),
    ("blk.0.ffn_down.weight", "FFN down  11008 ->  2048"),
    ("blk.0.attn_q.weight", "attn q     2048 ->  2048"),
    ("blk.0.attn_k.weight", "attn k     2048 ->   256"),
    ("output.weight", "salida     2048 -> 151936"),
];

const HILOS: &[usize] = &[1, 2, 4, 6, 12];

fn main() -> candle_core::Result<()> {
    let ruta = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "modelos/qwen2.5-3b-instruct-q3_k_m.gguf".into());

    let dev = Device::Cpu;
    let mut f = File::open(Path::new(&ruta)).expect("abrir el gguf");
    let ct = gguf_file::Content::read(&mut f).expect("leer el gguf");

    println!("\n  {ruta}");
    println!("  decode: una fila de entrada, que es el caso que domina la latencia\n");
    print!("  {:<26}", "matriz");
    for h in HILOS {
        print!(
            "{:>12}",
            format!("{h} hilo{}", if *h == 1 { "" } else { "s" })
        );
    }
    println!();
    print!("  {:-<26}", "");
    for _ in HILOS {
        print!("{:->12}", "");
    }
    println!();

    for (nombre, etiqueta) in MATRICES {
        let qt = match ct.tensor(&mut f, nombre, &dev) {
            Ok(t) => t,
            Err(_) => {
                println!("  {etiqueta:<26}  (no está en este modelo)");
                continue;
            }
        };
        // El ancho de la entrada sale de la matriz: candle guarda [salidas,
        // entradas] y multiplica x por su transpuesta, asi que la segunda
        // dimension es la que tiene que coincidir. Fijarlo a mano funciona
        // para las de 2048 y revienta con ffn_down, que entra con 11008.
        let k = qt.shape().dims()[1];
        let entrada = Tensor::zeros((1, k), candle_core::DType::F32, &dev)?;
        let mm = QMatMul::from_qtensor(qt)?;

        // La capa de salida es 30 veces mas grande que las demas: con las
        // mismas repeticiones tardaria eternidades sin decir nada nuevo.
        let reps = if nombre.starts_with("output") {
            20
        } else {
            200
        };

        let mut tiempos = Vec::new();
        for &h in HILOS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(h)
                .build()
                .expect("pool");

            // `install` hace que el `par_iter` interno de candle use ESTE pool
            // y no el global, que es lo que permite variar los hilos sin
            // relanzar el proceso.
            // Mediana de 3, no una corrida. Es la practica del resto de las
            // mediciones de este proyecto por un motivo medido: una sola dio
            // 21,6 y 32,3 tok/s para el mismo modelo. Sin esto, un pico de
            // ruido se lee como una conclusion.
            let mut muestras = Vec::new();
            for _ in 0..3 {
                let ms = pool.install(|| -> candle_core::Result<f64> {
                    let _ = mm.forward(&entrada)?; // calentar la cache
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        let _ = mm.forward(&entrada)?;
                    }
                    Ok(t0.elapsed().as_secs_f64() * 1000.0 / reps as f64)
                })?;
                muestras.push(ms);
            }
            muestras.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tiempos.push(muestras[1]);
        }

        print!("  {etiqueta:<26}");
        for (i, ms) in tiempos.iter().enumerate() {
            if i == 0 {
                print!("{:>12}", format!("{ms:.3} ms"));
            } else {
                print!("{:>12}", format!("{:.2}x", tiempos[0] / ms));
            }
        }
        println!();
    }

    // ── Segunda pregunta: ¿se reusan los pesos entre filas de un lote? ──
    //
    // La decodificacion especulativa se borro porque procesar 5 posiciones
    // costaba 3,74x en vez de ~1x, y de ahi se concluyo que el motor estaba
    // limitado por CALCULO. Esa conclusion solo se sostiene si el kernel lee
    // los pesos una vez para todo el lote. Si en cambio recorre fila por fila
    // releyendo la matriz entera cada vez, el 3,74x no dice nada sobre el
    // hardware: dice que la implementacion no amortiza.
    //
    // Se mide con la misma matriz y distinto numero de filas.
    println!();
    println!("  Coste por numero de filas de entrada, FFN gate, 6 hilos");
    println!(
        "  {:<10}{:>12}{:>12}{:>16}",
        "filas", "ms", "vs 1 fila", "ms por fila"
    );
    println!("  {:-<50}", "");

    let qt = ct.tensor(&mut f, "blk.0.ffn_gate.weight", &dev)?;
    let k = qt.shape().dims()[1];
    let mm = QMatMul::from_qtensor(qt)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .expect("pool");

    let mut base = 0.0f64;
    for filas in [1usize, 2, 4, 5, 8] {
        let x = Tensor::zeros((filas, k), candle_core::DType::F32, &dev)?;
        let mut muestras = Vec::new();
        for _ in 0..3 {
            let ms = pool.install(|| -> candle_core::Result<f64> {
                let _ = mm.forward(&x)?;
                let t0 = Instant::now();
                for _ in 0..50 {
                    let _ = mm.forward(&x)?;
                }
                Ok(t0.elapsed().as_secs_f64() * 1000.0 / 50.0)
            })?;
            muestras.push(ms);
        }
        muestras.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = muestras[1];
        if filas == 1 {
            base = ms;
        }
        println!(
            "  {:<10}{:>12}{:>12}{:>16}",
            filas,
            format!("{ms:.3}"),
            format!("{:.2}x", ms / base),
            format!("{:.3}", ms / filas as f64)
        );
    }

    println!();
    println!("  Si la columna de la derecha baja al crecer las filas, el kernel");
    println!("  amortiza la lectura de los pesos y el lote sale barato. Si se");
    println!("  queda plana, cada fila vuelve a leer la matriz entera y un lote");
    println!("  de K cuesta K, que es justo lo que la especulativa necesita que");
    println!("  NO pase.");

    println!();
    println!("  La primera columna es el tiempo absoluto; las demas, cuanto");
    println!("  acelera respecto a un hilo.");
    println!();
    println!("  Si las matrices grandes se acercan a 6x con 6 hilos, las");
    println!("  multiplicaciones NO son el cuello y hay que mirar lo que las");
    println!("  rodea. Si se quedan en 2,6x como el forward completo, el");
    println!("  cuello esta dentro del kernel o en la memoria.");

    Ok(())
}
