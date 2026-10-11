//! Compara la búsqueda de embeddings por filas contra descuantizar la tabla.
//!
//! La versión por filas ahorra 431 MB, pero si devuelve otra cosa no sirve de
//! nada. Esto lo comprueba directamente, sin pasar por la generación: ahí un
//! error se ve como texto raro y no dice dónde está.

use std::fs::File;

use candle_core::quantized::{gguf_file, QStorage, QTensor};
use candle_core::{Device, IndexOp, Tensor};

fn main() {
    let dev = Device::Cpu;
    let mut f = File::open("modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf").expect("abrir");
    let c = gguf_file::Content::read(&mut f).expect("leer");

    let tabla = c.tensor(&mut f, "token_embd.weight", &dev).expect("tensor");
    let dims = tabla.shape().dims().to_vec();
    let dtype = tabla.dtype();
    println!("\n  token_embd.weight  {dims:?}  {dtype:?}");
    println!(
        "  block_size {}  type_size {}",
        dtype.block_size(),
        dtype.type_size()
    );

    let cols = dims[1];
    let bytes_fila = cols / dtype.block_size() * dtype.type_size();
    println!("  bytes por fila: {bytes_fila}");

    // ── Camino original: descuantizar todo y seleccionar ───────────────────
    let completo = tabla.dequantize(&dev).expect("dequantize");
    println!("  tabla completa en f32: {:?}", completo.shape().dims());

    // Se prueba el 0 en la PRIMERA posicion y tambien en la tercera: si falla
    // solo cuando va primero, el problema es la posicion en el recorte; si
    // falla siempre, es el indice.
    let ids: Vec<u32> = vec![0, 1, 0, 100, 9707, 151643, 0];
    let idx = Tensor::new(ids.as_slice(), &dev).expect("ids");
    let esperado = completo.index_select(&idx, 0).expect("index_select");

    // ── Camino nuevo: recortar bytes y descuantizar solo eso ───────────────
    let datos = tabla.data().expect("data");
    let mut recorte = Vec::with_capacity(ids.len() * bytes_fila);
    for &i in &ids {
        let desde = i as usize * bytes_fila;
        recorte.extend_from_slice(&datos[desde..desde + bytes_fila]);
    }
    let storage = QStorage::Cpu(dtype.from_data(std::borrow::Cow::Borrowed(&recorte)));
    let chico = QTensor::new(storage, (ids.len(), cols)).expect("qtensor");
    let obtenido = chico.dequantize(&dev).expect("dequantize chico");

    // ── Comparación ────────────────────────────────────────────────────────
    println!("\n  fila | max |dif| | primeros valores esperados / obtenidos");
    let mut peor = 0.0f32;
    for (k, &i) in ids.iter().enumerate() {
        let e: Vec<f32> = esperado.i(k).unwrap().to_vec1().unwrap();
        let o: Vec<f32> = obtenido.i(k).unwrap().to_vec1().unwrap();
        let d = e
            .iter()
            .zip(&o)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        if d > peor {
            peor = d;
        }
        println!(
            "  {i:>6} | {d:>9.6} | {:.4} {:.4} {:.4}  /  {:.4} {:.4} {:.4}",
            e[0], e[1], e[2], o[0], o[1], o[2],
        );
    }

    println!("\n  diferencia maxima: {peor:.8}");
    if peor > 1e-6 {
        eprintln!("\n  La busqueda por filas NO reproduce la tabla: hay un bug.");
        std::process::exit(1);
    }
    println!("  La busqueda por filas reproduce la tabla exactamente.");
}
