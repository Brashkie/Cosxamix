//! ¿A dónde se van los 938 MB si el archivo pesa 468?
//!
//! El §4.1 de la guía dice que la carga convencional "duplica el consumo de RAM
//! física" y propone `mmap` como solución. Antes de aceptar esa explicación
//! conviene mirar dónde está de verdad la memoria: si el grueso no es la copia
//! del archivo sino otra cosa, `mmap` resolvería poco.

use std::fs::File;

use candle_core::quantized::gguf_file;

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1048576.0
}

fn main() {
    let ruta = "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let mut f = File::open(ruta).expect("abrir gguf");
    let c = gguf_file::Content::read(&mut f).expect("leer gguf");

    let en_disco = std::fs::metadata(ruta).unwrap().len() as f64 / 1048576.0;
    println!("\n  archivo en disco   {en_disco:>8.1} MB");

    // ── Lo que ocupa cada tensor tal como está guardado ─────────────────────
    let mut total_cuantizado = 0usize;
    let mut mayores: Vec<(String, usize, String)> = Vec::new();

    for (nombre, info) in &c.tensor_infos {
        let elementos: usize = info.shape.dims().iter().product();
        // Tamaño real del bloque cuantizado, no de los elementos sueltos.
        let bytes = elementos * info.ggml_dtype.type_size() / info.ggml_dtype.block_size();
        total_cuantizado += bytes;
        mayores.push((nombre.clone(), bytes, format!("{:?}", info.ggml_dtype)));
    }

    mayores.sort_by_key(|(_, b, _)| std::cmp::Reverse(*b));

    println!(
        "  suma de tensores   {:>8.1} MB  (cuantizados)",
        mb(total_cuantizado)
    );
    println!("\n  los 5 mas grandes:");
    for (n, b, t) in mayores.iter().take(5) {
        println!("    {:<28} {:>8.1} MB  {t}", n, mb(*b));
    }

    // ── El que se descuantiza ───────────────────────────────────────────────
    //
    // `from_gguf` hace `tok_embeddings.dequantize(device)`, o sea que esa tabla
    // pasa a f32 entera y queda residente. Y despues se vuelve a cargar
    // cuantizada para la capa de salida, cuando el gguf no trae `output.weight`
    // aparte.
    if let Some(info) = c.tensor_infos.get("token_embd.weight") {
        let dims = info.shape.dims();
        let elementos: usize = dims.iter().product();
        let cuantizado = elementos * info.ggml_dtype.type_size() / info.ggml_dtype.block_size();
        let en_f32 = elementos * 4;

        println!("\n  ─────────────────────────────────────────");
        println!("  token_embd.weight  {dims:?}  ({} elementos)", elementos);
        println!("    cuantizado       {:>8.1} MB", mb(cuantizado));
        println!(
            "    descuantizado    {:>8.1} MB  <- a f32, residente",
            mb(en_f32)
        );

        let tiene_salida = c.tensor_infos.contains_key("output.weight");
        println!(
            "    output.weight aparte: {}",
            if tiene_salida {
                "si"
            } else {
                "no, reusa el embedding"
            }
        );

        let copia_extra = if tiene_salida { 0 } else { cuantizado };
        let estimado = total_cuantizado + en_f32 + copia_extra;

        println!("\n  estimacion de residente:");
        println!(
            "    tensores cuantizados   {:>8.1} MB",
            mb(total_cuantizado)
        );
        println!("    + embedding en f32     {:>8.1} MB", mb(en_f32));
        if copia_extra > 0 {
            println!("    + 2a copia p/ salida   {:>8.1} MB", mb(copia_extra));
        }
        println!("    = total                {:>8.1} MB", mb(estimado));
        println!("\n  medido con tasklist:        938,1 MB");

        println!("\n  El grueso del exceso NO es la copia del archivo: es la tabla de");
        println!(
            "  embeddings pasada a f32. Son {:.0} MB para un modelo de {:.0}.",
            mb(en_f32),
            en_disco
        );
        println!("  Y en un forward solo hacen falta las filas de los tokens del");
        println!(
            "  prompt: {:.1} KB por token, no {:.0} MB.",
            dims[1] as f64 * 4.0 / 1024.0,
            mb(en_f32)
        );
    }
}
