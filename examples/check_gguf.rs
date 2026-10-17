//! ¿Nuestro lector de GGUF lee lo mismo que el de candle?
//!
//! El test unitario de `src/gguf.rs` usa un archivo sintetico y prueba que el
//! parser sea consistente consigo mismo. Eso no alcanza: un parser puede estar
//! equivocado de forma coherente. Lo que decide es contrastarlo contra una
//! implementacion que ya funciona, sobre los archivos de verdad.
//!
//! Candle queda como **oraculo** aunque mas adelante salga del binario. Que el
//! motor sea propio no significa tirar la unica forma barata de saber si esta
//! bien.
//!
//!     cargo run --release --features candle --example check_gguf

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let mut revisados = 0usize;
    let mut fallos = 0usize;

    for entrada in std::fs::read_dir(&dir)? {
        let ruta = entrada?.path();
        if ruta.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        revisados += 1;
        let nombre = ruta.file_name().unwrap().to_string_lossy().into_owned();
        println!();
        println!("  {nombre}");

        let nuestro = cosxamix::gguf::Gguf::abrir(&ruta)?;

        let mut f = std::fs::File::open(&ruta)?;
        let suyo = candle_core::quantized::gguf_file::Content::read(&mut f)?;

        let mut mal = Vec::new();

        if nuestro.tensores.len() != suyo.tensor_infos.len() {
            mal.push(format!(
                "cantidad de tensores: {} contra {}",
                nuestro.tensores.len(),
                suyo.tensor_infos.len()
            ));
        }

        // Cada tensor: forma, tipo y desplazamiento.
        for t in &nuestro.tensores {
            match suyo.tensor_infos.get(&t.nombre) {
                None => mal.push(format!("{}: candle no lo tiene", t.nombre)),
                Some(s) => {
                    let dims_suyos: Vec<u64> = s.shape.dims().iter().map(|d| *d as u64).collect();
                    // Candle invierte el orden de las dimensiones respecto al
                    // archivo. Se compara contra el nuestro al reves, que es
                    // justo lo que hay que verificar y no dar por sabido.
                    let nuestras_invertidas: Vec<u64> = t.dims.iter().rev().copied().collect();
                    if dims_suyos != nuestras_invertidas {
                        mal.push(format!(
                            "{}: dims {:?} contra {:?} de candle",
                            t.nombre, nuestras_invertidas, dims_suyos
                        ));
                    }
                    if t.bytes()
                        != s.shape.elem_count() / s.ggml_dtype.block_size()
                            * s.ggml_dtype.type_size()
                    {
                        mal.push(format!("{}: bytes distintos", t.nombre));
                    }
                    if t.offset != s.offset {
                        mal.push(format!(
                            "{}: offset {} contra {}",
                            t.nombre, t.offset, s.offset
                        ));
                    }
                }
            }
        }

        // Y los bytes crudos de un tensor, que es lo unico que prueba que la
        // alineacion de la zona de datos se calculo igual. Un desplazamiento
        // de unos pocos bytes no da error: da pesos equivocados.
        if let Some(t) = nuestro.tensores.iter().find(|t| t.bytes() < 20_000_000) {
            let nuestros = nuestro.datos_de(&ruta, t)?;
            let suyos = {
                let mut f = std::fs::File::open(&ruta)?;
                let qt = suyo.tensor(&mut f, &t.nombre, &candle_core::Device::Cpu)?;
                qt.data()?.to_vec()
            };
            if nuestros != suyos {
                let iguales = nuestros
                    .iter()
                    .zip(&suyos)
                    .take_while(|(a, b)| a == b)
                    .count();
                mal.push(format!(
                    "{}: los bytes difieren desde el {} de {}",
                    t.nombre,
                    iguales,
                    nuestros.len()
                ));
            } else {
                println!(
                    "    bytes de {} ({} KB): iguales",
                    t.nombre,
                    nuestros.len() / 1024
                );
            }
        }

        // Metadatos que importan para construir el modelo.
        for clave in [
            "qwen2.embedding_length",
            "qwen2.block_count",
            "qwen2.attention.head_count",
            "qwen2.attention.head_count_kv",
            "qwen2.feed_forward_length",
        ] {
            let n = nuestro.meta_entero(clave);
            let s = suyo.metadata.get(clave).and_then(|v| v.to_u32().ok());
            if n.map(|v| v as u32) != s {
                mal.push(format!("{clave}: {n:?} contra {s:?}"));
            }
        }

        if mal.is_empty() {
            println!(
                "    {} tensores, {} metadatos, datos desde {}: TODO IGUAL",
                nuestro.tensores.len(),
                nuestro.metadatos.len(),
                nuestro.inicio_datos
            );
        } else {
            fallos += 1;
            for m in mal.iter().take(6) {
                println!("    DIFIERE  {m}");
            }
        }
    }

    println!();
    if revisados == 0 {
        println!("  no habia ningun .gguf en {dir}");
        return Ok(());
    }
    println!("  {revisados} modelos revisados, {fallos} con diferencias");
    if fallos > 0 {
        std::process::exit(1);
    }
    Ok(())
}
