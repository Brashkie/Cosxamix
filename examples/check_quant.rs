//! ¿Nuestra descuantización da lo mismo que la de candle?
//!
//! Se exige **igualdad exacta**, no una tolerancia. Cada valor es una cuenta
//! determinista de enteros y una multiplicación: no hay nada que redondear
//! distinto. Un solo bit de diferencia significa que un desplazamiento está
//! mal, y en estos formatos un desplazamiento mal no da error — da pesos
//! plausibles pero equivocados, que es la peor forma de fallar.
//!
//! Se recorre cada modelo y, de cada tipo que aparezca, se toma un tensor.
//! Así los ocho formatos quedan cubiertos por los archivos de verdad y no por
//! bloques inventados.
//!
//!     cargo run --release --features candle --example check_quant

use std::collections::HashSet;

use cosxamix::gguf::Gguf;
use cosxamix::quant::descuantizar;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let mut fallos = 0usize;
    let mut tipos_vistos: HashSet<String> = HashSet::new();
    let mut comprobados = 0usize;

    for entrada in std::fs::read_dir(&dir)? {
        let ruta = entrada?.path();
        if ruta.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        println!();
        println!("  {}", ruta.file_name().unwrap().to_string_lossy());

        let g = Gguf::abrir(&ruta)?;
        let mut f = std::fs::File::open(&ruta)?;
        let suyo = candle_core::quantized::gguf_file::Content::read(&mut f)?;

        // Un tensor por tipo, el más chico de cada uno: alcanza para validar
        // el formato y no cuesta gigabytes.
        let mut por_tipo: std::collections::HashMap<String, &cosxamix::gguf::InfoTensor> =
            std::collections::HashMap::new();
        for t in &g.tensores {
            let clave = format!("{:?}", t.tipo);
            match por_tipo.get(&clave) {
                Some(y) if y.bytes() <= t.bytes() => {}
                _ => {
                    por_tipo.insert(clave, t);
                }
            }
        }

        let mut claves: Vec<_> = por_tipo.keys().cloned().collect();
        claves.sort();

        for clave in claves {
            let t = por_tipo[&clave];
            tipos_vistos.insert(clave.clone());
            comprobados += 1;

            let datos = g.datos_de(&ruta, t)?;
            let mut nuestro = vec![0f32; t.elementos()];
            descuantizar(t.tipo, &datos, &mut nuestro)?;

            let mut f = std::fs::File::open(&ruta)?;
            let qt = suyo.tensor(&mut f, &t.nombre, &candle_core::Device::Cpu)?;
            let suyos = qt
                .dequantize(&candle_core::Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            if nuestro.len() != suyos.len() {
                println!(
                    "    {clave:<6} LARGO distinto: {} vs {}",
                    nuestro.len(),
                    suyos.len()
                );
                fallos += 1;
                continue;
            }

            match nuestro.iter().zip(&suyos).position(|(a, b)| a != b) {
                None => println!(
                    "    {clave:<6} {:<28} {:>9} valores: IDENTICO",
                    t.nombre,
                    nuestro.len()
                ),
                Some(i) => {
                    println!(
                        "    {clave:<6} {:<28} DIFIERE en el valor {i}: {} contra {}",
                        t.nombre, nuestro[i], suyos[i]
                    );
                    fallos += 1;
                }
            }
        }
    }

    let mut lista: Vec<_> = tipos_vistos.into_iter().collect();
    lista.sort();
    println!();
    println!(
        "  {comprobados} tensores comprobados, {} tipos: {}",
        lista.len(),
        lista.join(", ")
    );
    if fallos > 0 {
        println!("  {fallos} con diferencias");
        std::process::exit(1);
    }
    println!("  cero diferencias");
    Ok(())
}
