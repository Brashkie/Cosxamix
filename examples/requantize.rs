//! Recuantiza un GGUF con nuestro cuantizador.
//!
//!     cargo run --release --example requantize -- origen.gguf destino.gguf q3k
//!     cargo run --release --example requantize -- origen.gguf destino.gguf como:otro.gguf
//!
//! Con `como:otro.gguf`, cada tensor va en el formato que tiene en `otro.gguf`:
//! es lo que permite comparar contra un archivo oficial en igualdad de
//! condiciones, porque un `Q3_K_M` deja algunos tensores en Q4_K a propósito.
//!
//! Lo que no sea una matriz con filas múltiplos de 256 —normas, sesgos— queda
//! en el formato que tenía.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use cosxamix::gguf::{Gguf, Pesos, TipoGgml, Valor};
use cosxamix::gguf_write::{Declaracion, Escritor};
use cosxamix::quant::{cuantizar_bloques, descuantizar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [origen, destino, pedido] = a.as_slice() else {
        println!("  uso: requantize origen.gguf destino.gguf (q2k|q3k|q4k|q8|como:otro.gguf)");
        return Ok(());
    };
    let (origen, destino) = (Path::new(origen), Path::new(destino));
    let g = Gguf::abrir(origen)?;
    let mapa = Pesos::abrir(&g, origen)?;

    let tipos: HashMap<String, TipoGgml> = if let Some(otro) = pedido.strip_prefix("como:") {
        let o = Gguf::abrir(Path::new(otro))?;
        o.tensores
            .iter()
            .map(|t| (t.nombre.clone(), t.tipo))
            .collect()
    } else {
        let tipo = match pedido.as_str() {
            "q2k" => TipoGgml::Q2K,
            "q3k" => TipoGgml::Q3K,
            "q4k" => TipoGgml::Q4K,
            "q8" => TipoGgml::Q8_0,
            otro => {
                return Err(format!("formato {otro}: q2k, q3k, q4k, q8 o como:otro.gguf").into())
            }
        };
        g.tensores
            .iter()
            .filter(|t| {
                t.dims.len() == 2 && t.dims[0].is_multiple_of(tipo.valores_por_bloque() as u64)
            })
            .map(|t| (t.nombre.clone(), tipo))
            .collect()
    };

    let mut e = Escritor::nuevo();
    let mut claves: Vec<&String> = g.metadatos.keys().collect();
    claves.sort();
    for clave in claves {
        if clave != "general.alignment" {
            e.meta(clave.clone(), g.metadatos[clave].clone());
        }
    }
    e.meta("cosxamix.requantized", Valor::Texto(pedido.clone()));
    for t in &g.tensores {
        let tipo = tipos.get(&t.nombre).copied().unwrap_or(t.tipo);
        e.tensor(Declaracion::nueva(t.nombre.clone(), &t.dims, tipo));
    }

    let t0 = Instant::now();
    let mut hechos = 0;
    let total = e.escribir(destino, |d, w| {
        let info = g
            .tensor(&d.nombre)
            .ok_or_else(|| format!("se perdió {}", d.nombre))?;
        let datos = mapa.de(info)?;
        // Lo que no cambia de formato se copia tal cual: volver a cuantizar
        // algo ya cuantizado solo agrega error.
        if info.tipo == d.tipo {
            return w.write_all(datos).map_err(|x| x.to_string());
        }
        let mut valores = vec![0f32; info.elementos()];
        descuantizar(info.tipo, datos, &mut valores)?;
        // De a tandas de bloques en todos los núcleos, y en orden.
        let por_bloque = d.tipo.valores_por_bloque();
        let partes: Vec<Result<Vec<u8>, String>> = valores
            .par_chunks(por_bloque * 1024)
            .map(|trozo| {
                let mut b = Vec::with_capacity(d.tipo.bytes_de(trozo.len()));
                cuantizar_bloques(d.tipo, trozo, &mut b)?;
                Ok(b)
            })
            .collect();
        for p in partes {
            w.write_all(&p?).map_err(|x| x.to_string())?;
        }
        hechos += 1;
        Ok(())
    })?;
    println!(
        "  {hechos} tensores recuantizados, {:.2} GB en {:.0} s",
        total as f64 / 1_073_741_824.0,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}
