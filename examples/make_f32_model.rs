//! Convierte un GGUF cuantizado a uno en `F32`, para poder comparar sin
//! cuantización de por medio.
//!
//! Existe para zanjar una pregunta concreta. El forward propio se aparta de
//! candle un 1,9% del rango de los logits, y forzar nuestro camino sin
//! cuantizar **no** lo acerca —sube a 2,2%—, lo que apunta a la cuantización
//! de activaciones de candle a Q8_K. Pero eso era una inferencia, no una
//! medición: podría ser un error nuestro.
//!
//! Con los pesos en `F32` los dos caminos dejan de cuantizar nada: candle hace
//! una multiplicación en coma flotante y nosotros también, **sobre exactamente
//! los mismos números**. Si ahí los logits coinciden hasta el ruido de `f32`,
//! el forward propio es correcto y toda la diferencia era cuantización. Si
//! siguen apartándose un 2%, hay un error y está en nuestro lado.
//!
//! Esto **no recupera información**: los pesos ya venían cuantizados y pasarlos
//! a `F32` no los mejora. Lo que quita es la cuantización de las
//! *activaciones*, que es la que estaba en discusión.
//!
//! Va tensor por tensor, así que convertir un modelo más grande que la RAM
//! funciona igual.
//!
//!     cargo run --release --example make_f32_model
//!     COSXAMIX_GGUF=qwen2.5-3b-instruct-q3_k_m.gguf cargo run --release --example make_f32_model

use std::io::Write;

use cosxamix::gguf::{Gguf, Pesos, TipoGgml, Valor};
use cosxamix::gguf_write::{Declaracion, Escritor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let archivo = std::env::var("COSXAMIX_GGUF")
        .unwrap_or_else(|_| "qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let origen = std::path::Path::new(&dir).join(&archivo);
    if !origen.exists() {
        println!("  falta {}", origen.display());
        return Ok(());
    }
    let destino = std::path::Path::new(&dir).join(archivo.replace(".gguf", "-f32.gguf"));

    let g = Gguf::abrir(&origen)?;
    let mapa = Pesos::abrir(&g, &origen)?;

    let mut e = Escritor::nuevo();
    // Los metadatos van tal cual, menos el tipo de archivo, que cambia.
    let mut claves: Vec<&String> = g.metadatos.keys().collect();
    claves.sort();
    for clave in claves {
        if clave == "general.alignment" || clave == "general.file_type" {
            continue;
        }
        e.meta(clave.clone(), g.metadatos[clave].clone());
    }
    e.meta("general.file_type", Valor::U32(TipoGgml::F32 as u32));

    for t in &g.tensores {
        e.tensor(Declaracion::nueva(t.nombre.clone(), &t.dims, TipoGgml::F32));
    }

    let origen_bytes: u64 = g.tensores.iter().map(|t| t.bytes() as u64).sum();
    println!();
    println!(
        "  {archivo}: {} tensores, {:.2} GB -> {:.2} GB en F32",
        g.tensores.len(),
        origen_bytes as f64 / 1_073_741_824.0,
        e.bytes_de_datos() as f64 / 1_073_741_824.0,
    );
    println!("  -> {}", destino.display());

    let t0 = std::time::Instant::now();
    let mut hecho = 0usize;
    let mut valores = Vec::new();
    let total = e.escribir(&destino, |d, w| {
        let info = g
            .tensor(&d.nombre)
            .ok_or_else(|| format!("se perdio {}", d.nombre))?;
        let n = info.elementos();
        valores.clear();
        valores.resize(n, 0.0f32);
        cosxamix::quant::descuantizar(info.tipo, mapa.de(info)?, &mut valores)?;

        // Se escribe por trozos y no de una: un tensor de 151.936 x 896
        // en F32 son 545 MB, y convertirlo entero a bytes de golpe
        // duplicaria ese pico sin ninguna necesidad.
        let mut bytes = Vec::with_capacity(1 << 20);
        for trozo in valores.chunks(1 << 18) {
            bytes.clear();
            cosxamix::quant::cuantizar_bloques(TipoGgml::F32, trozo, &mut bytes)?;
            w.write_all(&bytes).map_err(|x| x.to_string())?;
        }
        hecho += 1;
        if hecho.is_multiple_of(50) {
            print!(".");
            let _ = std::io::stdout().flush();
        }
        Ok(())
    })?;
    println!();
    println!(
        "  {hecho} tensores, {:.2} GB en {:.0} s ({:.0} MB/s)",
        total as f64 / 1_073_741_824.0,
        t0.elapsed().as_secs_f64(),
        total as f64 / 1_048_576.0 / t0.elapsed().as_secs_f64(),
    );

    // La comprobación: un tensor cualquiera tiene que leerse igual desde los
    // dos archivos. Si la conversión perdiera algo, acá se ve.
    let nuevo = Gguf::abrir(&destino)?;
    let mapa_nuevo = Pesos::abrir(&nuevo, &destino)?;
    let mut peor = 0f32;
    for nombre in [
        "token_embd.weight",
        "blk.0.ffn_gate.weight",
        "output_norm.weight",
    ] {
        let (Some(a), Some(b)) = (g.tensor(nombre), nuevo.tensor(nombre)) else {
            continue;
        };
        let mut va = vec![0f32; a.elementos()];
        let mut vb = vec![0f32; b.elementos()];
        cosxamix::quant::descuantizar(a.tipo, mapa.de(a)?, &mut va)?;
        cosxamix::quant::descuantizar(b.tipo, mapa_nuevo.de(b)?, &mut vb)?;
        let d = va
            .iter()
            .zip(&vb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        peor = peor.max(d);
        println!("    {nombre}: {} valores, peor diferencia {d}", va.len());
    }
    if peor == 0.0 {
        println!("  la conversion es exacta: los mismos numeros, otro formato");
    } else {
        return Err(format!("la conversion perdio {peor}").into());
    }

    println!();
    println!(
        "  Ahora: COSXAMIX_GGUF={} cargo run --release \\",
        destino.file_name().unwrap().to_string_lossy()
    );
    println!("    --features candle --example compare_forward");
    println!();
    println!("  Con los pesos en F32 ninguno de los dos cuantiza nada. Si los");
    println!("  logits coinciden hasta el ruido de f32, el forward propio es");
    println!("  correcto y la diferencia era cuantizacion. Si siguen apartandose");
    println!("  un 2 por ciento, hay un error y esta de nuestro lado.");
    println!();

    Ok(())
}
