//! Crea un modelo daiss y comprueba que el archivo sirve.
//!
//! Imprime primero la familia entera con sus cuentas, que es lo que permite
//! decidir qué tamaño tiene sentido en qué máquina, y después escribe el que
//! se le pida y lo vuelve a leer.
//!
//! La comprobación que vale es la última: con `--features candle` el archivo
//! se lee **también** con candle, que es una implementación independiente que
//! no comparte una línea con la nuestra. Que las dos lean lo mismo es la
//! diferencia entre escribir GGUF y escribir un archivo que solo nosotros
//! entendemos.
//!
//!     cargo run --release --example create_daiss
//!     COSXAMIX_DAISS=daiss-7b cargo run --release --example create_daiss
//!     cargo run --release --features candle --example create_daiss

use cosxamix::daiss::{Config, FAMILIA};
use cosxamix::gguf::{Gguf, TipoGgml};

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!();
    println!("  La familia daiss");
    println!();
    println!(
        "  {:<12}{:>8}{:>7}{:>7}{:>8}{:>10}{:>10}{:>10}{:>12}",
        "modelo", "params", "capas", "dim", "cab/kv", "Q8_0", "F16", "F32", "cache/token"
    );
    println!("  {:-<84}", "");
    for c in FAMILIA {
        println!(
            "  {:<12}{:>8}{:>7}{:>7}{:>8}{:>10}{:>10}{:>10}{:>12}",
            c.nombre,
            format!("{:.2}B", c.parametros() as f64 / 1e9),
            c.capas,
            c.dim,
            format!("{}/{}", c.cabezas, c.cabezas_kv),
            format!("{:.2} GB", gb(c.bytes_en(TipoGgml::Q8_0))),
            format!("{:.2} GB", gb(c.bytes_en(TipoGgml::F16))),
            format!("{:.2} GB", gb(c.bytes_en(TipoGgml::F32))),
            format!("{:.1} KB", c.bytes_de_cache_por_token() as f64 / 1024.0),
        );
    }
    println!();
    let sin: Vec<&str> = FAMILIA
        .iter()
        .filter(|c| !c.admite_kquants())
        .map(|c| c.nombre)
        .collect();
    if !sin.is_empty() {
        println!(
            "  Sin k-quants en toda la red: {}. Los k-quants piden filas de 256",
            sin.join(", ")
        );
        println!("  y ahi la atencion no las tiene; es lo mismo que le pasa al");
        println!("  Qwen2.5-0.5B real, cuyos GGUF mezclan formatos por este motivo.");
        println!();
    }

    let nombre = std::env::var("COSXAMIX_DAISS").unwrap_or_else(|_| "daiss-mini".into());
    let Some(cfg) = Config::de_nombre(&nombre) else {
        println!("  No existe {nombre}. Elegi uno de la tabla con COSXAMIX_DAISS.");
        return Ok(());
    };

    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    std::fs::create_dir_all(&dir)?;
    let ruta = std::path::Path::new(&dir).join(format!("{nombre}-q8_0.gguf"));
    let semilla: u64 = std::env::var("COSXAMIX_SEMILLA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2026);

    println!("  Escribiendo {nombre} con semilla {semilla}");
    println!("  -> {}", ruta.display());
    println!();

    // Con COSXAMIX_BPE el modelo usa NUESTRO vocabulario y lo lleva dentro.
    // Sin el, usa el prestado de Qwen2.5.
    let tok = match std::env::var("COSXAMIX_BPE") {
        Ok(p) => {
            let t = cosxamix::tokenizer::Bpe::cargar(std::path::Path::new(&p))?;
            println!("  tokenizador propio: {} tokens, de {p}", t.tamano());
            Some(t)
        }
        Err(_) => None,
    };
    let r = cosxamix::daiss::escribir_con(cfg, TipoGgml::Q8_0, semilla, tok.as_ref(), &ruta)?;
    println!(
        "  {} tensores, {:.2}B parametros, {:.2} GB en {:.1} s ({:.0} MB/s)",
        r.tensores,
        r.parametros as f64 / 1e9,
        gb(r.bytes),
        r.segundos,
        r.bytes as f64 / 1_048_576.0 / r.segundos,
    );

    // ── leerlo con el nuestro ───────────────────────────────────────────────
    let g = Gguf::abrir(&ruta)?;
    let suma: usize = g.tensores.iter().map(|t| t.bytes()).sum();
    println!();
    println!(
        "  Nuestro lector: {} tensores, {} metadatos",
        g.tensores.len(),
        g.metadatos.len()
    );
    println!(
        "    arquitectura {:?}, nombre {:?}",
        g.metadatos
            .get("general.architecture")
            .and_then(|v| v.texto()),
        g.metadatos.get("general.name").and_then(|v| v.texto()),
    );
    assert_eq!(g.tensores.len(), r.tensores);
    assert_eq!(g.meta_entero("qwen2.block_count"), Some(cfg.capas as u64));
    println!(
        "    la suma de los tensores es {:.2} GB y el archivo {:.2} GB",
        gb(suma as u64),
        gb(r.bytes)
    );

    // Si el modelo lleva tokenizador, el que sale del archivo tiene que
    // codificar igual que el que entro.
    if let Some(t) = &tok {
        let vuelta = cosxamix::daiss::tokenizador_de(&g)?.ok_or("no trajo el tokenizador")?;
        let muestra = "la multiplicacion fusionada gana con una fila";
        assert_eq!(vuelta.codificar(muestra), t.codificar(muestra));
        println!(
            "    el tokenizador viaja dentro y codifica igual: {} tokens",
            vuelta.tamano()
        );
    }

    // El ultimo tensor es el que descubre un offset mal calculado: si la
    // cuenta se corrio en algun punto, es el que se sale del archivo.
    let ultimo = g.tensores.last().unwrap();
    let bytes = g.datos_de(&ruta, ultimo)?;
    println!(
        "    el ultimo tensor, {}, se lee entero: {} bytes",
        ultimo.nombre,
        bytes.len()
    );

    #[cfg(feature = "candle")]
    {
        println!();
        println!("  candle, que no comparte una linea con nuestro escritor:");
        let mut f = std::fs::File::open(&ruta)?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut f)?;
        println!(
            "    {} tensores, {} metadatos",
            ct.tensor_infos.len(),
            ct.metadata.len()
        );
        assert_eq!(ct.tensor_infos.len(), r.tensores, "candle conto otra cosa");

        // Que lea la cabecera no alcanza: hay que cargar un tensor y
        // comprobar que los bytes son LOS MISMOS que lee el nuestro.
        let dev = candle_core::Device::Cpu;
        let qt = ct.tensor(&mut f, &ultimo.nombre, &dev)?;
        let suyo = qt.dequantize(&dev)?.flatten_all()?.to_vec1::<f32>()?;
        let mut nuestro = vec![0f32; ultimo.elementos()];
        cosxamix::quant::descuantizar(ultimo.tipo, &bytes, &mut nuestro)?;
        let distintos = nuestro.iter().zip(&suyo).filter(|(a, b)| a != b).count();
        println!(
            "    {} descuantizado: {} valores, {} distintos",
            ultimo.nombre,
            suyo.len(),
            distintos
        );
        assert_eq!(distintos, 0, "candle lee otros valores que nosotros");

        // Y lo ultimo: que el modelo entero se construya. Es lo que falla si
        // falta un tensor o si un nombre no es el que el cargador espera.
        //
        // Solo hasta 4 GB. Construirlo carga el modelo entero en memoria, y
        // esta maquina tiene 7,3 libres: con un daiss-7b fallaria por falta
        // de RAM, que no dice nada sobre si el archivo esta bien escrito.
        const TOPE_CARGA: u64 = 4 << 30;
        if r.bytes <= TOPE_CARGA {
            let mut f = std::fs::File::open(&ruta)?;
            let ct = candle_core::quantized::gguf_file::Content::read(&mut f)?;
            match cosxamix::model::qwen2::ModelWeights::from_gguf(ct, &mut f, &dev) {
                Ok(_) => println!("    el modelo se construye entero: daiss es cargable"),
                Err(e) => {
                    println!("    NO se construye: {e}");
                    return Err(e.into());
                }
            }
        } else {
            println!(
                "    {:.1} GB: no se construye aca, no entra en la RAM libre",
                gb(r.bytes)
            );
            // Lo que SI se puede comprobar sin cargarlo: que cada tensor que
            // el cargador va a pedir existe y tiene la forma que espera.
            let mut faltan = Vec::new();
            for d in cfg.declaraciones(TipoGgml::Q8_0) {
                match ct.tensor_infos.get(&d.nombre) {
                    Some(i) => {
                        let suyas: Vec<u64> = i.shape.dims().iter().map(|x| *x as u64).collect();
                        // candle devuelve la forma al reves que el archivo.
                        let mut nuestras = d.dims.clone();
                        nuestras.reverse();
                        if suyas != nuestras {
                            faltan.push(format!("{}: {suyas:?} vs {nuestras:?}", d.nombre));
                        }
                    }
                    None => faltan.push(format!("{}: no esta", d.nombre)),
                }
            }
            if faltan.is_empty() {
                println!(
                    "    los {} tensores que el cargador pide estan, con su forma",
                    cfg.declaraciones(TipoGgml::Q8_0).len()
                );
            } else {
                for f in faltan.iter().take(5) {
                    println!("    PROBLEMA {f}");
                }
                return Err(format!("{} tensores mal", faltan.len()).into());
            }
        }
    }

    #[cfg(not(feature = "candle"))]
    println!("\n  (con --features candle se comprueba ademas con su lector)");

    println!();
    println!("  Lo que esto NO es: un modelo entrenado. Los pesos son ruido con");
    println!("  la desviacion correcta, asi que genera texto sin sentido. Lo que");
    println!("  si es: el camino entero verificado, que es lo que hace falta");
    println!("  tener antes de poder entrenar algo.");
    println!();

    Ok(())
}
