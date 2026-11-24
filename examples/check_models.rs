//! Abre cada GGUF de una carpeta con la validación de límites de cada tensor
//! (`gguf::tramo_de_tensor`) y dice cuáles pasan.
//!
//! Sirve para comprobar que la regla no rechaza ningún archivo de verdad —los
//! oficiales, los daiss, los simulados—, que es lo que tiene que cumplir antes
//! que nada. Con `--features ada`, la regla la decide la versión demostrada
//! con SPARK.
//!
//!     cargo run --release --example check_models
//!     cargo run --release --features ada --example check_models -- otra/carpeta

use cosxamix::gguf::Gguf;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "modelos".into());
    let mut archivos: Vec<_> = std::fs::read_dir(&dir)
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
                .collect()
        })
        .unwrap_or_default();
    archivos.sort();
    let regla = if cfg!(feature = "ada") {
        "SPARK"
    } else {
        "Rust"
    };
    println!("  {} archivos en {dir}, regla en {regla}", archivos.len());
    let mut malos = 0;
    for a in &archivos {
        match Gguf::abrir(a) {
            Ok(g) => println!(
                "    ok     {} ({} tensores)",
                a.file_name().unwrap().to_string_lossy(),
                g.tensores.len()
            ),
            Err(e) => {
                malos += 1;
                println!(
                    "    MAL    {}: {e}",
                    a.file_name().unwrap().to_string_lossy()
                );
            }
        }
    }
    println!("  {} bien, {malos} rechazados", archivos.len() - malos);
}
