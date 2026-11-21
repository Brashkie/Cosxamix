//! ¿El modelo de entrenamiento calcula lo mismo que el de inferencia?
//!
//! Es la comprobacion que decide si entrenar sirve de algo. El modelo de
//! `src/training/` es una reimplementacion en f32 del motor propio
//! (`src/forward.rs`), escrita porque el gradiente no atraviesa una multiplicacion
//! cuantizada. Si esa reimplementacion tiene un error —el orden de RoPE, como
//! se repiten las cabezas KV, la escala de la atencion— el entrenamiento
//! optimizaria contra un modelo que no es el que despues se sirve, y nada lo
//! avisaria: la perdida bajaria igual.
//!
//! No se exige igualdad exacta: uno descuantiza a f32 y el otro multiplica en
//! enteros, asi que los numeros tienen que parecerse, no coincidir. Lo que si
//! se exige es que **elijan el mismo token**, que es lo unico que el modelo
//! produce de verdad.
//!
//!     cargo run --release --features training --example check_training_model

use std::path::Path;

use cosxamix::training::ModeloEntrenable;

fn main() -> Result<(), String> {
    let ruta = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());

    println!();
    println!("  {ruta}");
    println!("  descuantizando a f32...");
    let t0 = std::time::Instant::now();
    // Sin adaptadores (r=0 no vale, se usa r=1 y B en cero: no cambia nada).
    let f32m = ModeloEntrenable::cargar(Path::new(&ruta), 1, 16.0)?;
    println!("  listo en {:.1} s", t0.elapsed().as_secs_f64());
    let c = f32m.config();
    println!(
        "  oculto={} intermedio={} capas={} cabezas={}/{} vocab={}",
        c.oculto, c.intermedio, c.capas, c.cabezas, c.cabezas_kv, c.vocabulario
    );

    println!();
    println!("  el mismo modelo por el camino cuantizado, con el motor propio...");
    let qm = cosxamix::forward::Modelo::abrir(Path::new(&ruta))?;

    // Ids de un prompt real, no inventados: con ids al azar la distribucion
    // queda plana y el token mas probable lo decide el ruido.
    let ids: Vec<u32> = vec![
        151644, 872, 198, 3838, 374, 279, 6722, 315, 9870, 30, 151645, 198, 151644, 77091, 198,
    ];
    let l_f32 = f32m.logits(&ids);
    let a = &l_f32[l_f32.len() - c.vocabulario..];
    let mut cache = cosxamix::forward::Cache::nuevo(&qm.cfg);
    let b = qm.forward(&ids, &mut cache)?;
    let b = &b[b.len() - c.vocabulario..];

    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0
    };
    let rango = |v: &[f32]| -> f32 {
        v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
    };
    let dif_max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);

    println!();
    println!("  {:<30}{:>14}", "logits comparados", a.len());
    println!(
        "  {:<30}{:>14}",
        "rango de los logits",
        format!("{:.2}", rango(b))
    );
    println!(
        "  {:<30}{:>14}",
        "diferencia maxima",
        format!(
            "{dif_max:.3}  ({:.1}% del rango)",
            dif_max / rango(b) * 100.0
        )
    );
    let (ta, tb) = (argmax(a), argmax(b));
    println!(
        "  {:<30}{:>14}",
        "token mas probable",
        format!("{ta} f32 | {tb} cuant")
    );
    println!();
    if ta == tb {
        println!("  IGUAL: el modelo de entrenamiento es el mismo que el de inferencia.");
        Ok(())
    } else {
        println!("  DISTINTO: la reimplementacion en f32 tiene un error.");
        std::process::exit(1)
    }
}
