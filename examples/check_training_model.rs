//! ¿El modelo de entrenamiento calcula lo mismo que el de inferencia?
//!
//! Es la comprobacion que decide si entrenar sirve de algo. El modelo de
//! `src/training/` es una reimplementacion en f32 del que vive en
//! `src/model/`, escrita porque el gradiente no atraviesa una multiplicacion
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

use candle_core::{Device, IndexOp, Tensor};
use cosxamix::training::ModeloEntrenable;

fn main() -> candle_core::Result<()> {
    let ruta = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let dev = Device::Cpu;

    println!();
    println!("  {ruta}");
    println!("  descuantizando a f32...");
    let t0 = std::time::Instant::now();
    // Sin adaptadores (r=0 no vale, se usa r=1 y B en cero: no cambia nada).
    let f32m = ModeloEntrenable::cargar(Path::new(&ruta), 1, 16.0, &dev)?;
    println!("  listo en {:.1} s", t0.elapsed().as_secs_f64());
    let c = f32m.config();
    println!(
        "  oculto={} intermedio={} capas={} cabezas={}/{} vocab={}",
        c.oculto, c.intermedio, c.capas, c.cabezas, c.cabezas_kv, c.vocabulario
    );

    println!();
    println!("  el mismo modelo por el camino cuantizado...");
    let mut fh = std::fs::File::open(&ruta)?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut fh)?;
    let mut qm = cosxamix::model::qwen2::ModelWeights::from_gguf(ct, &mut fh, &dev)?;

    // Ids de un prompt real, no inventados: con ids al azar la distribucion
    // queda plana y el token mas probable lo decide el ruido.
    let ids: Vec<u32> = vec![
        151644, 872, 198, 3838, 374, 279, 6722, 315, 9870, 30, 151645, 198, 151644, 77091, 198,
    ];
    let t = ids.len();
    let entrada = Tensor::new(ids.as_slice(), &dev)?;

    let l_f32 = f32m.forward(&entrada)?;
    let l_q = qm.forward_todas(&entrada.unsqueeze(0)?, 0)?;

    let a = l_f32.i(t - 1)?.flatten_all()?.to_vec1::<f32>()?;
    let b = l_q
        .squeeze(0)?
        .i(t - 1)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

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
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);

    println!();
    println!("  {:<30}{:>14}", "logits comparados", a.len());
    println!(
        "  {:<30}{:>14}",
        "rango de los logits",
        format!("{:.2}", rango(&b))
    );
    println!(
        "  {:<30}{:>14}",
        "diferencia maxima",
        format!(
            "{dif_max:.3}  ({:.1}% del rango)",
            dif_max / rango(&b) * 100.0
        )
    );
    let (ta, tb) = (argmax(&a), argmax(&b));
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
