//! Entrenamiento: ajustar un modelo en una máquina sin GPU.
//!
//! Dos caminos, sobre el mismo transformer entrenable ([`transformer`]), con el
//! forward y el backward escritos a mano y comprobados contra diferencias
//! finitas:
//!
//! - **LoRA** ([`model`], [`lora`]): entrenar el 0,1% de un Qwen que ya sabe. En
//!   el 0,5B son 540.672 parámetros contra 494 millones, y el adaptador pesa
//!   ~2 MB en vez de 2 GB. La base va descuantizada a f32 y congelada.
//! - **Desde cero** ([`full`]): un daiss entero, con las tablas de vocabulario
//!   densas o en Tensor-Train, que se exporta a GGUF y lo corre el motor.
//!
//! Hasta la 0.10.0 esto lo hacía el autograd de candle; desde la 0.11.0 no hay
//! candle en ninguna parte.

/// Entrenar todos los parámetros, desde cero.
pub mod full;
/// Guardar y cargar los adaptadores LoRA.
pub mod lora;
/// Qwen2 descuantizado, con adaptadores LoRA.
pub mod model;
/// El transformer entrenable, con el backward escrito a mano, y AdamW.
pub mod transformer;

pub use lora::{cargar, guardar};
pub use model::{Config, ModeloEntrenable};
pub use transformer::AdamW;
