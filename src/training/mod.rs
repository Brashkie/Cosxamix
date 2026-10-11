//! Entrenamiento: ajustar un modelo en una maquina sin GPU.
//!
//! Es la mitad del objetivo del proyecto, y la que faltaba entera: hasta la
//! 0.2.0 Cosxamix solo infiere.
//!
//! Lo que hay aca es **LoRA**: entrenar el 0,1% de los parametros en vez del
//! 100%. En el 0,5B son 540.672 contra 494 millones, y el adaptador resultante
//! pesa ~2 MB en vez de 2 GB.
//!
//! Medido en un i5-10400 sin GPU, con una secuencia de 256 tokens: **~3,3 s
//! por paso**, o sea que mil pasos son menos de una hora. Esa cifra es la que
//! hace que esto tenga sentido y esta en el ROADMAP con su contexto.
//!
//! La base va en **f32 y no cuantizada**, y no por gusto: `QMatMul` de candle
//! no implementa `bwd`, asi que el gradiente no atraviesa una multiplicacion
//! cuantizada. Eso acota el alcance a modelos que entren en RAM en f32 — el
//! 0,5B entra, el 3B no.

pub mod lora;
pub mod model;

pub use lora::{cargar, guardar, LinealLora};
pub use model::{Config, ModeloEntrenable};
