//! Cosxamix — runtime de IA que sirve modelos de lenguaje por HTTP.
//!
//! El binario (`src/main.rs`) es una cáscara fina sobre esto. Está partido así
//! para que los tests y los bancos puedan medir los kernels sin levantar un
//! servidor: `benches/kernels.rs` compara backends llamando directamente a
//! `kernels::scalar::dot` y a los demás, y eso no se puede hacer contra un
//! binario.

pub mod engine;
pub mod kernels;

/// Lector de GGUF propio. No depende de candle: es el primer ladrillo del
/// motor de inferencia de la casa.
pub mod gguf;

/// Descuantizacion propia de los formatos de GGML, sobre bytes crudos y
/// escribiendo en un buffer que da quien llama.
pub mod quant;

/// Las operaciones del motor propio. Empieza por la multiplicacion de
/// matrices cuantizadas, que es donde se va el 99% del tiempo.
pub mod ops;

#[cfg(feature = "candle")]
pub mod model;

#[cfg(feature = "candle")]
pub mod engine_candle;

#[cfg(feature = "candle")]
pub mod engine_manager;

#[cfg(feature = "training")]
pub mod training;
