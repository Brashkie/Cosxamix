//! Cosxamix — runtime de IA que sirve modelos de lenguaje por HTTP.
//!
//! El binario (`src/main.rs`) es una cáscara fina sobre esto. Está partido así
//! para que los tests y los bancos puedan medir los kernels sin levantar un
//! servidor: `benches/kernels.rs` compara backends llamando directamente a
//! `kernels::scalar::dot` y a los demás, y eso no se puede hacer contra un
//! binario.

pub mod engine;
pub mod kernels;

#[cfg(feature = "candle")]
pub mod model;

#[cfg(feature = "candle")]
pub mod engine_candle;

#[cfg(feature = "candle")]
pub mod engine_manager;

#[cfg(feature = "training")]
pub mod training;
