//! El lado Rust de los kernels escritos en Zig.
//!
//! Solo existe con la feature `zig`. Sin ella, Cosxamix compila y funciona igual
//! —con la referencia y con AVX2— y no hace falta tener Zig instalado. Eso es
//! a propósito: que el proyecto use varios lenguajes no debe significar que
//! haya que instalarlos todos para compilarlo.

// La función que exporta `kernels/dot.zig`, con ABI de C.
//
// Los slices no cruzan la frontera: de este lado se descomponen en puntero y
// longitud, que sí son lo mismo en los dos lenguajes. (Comentario normal y no
// de documentación: rustdoc no documenta un bloque `extern`.)
unsafe extern "C" {
    fn cosxamix_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
}

/// Producto punto.
///
/// La parte `unsafe` está contenida acá y no se propaga: el llamador pasa dos
/// slices de Rust, y punteros y longitud salen de ellos, así que no hay forma
/// de que el tamaño no coincida con la memoria. Lo único que esto confía a
/// Zig es que no escriba donde no debe, y para eso están los tests que lo
/// comparan contra la referencia.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    if a.is_empty() {
        // Un puntero de un slice vacío puede ser el que marca "sin asignar".
        // Pasarlo a C es legal pero no vale la pena discutirlo por un caso que
        // siempre da cero.
        return 0.0;
    }

    unsafe { cosxamix_dot_f32(a.as_ptr(), b.as_ptr(), a.len()) }
}

unsafe extern "C" {
    fn cosxamix_rmsnorm_f32(
        salida: *mut f32,
        entrada: *const f32,
        peso: *const f32,
        n: usize,
        eps: f32,
    );
}

/// Normalización RMS.
pub fn rmsnorm(salida: &mut [f32], entrada: &[f32], peso: &[f32], eps: f32) {
    debug_assert_eq!(salida.len(), entrada.len());
    debug_assert_eq!(entrada.len(), peso.len());

    if entrada.is_empty() {
        return;
    }

    unsafe {
        cosxamix_rmsnorm_f32(
            salida.as_mut_ptr(),
            entrada.as_ptr(),
            peso.as_ptr(),
            entrada.len(),
            eps,
        )
    }
}

unsafe extern "C" {
    fn cosxamix_silu_f32(salida: *mut f32, entrada: *const f32, n: usize);
}

/// SiLU vectorizada.
///
/// Es el unico kernel de la frontera donde Zig aporta algo que no es cobertura
/// de plataforma: su `@exp` funciona **sobre un vector**, asi que ocho
/// exponenciales salen de una instruccion. Rust no tiene equivalente —
/// `f32::exp` es escalar— y vectorizarlo pide escribir a mano una
/// aproximacion polinomica con su reduccion de rango.
pub fn silu(salida: &mut [f32], entrada: &[f32]) {
    debug_assert_eq!(salida.len(), entrada.len());
    unsafe { cosxamix_silu_f32(salida.as_mut_ptr(), entrada.as_ptr(), entrada.len()) }
}

unsafe extern "C" {
    fn cosxamix_noop(x: f32) -> f32;
}

/// Cruza la FFI sin hacer trabajo. Solo para medir el coste de la frontera.
pub fn noop(x: f32) -> f32 {
    unsafe { cosxamix_noop(x) }
}
