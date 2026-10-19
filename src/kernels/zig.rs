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
    fn cosxamix_dot_i8_escalado_f32(
        a: *const i8,
        b: *const i8,
        esc_a: *const f32,
        esc_b: *const f32,
        n: usize,
        bloque: usize,
    ) -> f32;
}

/// Producto punto cuantizado por bloques.
///
/// Tiene una ventaja sobre el camino de AVX2 que no es de velocidad: **acepta
/// todo el rango de `i8`, incluido -128**. El de AVX2 usa `vpmaddubsw`, que
/// obliga a mover el signo con `vpsignb`, y -128 no tiene opuesto en i8 — el
/// producto sale con el signo cambiado. Zig ensancha a `i32` y multiplica
/// directo, asi que no hay precondicion que recordar.
///
/// Y es el unico camino vectorizado de esta operacion en ARM, donde el de
/// Rust usa `core::arch::x86_64` y no existe.
pub fn dot_i8_escalado(a: &[i8], b: &[i8], esc_a: &[f32], esc_b: &[f32], bloque: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    unsafe {
        cosxamix_dot_i8_escalado_f32(
            a.as_ptr(),
            b.as_ptr(),
            esc_a.as_ptr(),
            esc_b.as_ptr(),
            a.len(),
            bloque,
        )
    }
}

unsafe extern "C" {
    fn cosxamix_dot_q4k_i8_f32(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n: usize,
    ) -> f32;
}

/// El kernel **fusionado** de Q4_K: multiplica sin desempaquetar los pesos.
///
/// Existe porque el de AVX2 **no existe fuera de x86-64**. En ARM y en Termux
/// el camino de Rust usa `core::arch::x86_64` y no compila, así que sin esto
/// el despachador cae a la referencia escalar, byte a byte, en el bucle más
/// caliente del motor. Entra por el **segundo brazo de la regla 3** —habilita
/// una plataforma— y no por ganarle a nadie.
///
/// No reproduce el truco de `vpmaddubsw`, que suma pares de productos de 8
/// bits dentro de un registro de 16 y es lo que le da a AVX2 sus 32 carriles:
/// acá se ensancha a `i32` y se multiplica, que con NEON de 128 bits son 8.
/// A cambio, `@Vector` deja que LLVM elija la instrucción de cada
/// arquitectura, y en ARMv8.2 con `dotprod` puede salir `sdot`.
pub fn dot_q4k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 144);
    unsafe {
        cosxamix_dot_q4k_i8_f32(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len(),
        )
    }
}

unsafe extern "C" {
    fn cosxamix_noop(x: f32) -> f32;
}

/// Cruza la FFI sin hacer trabajo. Solo para medir el coste de la frontera.
pub fn noop(x: f32) -> f32 {
    unsafe { cosxamix_noop(x) }
}
