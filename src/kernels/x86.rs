//! Kernels con intrínsecos de x86-64: AVX2 + FMA.
//!
//! Están en Rust y no en otro lenguaje a propósito, y conviene decir por qué
//! antes de que alguien asuma que esto "debería" ser Zig o ensamblador: los
//! intrínsecos de `core::arch` son las **mismas instrucciones** que emitiría
//! cualquiera de los dos. No hay nada que Rust no pueda expresar acá.
//!
//! Lo que sí cambia entre lenguajes es la ergonomía de escribirlos y de
//! generar variantes por arquitectura. Esa es la discusión real, y se resuelve
//! midiendo: por eso existe el banco que compara backends.

use std::arch::x86_64::*;

/// Producto punto con AVX2 y FMA.
///
/// # Safety
///
/// (El encabezado va en inglés porque clippy busca ese literal.)
///
/// La CPU tiene que soportar `avx2` **y** `fma`. Quien llama lo garantiza;
/// `kernels::elegido()` solo devuelve `Backend::Avx2` tras comprobar las dos
/// con `is_x86_feature_detected!`. Ejecutar esto en una CPU sin ellas no da un
/// resultado malo: tira instrucción ilegal y el proceso muere.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    // Cuatro acumuladores y no uno: la latencia de un FMA es de varios ciclos,
    // y con un solo acumulador cada iteración espera a la anterior porque
    // depende de ella. Con cuatro cadenas independientes el procesador las
    // solapa. Es la diferencia entre usar la unidad vectorial y hacer cola
    // para ella.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();

    let mut i = 0;

    // 32 por vuelta: cuatro registros de 8 f32.
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            s3,
        );
        i += 32;
    }

    let mut acc = [0f32; 32];
    _mm256_storeu_ps(acc.as_mut_ptr(), s0);
    _mm256_storeu_ps(acc.as_mut_ptr().add(8), s1);
    _mm256_storeu_ps(acc.as_mut_ptr().add(16), s2);
    _mm256_storeu_ps(acc.as_mut_ptr().add(24), s3);

    reducir_y_cola(a, b, &acc, i)
}

/// La cola y la suma horizontal, compartidas con la versión en ensamblador.
///
/// Está factorizado para que las dos implementaciones usen **el mismo** código
/// aquí y la única diferencia entre ellas sea el bucle caliente. Comparar dos
/// kernels que además reducen distinto no diría cuál bucle es mejor, diría que
/// dan números distintos.
///
/// `desde` es cuántos elementos consumió ya el bucle de 32.
///
/// # Safety
///
/// Pide AVX2 y FMA, y que `acc` traiga los cuatro acumuladores de 8.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn reducir_y_cola(a: &[f32], b: &[f32], acc: &[f32; 32], desde: usize) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    let mut s0 = _mm256_loadu_ps(acc.as_ptr());
    let s1 = _mm256_loadu_ps(acc.as_ptr().add(8));
    let s2 = _mm256_loadu_ps(acc.as_ptr().add(16));
    let s3 = _mm256_loadu_ps(acc.as_ptr().add(24));

    // Lo que no llegó a 32 pero sí a 8.
    let mut i = desde;
    while i + 8 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        i += 8;
    }

    let mut total = suma_horizontal(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));

    // La cola, de a uno. Olvidarse de esto es el error clásico de un kernel
    // vectorizado: con n múltiplo de 8 todos los tests pasan y con 4097
    // elementos se pierde uno. Por eso el test prueba longitudes raras.
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }

    total
}

/// Suma los 8 `f32` de un registro de 256 bits.
#[target_feature(enable = "avx2")]
unsafe fn suma_horizontal(v: __m256) -> f32 {
    // Se parte en dos mitades de 128 y se suman; después se baja a 64 y a 32.
    // Es menos legible que sumar de a uno, pero una suma horizontal ingenua
    // cuesta más que el bucle entero cuando los vectores son cortos.
    let bajo = _mm256_castps256_ps128(v);
    let alto = _mm256_extractf128_ps(v, 1);
    let suma4 = _mm_add_ps(bajo, alto);

    let suma2 = _mm_add_ps(suma4, _mm_movehl_ps(suma4, suma4));
    let suma1 = _mm_add_ss(suma2, _mm_shuffle_ps(suma2, suma2, 0x55));

    _mm_cvtss_f32(suma1)
}

/// Normalización RMS con AVX2 + FMA.
///
/// # Safety
///
/// La CPU tiene que soportar `avx2` y `fma`, igual que `dot_avx2`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rmsnorm_avx2(salida: &mut [f32], entrada: &[f32], peso: &[f32], eps: f32) {
    debug_assert_eq!(salida.len(), entrada.len());
    debug_assert_eq!(entrada.len(), peso.len());

    let n = entrada.len();
    if n == 0 {
        return;
    }

    let pe = entrada.as_ptr();
    let pp = peso.as_ptr();
    let ps = salida.as_mut_ptr();

    // ── Primera pasada: suma de cuadrados ────────────────────────────────────
    //
    // Dos acumuladores y no cuatro como en `dot`: acá la segunda pasada domina
    // el tiempo, así que desenrollar más esta no compra nada y alarga el
    // código.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();

    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_loadu_ps(pe.add(i));
        let b = _mm256_loadu_ps(pe.add(i + 8));
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
        i += 16;
    }
    while i + 8 <= n {
        let a = _mm256_loadu_ps(pe.add(i));
        s0 = _mm256_fmadd_ps(a, a, s0);
        i += 8;
    }

    let mut suma = suma_horizontal(_mm256_add_ps(s0, s1));
    while i < n {
        let x = *pe.add(i);
        suma += x * x;
        i += 1;
    }

    // `sqrt` y la división se hacen UNA vez, en escalar. Vectorizarlas no
    // tendría sentido: son dos operaciones contra las 2n de las pasadas.
    let escala = 1.0 / (suma / n as f32 + eps).sqrt();
    let vescala = _mm256_set1_ps(escala);

    // ── Segunda pasada: escalar y aplicar el peso ────────────────────────────
    i = 0;
    while i + 8 <= n {
        let e = _mm256_loadu_ps(pe.add(i));
        let p = _mm256_loadu_ps(pp.add(i));
        _mm256_storeu_ps(ps.add(i), _mm256_mul_ps(_mm256_mul_ps(e, vescala), p));
        i += 8;
    }
    while i < n {
        *ps.add(i) = *pe.add(i) * escala * *pp.add(i);
        i += 1;
    }
}
