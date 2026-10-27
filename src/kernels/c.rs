//! El lado Rust de los kernels escritos en C.
//!
//! Solo existe con la feature `c`. Lo compila `zig cc` desde build.rs, sin
//! libc, así que no pide otro compilador que Zig. Un archivo por arquitectura:
//!
//! - **x86-64**, `kernels/q3k_avx2.c`: el fusionado de Q3_K con el mismo
//!   algoritmo que [`super::x86::dot_q3k_i8_avx2`], para comparar lenguajes.
//!   Pierde un 4% con Rust y no está en el despachador (DECISIONES §21);
//! - **ARM64**, `kernels/neon_dotprod.c`: los fusionados de Q4_K y Q3_K con
//!   `sdot`, que Rust estable no expone y Zig no genera. En el despachador
//!   desde la 0.5.1, en las CPU con `dotprod`: 2,4x sobre Zig en Q4_K y 5,4x
//!   sobre la referencia en Q3_K (DECISIONES §22).
//!
//! La especificación ejecutable son [`super::scalar::dot_q4k_i8`] y
//! [`super::scalar::dot_q3k_i8`].

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn cosxamix_q3k_c(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
    ) -> f32;
}

/// Producto punto de una fila Q3_K contra una entrada en `i8`, en C.
///
/// # Safety
///
/// La CPU tiene que tener AVX2, FMA y F16C ([`super::cpu_avx2`]).
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_q3k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 16);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 110);
    unsafe {
        cosxamix_q3k_c(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
        )
    }
}

// ─── ARM64, con `sdot` ───────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    fn cosxamix_q4k_neon(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
    ) -> f32;
    fn cosxamix_q3k_neon(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
    ) -> f32;
}

/// Si esta CPU tiene `sdot`. No la tienen el Cortex-A53 ni el A72 de la
/// Raspberry Pi 4; sí desde el A55 y el A76, y los Neoverse de los servidores.
#[cfg(target_arch = "aarch64")]
pub fn hay_dotprod() -> bool {
    use std::sync::OnceLock;
    static SI: OnceLock<bool> = OnceLock::new();
    *SI.get_or_init(|| std::arch::is_aarch64_feature_detected!("dotprod"))
}

/// El fusionado de Q4_K con `sdot`.
///
/// # Safety
///
/// La CPU tiene que tener `dotprod` ([`hay_dotprod`]): sin ella, `sdot` es una
/// instrucción ilegal.
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_q4k_i8_neon(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 144);
    unsafe {
        cosxamix_q4k_neon(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
        )
    }
}

/// El fusionado de Q3_K con `sdot`. `xs` no se usa —`sdot` multiplica con
/// signo y no hace falta el rodeo del +4—, pero se pide para que la firma sea
/// la de los otros caminos.
///
/// # Safety
///
/// La CPU tiene que tener `dotprod` ([`hay_dotprod`]).
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_q3k_i8_neon(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 16);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 110);
    unsafe {
        cosxamix_q3k_neon(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
        )
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::quant::f32_a_f16;

    /// Una fila Q4_K con bytes al azar y `d`, `dmin` válidos: con bytes al azar
    /// un `f16` puede salir NaN, y NaN no es igual a NaN.
    fn fila_q4k(n_bloques: usize, semilla: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
        let n = n_bloques * 256;
        let mut pesos: Vec<u8> = (0..n_bloques * 144)
            .map(|i| ((i * 131 + semilla) % 251) as u8)
            .collect();
        for bl in 0..n_bloques {
            let o = bl * 144;
            let d = f32_a_f16(0.01 + (bl % 50) as f32 * 0.003);
            let dmin = f32_a_f16(0.005 + (bl % 50) as f32 * 0.001);
            pesos[o..o + 2].copy_from_slice(&d.to_le_bytes());
            pesos[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
        }
        let x: Vec<f32> = (0..n)
            .map(|i| ((i + semilla) as f32 * 0.013).sin())
            .collect();
        let mut xq = vec![0i8; n];
        let mut xe = vec![0f32; n / 32];
        crate::ops::cuantizar_i8(&x, &mut xq, &mut xe);
        let xs: Vec<f32> = xq
            .chunks(32)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        (pesos, xq, xe, xs)
    }

    /// Lo mismo para Q3_K, con la entrada de a 16.
    fn fila_q3k(n_bloques: usize, semilla: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
        let n = n_bloques * 256;
        let mut pesos: Vec<u8> = (0..n_bloques * 110)
            .map(|i| ((i * 97 + semilla * 31) % 251) as u8)
            .collect();
        for bl in 0..n_bloques {
            let d = f32_a_f16(0.002 + (bl % 13) as f32 * 0.0007);
            pesos[bl * 110 + 108..bl * 110 + 110].copy_from_slice(&d.to_le_bytes());
        }
        let x: Vec<f32> = (0..n)
            .map(|i| ((i + semilla) as f32 * 0.017).sin())
            .collect();
        let mut xq = vec![0i8; n];
        let mut xe = vec![0f32; n / 16];
        crate::ops::cuantizar_i8_en(&x, &mut xq, &mut xe, 16);
        let xs: Vec<f32> = xq
            .chunks(16)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        (pesos, xq, xe, xs)
    }

    /// Uno, pocos y muchos bloques contra la referencia. Uno: el bucle puede
    /// dar una sola vuelta. 17 y 43: un paso de 144 o de 110 mal contado
    /// aparece acá y no con uno, y 43 son los bloques de una fila del FFN del
    /// 3B.
    #[test]
    fn q4k_coincide_con_la_referencia() {
        if !hay_dotprod() {
            return;
        }
        for n_bloques in [1usize, 2, 8, 17, 43] {
            let (pesos, xq, xe, xs) = fila_q4k(n_bloques, 17 + n_bloques);
            let esperado = crate::kernels::scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs);
            assert!(esperado.is_finite(), "datos mal armados: {esperado}");
            let nuestro = unsafe { dot_q4k_i8_neon(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
                "Q4_K, {n_bloques} bloques: sdot dio {nuestro} y la referencia {esperado}"
            );
        }
    }

    #[test]
    fn q3k_coincide_con_la_referencia() {
        if !hay_dotprod() {
            return;
        }
        for n_bloques in [1usize, 2, 8, 17, 43] {
            let (pesos, xq, xe, xs) = fila_q3k(n_bloques, 7 + n_bloques);
            let esperado = crate::kernels::scalar::dot_q3k_i8(&pesos, &xq, &xe, &xs);
            assert!(esperado.is_finite(), "datos mal armados: {esperado}");
            let nuestro = unsafe { dot_q3k_i8_neon(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
                "Q3_K, {n_bloques} bloques: sdot dio {nuestro} y la referencia {esperado}"
            );
        }
    }

    /// Y lo mismo que el de Zig, que es el camino vectorizado de ARM hoy y
    /// contra quien compite.
    #[cfg(feature = "zig")]
    #[test]
    fn q4k_coincide_con_zig() {
        if !hay_dotprod() {
            return;
        }
        let (pesos, xq, xe, xs) = fila_q4k(8, 17);
        let suyo = crate::kernels::zig::dot_q4k_i8(&pesos, &xq, &xe, &xs);
        let nuestro = unsafe { dot_q4k_i8_neon(&pesos, &xq, &xe, &xs) };
        assert!(
            (nuestro - suyo).abs() / suyo.abs().max(1.0) < 1e-4,
            "sdot dio {nuestro} y zig {suyo}"
        );
    }
}
