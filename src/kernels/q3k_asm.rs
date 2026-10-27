//! El kernel fusionado de Q3_K en ensamblador.
//!
//! El cuerpo está en `kernels/q3k_avx2_win64.s` y `kernels/q3k_avx2_sysv.s`,
//! uno por convención de llamada, con el mismo algoritmo que
//! [`super::x86::dot_q3k_i8_avx2`] y que `kernels/q3k_avx2.c`. Va detrás de la
//! feature `asm_q3k`, apagada por defecto:
//!
//! ```text
//! cargo test  --features asm_q3k
//! cargo bench --features asm_q3k --bench kernels -- q3k
//! ```
//!
//! ## El contrato
//!
//! ```c
//! float cosxamix_q3k_completo(const uint8_t *bloques,  // 110 bytes por bloque
//!                             const int8_t  *xq,       // 256 por bloque
//!                             const float   *xe,       // 16 por bloque
//!                             const float   *xs,       // 16 por bloque
//!                             size_t         n_bloques);
//! ```
//!
//! A diferencia del de Q4_K no hay acumulador que entre y salga: la fila se
//! recorre en una llamada y el resultado vuelve reducido en `xmm0`, que es
//! donde las dos convenciones devuelven un `float`.
//!
//! Usa los dieciséis `ymm`: siete constantes del bucle viven en ymm9-ymm15 de
//! punta a punta. En Windows, xmm6-xmm15 son del llamador y el `.s` los guarda
//! y los restaura; en System V no hace falta.
//!
//! **La especificación ejecutable es [`super::scalar::dot_q3k_i8`].**

#![cfg(all(feature = "asm_q3k", target_arch = "x86_64"))]

use std::arch::global_asm;

#[cfg(target_os = "windows")]
global_asm!(
    include_str!("../../kernels/q3k_avx2_win64.s"),
    options(att_syntax)
);

#[cfg(not(target_os = "windows"))]
global_asm!(
    include_str!("../../kernels/q3k_avx2_sysv.s"),
    options(att_syntax)
);

unsafe extern "C" {
    fn cosxamix_q3k_completo(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
    ) -> f32;
}

/// Producto punto de una fila Q3_K contra una entrada en `i8`, en ensamblador.
///
/// # Safety
///
/// La CPU tiene que tener AVX2, FMA y F16C ([`super::cpu_avx2`]).
pub unsafe fn dot_q3k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(xq.len().is_multiple_of(256));
    assert_eq!(xe.len(), xq.len() / 16);
    assert_eq!(xs.len(), xe.len());
    assert!(bloques.len() >= xq.len() / 256 * 110);
    unsafe {
        cosxamix_q3k_completo(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::quant::f32_a_f16;

    /// Una fila de `n_bloques` con escalas y `d` razonables, y la entrada
    /// cuantizada de a 16 como lo hace el multiplicador.
    fn fila(n_bloques: usize, semilla: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
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

    /// Uno, pocos y muchos bloques, contra la referencia y contra los
    /// intrínsecos. Uno: el bucle puede dar una sola vuelta. Diecisiete y
    /// cuarenta y tres: los pasos de 110, 256 y 64 bytes tienen que estar bien
    /// contados, y 43 son los bloques de una fila del FFN del 3B.
    #[test]
    fn coincide_con_la_referencia_y_los_intrinsecos() {
        if !crate::kernels::cpu_avx2() {
            return;
        }
        for n_bloques in [1usize, 2, 8, 17, 43] {
            let (pesos, xq, xe, xs) = fila(n_bloques, 7 + n_bloques);
            let esperado = crate::kernels::scalar::dot_q3k_i8(&pesos, &xq, &xe, &xs);
            assert!(esperado.is_finite(), "datos mal armados: {esperado}");
            let escala = esperado.abs().max(1.0);
            let nuestro = unsafe { super::dot_q3k_i8(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - esperado).abs() / escala < 1e-5,
                "{n_bloques} bloques: el ensamblador dio {nuestro} y la referencia {esperado}"
            );
            let intrinsecos =
                unsafe { crate::kernels::x86::dot_q3k_i8_avx2(&pesos, &xq, &xe, &xs) };
            assert_eq!(
                nuestro.to_bits(),
                intrinsecos.to_bits(),
                "{n_bloques} bloques: mismo algoritmo y mismo orden, tiene que dar igual bit a bit"
            );
        }
    }
}
