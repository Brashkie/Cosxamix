//! El kernel fusionado de Q4_K en ensamblador para ARM64, con `sdot`.
//!
//! ## Por qué en ensamblador, y no es por un 4%
//!
//! En x86 el ensamblador compite contra intrínsecos que usan las mismas
//! instrucciones, y gana por poco. En ARM la situación es otra: **es la única
//! forma que tiene este proyecto de usar `sdot`**, la instrucción que hace el
//! producto punto de cuatro `i8` en un `i32`:
//!
//! - **Rust estable no la expone.** `vdotq_s32` sigue detrás de
//!   `stdarch_neon_dotprod` en la 1.97 —comprobado compilando—;
//! - **Zig no la genera.** Su kernel compilado para un Cortex-A76, que la
//!   tiene, sale con 26 `smull`/`smlal` y ningún `sdot`: LLVM no reconoce el
//!   patrón de ensanchar y multiplicar —comprobado en el ensamblador que
//!   emite—.
//!
//! Así que el camino vectorizado de ARM hoy es el de Zig, que ensancha a `i32`
//! con 8 carriles: 2,25x sobre el escalar pero 1,7x por debajo de AVX2.
//! `sdot` hace 16 productos de 8 bits por instrucción.
//!
//! **Hoy hay otra forma: C.** `arm_neon.h` trae `vdotq_s32` estable, y
//! `kernels/neon_dotprod.c` (feature `c`, ver `src/kernels/c.rs`) tiene los
//! fusionados de Q4_K y Q3_K con `sdot`. Este andamio queda para el día en que
//! haga falta ganarle al compilador de C a mano, como pasó en x86 con Q4_K.
//!
//! ## Detrás de qué, y cuándo
//!
//! Va detrás de la feature `asm_q4k`, la misma que el de x86, y solo se compila
//! si `build.rs` encuentra la etiqueta `cosxamix_q4k_completo:` en
//! `kernels/q4k_neon.s`. Hasta que el archivo exista, nada cambia.
//!
//! **No está en el despachador** y no va a estar hasta que se mida en ARM de
//! verdad: esta máquina de desarrollo es x86-64 y no puede ejecutarlo. La
//! medición que decide es la del teléfono con Termux, que además es la
//! plataforma objetivo. `sdot` no existe en todos los ARM —no la tienen el
//! Cortex-A53 ni el A72 de la Raspberry Pi 4—, así que cuando entre lo hará
//! preguntando `is_aarch64_feature_detected!("dotprod")`, que sí es estable.
//!
//! ## El contrato
//!
//! El mismo `completo` del de x86, con la convención de ARM:
//!
//! ```c
//! void cosxamix_q4k_completo(const uint8_t *bloques,  // x0: 144 bytes por bloque
//!                            const int8_t  *xq,       // x1: 256 valores por bloque
//!                            const float   *xe,       // x2: 8 por bloque
//!                            const float   *xs,       // x3: 8 por bloque
//!                            size_t         n_bloques,// x4
//!                            float         *acc);     // x5: 16 f32, lee y escribe
//! ```
//!
//! Lo que calcula por bloque es lo de [`super::scalar::dot_q4k_i8`], que es la
//! especificación ejecutable, y lo de `escalas_q4k` en `x86.rs`, que es la
//! versión vectorial comentada:
//!
//! | bytes del bloque | qué es |
//! |---|---|
//! | 0–1 | `d`, en `f16` |
//! | 2–3 | `dmin`, en `f16` |
//! | 4–15 | las 8 escalas y los 8 mínimos, de 6 bits, empaquetados |
//! | 16–143 | `qs`: 4 pares de 32 bytes, nibbles bajos el sub-bloque par y altos el impar |
//!
//! - **`d` y `dmin`**: `fcvtl` pasa `f16` a `f32`, y es ARMv8.0 básico, no
//!   necesita ninguna extensión;
//! - **escalas `sc[j]` y mínimos `mn[j]`**: el mismo desempaquetado sin ramas
//!   que en x86, tres palabras de 32 bits y máscaras;
//! - **escala del sub-bloque `j`**: `d · sc[j] · xe[j]`;
//! - **desplazamiento**: `dmin · mn[j] · xe[j] · xs[j]`, acumulado **en
//!   positivo**.
//!
//! `acc` son 16 `f32`. Rust calcula **`Σ acc[0..8] − Σ acc[8..16]`** y nada más,
//! así que cómo se reparten los productos entre los 8 primeros carriles y el
//! desplazamiento entre los 8 últimos es decisión del kernel: con NEON de 128
//! bits, 8 `f32` son dos registros, y lo natural es dos acumuladores para cada
//! cosa. Los dos bloques de 8 entran y salen, así que se puede llamar por
//! tandas.
//!
//! ## Lo que es distinto de x86
//!
//! 1. **Una sola convención para los tres sistemas.** Linux, Android y macOS
//!    usan AAPCS64: argumentos en `x0`–`x7`. Un solo archivo, como el
//!    `sysv.s`: sin `.type`, que Mach-O rechaza, y con el símbolo con y sin
//!    guion bajo, porque Mach-O lo decora;
//! 2. **qué registros preservar.** `v0`–`v7` y `v16`–`v31` son volátiles: son
//!    24 registros de 128 bits libres, mucho más que los 6 `ymm` que usa el de
//!    x86. De `v8`–`v15` hay que preservar los 64 bits bajos, así que conviene
//!    no tocarlos. Y en Apple **`x18` está reservado** por la plataforma: no se
//!    usa en ningún sistema, para que el archivo sirva a los tres;
//! 3. **`sdot` hay que habilitarlo en el archivo.** El crate se compila para
//!    ARMv8.0 sin extensiones, que es lo portable, y el ensamblador de LLVM
//!    rechaza `sdot` si `dotprod` no está habilitada. Va
//!    `.arch_extension dotprod` al principio del `.s`. Eso habilita la
//!    instrucción en el ensamblador, no en la CPU: quien llame tiene que haber
//!    comprobado `dotprod` antes;
//! 4. **`sdot` multiplica con signo los dos operandos**, a diferencia de
//!    `vpmaddubsw` en x86, que pide uno sin signo. Los nibbles de 0 a 15
//!    entran igual como `i8` positivos, así que acá no hace falta ningún rodeo.
//!
//! La sintaxis es la de ARM64 de LLVM; no hay que declararla.

#![cfg(all(feature = "asm_q4k", hay_q4k_neon, target_arch = "aarch64"))]

use std::arch::global_asm;

global_asm!(include_str!("../../kernels/q4k_neon.s"));

unsafe extern "C" {
    /// Contrato completo: `n_bloques` bloques enteros, `acc` de 16 `f32`, entran
    /// y salen.
    fn cosxamix_q4k_completo(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
        acc: *mut f32,
    );
}

/// El fusionado de Q4_K entero en ensamblador NEON, con `sdot`.
///
/// # Safety
///
/// Pide NEON y `dotprod`. Quien llame tiene que haber comprobado
/// `is_aarch64_feature_detected!("dotprod")`: en un ARM sin la extensión,
/// `sdot` es una instrucción ilegal.
pub unsafe fn dot_q4k_i8_neon(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 144);

    let mut acc = [0f32; 16];
    unsafe {
        cosxamix_q4k_completo(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
            acc.as_mut_ptr(),
        );
    }
    acc[..8].iter().sum::<f32>() - acc[8..].iter().sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hay_dotprod() -> bool {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    /// Bloques Q4_K con bytes al azar pero `d` y `dmin` válidos: con bytes al
    /// azar los `f16` pueden salir infinito o NaN, y NaN no es igual a NaN.
    fn fila(n_bloques: usize, semilla: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
        let n = n_bloques * 256;
        let mut pesos: Vec<u8> = (0..n_bloques * 144)
            .map(|i| ((i * 131 + semilla) % 251) as u8)
            .collect();
        for bl in 0..n_bloques {
            let o = bl * 144;
            let d = crate::quant::f32_a_f16(0.01 + (bl % 50) as f32 * 0.003);
            let dmin = crate::quant::f32_a_f16(0.005 + (bl % 50) as f32 * 0.001);
            pesos[o..o + 2].copy_from_slice(&d.to_le_bytes());
            pesos[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
        }
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut xq = vec![0i8; n];
        let mut xe = vec![0f32; n / 32];
        crate::ops::cuantizar_i8(&x, &mut xq, &mut xe);
        let xs: Vec<f32> = xq
            .chunks(32)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        (pesos, xq, xe, xs)
    }

    /// Tiene que dar lo mismo que la referencia escalar, que es la
    /// especificación.
    #[test]
    fn coincide_con_la_referencia() {
        if !hay_dotprod() {
            return;
        }
        let (pesos, xq, xe, xs) = fila(8, 17);
        let esperado = crate::kernels::scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs);
        let nuestro = unsafe { dot_q4k_i8_neon(&pesos, &xq, &xe, &xs) };
        assert!(
            (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
            "neon dio {nuestro} y la referencia {esperado}"
        );
    }

    /// Y lo mismo que el de Zig, que es contra quien compite en ARM.
    #[cfg(feature = "zig")]
    #[test]
    fn coincide_con_zig() {
        if !hay_dotprod() {
            return;
        }
        let (pesos, xq, xe, xs) = fila(8, 17);
        let suyo = crate::kernels::zig::dot_q4k_i8(&pesos, &xq, &xe, &xs);
        let nuestro = unsafe { dot_q4k_i8_neon(&pesos, &xq, &xe, &xs) };
        assert!(
            (nuestro - suyo).abs() / suyo.abs().max(1.0) < 1e-4,
            "neon dio {nuestro} y zig {suyo}"
        );
    }

    /// Un bloque, pocos y muchos: el bucle puede dar una sola vuelta, y un paso
    /// de 144 mal contado aparece con 17 y no con 1.
    #[test]
    fn aguanta_uno_pocos_y_muchos() {
        if !hay_dotprod() {
            return;
        }
        for n_bloques in [1usize, 2, 17, 80] {
            let (pesos, xq, xe, xs) = fila(n_bloques, 41);
            let esperado = crate::kernels::scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs);
            assert!(
                esperado.is_finite(),
                "con {n_bloques} bloques la referencia dio {esperado}"
            );
            let nuestro = unsafe { dot_q4k_i8_neon(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
                "con {n_bloques} bloques neon dio {nuestro} y la referencia {esperado}"
            );
        }
    }
}
