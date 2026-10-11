//! `dot` con el bucle caliente en un archivo de ensamblador de verdad.
//!
//! ## Por qué existe
//!
//! El proyecto venía **razonando** sobre el ensamblador —que el bucle que
//! importa ya está escrito a mano en candle, que haría falta un perfilado que
//! señale algo— sin haber escrito ninguno. Razonar no es medir.
//!
//! La pregunta concreta: los intrínsecos de `x86.rs` le dejan a LLVM elegir el
//! orden de las instrucciones, el desenrollado y los registros. **¿Queda algo
//! sobre la mesa que un humano pueda recoger?**
//!
//! ## Dónde está el código
//!
//! En [`kernels/dot_avx2_win64.s`] y [`kernels/dot_avx2_sysv.s`], archivos de
//! ensamblador sueltos, no `asm!` metido entre Rust. Este módulo solo los
//! incluye y declara la firma.
//!
//! **Son dos archivos y no uno porque el ensamblador es el único lenguaje de
//! esta frontera que tiene que saber la convención de llamada.** Rust, Zig y
//! Fortran reciben sus argumentos donde el compilador decida; acá hay que
//! escribir en qué registro llega cada uno, y Windows y System V no coinciden:
//!
//! ```text
//!   Windows x64:  rcx  rdx  r8   r9
//!   System V:     rdi  rsi  rdx  rcx
//! ```
//!
//! Se ensamblan con LLVM, por `global_asm!`, así que **no hace falta ningún
//! ensamblador externo**: compila con el mismo `cargo build` de siempre. Es la
//! razón por la que esto entra sin feature propia, a diferencia de Zig.
//!
//! ## Cómo está hecho para que la comparación sea limpia
//!
//! Solo el bucle de 32 elementos por vuelta está en ensamblador. La cola y la
//! suma horizontal son **el mismo código** que la versión de intrínsecos,
//! llamando a la misma función. Así la única diferencia entre las dos es el
//! bucle caliente, y el resultado sale **bit a bit idéntico** — que es lo que
//! comprueba el test. Si se reescribiera todo, un resultado distinto no diría
//! si es más rápido o si está mal.

#![cfg(target_arch = "x86_64")]

use std::arch::global_asm;

#[cfg(target_os = "windows")]
global_asm!(include_str!("../../kernels/dot_avx2_win64.s"));

#[cfg(not(target_os = "windows"))]
global_asm!(include_str!("../../kernels/dot_avx2_sysv.s"));

extern "C" {
    /// Consume `bloques * 32` elementos y deja cuatro acumuladores de 8 en
    /// `acc`. Implementada en los `.s` de arriba.
    fn cosxamix_dot_bloques(a: *const f32, b: *const f32, acc: *mut f32, bloques: usize);
}

/// Producto punto con el bucle principal en ensamblador.
///
/// # Safety
///
/// Pide AVX2 y FMA. Quien llame tiene que haberlo comprobado — de eso se
/// encarga `kernels::elegido()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_asm(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let bloques = a.len() / 32;
    let mut acc = [0f32; 32];
    cosxamix_dot_bloques(a.as_ptr(), b.as_ptr(), acc.as_mut_ptr(), bloques);

    super::x86::reducir_y_cola(a, b, &acc, bloques * 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mismo número, bit a bit, que la versión de intrínsecos.
    ///
    /// No vale una tolerancia: las dos hacen las mismas operaciones en el
    /// mismo orden, así que un bit de diferencia significaría que el bucle en
    /// ensamblador leyó algo de otro sitio o acumuló distinto. Se prueban
    /// largos raros porque olvidarse de la cola es el error clásico: con `n`
    /// múltiplo de 32 todo pasa.
    #[test]
    fn identico_a_los_intrinsecos() {
        if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
            return;
        }
        for n in [0usize, 1, 7, 8, 31, 32, 33, 64, 100, 1024, 4097] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..n)
                .map(|i| (i as f32 * 0.11).cos() * 2.5 - 1.0)
                .collect();
            let (x, y) = unsafe { (super::super::x86::dot_avx2(&a, &b), dot_asm(&a, &b)) };
            assert_eq!(
                x, y,
                "n={n}: el ensamblador no da lo mismo que los intrinsecos"
            );
        }
    }
}
