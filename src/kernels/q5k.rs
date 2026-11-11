//! El kernel fusionado de Q5_K, en todos los lenguajes de la frontera.
//!
//! Un bloque son 176 bytes por cada 256 valores: `d` y `dmin` en `f16`, 12 bytes
//! con 8 escalas y 8 mínimos de 6 bits —el encabezado de Q4_K—, 32 bytes `qh`
//! con el quinto bit de cada valor y 128 de nibbles. Cada valor va de 0 a 31.
//!
//! - la referencia en Rust: [`super::scalar::dot_q5k_i8`];
//! - intrínsecos AVX2: [`super::x86::dot_q5k_i8_avx2`], que arma el quinto bit
//!   con `and`, `cmpeq` y `and`;
//! - ensamblador (`kernels/q5k_avx2_*.s`) y C (`kernels/q5k_avx2.c`), que lo arman
//!   corriendo `qh` hasta la posición 4 y con un solo `and`;
//! - Zig (`kernels/q5k.zig`), escrito simple, el camino de ARM sin `sdot`.
//!
//! `COSXAMIX_Q5K` elige uno, para medir: `escalar`, `avx2`, `asm`, `c` o `zig`.

use std::sync::OnceLock;

/// Los lenguajes en los que está escrito.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Copia {
    Escalar,
    Avx2,
    Asm,
    C,
    Zig,
}

#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
std::arch::global_asm!(include_str!("../../kernels/q5k_avx2_win64.s"));

#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
std::arch::global_asm!(include_str!("../../kernels/q5k_avx2_sysv.s"));

/// La forma de todas las copias que no están en Rust.
type Kernel = unsafe extern "C" fn(*const u8, *const i8, *const f32, *const f32, usize) -> f32;

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn cosxamix_q5k_asm(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(all(feature = "c", target_arch = "x86_64"))]
unsafe extern "C" {
    fn cosxamix_q5k_c(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(feature = "zig")]
unsafe extern "C" {
    fn cosxamix_q5k_zig(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(all(feature = "zig", cosxamix_v3))]
unsafe extern "C" {
    fn cosxamix_q5k_zig_v3(
        b: *const u8,
        q: *const i8,
        e: *const f32,
        s: *const f32,
        n: usize,
    ) -> f32;
}

/// Las copias que esta compilación trae y esta CPU puede correr.
pub fn disponibles() -> Vec<Copia> {
    #[allow(unused_mut)]
    let mut v = vec![Copia::Escalar];
    #[cfg(target_arch = "x86_64")]
    if super::cpu_avx2() {
        v.push(Copia::Avx2);
        v.push(Copia::Asm);
        #[cfg(feature = "c")]
        v.push(Copia::C);
    }
    #[cfg(feature = "zig")]
    v.push(Copia::Zig);
    v
}

/// La que se usa: la de `COSXAMIX_Q5K` si está disponible; si no, la que respeta
/// el `COSXAMIX_KERNEL` del proyecto y entre esas la que rinde.
pub fn elegida() -> Copia {
    static COPIA: OnceLock<Copia> = OnceLock::new();
    *COPIA.get_or_init(|| {
        let hay = disponibles();
        let pedida =
            std::env::var("COSXAMIX_Q5K")
                .ok()
                .and_then(|v| match v.to_lowercase().as_str() {
                    "escalar" => Some(Copia::Escalar),
                    "avx2" => Some(Copia::Avx2),
                    "asm" => Some(Copia::Asm),
                    "c" => Some(Copia::C),
                    "zig" => Some(Copia::Zig),
                    _ => None,
                });
        if let Some(p) = pedida.filter(|p| hay.contains(p)) {
            return p;
        }
        if super::elegido() == super::Backend::Escalar {
            return Copia::Escalar;
        }
        for preferida in [Copia::Avx2, Copia::Zig] {
            if hay.contains(&preferida) {
                return preferida;
            }
        }
        Copia::Escalar
    })
}

/// El kernel de una copia que no está en Rust, si es de esas.
fn kernel_de(copia: Copia) -> Option<Kernel> {
    match copia {
        #[cfg(target_arch = "x86_64")]
        Copia::Asm => Some(cosxamix_q5k_asm),
        #[cfg(all(feature = "c", target_arch = "x86_64"))]
        Copia::C => Some(cosxamix_q5k_c),
        #[cfg(feature = "zig")]
        Copia::Zig => {
            #[cfg(cosxamix_v3)]
            if super::zig::v3() {
                return Some(cosxamix_q5k_zig_v3);
            }
            Some(cosxamix_q5k_zig)
        }
        _ => None,
    }
}

/// Una fila con una copia dada.
pub fn dot_con(copia: Copia, bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(
        xq.len().is_multiple_of(256),
        "dot_q5k_i8: la fila tiene que ser múltiplo de 256"
    );
    assert_eq!(xe.len(), xq.len() / 32, "dot_q5k_i8: faltan escalas");
    assert_eq!(xs.len(), xe.len(), "dot_q5k_i8: faltan sumas");
    assert!(
        bloques.len() >= xq.len() / 256 * 176,
        "dot_q5k_i8: faltan bytes de pesos"
    );
    if let Some(k) = kernel_de(copia) {
        // SAFETY: las longitudes se comprobaron arriba, y `disponibles` solo
        // ofrece lo que la CPU puede correr.
        return unsafe {
            k(
                bloques.as_ptr(),
                xq.as_ptr(),
                xe.as_ptr(),
                xs.as_ptr(),
                xq.len() / 256,
            )
        };
    }
    #[cfg(target_arch = "x86_64")]
    if copia == Copia::Avx2 {
        // SAFETY: `Avx2` solo se ofrece si la CPU lo tiene.
        return unsafe { super::x86::dot_q5k_i8_avx2(bloques, xq, xe, xs) };
    }
    super::scalar::dot_q5k_i8(bloques, xq, xe, xs)
}

/// Una fila con la copia elegida.
#[inline]
pub fn dot(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    dot_con(elegida(), bloques, xq, xe, xs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::TipoGgml;
    use crate::quant::{cuantizar_bloques, descuantizar};

    /// Una fila de verdad, cuantizada, y una entrada con su escala y su suma.
    fn fila(n_bloques: usize, semilla: u64) -> (Vec<u8>, Vec<i8>, Vec<f32>) {
        let n = n_bloques * 256;
        let mut azar = crate::daiss::Azar::con_semilla(semilla);
        let w: Vec<f32> = (0..n).map(|_| azar.normales(0.02).0).collect();
        let mut pesos = Vec::new();
        cuantizar_bloques(TipoGgml::Q5K, &w, &mut pesos).unwrap();
        let xq: Vec<i8> = (0..n).map(|_| azar.siguiente() as i8).collect();
        let xe: Vec<f32> = (0..n / 32).map(|_| azar.normales(0.1).0.abs()).collect();
        (pesos, xq, xe)
    }

    fn sumas(xq: &[i8]) -> Vec<f32> {
        xq.chunks(32)
            .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
            .collect()
    }

    /// La referencia, contra la definición: descuantizar y multiplicar a mano.
    #[test]
    fn la_referencia_da_lo_mismo_que_descuantizar() {
        for n_bloques in [1, 2, 8] {
            let (pesos, xq, xe) = fila(n_bloques, 5 + n_bloques as u64);
            let xs = sumas(&xq);
            let mut w = vec![0f32; n_bloques * 256];
            descuantizar(TipoGgml::Q5K, &pesos, &mut w).unwrap();
            let mut esperado = 0f64;
            for (i, (p, q)) in w.iter().zip(&xq).enumerate() {
                esperado += f64::from(*p) * f64::from(*q) * f64::from(xe[i / 32]);
            }
            let r = f64::from(crate::kernels::scalar::dot_q5k_i8(&pesos, &xq, &xe, &xs));
            assert!(
                (r - esperado).abs() / esperado.abs().max(1.0) < 1e-3,
                "{r} contra {esperado}"
            );
        }
    }

    /// Todas las copias contra la referencia, también con los extremos de `i8`
    /// y con `qh` en todos unos, que es donde el quinto bit pesa más.
    #[test]
    fn todas_las_copias_dan_lo_mismo_que_la_referencia() {
        for n_bloques in [1, 2, 3, 8, 24] {
            let (mut pesos, mut xq, xe) = fila(n_bloques, 13 + n_bloques as u64);
            for caso in 0..4 {
                if caso == 1 {
                    xq.iter_mut().enumerate().for_each(|(i, v)| {
                        *v = if i % 2 == 0 { 127 } else { -128 };
                    });
                }
                if caso == 2 {
                    for bl in pesos.as_chunks_mut::<176>().0 {
                        bl[16..48].fill(0xFF);
                    }
                }
                if caso == 3 {
                    for bl in pesos.as_chunks_mut::<176>().0 {
                        for (i, b) in bl[4..176].iter_mut().enumerate() {
                            *b = (i * 73 + 29) as u8;
                        }
                    }
                }
                let xs = sumas(&xq);
                let r = crate::kernels::scalar::dot_q5k_i8(&pesos, &xq, &xe, &xs);
                assert!(r.is_finite());
                for copia in disponibles() {
                    let v = dot_con(copia, &pesos, &xq, &xe, &xs);
                    assert!(
                        (v - r).abs() / r.abs().max(1.0) < 1e-4,
                        "{copia:?}, {n_bloques} bloques, caso {caso}: {v} contra {r}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "faltan bytes")]
    fn sin_bytes_suficientes_no_lee_de_mas() {
        let xq = vec![0i8; 256];
        let (xe, xs) = (vec![0f32; 8], vec![0f32; 8]);
        dot(&[0u8; 175], &xq, &xe, &xs);
    }
}
