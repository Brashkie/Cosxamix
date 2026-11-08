//! El kernel fusionado de Q2_K: una fila contra la entrada cuantizada, sin
//! desempaquetar la fila a ningún buffer.
//!
//! Es lo que Chavín necesita: Q2_K es el único formato en el que entra en
//! memoria, y era el único de los que importan sin kernel fusionado.
//!
//! Un bloque son 84 bytes por cada 256 valores: 16 escalas de un byte —4 bits de
//! escala y 4 de mínimo—, 64 bytes de valores de 2 bits y `d` y `dmin` en `f16`.
//! Cada valor vale `d * esc * q - dmin * min`, con `q` de 0 a 3, y como la entrada
//! se cuantizó por sub-bloques de 16:
//!
//! ```text
//! suma = Σ xe * (d * esc * Σ(x * q) - dmin * min * Σ x)
//! ```
//!
//! `Σ x` ya viene calculada en `xs`: ese término no toca los bytes de la fila.
//!
//! Está en todos los lenguajes de la frontera, con el mismo algoritmo: la
//! referencia en Rust ([`escalar`]), intrínsecos AVX2 ([`avx2`]), ensamblador
//! (`kernels/q2k_avx2_*.s`), C (`kernels/q2k_avx2.c` y, en ARM64 con `sdot`,
//! `kernels/q2k_neon.c`) y Zig (`kernels/q2k.zig`, también NEON sin `sdot`).
//! `COSXAMIX_Q2K` elige uno, para medir: `escalar`, `avx2`, `asm`, `c`, `neon` o
//! `zig`.

use std::sync::OnceLock;

use crate::quant::leer_f16;

/// Los lenguajes en los que está escrito.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Copia {
    Escalar,
    Avx2,
    Asm,
    C,
    Neon,
    Zig,
}

/// La referencia. Es la especificación de las demás.
pub fn escalar(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    let mut total = 0f32;
    for bl in 0..xq.len() / 256 {
        let o = bl * 84;
        let esc = &bloques[o..o + 16];
        let qs = &bloques[o + 16..o + 80];
        let d = leer_f16(bloques, o + 80);
        let dm = leer_f16(bloques, o + 82);
        // Ocho grupos de 32 valores; en cada uno cambia el corrimiento.
        for g in 0..8 {
            let qsg = &qs[(g / 4) * 32..][..32];
            let corrimiento = 2 * (g % 4);
            let x = &xq[bl * 256 + g * 32..][..32];
            for medio in 0..2 {
                let s = g * 2 + medio;
                let sub = bl * 16 + s;
                let mut entero = 0i32;
                for l in 0..16 {
                    let idx = medio * 16 + l;
                    entero += i32::from(x[idx]) * i32::from((qsg[idx] >> corrimiento) & 3);
                }
                let e = d * f32::from(esc[s] & 0xF);
                let m = dm * f32::from(esc[s] >> 4);
                total += xe[sub] * (e * entero as f32 - m * xs[sub]);
            }
        }
    }
    total
}

/// Con intrínsecos de AVX2.
///
/// # Safety
///
/// La CPU tiene que tener AVX2, FMA y F16C ([`super::cpu_avx2`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn avx2(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let unos = _mm256_set1_epi16(1);
    let dos_bits = _mm256_set1_epi8(0x03);
    let nibble = _mm_set1_epi8(0x0F);
    let r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    let r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    let r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut correccion = _mm256_setzero_ps();

    for bl in 0..xq.len() / 256 {
        // SAFETY: quien llama garantiza 84 bytes por bloque, 256 valores y 16
        // escalas y sumas por bloque (`dot_q2k_i8` lo comprueba).
        unsafe {
            let b = bloques.as_ptr().add(bl * 84);
            let esc = _mm_loadu_si128(b.cast());
            let bajos = _mm_and_si128(esc, nibble);
            let altos = _mm_and_si128(_mm_srli_epi16(esc, 4), nibble);
            let f8 = |v: __m128i| _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));

            let f16_en = |o: usize| {
                let h = (b.add(o) as *const u16).read_unaligned();
                _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(h))))
            };
            let (d, dm) = (f16_en(80), f16_en(82));

            let xe_p = xe.as_ptr().add(bl * 16);
            let xs_p = xs.as_ptr().add(bl * 16);
            let (e0, e1) = (_mm256_loadu_ps(xe_p), _mm256_loadu_ps(xe_p.add(8)));
            let v0 = _mm256_mul_ps(_mm256_mul_ps(d, f8(bajos)), e0);
            let v1 = _mm256_mul_ps(_mm256_mul_ps(d, f8(_mm_srli_si128(bajos, 8))), e1);
            let m0 = _mm256_mul_ps(_mm256_mul_ps(dm, f8(altos)), e0);
            let m1 = _mm256_mul_ps(_mm256_mul_ps(dm, f8(_mm_srli_si128(altos, 8))), e1);
            correccion = _mm256_fmadd_ps(m0, _mm256_loadu_ps(xs_p), correccion);
            correccion = _mm256_fmadd_ps(m1, _mm256_loadu_ps(xs_p.add(8)), correccion);

            let qs0 = _mm256_loadu_si256(b.add(16).cast());
            let qs1 = _mm256_loadu_si256(b.add(48).cast());
            let base = bl * 256;

            macro_rules! grupo {
                ($acc:ident, $g:literal, $qs:ident, $lo:literal, $v:ident, $r:ident) => {{
                    let q = _mm256_and_si256(_mm256_srli_epi16($qs, $lo), dos_bits);
                    let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32).cast());
                    let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                    let escala = _mm256_permutevar8x32_ps($v, $r);
                    $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, $acc);
                }};
            }
            grupo!(acc0, 0, qs0, 0, v0, r0);
            grupo!(acc1, 1, qs0, 2, v0, r1);
            grupo!(acc0, 2, qs0, 4, v0, r2);
            grupo!(acc1, 3, qs0, 6, v0, r3);
            grupo!(acc0, 4, qs1, 0, v1, r0);
            grupo!(acc1, 5, qs1, 2, v1, r1);
            grupo!(acc0, 6, qs1, 4, v1, r2);
            grupo!(acc1, 7, qs1, 6, v1, r3);
        }
    }
    // SAFETY: AVX2 lo garantiza quien llama.
    unsafe {
        super::x86::reducir_f32(_mm256_add_ps(acc0, acc1)) - super::x86::reducir_f32(correccion)
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
std::arch::global_asm!(include_str!("../../kernels/q2k_avx2_win64.s"));

#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
std::arch::global_asm!(include_str!("../../kernels/q2k_avx2_sysv.s"));

/// La forma de todas las copias que no están en Rust.
type Kernel = unsafe extern "C" fn(*const u8, *const i8, *const f32, *const f32, usize) -> f32;

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn cosxamix_q2k_asm(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(all(feature = "c", target_arch = "x86_64"))]
unsafe extern "C" {
    fn cosxamix_q2k_c(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(all(feature = "c", target_arch = "aarch64"))]
unsafe extern "C" {
    fn cosxamix_q2k_neon(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize)
        -> f32;
}

#[cfg(feature = "zig")]
unsafe extern "C" {
    fn cosxamix_q2k_zig(b: *const u8, q: *const i8, e: *const f32, s: *const f32, n: usize) -> f32;
}

#[cfg(all(feature = "zig", cosxamix_v3))]
unsafe extern "C" {
    fn cosxamix_q2k_zig_v3(
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
    #[cfg(all(feature = "c", target_arch = "aarch64"))]
    if super::c::hay_dotprod() {
        v.push(Copia::Neon);
    }
    #[cfg(feature = "zig")]
    v.push(Copia::Zig);
    v
}

/// La que se usa: la de `COSXAMIX_Q2K` si está disponible; si no, la que respeta
/// el `COSXAMIX_KERNEL` del proyecto y entre esas la que rinde.
pub fn elegida() -> Copia {
    static COPIA: OnceLock<Copia> = OnceLock::new();
    *COPIA.get_or_init(|| {
        let hay = disponibles();
        let pedida =
            std::env::var("COSXAMIX_Q2K")
                .ok()
                .and_then(|v| match v.to_lowercase().as_str() {
                    "escalar" => Some(Copia::Escalar),
                    "avx2" => Some(Copia::Avx2),
                    "asm" => Some(Copia::Asm),
                    "c" => Some(Copia::C),
                    "neon" => Some(Copia::Neon),
                    "zig" => Some(Copia::Zig),
                    _ => None,
                });
        if let Some(p) = pedida.filter(|p| hay.contains(p)) {
            return p;
        }
        if super::elegido() == super::Backend::Escalar {
            return Copia::Escalar;
        }
        for preferida in [Copia::Avx2, Copia::Neon, Copia::Zig] {
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
        Copia::Asm => Some(cosxamix_q2k_asm),
        #[cfg(all(feature = "c", target_arch = "x86_64"))]
        Copia::C => Some(cosxamix_q2k_c),
        #[cfg(all(feature = "c", target_arch = "aarch64"))]
        Copia::Neon => Some(cosxamix_q2k_neon),
        #[cfg(feature = "zig")]
        Copia::Zig => {
            #[cfg(cosxamix_v3)]
            if super::zig::v3() {
                return Some(cosxamix_q2k_zig_v3);
            }
            Some(cosxamix_q2k_zig)
        }
        _ => None,
    }
}

/// Una fila con una copia dada.
pub fn dot_con(copia: Copia, bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(
        xq.len().is_multiple_of(256),
        "dot_q2k_i8: la fila tiene que ser múltiplo de 256"
    );
    assert_eq!(xe.len(), xq.len() / 16, "dot_q2k_i8: faltan escalas");
    assert_eq!(xs.len(), xe.len(), "dot_q2k_i8: faltan sumas");
    assert!(
        bloques.len() >= xq.len() / 256 * 84,
        "dot_q2k_i8: faltan bytes de pesos"
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
        return unsafe { avx2(bloques, xq, xe, xs) };
    }
    escalar(bloques, xq, xe, xs)
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
    fn fila(n_bloques: usize, semilla: u64) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
        let n = n_bloques * 256;
        let mut azar = crate::daiss::Azar::con_semilla(semilla);
        let w: Vec<f32> = (0..n).map(|_| azar.normales(0.02).0).collect();
        let mut pesos = Vec::new();
        cuantizar_bloques(TipoGgml::Q2K, &w, &mut pesos).unwrap();
        let xq: Vec<i8> = (0..n).map(|_| azar.siguiente() as i8).collect();
        let xe: Vec<f32> = (0..n / 16).map(|_| azar.normales(0.1).0.abs()).collect();
        let xs: Vec<f32> = xq
            .chunks(16)
            .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
            .collect();
        (pesos, xq, xe, xs)
    }

    /// La referencia, contra la definición: descuantizar la fila y multiplicar
    /// a mano. Es lo que dice que el formato se leyó bien.
    #[test]
    fn la_referencia_da_lo_mismo_que_descuantizar() {
        for n_bloques in [1, 2, 8] {
            let (pesos, xq, xe, xs) = fila(n_bloques, 3 + n_bloques as u64);
            let mut w = vec![0f32; n_bloques * 256];
            descuantizar(TipoGgml::Q2K, &pesos, &mut w).unwrap();
            let mut esperado = 0f64;
            for (i, (p, q)) in w.iter().zip(&xq).enumerate() {
                esperado += f64::from(*p) * f64::from(*q) * f64::from(xe[i / 16]);
            }
            let r = f64::from(escalar(&pesos, &xq, &xe, &xs));
            assert!(
                (r - esperado).abs() / esperado.abs().max(1.0) < 1e-3,
                "{r} contra {esperado}"
            );
        }
    }

    /// Todas las copias, contra la referencia, también con los bytes más raros:
    /// valores extremos de `i8` y bytes de escala al azar.
    #[test]
    fn todas_las_copias_dan_lo_mismo_que_la_referencia() {
        for n_bloques in [1, 2, 3, 8, 24] {
            let (mut pesos, mut xq, xe, _) = fila(n_bloques, 11 + n_bloques as u64);
            // La mitad de las veces, los extremos: es donde `maddubs` podría
            // saturar si la cuenta estuviera mal.
            for caso in 0..3 {
                if caso == 1 {
                    xq.iter_mut().enumerate().for_each(|(i, v)| {
                        *v = if i % 2 == 0 { 127 } else { -128 };
                    });
                }
                if caso == 2 {
                    for bl in pesos.as_chunks_mut::<84>().0 {
                        for (i, b) in bl[16..80].iter_mut().enumerate() {
                            *b = (i * 37) as u8 | 0x11;
                        }
                    }
                }
                let xs: Vec<f32> = xq
                    .chunks(16)
                    .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
                    .collect();
                let r = escalar(&pesos, &xq, &xe, &xs);
                assert!(r.is_finite());
                for copia in disponibles() {
                    let v = dot_con(copia, &pesos, &xq, &xe, &xs);
                    assert!(
                        (v - r).abs() / r.abs().max(1.0) < 1e-3,
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
        let (xe, xs) = (vec![0f32; 16], vec![0f32; 16]);
        dot(&[0u8; 83], &xq, &xe, &xs);
    }
}
