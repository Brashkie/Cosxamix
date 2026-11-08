//! Clasificar bytes de a 32: letras ASCII, dígitos y bytes altos.
//!
//! Es el kernel de la pre-tokenización. En vez de mirar un byte a la vez,
//! saca tres máscaras por bloque, y encontrar dónde termina una palabra es
//! buscar el primer cero con `tzcnt`.
//!
//! Está en todos los lenguajes de la frontera, con el mismo algoritmo y el
//! mismo resultado, bit a bit: la referencia en Rust ([`escalar`]), intrínsecos
//! AVX2, ensamblador (`kernels/classify_avx2_*.s`), Zig
//! (`kernels/classify.zig`, también NEON) y C (`kernels/classify_avx2.c`).
//! `COSXAMIX_CLASIFICAR` elige uno, para medir: `escalar`, `avx2`, `asm`, `zig`
//! o `c`.

use std::sync::OnceLock;

/// Los lenguajes en los que está escrito.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Copia {
    Escalar,
    Avx2,
    Asm,
    Zig,
    C,
}

/// La referencia, un byte a la vez. Es la especificación de las demás.
pub fn escalar(bytes: &[u8], letras: &mut [u32], digitos: &mut [u32], altos: &mut [u32]) {
    for (i, bloque) in bytes.as_chunks::<32>().0.iter().enumerate() {
        let (mut l, mut d, mut h) = (0u32, 0u32, 0u32);
        for (k, b) in bloque.iter().enumerate() {
            l |= u32::from(b.is_ascii_alphabetic()) << k;
            d |= u32::from(b.is_ascii_digit()) << k;
            h |= u32::from(*b >= 0x80) << k;
        }
        letras[i] = l;
        digitos[i] = d;
        altos[i] = h;
    }
}

/// # Safety
///
/// La CPU tiene que tener AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2(bytes: &[u8], letras: &mut [u32], digitos: &mut [u32], altos: &mut [u32]) {
    use std::arch::x86_64::*;
    let bit5 = _mm256_set1_epi8(0x20);
    let a = _mm256_set1_epi8(b'a' as i8);
    let veinticinco = _mm256_set1_epi8(25);
    let cero = _mm256_set1_epi8(b'0' as i8);
    let nueve = _mm256_set1_epi8(9);
    for (i, bloque) in bytes.as_chunks::<32>().0.iter().enumerate() {
        // SAFETY: `chunks_exact(32)` da 32 bytes, y la carga no pide alineación.
        let v = unsafe { _mm256_loadu_si256(bloque.as_ptr().cast()) };
        // Sin signo, `x <= k` es `min(x, k) == x`: AVX2 no compara sin signo.
        let l = _mm256_sub_epi8(_mm256_or_si256(v, bit5), a);
        letras[i] =
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_min_epu8(l, veinticinco), l)) as u32;
        let d = _mm256_sub_epi8(v, cero);
        digitos[i] = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_min_epu8(d, nueve), d)) as u32;
        altos[i] = _mm256_movemask_epi8(v) as u32;
    }
}

#[cfg(all(target_arch = "x86_64", target_os = "windows"))]
std::arch::global_asm!(include_str!("../../kernels/classify_avx2_win64.s"));

#[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
std::arch::global_asm!(include_str!("../../kernels/classify_avx2_sysv.s"));

/// La forma de todas las copias que no están en Rust.
type Kernel = unsafe extern "C" fn(*const u8, usize, *mut u32, *mut u32, *mut u32);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn cosxamix_clasificar_asm(
        p: *const u8,
        bloques: usize,
        letras: *mut u32,
        digitos: *mut u32,
        altos: *mut u32,
    );
}

#[cfg(all(feature = "c", target_arch = "x86_64"))]
unsafe extern "C" {
    fn cosxamix_clasificar_c(
        p: *const u8,
        bloques: usize,
        letras: *mut u32,
        digitos: *mut u32,
        altos: *mut u32,
    );
}

#[cfg(feature = "zig")]
unsafe extern "C" {
    fn cosxamix_clasificar(
        p: *const u8,
        bloques: usize,
        letras: *mut u32,
        digitos: *mut u32,
        altos: *mut u32,
    );
}

#[cfg(all(feature = "zig", cosxamix_v3))]
unsafe extern "C" {
    fn cosxamix_clasificar_v3(
        p: *const u8,
        bloques: usize,
        letras: *mut u32,
        digitos: *mut u32,
        altos: *mut u32,
    );
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

/// La que se usa: la de `COSXAMIX_CLASIFICAR` si hay una disponible; si no, la
/// primera de las que rinden (la decisión medida está en DECISIONES).
pub fn elegida() -> Copia {
    static COPIA: OnceLock<Copia> = OnceLock::new();
    *COPIA.get_or_init(|| {
        let hay = disponibles();
        let pedida = std::env::var("COSXAMIX_CLASIFICAR").ok().and_then(|v| {
            match v.to_lowercase().as_str() {
                "escalar" => Some(Copia::Escalar),
                "avx2" => Some(Copia::Avx2),
                "asm" => Some(Copia::Asm),
                "zig" => Some(Copia::Zig),
                "c" => Some(Copia::C),
                _ => None,
            }
        });
        if let Some(p) = pedida.filter(|p| hay.contains(p)) {
            return p;
        }
        for preferida in [Copia::Asm, Copia::Zig] {
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
        Copia::Asm => Some(cosxamix_clasificar_asm),
        #[cfg(all(feature = "c", target_arch = "x86_64"))]
        Copia::C => Some(cosxamix_clasificar_c),
        #[cfg(feature = "zig")]
        Copia::Zig => {
            #[cfg(cosxamix_v3)]
            if super::zig::v3() {
                return Some(cosxamix_clasificar_v3);
            }
            Some(cosxamix_clasificar)
        }
        _ => None,
    }
}

/// Las máscaras de `bytes` con una copia dada. Cada una mide un `u32` por
/// bloque de 32 bytes, contando el último, que puede venir incompleto: lo que
/// le falta cuenta como un byte que no es nada de eso.
pub fn clasificar_con(
    copia: Copia,
    bytes: &[u8],
    letras: &mut [u32],
    digitos: &mut [u32],
    altos: &mut [u32],
) {
    let bloques = bytes.len() / 32;
    let total = bytes.len().div_ceil(32);
    assert!(letras.len() >= total && digitos.len() >= total && altos.len() >= total);
    let enteros = &bytes[..bloques * 32];

    if let Some(k) = kernel_de(copia) {
        // SAFETY: `disponibles` solo ofrece lo que la CPU puede correr; `bytes`
        // tiene `bloques * 32` bytes y los tres destinos, lugar para `bloques`.
        unsafe {
            k(
                bytes.as_ptr(),
                bloques,
                letras.as_mut_ptr(),
                digitos.as_mut_ptr(),
                altos.as_mut_ptr(),
            )
        };
    } else {
        #[cfg(target_arch = "x86_64")]
        if copia == Copia::Avx2 {
            // SAFETY: `Avx2` solo se ofrece si la CPU lo tiene.
            unsafe { avx2(enteros, letras, digitos, altos) };
        } else {
            escalar(enteros, letras, digitos, altos);
        }
        #[cfg(not(target_arch = "x86_64"))]
        escalar(enteros, letras, digitos, altos);
    }

    if total > bloques {
        let mut cola = [0u8; 32];
        cola[..bytes.len() - enteros.len()].copy_from_slice(&bytes[enteros.len()..]);
        let (mut l, mut d, mut h) = ([0u32], [0u32], [0u32]);
        escalar(&cola, &mut l, &mut d, &mut h);
        letras[bloques] = l[0];
        digitos[bloques] = d[0];
        altos[bloques] = h[0];
    }
}

/// Las máscaras de un texto, con la copia elegida.
#[derive(Debug, Default)]
pub struct Mascaras {
    pub letras: Vec<u32>,
    pub digitos: Vec<u32>,
    pub altos: Vec<u32>,
}

impl Mascaras {
    pub fn de(bytes: &[u8]) -> Self {
        Self::con(elegida(), bytes)
    }

    pub fn con(copia: Copia, bytes: &[u8]) -> Self {
        // Uno más al final, en cero: buscar el fin de una corrida siempre
        // encuentra uno sin mirar si se terminó el texto.
        let n = bytes.len().div_ceil(32) + 1;
        let mut m = Self {
            letras: vec![0; n],
            digitos: vec![0; n],
            altos: vec![0; n],
        };
        clasificar_con(copia, bytes, &mut m.letras, &mut m.digitos, &mut m.altos);
        m
    }
}

/// La primera posición `>= i` cuyo bit está en cero. La máscara termina con un
/// bloque en cero, así que siempre hay una.
#[inline]
pub fn primer_cero(mascara: &[u32], i: usize) -> usize {
    let mut b = i / 32;
    let m = !mascara[b] >> (i % 32);
    if m != 0 {
        return i + m.trailing_zeros() as usize;
    }
    loop {
        b += 1;
        let m = !mascara[b];
        if m != 0 {
            return b * 32 + m.trailing_zeros() as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Todas las copias, con todos los bytes en todas las posiciones de un
    /// bloque, y con largos que no son múltiplos de 32.
    #[test]
    fn todas_las_copias_dan_lo_mismo_que_la_referencia() {
        let mut datos: Vec<u8> = Vec::new();
        for desplazado in 0..32 {
            datos.extend(std::iter::repeat_n(b' ', desplazado));
            datos.extend(0..=255u8);
        }
        let mut azar = 0x0123_4567_89AB_CDEF_u64;
        for _ in 0..4096 {
            azar ^= azar << 13;
            azar ^= azar >> 7;
            azar ^= azar << 17;
            datos.push(azar as u8);
        }
        for largo in [0, 1, 31, 32, 33, 63, 64, 65, 255, 256, 1000, datos.len()] {
            let b = &datos[..largo];
            let r = Mascaras::con(Copia::Escalar, b);
            // La referencia, contra la definición.
            for (i, x) in b.iter().enumerate() {
                let bit = |m: &[u32]| m[i / 32] >> (i % 32) & 1 == 1;
                assert_eq!(bit(&r.letras), x.is_ascii_alphabetic(), "byte {x}");
                assert_eq!(bit(&r.digitos), x.is_ascii_digit(), "byte {x}");
                assert_eq!(bit(&r.altos), *x >= 0x80, "byte {x}");
            }
            for copia in disponibles() {
                let m = Mascaras::con(copia, b);
                assert_eq!(m.letras, r.letras, "{copia:?} largo {largo}");
                assert_eq!(m.digitos, r.digitos, "{copia:?} largo {largo}");
                assert_eq!(m.altos, r.altos, "{copia:?} largo {largo}");
            }
        }
    }

    #[test]
    fn primer_cero_cruza_bloques() {
        let m = [u32::MAX, u32::MAX, 0b0111, 0];
        assert_eq!(primer_cero(&m, 0), 64 + 3);
        assert_eq!(primer_cero(&m, 40), 64 + 3);
        assert_eq!(primer_cero(&m, 64 + 3), 64 + 3);
        assert_eq!(primer_cero(&m, 64 + 4), 64 + 4);
        assert_eq!(primer_cero(&m, 96), 96);
    }
}
