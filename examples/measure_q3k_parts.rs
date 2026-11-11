//! ¿Qué parte del kernel de Q3_K cuesta, por hilo?
//!
//! El kernel fusionado de Q3_K procesa ~8 GB/s por hilo con los pesos en caché, y
//! el de Q4_K ~11 (DECISIONES §34). Antes de cambiarlo se mide por partes, que es
//! la lección de §24: se copia el kernel y se le quita una cosa por vez. Lo que
//! baja el tiempo al quitar una parte es lo que esa parte cuesta. Las variantes
//! **dan resultados incorrectos a propósito**: sirven solo para medir.
//!
//! - `completo`: el kernel de `src/kernels/x86.rs`, tal cual;
//! - `sin escalas`: las 16 escalas se calculan una vez, fuera del bucle;
//! - `sin tercer bit`: los valores son solo los 2 bits bajos;
//! - `sin f32 por grupo`: suma los `i32` de todo el bloque y convierte una vez,
//!   como las escalas enteras que §24 descartó por precisión;
//! - `solo cargar`: lee los bytes del bloque y la entrada, sin cuentas.
//!
//! Y dos formas de esconder lo que cuestan las escalas **sin cambiar el
//! resultado**, que sí tienen que dar lo mismo que el kernel:
//!
//! - `adelantadas`: las escalas del bloque siguiente se calculan antes de los
//!   grupos del actual;
//! - `dos pasadas`: primero las de toda la fila, ya repartidas en un vector por
//!   grupo, y después los grupos las leen de memoria, sin permutar.
//!
//!     cargo run --release --example measure_q3k_parts

#[cfg(target_arch = "x86_64")]
mod medir {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    #[derive(Clone, Copy, PartialEq)]
    pub enum Variante {
        Completo,
        SinEscalas,
        SinTercerBit,
        SinF32PorGrupo,
        SoloCargar,
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn reducir(v: __m256) -> f32 {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55)))
    }

    /// El kernel de `dot_q3k_i8_avx2`, con la variante elegida en tiempo de
    /// compilación para que el compilador quite lo que no se usa.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn q3k<const V: u8>(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
        let v = match V {
            0 => Variante::Completo,
            1 => Variante::SinEscalas,
            2 => Variante::SinTercerBit,
            3 => Variante::SinF32PorGrupo,
            _ => Variante::SoloCargar,
        };
        let unos = _mm256_set1_epi16(1);
        let dos_bits = _mm256_set1_epi8(0x03);
        let cuatro = _mm256_set1_epi8(4);
        let nibble = _mm_set1_epi8(0x0F);
        let tres = _mm_set1_epi8(3);
        let treinta_y_dos = _mm_set1_epi8(32);
        let corrimientos = _mm_setr_epi32(0, 2, 4, 6);
        let r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
        let r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
        let r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
        let r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut correccion = _mm256_setzero_ps();
        let mut enteros = _mm256_setzero_si256();

        let escalas_de = |b: *const u8, bl: usize| {
            let lo8 = _mm_loadl_epi64(b.add(96) as *const __m128i);
            let bajos = _mm_unpacklo_epi64(
                _mm_and_si128(lo8, nibble),
                _mm_and_si128(_mm_srli_epi16(lo8, 4), nibble),
            );
            let t = (b.add(104) as *const i32).read_unaligned();
            let altos = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(t), corrimientos), tres);
            let esc8 = _mm_sub_epi8(_mm_or_si128(bajos, _mm_slli_epi16(altos, 4)), treinta_y_dos);
            let h = (b.add(108) as *const u16).read_unaligned();
            let d = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)));
            let xe_p = xe.as_ptr().add(bl * 16);
            let e0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(esc8));
            let e1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(esc8, 8)));
            (
                _mm256_mul_ps(_mm256_mul_ps(d, e0), _mm256_loadu_ps(xe_p)),
                _mm256_mul_ps(_mm256_mul_ps(d, e1), _mm256_loadu_ps(xe_p.add(8))),
            )
        };
        let fijas = escalas_de(bloques.as_ptr(), 0);

        for bl in 0..xq.len() / 256 {
            let b = bloques.as_ptr().add(bl * 110);
            let base = bl * 256;
            let hm = _mm256_loadu_si256(b as *const __m256i);
            let qs0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
            let qs1 = _mm256_loadu_si256(b.add(64) as *const __m256i);

            if v == Variante::SoloCargar {
                let mut s = _mm256_xor_si256(hm, _mm256_xor_si256(qs0, qs1));
                for g in 0..8 {
                    s = _mm256_xor_si256(
                        s,
                        _mm256_loadu_si256(xq.as_ptr().add(base + g * 32) as *const __m256i),
                    );
                }
                enteros = _mm256_add_epi32(enteros, s);
                continue;
            }

            let (v0, v1) = if v == Variante::SinEscalas {
                fijas
            } else {
                let (v0, v1) = escalas_de(b, bl);
                let xs_p = xs.as_ptr().add(bl * 16);
                correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
                correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p.add(8)), correccion);
                (v0, v1)
            };

            macro_rules! grupo {
                ($acc:ident, $g:literal, $qs:ident, $lo:literal, $alto:expr, $v:ident, $r:ident) => {{
                    let bajos = _mm256_and_si256(_mm256_srli_epi16($qs, $lo), dos_bits);
                    let q = if v == Variante::SinTercerBit {
                        bajos
                    } else {
                        _mm256_or_si256(bajos, _mm256_and_si256($alto, cuatro))
                    };
                    let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32) as *const __m256i);
                    let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                    if v == Variante::SinF32PorGrupo {
                        enteros = _mm256_add_epi32(enteros, p32);
                    } else {
                        let escala = _mm256_permutevar8x32_ps($v, $r);
                        $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, $acc);
                    }
                }};
            }
            grupo!(acc0, 0, qs0, 0, _mm256_slli_epi16(hm, 2), v0, r0);
            grupo!(acc1, 1, qs0, 2, _mm256_slli_epi16(hm, 1), v0, r1);
            grupo!(acc0, 2, qs0, 4, hm, v0, r2);
            grupo!(acc1, 3, qs0, 6, _mm256_srli_epi16(hm, 1), v0, r3);
            grupo!(acc0, 4, qs1, 0, _mm256_srli_epi16(hm, 2), v1, r0);
            grupo!(acc1, 5, qs1, 2, _mm256_srli_epi16(hm, 3), v1, r1);
            grupo!(acc0, 6, qs1, 4, _mm256_srli_epi16(hm, 4), v1, r2);
            grupo!(acc1, 7, qs1, 6, _mm256_srli_epi16(hm, 5), v1, r3);
            if v == Variante::SinF32PorGrupo {
                acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(enteros), v0, acc0);
                enteros = _mm256_setzero_si256();
            }
        }
        reducir(_mm256_add_ps(acc0, acc1)) - 4.0 * reducir(correccion)
            + _mm256_extract_epi32(enteros, 0) as f32
    }

    /// Las escalas de un bloque: `(v0, v1)` ya multiplicadas por `d` y por las de
    /// la entrada, igual que en el kernel.
    #[inline(always)]
    unsafe fn escalas(b: *const u8, xe: *const f32) -> (__m256, __m256) {
        let nibble = _mm_set1_epi8(0x0F);
        let lo8 = _mm_loadl_epi64(b.add(96) as *const __m128i);
        let bajos = _mm_unpacklo_epi64(
            _mm_and_si128(lo8, nibble),
            _mm_and_si128(_mm_srli_epi16(lo8, 4), nibble),
        );
        let t = (b.add(104) as *const i32).read_unaligned();
        let altos = _mm_and_si128(
            _mm_srlv_epi32(_mm_set1_epi32(t), _mm_setr_epi32(0, 2, 4, 6)),
            _mm_set1_epi8(3),
        );
        let esc8 = _mm_sub_epi8(
            _mm_or_si128(bajos, _mm_slli_epi16(altos, 4)),
            _mm_set1_epi8(32),
        );
        let h = (b.add(108) as *const u16).read_unaligned();
        let d = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)));
        let e0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(esc8));
        let e1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(esc8, 8)));
        (
            _mm256_mul_ps(_mm256_mul_ps(d, e0), _mm256_loadu_ps(xe)),
            _mm256_mul_ps(_mm256_mul_ps(d, e1), _mm256_loadu_ps(xe.add(8))),
        )
    }

    /// Las escalas del bloque siguiente, antes de los grupos del actual.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn adelantadas(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
        let unos = _mm256_set1_epi16(1);
        let dos_bits = _mm256_set1_epi8(0x03);
        let cuatro = _mm256_set1_epi8(4);
        let r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
        let r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
        let r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
        let r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut correccion = _mm256_setzero_ps();
        let n = xq.len() / 256;
        let mut sig = escalas(bloques.as_ptr(), xe.as_ptr());
        for bl in 0..n {
            let b = bloques.as_ptr().add(bl * 110);
            let base = bl * 256;
            let (v0, v1) = sig;
            if bl + 1 < n {
                sig = escalas(b.add(110), xe.as_ptr().add((bl + 1) * 16));
            }
            let xs_p = xs.as_ptr().add(bl * 16);
            correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
            correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p.add(8)), correccion);
            let hm = _mm256_loadu_si256(b as *const __m256i);
            let qs0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
            let qs1 = _mm256_loadu_si256(b.add(64) as *const __m256i);
            macro_rules! grupo {
                ($acc:ident, $g:literal, $qs:ident, $lo:literal, $alto:expr, $v:ident, $r:ident) => {{
                    let bajos = _mm256_and_si256(_mm256_srli_epi16($qs, $lo), dos_bits);
                    let q = _mm256_or_si256(bajos, _mm256_and_si256($alto, cuatro));
                    let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32) as *const __m256i);
                    let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                    let escala = _mm256_permutevar8x32_ps($v, $r);
                    $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, $acc);
                }};
            }
            grupo!(acc0, 0, qs0, 0, _mm256_slli_epi16(hm, 2), v0, r0);
            grupo!(acc1, 1, qs0, 2, _mm256_slli_epi16(hm, 1), v0, r1);
            grupo!(acc0, 2, qs0, 4, hm, v0, r2);
            grupo!(acc1, 3, qs0, 6, _mm256_srli_epi16(hm, 1), v0, r3);
            grupo!(acc0, 4, qs1, 0, _mm256_srli_epi16(hm, 2), v1, r0);
            grupo!(acc1, 5, qs1, 2, _mm256_srli_epi16(hm, 3), v1, r1);
            grupo!(acc0, 6, qs1, 4, _mm256_srli_epi16(hm, 4), v1, r2);
            grupo!(acc1, 7, qs1, 6, _mm256_srli_epi16(hm, 5), v1, r3);
        }
        reducir(_mm256_add_ps(acc0, acc1)) - 4.0 * reducir(correccion)
    }

    /// Primero todas las escalas de la fila, repartidas en un vector por grupo
    /// (`[s s s s t t t t]`), y después los grupos con la escala de memoria.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dos_pasadas(
        bloques: &[u8],
        xq: &[i8],
        xe: &[f32],
        xs: &[f32],
        tabla: &mut Vec<__m256>,
    ) -> f32 {
        let unos = _mm256_set1_epi16(1);
        let dos_bits = _mm256_set1_epi8(0x03);
        let cuatro = _mm256_set1_epi8(4);
        let r = [
            _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1),
            _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3),
            _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5),
            _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7),
        ];
        let n = xq.len() / 256;
        tabla.clear();
        let mut correccion = _mm256_setzero_ps();
        for bl in 0..n {
            let (v0, v1) = escalas(bloques.as_ptr().add(bl * 110), xe.as_ptr().add(bl * 16));
            let xs_p = xs.as_ptr().add(bl * 16);
            correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
            correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p.add(8)), correccion);
            for v in [v0, v1] {
                for ri in r {
                    tabla.push(_mm256_permutevar8x32_ps(v, ri));
                }
            }
        }
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let t = tabla.as_ptr();
        for bl in 0..n {
            let b = bloques.as_ptr().add(bl * 110);
            let base = bl * 256;
            let e = t.add(bl * 8);
            let hm = _mm256_loadu_si256(b as *const __m256i);
            let qs0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
            let qs1 = _mm256_loadu_si256(b.add(64) as *const __m256i);
            macro_rules! grupo {
                ($acc:ident, $g:literal, $qs:ident, $lo:literal, $alto:expr) => {{
                    let bajos = _mm256_and_si256(_mm256_srli_epi16($qs, $lo), dos_bits);
                    let q = _mm256_or_si256(bajos, _mm256_and_si256($alto, cuatro));
                    let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32) as *const __m256i);
                    let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                    $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), *e.add($g), $acc);
                }};
            }
            grupo!(acc0, 0, qs0, 0, _mm256_slli_epi16(hm, 2));
            grupo!(acc1, 1, qs0, 2, _mm256_slli_epi16(hm, 1));
            grupo!(acc0, 2, qs0, 4, hm);
            grupo!(acc1, 3, qs0, 6, _mm256_srli_epi16(hm, 1));
            grupo!(acc0, 4, qs1, 0, _mm256_srli_epi16(hm, 2));
            grupo!(acc1, 5, qs1, 2, _mm256_srli_epi16(hm, 3));
            grupo!(acc0, 6, qs1, 4, _mm256_srli_epi16(hm, 4));
            grupo!(acc1, 7, qs1, 6, _mm256_srli_epi16(hm, 5));
        }
        reducir(_mm256_add_ps(acc0, acc1)) - 4.0 * reducir(correccion)
    }

    /// La mediana de varias tandas, en ns por fila.
    pub fn medir(mut f: impl FnMut() -> f32) -> f64 {
        const VECES: u32 = 20_000;
        let mut v: Vec<f64> = (0..11)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..VECES {
                    black_box(f());
                }
                t.elapsed().as_secs_f64() * 1e9 / f64::from(VECES)
            })
            .collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use cosxamix::gguf::TipoGgml;
    use cosxamix::quant::cuantizar_bloques;
    use medir::*;

    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        println!("  hace falta AVX2 y FMA");
        return;
    }
    let ancho: usize = std::env::var("COSXAMIX_ANCHO")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(11008);
    let mut azar = cosxamix::daiss::Azar::con_semilla(5);
    let w: Vec<f32> = (0..ancho).map(|_| azar.normales(0.02).0).collect();
    let mut pesos = Vec::new();
    cuantizar_bloques(TipoGgml::Q3K, &w, &mut pesos).unwrap();
    let xq: Vec<i8> = (0..ancho).map(|_| azar.siguiente() as i8).collect();
    let xe: Vec<f32> = (0..ancho / 16)
        .map(|_| azar.normales(0.1).0.abs())
        .collect();
    let xs: Vec<f32> = xq
        .chunks(16)
        .map(|c| c.iter().map(|v| i32::from(*v)).sum::<i32>() as f32)
        .collect();

    // El kernel de verdad, para comprobar que la copia es la misma.
    let real = cosxamix::kernels::dot_q3k_i8(&pesos, &xq, &xe, &xs);
    let copia = unsafe { q3k::<0>(&pesos, &xq, &xe, &xs) };
    assert!(
        (real - copia).abs() <= real.abs() * 1e-5,
        "{real} contra {copia}"
    );

    println!(
        "  una fila de {ancho} valores en Q3_K, {} bytes, en caché",
        pesos.len()
    );
    let gbs = |ns: f64| pesos.len() as f64 / ns;
    let completo = medir(|| unsafe { q3k::<0>(&pesos, &xq, &xe, &xs) });
    println!(
        "  {:<20}{completo:>8.0} ns {:>6.1} GB/s",
        "completo",
        gbs(completo)
    );
    for (nombre, ns) in [
        (
            "sin escalas",
            medir(|| unsafe { q3k::<1>(&pesos, &xq, &xe, &xs) }),
        ),
        (
            "sin tercer bit",
            medir(|| unsafe { q3k::<2>(&pesos, &xq, &xe, &xs) }),
        ),
        (
            "sin f32 por grupo",
            medir(|| unsafe { q3k::<3>(&pesos, &xq, &xe, &xs) }),
        ),
        (
            "solo cargar",
            medir(|| unsafe { q3k::<4>(&pesos, &xq, &xe, &xs) }),
        ),
    ] {
        println!(
            "  {nombre:<20}{ns:>8.0} ns {:>6.1} GB/s   esa parte cuesta {:>3.0}%",
            gbs(ns),
            (1.0 - ns / completo) * 100.0
        );
    }

    // Las dos formas que no cambian el resultado: primero, que den lo mismo.
    let mut tabla = Vec::new();
    let ad = unsafe { adelantadas(&pesos, &xq, &xe, &xs) };
    let dp = unsafe { dos_pasadas(&pesos, &xq, &xe, &xs, &mut tabla) };
    for (nombre, r) in [("adelantadas", ad), ("dos pasadas", dp)] {
        assert!(
            (real - r).abs() <= real.abs() * 1e-5,
            "{nombre}: {r} contra {real}"
        );
    }
    println!();
    for (nombre, ns) in [
        (
            "adelantadas",
            medir(|| unsafe { adelantadas(&pesos, &xq, &xe, &xs) }),
        ),
        (
            "dos pasadas",
            medir(|| unsafe { dos_pasadas(&pesos, &xq, &xe, &xs, &mut tabla) }),
        ),
    ] {
        println!(
            "  {nombre:<20}{ns:>8.0} ns {:>6.1} GB/s   {:.2}x",
            gbs(ns),
            completo / ns
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("  solo en x86-64");
}
