//! Multiplicación de matrices en f32: `C += A · B`.
//!
//! Es el kernel de todo lo que no son pesos cuantizados: las convoluciones y
//! las capas lineales del UNet y del codificador de texto de las imágenes
//! (`image_gen`), unos 400 mil millones de multiplicaciones por imagen. Antes
//! las hacía candle; desde la 0.11.0, esto.
//!
//! Al estilo de BLIS:
//!
//! - **un microkernel de 6×16** que lleva 12 acumuladores AVX2 (6 filas × 2
//!   registros de 8) y recorre `k` con una carga de 16 valores de B y 6
//!   difusiones de A por paso: 12 FMA por cada 8 cargas;
//! - **paneles empaquetados**: un bloque de A en paneles de 6 filas y uno de B
//!   en paneles de 16 columnas, contiguos en el orden en que el microkernel los
//!   lee. El empaquetado es también donde **los pesos en f16 pasan a f32**, así
//!   que en memoria se quedan en f16 (la mitad) y la cuenta va en f32;
//! - **A y B se leen a través de un trait** ([`FuenteA`], [`FuenteB`]): una
//!   matriz común con sus pasos, o una ventana de convolución que arma el
//!   im2col al empaquetar, sin materializarlo entero.
//!
//! El trabajo se reparte en bloques de `MC × NC` sobre los núcleos, cada uno
//! con sus propios paneles. Sin AVX2 (o en ARM), el microkernel escalar, que es
//! también la referencia de los tests.

use rayon::prelude::*;

pub const MR: usize = 6;
pub const NR: usize = 16;
const MC: usize = 96;
const NC: usize = 256;
const KC: usize = 256;

/// De dónde salen los valores de A (`m × k`), panel por panel.
pub trait FuenteA: Sync {
    /// Escribe `dst[kk * MR + ii] = A[i0 + ii, k0 + kk]` para `ii < filas`,
    /// `kk < kc`, y ceros en las filas que faltan hasta `MR`.
    fn empacar_a(&self, i0: usize, filas: usize, k0: usize, kc: usize, dst: &mut [f32]);
}

/// De dónde salen los valores de B (`k × n`), panel por panel.
pub trait FuenteB: Sync {
    /// Escribe `dst[kk * NR + jj] = B[k0 + kk, j0 + jj]` para `jj < columnas`,
    /// `kk < kc`, y ceros en las columnas que faltan hasta `NR`.
    fn empacar_b(&self, k0: usize, kc: usize, j0: usize, columnas: usize, dst: &mut [f32]);
}

/// Una matriz f32 con sus pasos: `X[i, j] = datos[i * fila + j * col]`. Con
/// `fila`/`col` cambiados es la traspuesta, sin copiarla.
pub struct Mat<'a> {
    pub datos: &'a [f32],
    pub fila: usize,
    pub col: usize,
}

/// Pesos en f16 fila por fila: `W[i, j] = f16(datos[i * fila + j])`. Como A
/// es `W`; como B, `W` traspuesta (ver [`MatF16T`]).
pub struct MatF16<'a> {
    pub datos: &'a [u16],
    pub fila: usize,
}

/// `Wᵀ` de unos pesos f16 guardados fila por fila (`out × in`), como B: lo
/// que pide una capa lineal `y = x · Wᵀ`.
pub struct MatF16T<'a> {
    pub datos: &'a [u16],
    pub fila: usize,
}

impl FuenteA for Mat<'_> {
    fn empacar_a(&self, i0: usize, filas: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        for kk in 0..kc {
            let d = &mut dst[kk * MR..kk * MR + MR];
            for (ii, x) in d.iter_mut().enumerate() {
                *x = if ii < filas {
                    self.datos[(i0 + ii) * self.fila + (k0 + kk) * self.col]
                } else {
                    0.0
                };
            }
        }
    }
}

impl FuenteB for Mat<'_> {
    fn empacar_b(&self, k0: usize, kc: usize, j0: usize, columnas: usize, dst: &mut [f32]) {
        for kk in 0..kc {
            let d = &mut dst[kk * NR..kk * NR + NR];
            let base = (k0 + kk) * self.fila + j0 * self.col;
            if self.col == 1 {
                d[..columnas].copy_from_slice(&self.datos[base..base + columnas]);
                d[columnas..].fill(0.0);
            } else {
                for (jj, x) in d.iter_mut().enumerate() {
                    *x = if jj < columnas {
                        self.datos[base + jj * self.col]
                    } else {
                        0.0
                    };
                }
            }
        }
    }
}

impl FuenteA for MatF16<'_> {
    fn empacar_a(&self, i0: usize, filas: usize, k0: usize, kc: usize, dst: &mut [f32]) {
        // Fila por fila: cada una se convierte entera y se reparte en el panel.
        let mut tmp = [0f32; KC];
        for ii in 0..MR {
            if ii < filas {
                let base = (i0 + ii) * self.fila + k0;
                f16_a_f32(&self.datos[base..base + kc], &mut tmp[..kc]);
                for kk in 0..kc {
                    dst[kk * MR + ii] = tmp[kk];
                }
            } else {
                for kk in 0..kc {
                    dst[kk * MR + ii] = 0.0;
                }
            }
        }
    }
}

impl FuenteB for MatF16T<'_> {
    fn empacar_b(&self, k0: usize, kc: usize, j0: usize, columnas: usize, dst: &mut [f32]) {
        // B[k, j] = W[j, k]: cada columna del panel es un tramo de una fila.
        let mut tmp = [0f32; KC];
        for jj in 0..NR {
            if jj < columnas {
                let base = (j0 + jj) * self.fila + k0;
                f16_a_f32(&self.datos[base..base + kc], &mut tmp[..kc]);
                for kk in 0..kc {
                    dst[kk * NR + jj] = tmp[kk];
                }
            } else {
                for kk in 0..kc {
                    dst[kk * NR + jj] = 0.0;
                }
            }
        }
    }
}

/// f16 → f32, de a 8 con F16C si la CPU lo tiene.
pub fn f16_a_f32(src: &[u16], dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::cpu_avx2() && std::arch::is_x86_feature_detected!("f16c") {
        // SAFETY: la CPU tiene F16C (comprobado arriba).
        unsafe { f16_a_f32_f16c(src, dst) };
        return;
    }
    for (d, s) in dst.iter_mut().zip(src) {
        *d = f16_escalar(*s);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn f16_a_f32_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = src.len().min(dst.len());
    let mut i = 0;
    while i + 8 <= n {
        let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..n {
        dst[j] = f16_escalar(src[j]);
    }
}

/// Un f16 a f32, a mano: signo, exponente y mantisa, con subnormales.
pub fn f16_escalar(h: u16) -> f32 {
    let signo = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => signo,
        0 => {
            // Subnormal: se normaliza.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            signo | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        31 => signo | 0x7f80_0000 | (man << 13),
        _ => signo | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// `C[m × n] += A[m × k] · B[k × n]`, con `C[i, j] = c[i * ldc + j]`.
pub fn sgemm(
    (m, n, k): (usize, usize, usize),
    a: &dyn FuenteA,
    b: &dyn FuenteB,
    c: &mut [f32],
    ldc: usize,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    assert!(
        ldc >= n && c.len() >= (m - 1) * ldc + n,
        "sgemm: C no alcanza"
    );
    let bloques_m = m.div_ceil(MC);
    let bloques_n = n.div_ceil(NC);
    let c_ptr = c.as_mut_ptr() as usize;
    // Cada bloque (MC × NC) de C lo escribe una sola tarea: se pueden repartir.
    (0..bloques_m * bloques_n).into_par_iter().for_each(|t| {
        let (bm, bn) = (t / bloques_n, t % bloques_n);
        let (i0, j0) = (bm * MC, bn * NC);
        let (mc, nc) = ((m - i0).min(MC), (n - j0).min(NC));
        let mut pa = vec![0f32; MC.div_ceil(MR) * MR * KC];
        let mut pb = vec![0f32; NC.div_ceil(NR) * NR * KC];
        let mut k0 = 0;
        while k0 < k {
            let kc = (k - k0).min(KC);
            for (p, ii) in (0..mc).step_by(MR).enumerate() {
                let filas = (mc - ii).min(MR);
                a.empacar_a(
                    i0 + ii,
                    filas,
                    k0,
                    kc,
                    &mut pa[p * MR * KC..p * MR * KC + MR * kc],
                );
            }
            for (p, jj) in (0..nc).step_by(NR).enumerate() {
                let cols = (nc - jj).min(NR);
                b.empacar_b(
                    k0,
                    kc,
                    j0 + jj,
                    cols,
                    &mut pb[p * NR * KC..p * NR * KC + NR * kc],
                );
            }
            for (pi, ii) in (0..mc).step_by(MR).enumerate() {
                let filas = (mc - ii).min(MR);
                for (pj, jj) in (0..nc).step_by(NR).enumerate() {
                    let cols = (nc - jj).min(NR);
                    let ap = &pa[pi * MR * KC..pi * MR * KC + MR * kc];
                    let bp = &pb[pj * NR * KC..pj * NR * KC + NR * kc];
                    // SAFETY: este bloque de C es solo de esta tarea, y
                    // (i0 + ii + filas, j0 + jj + cols) cae dentro de C.
                    unsafe {
                        let dst = (c_ptr as *mut f32).add((i0 + ii) * ldc + j0 + jj);
                        micro(kc, ap, bp, dst, ldc, filas, cols);
                    }
                }
            }
            k0 += kc;
        }
    });
}

/// Un bloque de MR × NR: el de AVX2 si la CPU lo tiene, el escalar si no.
///
/// # Safety
/// `dst` apunta a `filas` filas de paso `ldc` con `cols` valores cada una.
unsafe fn micro(
    kc: usize,
    a: &[f32],
    b: &[f32],
    dst: *mut f32,
    ldc: usize,
    filas: usize,
    cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if super::cpu_avx2() {
        return micro_avx2(kc, a, b, dst, ldc, filas, cols);
    }
    micro_escalar(kc, a, b, dst, ldc, filas, cols)
}

/// La referencia: la misma cuenta, valor por valor.
///
/// # Safety
/// Como [`micro`].
pub unsafe fn micro_escalar(
    kc: usize,
    a: &[f32],
    b: &[f32],
    dst: *mut f32,
    ldc: usize,
    filas: usize,
    cols: usize,
) {
    let mut acc = [[0f32; NR]; MR];
    for kk in 0..kc {
        let fa = &a[kk * MR..kk * MR + MR];
        let fb = &b[kk * NR..kk * NR + NR];
        for (i, fila) in acc.iter_mut().enumerate() {
            for (j, x) in fila.iter_mut().enumerate() {
                *x = fa[i].mul_add(fb[j], *x);
            }
        }
    }
    for (i, fila) in acc.iter().enumerate().take(filas) {
        for (j, x) in fila.iter().enumerate().take(cols) {
            *dst.add(i * ldc + j) += x;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(
    kc: usize,
    a: &[f32],
    b: &[f32],
    dst: *mut f32,
    ldc: usize,
    filas: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    let mut c = [_mm256_setzero_ps(); 12];
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    for kk in 0..kc {
        let b0 = _mm256_loadu_ps(pb.add(kk * NR));
        let b1 = _mm256_loadu_ps(pb.add(kk * NR + 8));
        let fa = pa.add(kk * MR);
        let a0 = _mm256_broadcast_ss(&*fa);
        c[0] = _mm256_fmadd_ps(a0, b0, c[0]);
        c[1] = _mm256_fmadd_ps(a0, b1, c[1]);
        let a1 = _mm256_broadcast_ss(&*fa.add(1));
        c[2] = _mm256_fmadd_ps(a1, b0, c[2]);
        c[3] = _mm256_fmadd_ps(a1, b1, c[3]);
        let a2 = _mm256_broadcast_ss(&*fa.add(2));
        c[4] = _mm256_fmadd_ps(a2, b0, c[4]);
        c[5] = _mm256_fmadd_ps(a2, b1, c[5]);
        let a3 = _mm256_broadcast_ss(&*fa.add(3));
        c[6] = _mm256_fmadd_ps(a3, b0, c[6]);
        c[7] = _mm256_fmadd_ps(a3, b1, c[7]);
        let a4 = _mm256_broadcast_ss(&*fa.add(4));
        c[8] = _mm256_fmadd_ps(a4, b0, c[8]);
        c[9] = _mm256_fmadd_ps(a4, b1, c[9]);
        let a5 = _mm256_broadcast_ss(&*fa.add(5));
        c[10] = _mm256_fmadd_ps(a5, b0, c[10]);
        c[11] = _mm256_fmadd_ps(a5, b1, c[11]);
    }
    if filas == MR && cols == NR {
        for i in 0..MR {
            let p = dst.add(i * ldc);
            _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), c[2 * i]));
            _mm256_storeu_ps(
                p.add(8),
                _mm256_add_ps(_mm256_loadu_ps(p.add(8)), c[2 * i + 1]),
            );
        }
    } else {
        // Un borde: a un búfer y de ahí lo que corresponde.
        let mut t = [0f32; MR * NR];
        for i in 0..MR {
            _mm256_storeu_ps(t.as_mut_ptr().add(i * NR), c[2 * i]);
            _mm256_storeu_ps(t.as_mut_ptr().add(i * NR + 8), c[2 * i + 1]);
        }
        for i in 0..filas {
            for j in 0..cols {
                *dst.add(i * ldc + j) += t[i * NR + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn azar(n: usize, semilla: u64) -> Vec<f32> {
        let mut a = crate::daiss::Azar::con_semilla(semilla);
        (0..n).map(|_| a.normales(1.0).0).collect()
    }

    fn ingenua(
        m: usize,
        n: usize,
        k: usize,
        a: impl Fn(usize, usize) -> f32,
        b: impl Fn(usize, usize) -> f32,
    ) -> Vec<f64> {
        let mut c = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k)
                    .map(|p| f64::from(a(i, p)) * f64::from(b(p, j)))
                    .sum();
            }
        }
        c
    }

    fn cerca(c: &[f32], r: &[f64], k: usize) {
        for (x, y) in c.iter().zip(r) {
            let tol = 1e-4 * (k as f64).sqrt() * (1.0 + y.abs());
            assert!((f64::from(*x) - y).abs() <= tol, "{x} contra {y}");
        }
    }

    /// Contra la definición, en tamaños con bordes por todos lados: menos de
    /// un bloque, justo un bloque, y más de uno en las tres dimensiones.
    #[test]
    fn da_lo_mismo_que_la_definicion() {
        for &(m, n, k) in &[
            (1, 1, 1),
            (5, 7, 3),
            (6, 16, 1),
            (7, 17, 300),
            (97, 257, 513),
            (200, 33, 70),
        ] {
            let a = azar(m * k, 1);
            let b = azar(k * n, 2);
            let mut c = vec![0f32; m * n];
            sgemm(
                (m, n, k),
                &Mat {
                    datos: &a,
                    fila: k,
                    col: 1,
                },
                &Mat {
                    datos: &b,
                    fila: n,
                    col: 1,
                },
                &mut c,
                n,
            );
            cerca(
                &c,
                &ingenua(m, n, k, |i, p| a[i * k + p], |p, j| b[p * n + j]),
                k,
            );
        }
    }

    /// `C +=`: lo que había en C se suma, y un `ldc` mayor que `n` deja la
    /// columna de más sin tocar.
    #[test]
    fn acumula_y_respeta_ldc() {
        let (m, n, k, ldc) = (13, 20, 9, 23);
        let a = azar(m * k, 3);
        let b = azar(k * n, 4);
        let mut c = vec![1.5f32; m * ldc];
        sgemm(
            (m, n, k),
            &Mat {
                datos: &a,
                fila: k,
                col: 1,
            },
            &Mat {
                datos: &b,
                fila: n,
                col: 1,
            },
            &mut c,
            ldc,
        );
        let r = ingenua(m, n, k, |i, p| a[i * k + p], |p, j| b[p * n + j]);
        for i in 0..m {
            for j in 0..ldc {
                let x = c[i * ldc + j];
                if j < n {
                    assert!((f64::from(x) - (r[i * n + j] + 1.5)).abs() < 1e-3);
                } else {
                    assert_eq!(x, 1.5);
                }
            }
        }
    }

    /// Pesos en f16, como A y traspuestos como B: lo mismo que convertirlos
    /// antes y multiplicar.
    #[test]
    fn los_pesos_en_f16_dan_lo_mismo_que_convertidos() {
        let (m, n, k) = (19, 40, 130);
        let w: Vec<u16> = azar(m * k, 5).iter().map(|x| f32_a_f16(*x)).collect();
        let wf: Vec<f32> = w.iter().map(|h| f16_escalar(*h)).collect();
        let b = azar(k * n, 6);
        let mut c = vec![0f32; m * n];
        sgemm(
            (m, n, k),
            &MatF16 { datos: &w, fila: k },
            &Mat {
                datos: &b,
                fila: n,
                col: 1,
            },
            &mut c,
            n,
        );
        cerca(
            &c,
            &ingenua(m, n, k, |i, p| wf[i * k + p], |p, j| b[p * n + j]),
            k,
        );

        // y = x · Wᵀ, con W de (out = n) × (in = k).
        let (t, ent, sal) = (7, 130, 19);
        let x = azar(t * ent, 7);
        let mut y = vec![0f32; t * sal];
        sgemm(
            (t, sal, ent),
            &Mat {
                datos: &x,
                fila: ent,
                col: 1,
            },
            &MatF16T {
                datos: &w,
                fila: ent,
            },
            &mut y,
            sal,
        );
        cerca(
            &y,
            &ingenua(t, sal, ent, |i, p| x[i * ent + p], |p, j| wf[j * ent + p]),
            ent,
        );
    }

    #[test]
    fn f16_conocidos() {
        for (h, f) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x7bff, 65504.0),
            (0x0001, 5.960_464_5e-8),
            (0x0000, 0.0),
        ] {
            assert_eq!(f16_escalar(h), f, "{h:#x}");
        }
        let src: Vec<u16> = (0..1000u16).map(|i| i.wrapping_mul(61)).collect();
        let mut v = vec![0f32; src.len()];
        f16_a_f32(&src, &mut v);
        for (h, x) in src.iter().zip(&v) {
            let e = f16_escalar(*h);
            assert!(e == *x || (e.is_nan() && x.is_nan()), "{h:#x}");
        }
    }

    fn f32_a_f16(x: f32) -> u16 {
        // Solo para los tests: redondeo simple, valores normales.
        let b = x.to_bits();
        let signo = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        if exp <= 0 {
            return signo;
        }
        let man = ((b >> 13) & 0x3ff) as u16;
        signo | ((exp as u16) << 10) | man
    }
}
