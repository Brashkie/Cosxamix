//! Las piezas con las que se arman CLIP, el UNet y TAESD, sobre
//! [`crate::kernels::sgemm`]. Todo en f32; los pesos grandes quedan en f16 en el
//! mapeo del archivo y se convierten al empaquetar.
//!
//! Las imágenes van en CHW (canal, fila, columna) y las secuencias en filas de
//! `dim` valores (un token por fila).

use std::sync::Arc;

use rayon::prelude::*;

use super::safetensors::{Safetensors, P16};
use crate::kernels::sgemm::{sgemm, FuenteB, Mat, MatF16, MatF16T, NR};

type Resultado<T> = Result<T, String>;

/// Pesos de una capa: en f16 dentro del archivo, o en f32 (los de TAESD).
pub enum Pesos {
    F16(P16),
    F32(Vec<f32>),
}

impl Pesos {
    pub fn cargar(st: &Arc<Safetensors>, nombre: &str, forma: &[usize]) -> Resultado<Self> {
        match Safetensors::f16(st, nombre, forma) {
            Ok(p) => Ok(Pesos::F16(p)),
            Err(_) => Ok(Pesos::F32(st.f32(nombre, forma)?)),
        }
    }
}

/// `y = x · Wᵀ + b`, con `W` de `sal × ent`.
pub struct Lineal {
    w: Pesos,
    ent: usize,
    pub sal: usize,
    b: Option<Vec<f32>>,
}

impl Lineal {
    pub fn cargar(
        st: &Arc<Safetensors>,
        prefijo: &str,
        ent: usize,
        sal: usize,
        sesgo: bool,
    ) -> Resultado<Self> {
        Ok(Self {
            w: Pesos::cargar(st, &format!("{prefijo}.weight"), &[sal, ent])?,
            ent,
            sal,
            b: if sesgo {
                Some(st.f32(&format!("{prefijo}.bias"), &[sal])?)
            } else {
                None
            },
        })
    }

    /// `filas` tokens de `ent` valores → `filas × sal`.
    pub fn aplicar(&self, x: &[f32], filas: usize) -> Vec<f32> {
        assert_eq!(x.len(), filas * self.ent, "Lineal: entrada de otro tamaño");
        let mut y = match &self.b {
            Some(b) => b.iter().copied().cycle().take(filas * self.sal).collect(),
            None => vec![0f32; filas * self.sal],
        };
        let a = Mat {
            datos: x,
            fila: self.ent,
            col: 1,
        };
        match &self.w {
            Pesos::F16(w) => sgemm(
                (filas, self.sal, self.ent),
                &a,
                &MatF16T {
                    datos: w.datos(),
                    fila: self.ent,
                },
                &mut y,
                self.sal,
            ),
            Pesos::F32(w) => sgemm(
                (filas, self.sal, self.ent),
                &a,
                &Mat {
                    datos: w,
                    fila: 1,
                    col: self.ent,
                },
                &mut y,
                self.sal,
            ),
        }
        y
    }
}

/// La ventana de una convolución 3×3 con relleno 1, como B de un GEMM:
/// `B[ci·9 + ky·3 + kx, oy·ow + ox] = x[ci, oy·paso + ky − 1, ox·paso + kx − 1]`
/// (cero fuera de la imagen). Así el im2col se arma panel por panel, al
/// empaquetar, y nunca entero.
struct Ventana<'a> {
    x: &'a [f32],
    h: usize,
    w: usize,
    paso: usize,
    ow: usize,
}

impl FuenteB for Ventana<'_> {
    fn empacar_b(&self, k0: usize, kc: usize, j0: usize, columnas: usize, dst: &mut [f32]) {
        for kk in 0..kc {
            let k = k0 + kk;
            let (ci, ky, kx) = (k / 9, k % 9 / 3, k % 3);
            let canal = &self.x[ci * self.h * self.w..(ci + 1) * self.h * self.w];
            let d = &mut dst[kk * NR..kk * NR + NR];
            for (jj, v) in d.iter_mut().enumerate() {
                *v = if jj < columnas {
                    let j = j0 + jj;
                    let (oy, ox) = (j / self.ow, j % self.ow);
                    let y = (oy * self.paso + ky) as isize - 1;
                    let x = (ox * self.paso + kx) as isize - 1;
                    if y < 0 || x < 0 || y >= self.h as isize || x >= self.w as isize {
                        0.0
                    } else {
                        canal[y as usize * self.w + x as usize]
                    }
                } else {
                    0.0
                };
            }
        }
    }
}

/// Convolución 3×3 (relleno 1, paso 1 o 2) o 1×1.
pub struct Conv {
    w: Pesos,
    pub ent: usize,
    pub sal: usize,
    lado: usize,
    paso: usize,
    b: Option<Vec<f32>>,
}

impl Conv {
    pub fn cargar(
        st: &Arc<Safetensors>,
        prefijo: &str,
        (ent, sal, lado, paso): (usize, usize, usize, usize),
        sesgo: bool,
    ) -> Resultado<Self> {
        Ok(Self {
            w: Pesos::cargar(st, &format!("{prefijo}.weight"), &[sal, ent, lado, lado])?,
            ent,
            sal,
            lado,
            paso,
            b: if sesgo {
                Some(st.f32(&format!("{prefijo}.bias"), &[sal])?)
            } else {
                None
            },
        })
    }

    /// `x` en CHW de `h × w` → la salida en CHW y su tamaño.
    pub fn aplicar(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        assert_eq!(x.len(), self.ent * h * w, "Conv: entrada de otro tamaño");
        let (oh, ow) = if self.lado == 3 {
            ((h - 1) / self.paso + 1, (w - 1) / self.paso + 1)
        } else {
            (h, w)
        };
        let hw = oh * ow;
        let mut y = match &self.b {
            Some(b) => b.iter().flat_map(|v| std::iter::repeat_n(*v, hw)).collect(),
            None => vec![0f32; self.sal * hw],
        };
        let k = self.ent * self.lado * self.lado;
        let ventana = Ventana {
            x,
            h,
            w,
            paso: self.paso,
            ow,
        };
        let plano = Mat {
            datos: x,
            fila: hw,
            col: 1,
        };
        let b: &dyn FuenteB = if self.lado == 3 { &ventana } else { &plano };
        match &self.w {
            Pesos::F16(p) => sgemm(
                (self.sal, hw, k),
                &MatF16 {
                    datos: p.datos(),
                    fila: k,
                },
                b,
                &mut y,
                hw,
            ),
            Pesos::F32(p) => sgemm(
                (self.sal, hw, k),
                &Mat {
                    datos: p,
                    fila: k,
                    col: 1,
                },
                b,
                &mut y,
                hw,
            ),
        }
        (y, oh, ow)
    }
}

/// Los pesos de una normalización: escala y desplazamiento.
pub struct Norma {
    g: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl Norma {
    pub fn cargar(st: &Arc<Safetensors>, prefijo: &str, dim: usize, eps: f32) -> Resultado<Self> {
        Ok(Self {
            g: st.f32(&format!("{prefijo}.weight"), &[dim])?,
            b: st.f32(&format!("{prefijo}.bias"), &[dim])?,
            eps,
        })
    }

    /// LayerNorm de cada fila de `dim` valores.
    pub fn capa(&self, x: &mut [f32]) {
        let dim = self.g.len();
        x.par_chunks_mut(dim).for_each(|f| {
            let media = f.iter().map(|v| f64::from(*v)).sum::<f64>() / dim as f64;
            let var = f
                .iter()
                .map(|v| (f64::from(*v) - media).powi(2))
                .sum::<f64>()
                / dim as f64;
            let inv = 1.0 / (var + f64::from(self.eps)).sqrt();
            for (i, v) in f.iter_mut().enumerate() {
                *v = ((f64::from(*v) - media) * inv) as f32 * self.g[i] + self.b[i];
            }
        });
    }

    /// GroupNorm de una imagen CHW de `hw` píxeles por canal, en `grupos`.
    pub fn grupos(&self, x: &mut [f32], hw: usize, grupos: usize) {
        let c = self.g.len();
        let por_grupo = c / grupos;
        x.par_chunks_mut(por_grupo * hw)
            .enumerate()
            .for_each(|(gi, g)| {
                let n = g.len() as f64;
                let media = g.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
                let var = g
                    .iter()
                    .map(|v| (f64::from(*v) - media).powi(2))
                    .sum::<f64>()
                    / n;
                let inv = 1.0 / (var + f64::from(self.eps)).sqrt();
                for (ci, canal) in g.chunks_mut(hw).enumerate() {
                    let cc = gi * por_grupo + ci;
                    let (e, d) = (self.g[cc], self.b[cc]);
                    for v in canal {
                        *v = ((f64::from(*v) - media) * inv) as f32 * e + d;
                    }
                }
            });
    }
}

pub fn silu(x: &mut [f32]) {
    x.par_chunks_mut(4096).for_each(|c| {
        for v in c {
            *v /= 1.0 + (-*v).exp();
        }
    });
}

/// La función de error, con el error máximo de 1,2e-7 de la aproximación de
/// Chebyshev de Numerical Recipes (`erfc`).
pub fn erf(x: f32) -> f32 {
    let z = f64::from(x).abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    (if x >= 0.0 { 1.0 - r } else { r - 1.0 }) as f32
}

/// GELU exacta: `x · Φ(x)`, como la de PyTorch por defecto.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Atención por cabezas: `q` de `tq × c`, `k` y `v` de `tk × c`, cabezas de
/// `c / cabezas`. Con `causal`, el token `i` solo ve hasta el `i`.
pub fn atencion(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (tq, tk, c): (usize, usize, usize),
    cabezas: usize,
    causal: bool,
) -> Vec<f32> {
    let d = c / cabezas;
    let escala = 1.0 / (d as f32).sqrt();
    let mut salida = vec![0f32; tq * c];
    // Las cabezas de a una: cada una usa todos los núcleos en sus GEMM, y la
    // matriz de puntajes (tq × tk) no se repite por cabeza en memoria.
    let mut s = vec![0f32; tq * tk];
    for h in 0..cabezas {
        s.fill(0.0);
        // S = Q_h · K_hᵀ: A(i, p) = q[i·c + h·d + p], B(p, j) = k[j·c + h·d + p].
        sgemm(
            (tq, tk, d),
            &Mat {
                datos: &q[h * d..],
                fila: c,
                col: 1,
            },
            &Mat {
                datos: &k[h * d..],
                fila: 1,
                col: c,
            },
            &mut s,
            tk,
        );
        s.par_chunks_mut(tk).enumerate().for_each(|(i, f)| {
            let visibles = if causal { i + 1 } else { tk };
            let mut max = f32::MIN;
            for x in &mut f[..visibles] {
                *x *= escala;
                max = max.max(*x);
            }
            let mut suma = 0f32;
            for x in &mut f[..visibles] {
                *x = (*x - max).exp();
                suma += *x;
            }
            for x in &mut f[..visibles] {
                *x /= suma;
            }
            f[visibles..].fill(0.0);
        });
        // O_h = P · V_h, escrito en sus columnas de la salida.
        sgemm(
            (tq, d, tk),
            &Mat {
                datos: &s,
                fila: tk,
                col: 1,
            },
            &Mat {
                datos: &v[h * d..],
                fila: c,
                col: 1,
            },
            &mut salida[h * d..],
            c,
        );
    }
    salida
}

/// CHW → (HW × C): una fila por píxel.
pub fn a_filas(x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut y = vec![0f32; c * hw];
    y.par_chunks_mut(c).enumerate().for_each(|(p, f)| {
        for (ci, v) in f.iter_mut().enumerate() {
            *v = x[ci * hw + p];
        }
    });
    y
}

/// (HW × C) → CHW.
pub fn a_canales(x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut y = vec![0f32; c * hw];
    y.par_chunks_mut(hw).enumerate().for_each(|(ci, canal)| {
        for (p, v) in canal.iter_mut().enumerate() {
            *v = x[p * c + ci];
        }
    });
    y
}

/// El vecino más cercano ×2, en CHW.
pub fn ampliar(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (h2, w2) = (2 * h, 2 * w);
    let mut y = vec![0f32; c * h2 * w2];
    y.par_chunks_mut(h2 * w2)
        .enumerate()
        .for_each(|(ci, canal)| {
            let src = &x[ci * h * w..(ci + 1) * h * w];
            for (yy, fila) in canal.chunks_mut(w2).enumerate() {
                let s = &src[(yy / 2) * w..(yy / 2 + 1) * w];
                for (xx, v) in fila.iter_mut().enumerate() {
                    *v = s[xx / 2];
                }
            }
        });
    y
}

pub fn sumar(a: &mut [f32], b: &[f32]) {
    a.par_chunks_mut(4096)
        .zip(b.par_chunks(4096))
        .for_each(|(x, y)| {
            for (u, v) in x.iter_mut().zip(y) {
                *u += v;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_y_gelu_conocidos() {
        for (x, e) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-2.0, -0.995_322_3),
            (3.0, 0.999_977_9),
        ] {
            assert!((erf(x) - e).abs() < 2e-7, "erf({x}) = {} y no {e}", erf(x));
        }
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_3).abs() < 1e-6);
    }

    /// La ventana de la convolución contra la definición, con paso 1 y 2 y
    /// una imagen no cuadrada, también con canales = alto = ancho (el caso que
    /// candle leía mal).
    #[test]
    fn la_convolucion_da_lo_que_dice_la_definicion() {
        let mut a = crate::daiss::Azar::con_semilla(3);
        for &(ent, sal, h, w, paso) in &[
            (3usize, 5usize, 7usize, 6usize, 1usize),
            (4, 4, 4, 4, 1),
            (3, 2, 9, 8, 2),
        ] {
            let x: Vec<f32> = (0..ent * h * w).map(|_| a.normales(1.0).0).collect();
            let pesos: Vec<f32> = (0..sal * ent * 9).map(|_| a.normales(0.5).0).collect();
            let b: Vec<f32> = (0..sal).map(|_| a.normales(0.1).0).collect();
            let conv = Conv {
                w: Pesos::F32(pesos.clone()),
                ent,
                sal,
                lado: 3,
                paso,
                b: Some(b.clone()),
            };
            let (y, oh, ow) = conv.aplicar(&x, h, w);
            for o in 0..sal {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut s = f64::from(b[o]);
                        for ci in 0..ent {
                            for ky in 0..3 {
                                for kx in 0..3 {
                                    let (yy, xx) = (
                                        (i * paso + ky) as isize - 1,
                                        (j * paso + kx) as isize - 1,
                                    );
                                    if yy < 0 || xx < 0 || yy >= h as isize || xx >= w as isize {
                                        continue;
                                    }
                                    s += f64::from(x[(ci * h + yy as usize) * w + xx as usize])
                                        * f64::from(pesos[((o * ent + ci) * 3 + ky) * 3 + kx]);
                                }
                            }
                        }
                        let got = f64::from(y[(o * oh + i) * ow + j]);
                        assert!((got - s).abs() < 1e-4, "({o},{i},{j}): {got} contra {s}");
                    }
                }
            }
        }
    }

    /// La atención contra la definición, causal y sin máscara.
    #[test]
    fn la_atencion_da_lo_que_dice_la_definicion() {
        let mut a = crate::daiss::Azar::con_semilla(9);
        let (tq, tk, c, cabezas) = (5usize, 7usize, 8usize, 2usize);
        for causal in [false, true] {
            let tk = if causal { tq } else { tk };
            let q: Vec<f32> = (0..tq * c).map(|_| a.normales(1.0).0).collect();
            let k: Vec<f32> = (0..tk * c).map(|_| a.normales(1.0).0).collect();
            let v: Vec<f32> = (0..tk * c).map(|_| a.normales(1.0).0).collect();
            let y = atencion(&q, &k, &v, (tq, tk, c), cabezas, causal);
            let d = c / cabezas;
            for h in 0..cabezas {
                for i in 0..tq {
                    let vis = if causal { i + 1 } else { tk };
                    let p: Vec<f64> = (0..vis)
                        .map(|j| {
                            (0..d)
                                .map(|t| {
                                    f64::from(q[i * c + h * d + t])
                                        * f64::from(k[j * c + h * d + t])
                                })
                                .sum::<f64>()
                                / (d as f64).sqrt()
                        })
                        .collect();
                    let m = p.iter().cloned().fold(f64::MIN, f64::max);
                    let e: Vec<f64> = p.iter().map(|x| (x - m).exp()).collect();
                    let s: f64 = e.iter().sum();
                    for t in 0..d {
                        let r: f64 = (0..vis)
                            .map(|j| e[j] / s * f64::from(v[j * c + h * d + t]))
                            .sum();
                        assert!((f64::from(y[i * c + h * d + t]) - r).abs() < 1e-5);
                    }
                }
            }
        }
    }
}
