//! TAESD (github.com/madebyollin/taesd, MIT): el decodificador chico que lee
//! los mismos latentes que el VAE de SD 1.x y 2.x, 4,9 MB.
//!
//! Convoluciones de 64 canales: una de entrada, tres bloques por escala y tres
//! ampliaciones ×2, y una de salida a RGB. Los tensores se llaman por su índice
//! en el `nn.Sequential` del original (`taesd.py`).

use std::path::Path;

use super::nn::{ampliar, Conv};
use super::safetensors::Safetensors;

type Resultado<T> = Result<T, String>;

enum Capa {
    Conv(Conv),
    Relu,
    Bloque(Conv, Conv, Conv),
    Ampliar,
}

pub struct Taesd {
    capas: Vec<Capa>,
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

impl Taesd {
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let st = Safetensors::abrir(ruta)?;
        let conv =
            |i: usize, ent, sal, sesgo| Conv::cargar(&st, &i.to_string(), (ent, sal, 3, 1), sesgo);
        let bloque = |i: usize| -> Resultado<Capa> {
            let c = |n: usize| Conv::cargar(&st, &format!("{i}.conv.{n}"), (64, 64, 3, 1), true);
            Ok(Capa::Bloque(c(0)?, c(2)?, c(4)?))
        };
        let mut capas = vec![Capa::Conv(conv(1, 4, 64, true)?), Capa::Relu];
        let mut i = 3;
        for _ in 0..3 {
            for _ in 0..3 {
                capas.push(bloque(i)?);
                i += 1;
            }
            capas.push(Capa::Ampliar);
            capas.push(Capa::Conv(conv(i + 1, 64, 64, false)?));
            i += 2;
        }
        capas.push(bloque(18)?);
        capas.push(Capa::Conv(conv(19, 64, 3, true)?));
        Ok(Self { capas })
    }

    /// Los latentes del UNet (4 × h × w) → RGB en [0, 1], CHW de 8h × 8w.
    pub fn decodificar(&self, lat: &[f32], (h, w): (usize, usize)) -> (Vec<f32>, usize, usize) {
        let mut x: Vec<f32> = lat.iter().map(|v| (v / 3.0).tanh() * 3.0).collect();
        let (mut h, mut w) = (h, w);
        for capa in &self.capas {
            match capa {
                Capa::Conv(c) => x = c.aplicar(&x, h, w).0,
                Capa::Relu => relu(&mut x),
                Capa::Ampliar => {
                    let c = x.len() / (h * w);
                    x = ampliar(&x, c, h, w);
                    (h, w) = (2 * h, 2 * w);
                }
                Capa::Bloque(a, b, c) => {
                    let mut y = a.aplicar(&x, h, w).0;
                    relu(&mut y);
                    let mut y = b.aplicar(&y, h, w).0;
                    relu(&mut y);
                    let mut y = c.aplicar(&y, h, w).0;
                    for (u, v) in y.iter_mut().zip(&x) {
                        *u = (*u + v).max(0.0);
                    }
                    x = y;
                }
            }
        }
        for v in &mut x {
            *v = v.clamp(0.0, 1.0);
        }
        (x, h, w)
    }
}
