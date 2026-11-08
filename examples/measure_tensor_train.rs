//! ¿Cuánto se puede achicar el vocabulario de un daiss entrenado, y con qué?
//!
//! Las tablas de vocabulario (`token_embd` y `output`) son el 47% de un daiss-mini
//! con BPE 8.192. TensorGPT dice que en Tensor-Train ocupan 39 a 65 veces menos.
//! Acá se mide, sobre un modelo propio entrenado, y contra las dos alternativas que
//! hay que ganarle para que valga:
//!
//! - **Tensor-Train** (TT-SVD exacta, como TensorGPT): la matriz `V × D` se mira
//!   como un tensor de 4 índices, `(8·4)(8·4)(8·4)(16·4)`, y se corta en una
//!   cadena de 4 núcleos con rango a lo sumo `r`.
//! - **SVD de rango bajo**, `U·S·Vᵀ` con rango `k`: lo clásico.
//! - **Cuantizar** en Q8_0, Q4_K y Q2_K: lo que el motor ya hace.
//!
//! La medida es la que importa, **bits por byte sobre todo el texto de prueba**,
//! y no la energía ni el error de la matriz: una matriz puede quedar con poco error
//! y el modelo, peor. Se comprime una tabla por vez, y la otra queda como estaba.
//!
//!     cargo run --release --features training --example measure_tensor_train -- modelo.gguf corpus.txt
//!
//! El corpus se corta igual que en `train_daiss`: lo de prueba es lo que no vio.

use std::path::Path;
use std::time::Instant;

use candle_core::{Device, Tensor};
use rayon::prelude::*;

use cosxamix::daiss::{tokenizador_de, Config};
use cosxamix::gguf::{Gguf, TipoGgml};
use cosxamix::training::full::ModeloCompleto;

/// Igual que en `train_daiss`.
fn cuerpo(t: &str) -> &str {
    let desde = t
        .find("*** START OF")
        .and_then(|i| t[i..].find('\n').map(|j| i + j + 1))
        .unwrap_or(0);
    let hasta = t.find("*** END OF").unwrap_or(t.len());
    &t[desde..hasta]
}

/// Bits por byte sobre todo el texto, en ventanas de `t` sin solapar.
fn bits_en_todo(
    m: &ModeloCompleto,
    ids: &[u32],
    t: usize,
    bytes_por_token: f64,
    dev: &Device,
) -> candle_core::Result<f64> {
    let ventanas: Vec<&[u32]> = ids.chunks_exact(t).collect();
    let sumas = ventanas
        .par_chunks((512 / t).max(1))
        .map(|tanda| {
            let planos: Vec<u32> = tanda.iter().flat_map(|v| v.iter().copied()).collect();
            let l = Tensor::from_vec(planos, (tanda.len(), t), dev)?;
            Ok(f64::from(m.perdida(&l)?.to_scalar::<f32>()?) * tanda.len() as f64)
        })
        .collect::<candle_core::Result<Vec<f64>>>()?;
    let media = sumas.iter().sum::<f64>() / ventanas.len() as f64;
    Ok(media / std::f64::consts::LN_2 / bytes_por_token)
}

/// Autovalores y autovectores de una matriz simétrica `n × n`, por Jacobi cíclico:
/// lento, `O(n³)` por barrida, pero exacto y sin dependencias. Devuelve los
/// autovalores de mayor a menor y los autovectores en columnas, en ese orden.
fn jacobi(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let total: f64 = a.iter().map(|x| x * x).sum();
    for _ in 0..60 {
        let mut fuera = 0f64;
        for p in 0..n {
            for q in p + 1..n {
                fuera += a[p * n + q] * a[p * n + q];
            }
        }
        if fuera <= 1e-26 * total {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut orden: Vec<usize> = (0..n).collect();
    orden.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let valores = orden.iter().map(|&i| a[i * n + i]).collect();
    let mut vectores = vec![0f64; n * n];
    for (col, &i) in orden.iter().enumerate() {
        for k in 0..n {
            vectores[k * n + col] = v[k * n + i];
        }
    }
    (valores, vectores)
}

/// `G = A·Aᵀ` de una matriz `m × n`, en paralelo por filas.
fn gram_filas(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let filas: Vec<Vec<f64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let fi = &a[i * n..(i + 1) * n];
            (0..m)
                .map(|j| {
                    fi.iter()
                        .zip(&a[j * n..(j + 1) * n])
                        .map(|(x, y)| x * y)
                        .sum()
                })
                .collect()
        })
        .collect();
    filas.concat()
}

fn traspuesta(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut t = vec![0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// La SVD truncada a rango `r` de una matriz `m × n`: `(U, S·Vᵀ, k)`, con `U` de
/// `m × k` y `S·Vᵀ` de `k × n`. Por los autovectores del Gram del lado chico.
fn svd(a: &[f64], m: usize, n: usize, r: usize) -> (Vec<f64>, Vec<f64>, usize) {
    if m > n {
        // Por la traspuesta: si Aᵀ = X·Y, entonces A = Yᵀ·Xᵀ. La escala queda a la
        // izquierda en vez de a la derecha; para TT-SVD da igual, el producto y el
        // rango son los mismos.
        let (x, y, k) = svd(&traspuesta(a, m, n), n, m, r);
        return (traspuesta(&y, k, m), traspuesta(&x, n, k), k);
    }
    let (_, vectores) = jacobi(gram_filas(a, m, n), m);
    let k = r.min(m).min(n);
    let mut u = vec![0f64; m * k];
    for i in 0..m {
        for c in 0..k {
            u[i * k + c] = vectores[i * m + c];
        }
    }
    // S·Vᵀ = Uᵀ·A.
    let mut svt = vec![0f64; k * n];
    for c in 0..k {
        let fila = &mut svt[c * n..(c + 1) * n];
        for i in 0..m {
            let ui = u[i * k + c];
            if ui != 0.0 {
                for (f, x) in fila.iter_mut().zip(&a[i * n..(i + 1) * n]) {
                    *f += ui * x;
                }
            }
        }
    }
    (u, svt, k)
}

/// `X (p × q) · Y (q × s)`.
fn producto(x: &[f64], y: &[f64], p: usize, q: usize, s: usize) -> Vec<f64> {
    let filas: Vec<Vec<f64>> = (0..p)
        .into_par_iter()
        .map(|i| {
            let mut f = vec![0f64; s];
            for j in 0..q {
                let xij = x[i * q + j];
                for (o, yv) in f.iter_mut().zip(&y[j * s..(j + 1) * s]) {
                    *o += xij * yv;
                }
            }
            f
        })
        .collect();
    filas.concat()
}

const FILAS: [usize; 4] = [8, 8, 8, 16];
const COLUMNAS: [usize; 4] = [4, 4, 4, 4];

/// La posición en el tensor `(i₁j₁)(i₂j₂)(i₃j₃)(i₄j₄)` de cada valor de la matriz.
fn posiciones(v: usize, d: usize) -> Vec<usize> {
    let mut p = Vec::with_capacity(v * d);
    for fila in 0..v {
        let mut i = [0usize; 4];
        let mut resto = fila;
        for k in (0..4).rev() {
            i[k] = resto % FILAS[k];
            resto /= FILAS[k];
        }
        for col in 0..d {
            let mut j = [0usize; 4];
            let mut resto = col;
            for k in (0..4).rev() {
                j[k] = resto % COLUMNAS[k];
                resto /= COLUMNAS[k];
            }
            let mut idx = 0;
            for k in 0..4 {
                idx = idx * FILAS[k] * COLUMNAS[k] + i[k] * COLUMNAS[k] + j[k];
            }
            p.push(idx);
        }
    }
    p
}

/// TT-SVD de la matriz con rango máximo `r`: devuelve la matriz reconstruida y
/// cuántos parámetros tienen los núcleos.
fn tensor_train(w: &[f32], pos: &[usize], r: usize) -> (Vec<f32>, usize) {
    let s: Vec<usize> = (0..4).map(|k| FILAS[k] * COLUMNAS[k]).collect();
    let mut c = vec![0f64; w.len()];
    for (x, &p) in w.iter().zip(pos) {
        c[p] = f64::from(*x);
    }
    let mut nucleos: Vec<(usize, usize, usize, Vec<f64>)> = Vec::new();
    let mut r_prev = 1;
    for &sk in &s[..3] {
        let filas = r_prev * sk;
        let cols = c.len() / filas;
        let (u, svt, k) = svd(&c, filas, cols, r);
        nucleos.push((r_prev, sk, k, u));
        c = svt;
        r_prev = k;
    }
    nucleos.push((r_prev, s[3], 1, c));
    let parametros = nucleos.iter().map(|(a, b, c, _)| a * b * c).sum();

    // Reconstruir: contraer la cadena de izquierda a derecha.
    let (_, s0, r1, x0) = &nucleos[0];
    let (mut x, mut p, mut r_actual) = (x0.clone(), *s0, *r1);
    for (rk, sk, rn, g) in &nucleos[1..] {
        debug_assert_eq!(*rk, r_actual);
        x = producto(&x, g, p, *rk, sk * rn);
        p *= sk;
        r_actual = *rn;
    }
    let reconstruida = pos.iter().map(|&i| x[i] as f32).collect();
    (reconstruida, parametros)
}

/// La mejor aproximación de rango `k`, `U·S·Vᵀ`.
fn rango_bajo(w: &[f32], v: usize, d: usize, k: usize) -> Vec<f32> {
    let a: Vec<f64> = w.iter().map(|x| f64::from(*x)).collect();
    let (us, vt, k) = svd(&a, v, d, k);
    producto(&us, &vt, v, k, d)
        .into_iter()
        .map(|x| x as f32)
        .collect()
}

fn cuantizada(w: &[f32], d: usize, tipo: TipoGgml) -> Result<(Vec<f32>, usize), String> {
    let mut b = Vec::new();
    for fila in w.chunks(d) {
        cosxamix::quant::cuantizar_bloques(tipo, fila, &mut b)?;
    }
    let mut salida = vec![0f32; w.len()];
    cosxamix::quant::descuantizar(tipo, &b, &mut salida)?;
    Ok((salida, b.len()))
}

fn error_relativo(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += f64::from(x - y).powi(2);
        den += f64::from(*x).powi(2);
    }
    (num / den).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ruta_modelo = args.next().expect("falta el modelo");
    let ruta_corpus = args.next().expect("falta el corpus");
    let ruta = Path::new(&ruta_modelo);

    let g = Gguf::abrir(ruta)?;
    let tok = tokenizador_de(&g)?.ok_or("el modelo no trae su tokenizador")?;
    let base = Config::de_nombre("daiss-mini").expect("existe");
    let cfg = Config {
        vocab: tok.tamano(),
        ..*base
    };
    let (v, d) = (cfg.vocab, cfg.dim);
    if v != FILAS.iter().product::<usize>() || d != COLUMNAS.iter().product::<usize>() {
        return Err(format!("la forma {v} x {d} no es la de daiss-mini con BPE 8.192").into());
    }

    let crudo = String::from_utf8_lossy(&std::fs::read(&ruta_corpus)?).replace("\r\n", "\n");
    let texto = cuerpo(&crudo);
    let corte = texto[..texto.len() / 10 * 9].rfind('\n').unwrap_or(0) + 1;
    let prueba = &texto[corte..];
    let ids = tok.codificar(prueba);
    let bytes_por_token = prueba.len() as f64 / ids.len() as f64;

    let t = 128;
    let dev = Device::Cpu;
    let m = ModeloCompleto::nuevo(&cfg, 0, t, &dev)?;
    m.cargar_pesos(ruta)?;
    let total = m.parametros();
    let t0 = Instant::now();
    let original = bits_en_todo(&m, &ids, t, bytes_por_token, &dev)?;
    println!(
        "  {ruta_modelo}: {:.2} M parámetros, {} KB de prueba",
        total as f64 / 1e6,
        prueba.len() >> 10
    );
    println!(
        "  sin tocar: {original:.4} bits por byte ({:.0} s por medida)",
        t0.elapsed().as_secs_f64()
    );

    let pos = posiciones(v, d);
    for nombre in ["token_embd.weight", "output.weight"] {
        let var = m.peso(nombre).ok_or("falta el peso")?;
        let w: Vec<f32> = var.as_tensor().flatten_all()?.to_vec1()?;
        let bytes_f32 = w.len() * 4;
        println!();
        println!(
            "  {nombre}, {v} x {d}: {:.2} M valores, {} KB en F32",
            w.len() as f64 / 1e6,
            bytes_f32 >> 10
        );
        println!(
            "  {:<12}{:>11}{:>10}{:>9}{:>10}{:>11}{:>8}",
            "", "parámetros", "KB", "veces", "error", "bits/byte", "Δ"
        );

        let medir = |etiqueta: String, nueva: Vec<f32>, bytes: usize, params: usize| {
            let error = error_relativo(&w, &nueva);
            var.set(&Tensor::from_vec(nueva, (v, d), &dev)?)?;
            let bits = bits_en_todo(&m, &ids, t, bytes_por_token, &dev)?;
            println!(
                "  {etiqueta:<12}{params:>11}{:>10}{:>8.1}x{:>9.1}%{bits:>11.4}{:>+8.4}",
                bytes >> 10,
                bytes_f32 as f64 / bytes as f64,
                error * 100.0,
                bits - original
            );
            Ok::<_, candle_core::Error>(())
        };

        for tipo in [TipoGgml::Q8_0, TipoGgml::Q4K, TipoGgml::Q2K] {
            let (nueva, bytes) = cuantizada(&w, d, tipo)?;
            medir(format!("{tipo:?}"), nueva, bytes, w.len())?;
        }
        for k in [128, 64, 32, 16] {
            let nueva = rango_bajo(&w, v, d, k);
            let params = k * (v + d);
            medir(format!("SVD k={k}"), nueva, params * 4, params)?;
        }
        for r in [256, 128, 64, 32, 16, 8] {
            let (nueva, params) = tensor_train(&w, &pos, r);
            medir(format!("TT r={r}"), nueva, params * 4, params)?;
        }
        var.set(&Tensor::from_vec(w, (v, d), &dev)?)?;
    }
    println!();
    println!("  Los bytes de SVD y TT son en F32: cuantizados, bajarían otro tanto.");
    Ok(())
}
