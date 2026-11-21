//! Un Qwen2 entrenable, con el backward escrito a mano y sin candle.
//!
//! Es la arquitectura de daiss y de Qwen2.5: embeddings, y por capa RMSNorm,
//! atención con RoPE y cabezas agrupadas (q, k y v con sesgo), RMSNorm y
//! SwiGLU; al final RMSNorm y la proyección al vocabulario. Hasta la 0.10.0 lo
//! entrenaba el autograd de candle; desde la 0.11.0, esto: un forward que guarda
//! lo que necesita su vuelta y un backward por operación, comprobados contra
//! diferencias finitas en los tests.
//!
//! Los parámetros van en un registro plano ([`Params`]): cada uno con su nombre
//! —el del GGUF—, su forma, sus valores y si entrena. Así lo mismo sirve para
//! entrenar todo (daiss desde cero) o solo adaptadores LoRA con la base
//! congelada (Qwen). Las multiplicaciones van por [`crate::kernels::sgemm`], y
//! un lote se reparte entre los núcleos por secuencia.

use rayon::prelude::*;

use crate::kernels::sgemm::{sgemm, Mat};

/// La forma del modelo.
#[derive(Debug, Clone, Copy)]
pub struct Forma {
    pub dim: usize,
    pub ffn: usize,
    pub capas: usize,
    pub cabezas: usize,
    pub cabezas_kv: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope: f32,
}

impl Forma {
    pub fn dim_cabeza(&self) -> usize {
        self.dim / self.cabezas
    }
    pub fn dim_kv(&self) -> usize {
        self.cabezas_kv * self.dim_cabeza()
    }
}

/// Todos los parámetros, en orden.
#[derive(Default)]
pub struct Params {
    pub valores: Vec<Vec<f32>>,
    pub nombres: Vec<String>,
    pub formas: Vec<Vec<usize>>,
    pub entrena: Vec<bool>,
}

impl Params {
    pub fn agregar(
        &mut self,
        nombre: impl Into<String>,
        forma: Vec<usize>,
        valores: Vec<f32>,
        entrena: bool,
    ) -> usize {
        assert_eq!(valores.len(), forma.iter().product::<usize>());
        self.valores.push(valores);
        self.nombres.push(nombre.into());
        self.formas.push(forma);
        self.entrena.push(entrena);
        self.valores.len() - 1
    }

    pub fn indice(&self, nombre: &str) -> Option<usize> {
        self.nombres.iter().position(|n| n == nombre)
    }

    /// Gradientes en cero para lo que entrena; vacíos para lo congelado.
    pub fn gradientes(&self) -> Vec<Vec<f32>> {
        self.valores
            .iter()
            .zip(&self.entrena)
            .map(|(v, e)| if *e { vec![0f32; v.len()] } else { Vec::new() })
            .collect()
    }

    /// Cuántos valores entrenan.
    pub fn entrenables(&self) -> usize {
        self.valores
            .iter()
            .zip(&self.entrena)
            .filter(|(_, e)| **e)
            .map(|(v, _)| v.len())
            .sum()
    }
}

/// Un adaptador LoRA: `y += escala · (x Aᵀ) Bᵀ`, con A de `r × ent` y B de `sal × r`.
#[derive(Clone, Copy)]
pub struct Lora {
    pub a: usize,
    pub b: usize,
    pub r: usize,
    pub escala: f32,
}

/// `y = x Wᵀ + b (+ LoRA)`, con W de `sal × ent`.
#[derive(Clone, Copy)]
pub struct Lineal {
    pub w: usize,
    pub b: Option<usize>,
    pub lora: Option<Lora>,
    pub ent: usize,
    pub sal: usize,
}

pub struct Capa {
    pub norma_atencion: usize,
    pub q: Lineal,
    pub k: Lineal,
    pub v: Lineal,
    pub o: Lineal,
    pub norma_ffn: usize,
    pub gate: Lineal,
    pub up: Lineal,
    pub down: Lineal,
}

/// El modelo: la forma, los parámetros y cómo se arman.
pub struct Transformer {
    pub forma: Forma,
    pub params: Params,
    pub capas: Vec<Capa>,
    pub norma_final: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pub largo_maximo: usize,
}

/// Las tablas de vocabulario de una pasada: la de entrada y la de salida
/// (pueden ser la misma, o venir de una cadena de Tensor-Train).
pub struct Tablas<'a> {
    pub entrada: &'a [f32],
    pub salida: &'a [f32],
    /// Si se calculan sus gradientes. Con las tablas congeladas (LoRA) no: en
    /// Qwen 0,5B serían 545 MB por tabla y por secuencia, para tirarlos.
    pub gradientes: bool,
}

/// Los gradientes de una pasada: por parámetro, y los de las dos tablas.
pub struct Gradientes {
    pub params: Vec<Vec<f32>>,
    pub entrada: Vec<f32>,
    pub salida: Vec<f32>,
}

impl Gradientes {
    fn sumar(mut self, otro: Self) -> Self {
        for (a, b) in self.params.iter_mut().zip(&otro.params) {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        for (a, b) in [
            (&mut self.entrada, &otro.entrada),
            (&mut self.salida, &otro.salida),
        ] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
        self
    }
}

fn mm(m: usize, n: usize, k: usize, a: Mat, b: Mat, c: &mut [f32], ldc: usize) {
    sgemm((m, n, k), &a, &b, c, ldc);
}

fn mat(d: &[f32], fila: usize, col: usize) -> Mat<'_> {
    Mat {
        datos: d,
        fila,
        col,
    }
}

impl Lineal {
    /// La salida y, si hay LoRA, `x Aᵀ` (lo que pide su vuelta).
    fn ida(&self, p: &Params, x: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let w = &p.valores[self.w];
        let mut y = match self.b {
            Some(b) => p.valores[b]
                .iter()
                .copied()
                .cycle()
                .take(t * self.sal)
                .collect(),
            None => vec![0f32; t * self.sal],
        };
        mm(
            t,
            self.sal,
            self.ent,
            mat(x, self.ent, 1),
            mat(w, 1, self.ent),
            &mut y,
            self.sal,
        );
        let mut h = Vec::new();
        if let Some(l) = self.lora {
            h = vec![0f32; t * l.r];
            mm(
                t,
                l.r,
                self.ent,
                mat(x, self.ent, 1),
                mat(&p.valores[l.a], 1, self.ent),
                &mut h,
                l.r,
            );
            let hs: Vec<f32> = h.iter().map(|v| v * l.escala).collect();
            mm(
                t,
                self.sal,
                l.r,
                mat(&hs, l.r, 1),
                mat(&p.valores[l.b], 1, l.r),
                &mut y,
                self.sal,
            );
        }
        (y, h)
    }

    /// Suma a `dx` lo que vuelve por `x`, y acumula los gradientes de lo que entrena.
    /// `(x, h)` son la entrada y el `x Aᵀ` que guardó la ida.
    fn vuelta(
        &self,
        p: &Params,
        g: &mut [Vec<f32>],
        (x, h): (&[f32], &[f32]),
        dy: &[f32],
        dx: &mut [f32],
        t: usize,
    ) {
        if p.entrena[self.w] {
            mm(
                self.sal,
                self.ent,
                t,
                mat(dy, 1, self.sal),
                mat(x, self.ent, 1),
                &mut g[self.w],
                self.ent,
            );
        }
        if let Some(b) = self.b {
            if p.entrena[b] {
                for fila in dy.chunks(self.sal) {
                    for (gb, v) in g[b].iter_mut().zip(fila) {
                        *gb += v;
                    }
                }
            }
        }
        mm(
            t,
            self.ent,
            self.sal,
            mat(dy, self.sal, 1),
            mat(&p.valores[self.w], self.ent, 1),
            dx,
            self.ent,
        );
        if let Some(l) = self.lora {
            let hs: Vec<f32> = h.iter().map(|v| v * l.escala).collect();
            if p.entrena[l.b] {
                mm(
                    self.sal,
                    l.r,
                    t,
                    mat(dy, 1, self.sal),
                    mat(&hs, l.r, 1),
                    &mut g[l.b],
                    l.r,
                );
            }
            let mut dh = vec![0f32; t * l.r];
            mm(
                t,
                l.r,
                self.sal,
                mat(dy, self.sal, 1),
                mat(&p.valores[l.b], l.r, 1),
                &mut dh,
                l.r,
            );
            for v in &mut dh {
                *v *= l.escala;
            }
            if p.entrena[l.a] {
                mm(
                    l.r,
                    self.ent,
                    t,
                    mat(&dh, 1, l.r),
                    mat(x, self.ent, 1),
                    &mut g[l.a],
                    self.ent,
                );
            }
            mm(
                t,
                self.ent,
                l.r,
                mat(&dh, l.r, 1),
                mat(&p.valores[l.a], self.ent, 1),
                dx,
                self.ent,
            );
        }
    }
}

/// RMSNorm: la salida y el `1/rms` de cada fila.
fn rms_ida(x: &[f32], g: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let d = g.len();
    let mut y = vec![0f32; x.len()];
    let mut r = vec![0f32; x.len() / d];
    for ((fy, fx), ri) in y.chunks_mut(d).zip(x.chunks(d)).zip(&mut r) {
        let m = fx.iter().map(|v| v * v).sum::<f32>() / d as f32;
        *ri = 1.0 / (m + eps).sqrt();
        for ((a, b), w) in fy.iter_mut().zip(fx).zip(g) {
            *a = b * *ri * w;
        }
    }
    (y, r)
}

fn rms_vuelta(x: &[f32], g: &[f32], r: &[f32], dy: &[f32], dx: &mut [f32], dg: Option<&mut [f32]>) {
    let d = g.len();
    let mut dg_local = vec![0f32; d];
    for (((fx, fdy), fdx), ri) in x.chunks(d).zip(dy.chunks(d)).zip(dx.chunks_mut(d)).zip(r) {
        let mut ux = 0f32;
        for ((a, b), w) in fx.iter().zip(fdy).zip(g) {
            ux += b * w * a;
        }
        let k = ri * ri * ri * ux / d as f32;
        for (i, ((a, b), o)) in fx.iter().zip(fdy).zip(fdx.iter_mut()).enumerate() {
            *o += ri * b * g[i] - a * k;
            dg_local[i] += b * a * ri;
        }
    }
    if let Some(dg) = dg {
        for (a, b) in dg.iter_mut().zip(&dg_local) {
            *a += b;
        }
    }
}

/// Lo que la pasada de una capa guarda para su vuelta.
struct Guardado {
    x: Vec<f32>,
    h1: Vec<f32>,
    r1: Vec<f32>,
    hq: Vec<f32>,
    hk: Vec<f32>,
    hv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    p: Vec<f32>,
    y: Vec<f32>,
    ho: Vec<f32>,
    x2: Vec<f32>,
    h2: Vec<f32>,
    r2: Vec<f32>,
    gp: Vec<f32>,
    hg: Vec<f32>,
    u: Vec<f32>,
    hu: Vec<f32>,
    m: Vec<f32>,
    hd: Vec<f32>,
}

impl Transformer {
    pub fn nuevo(
        forma: Forma,
        params: Params,
        capas: Vec<Capa>,
        norma_final: usize,
        largo_maximo: usize,
    ) -> Self {
        let (cos, sin) = crate::ops::frecuencias_rope(forma.dim_cabeza(), forma.rope, largo_maximo);
        Self {
            forma,
            params,
            capas,
            norma_final,
            cos,
            sin,
            largo_maximo,
        }
    }

    /// Atención causal con cabezas agrupadas: la salida y las probabilidades.
    fn atencion_ida(&self, q: &[f32], k: &[f32], v: &[f32], t: usize) -> (Vec<f32>, Vec<f32>) {
        let f = &self.forma;
        let (dc, hd, kvd) = (f.dim_cabeza(), f.dim, f.dim_kv());
        let rep = f.cabezas / f.cabezas_kv;
        let escala = 1.0 / (dc as f32).sqrt();
        let mut y = vec![0f32; t * hd];
        let mut p = vec![0f32; f.cabezas * t * t];
        for h in 0..f.cabezas {
            let gi = h / rep;
            let s = &mut p[h * t * t..(h + 1) * t * t];
            mm(
                t,
                t,
                dc,
                mat(&q[h * dc..], hd, 1),
                mat(&k[gi * dc..], 1, kvd),
                s,
                t,
            );
            for (i, fila) in s.chunks_mut(t).enumerate() {
                let mut max = f32::MIN;
                for x in &mut fila[..=i] {
                    *x *= escala;
                    max = max.max(*x);
                }
                let mut suma = 0f32;
                for x in &mut fila[..=i] {
                    *x = (*x - max).exp();
                    suma += *x;
                }
                for x in &mut fila[..=i] {
                    *x /= suma;
                }
                fila[i + 1..].fill(0.0);
            }
            mm(
                t,
                dc,
                t,
                mat(s, t, 1),
                mat(&v[gi * dc..], kvd, 1),
                &mut y[h * dc..],
                hd,
            );
        }
        (y, p)
    }

    fn atencion_vuelta(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        p: &[f32],
        dy: &[f32],
        t: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let f = &self.forma;
        let (dc, hd, kvd) = (f.dim_cabeza(), f.dim, f.dim_kv());
        let rep = f.cabezas / f.cabezas_kv;
        let escala = 1.0 / (dc as f32).sqrt();
        let (mut dq, mut dk, mut dv) =
            (vec![0f32; t * hd], vec![0f32; t * kvd], vec![0f32; t * kvd]);
        let mut ds = vec![0f32; t * t];
        for h in 0..f.cabezas {
            let gi = h / rep;
            let ph = &p[h * t * t..(h + 1) * t * t];
            // dV += Pᵀ dO
            mm(
                t,
                dc,
                t,
                mat(ph, 1, t),
                mat(&dy[h * dc..], hd, 1),
                &mut dv[gi * dc..],
                kvd,
            );
            // dP = dO Vᵀ, y dS = P ⊙ (dP − Σ dP·P), ya por la escala.
            ds.fill(0.0);
            mm(
                t,
                t,
                dc,
                mat(&dy[h * dc..], hd, 1),
                mat(&v[gi * dc..], 1, kvd),
                &mut ds,
                t,
            );
            for (fila_s, fila_p) in ds.chunks_mut(t).zip(ph.chunks(t)) {
                let dot: f32 = fila_s.iter().zip(fila_p).map(|(a, b)| a * b).sum();
                for (a, b) in fila_s.iter_mut().zip(fila_p) {
                    *a = b * (*a - dot) * escala;
                }
            }
            mm(
                t,
                dc,
                t,
                mat(&ds, t, 1),
                mat(&k[gi * dc..], kvd, 1),
                &mut dq[h * dc..],
                hd,
            );
            mm(
                t,
                dc,
                t,
                mat(&ds, 1, t),
                mat(&q[h * dc..], hd, 1),
                &mut dk[gi * dc..],
                kvd,
            );
        }
        (dq, dk, dv)
    }

    fn rope(&self, x: &mut [f32], cabezas: usize, inversa: bool) {
        let t = x.len() / (cabezas * self.forma.dim_cabeza());
        if inversa {
            let menos: Vec<f32> = self.sin.iter().map(|s| -s).collect();
            crate::ops::rope(
                x,
                (t, cabezas, self.forma.dim_cabeza()),
                &self.cos,
                &menos,
                0,
            )
            .expect("rope");
        } else {
            crate::ops::rope(
                x,
                (t, cabezas, self.forma.dim_cabeza()),
                &self.cos,
                &self.sin,
                0,
            )
            .expect("rope");
        }
    }

    /// Los logits de `ids` (`t × vocab`), sin gradientes.
    pub fn logits(&self, ids: &[u32], tablas: &Tablas) -> Vec<f32> {
        let x = self.cuerpo(ids, tablas, &mut Vec::new());
        let (xf, _) = rms_ida(&x, &self.params.valores[self.norma_final], self.forma.eps);
        let t = ids.len();
        let mut l = vec![0f32; t * self.forma.vocab];
        mm(
            t,
            self.forma.vocab,
            self.forma.dim,
            mat(&xf, self.forma.dim, 1),
            mat(tablas.salida, 1, self.forma.dim),
            &mut l,
            self.forma.vocab,
        );
        l
    }

    /// Las capas, guardando lo que piden sus vueltas si `guardar` viene vacío
    /// pero con capacidad (se llena siempre; `logits` lo descarta).
    fn cuerpo(&self, ids: &[u32], tablas: &Tablas, guardar: &mut Vec<Guardado>) -> Vec<f32> {
        let f = self.forma;
        let t = ids.len();
        assert!(
            t <= self.largo_maximo,
            "{t} tokens y el máximo es {}",
            self.largo_maximo
        );
        let p = &self.params;
        let mut x = vec![0f32; t * f.dim];
        for (fila, &id) in x.chunks_mut(f.dim).zip(ids) {
            fila.copy_from_slice(&tablas.entrada[id as usize * f.dim..(id as usize + 1) * f.dim]);
        }
        guardar.clear();
        for c in &self.capas {
            let (h1, r1) = rms_ida(&x, &p.valores[c.norma_atencion], f.eps);
            let (mut q, hq) = c.q.ida(p, &h1, t);
            let (mut k, hk) = c.k.ida(p, &h1, t);
            let (v, hv) = c.v.ida(p, &h1, t);
            self.rope(&mut q, f.cabezas, false);
            self.rope(&mut k, f.cabezas_kv, false);
            let (y, pr) = self.atencion_ida(&q, &k, &v, t);
            let (a, ho) = c.o.ida(p, &y, t);
            let x2: Vec<f32> = x.iter().zip(&a).map(|(u, w)| u + w).collect();
            let (h2, r2) = rms_ida(&x2, &p.valores[c.norma_ffn], f.eps);
            let (gp, hg) = c.gate.ida(p, &h2, t);
            let (u, hu) = c.up.ida(p, &h2, t);
            let m: Vec<f32> = gp
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let (d, hd) = c.down.ida(p, &m, t);
            let x3: Vec<f32> = x2.iter().zip(&d).map(|(u, w)| u + w).collect();
            guardar.push(Guardado {
                x,
                h1,
                r1,
                hq,
                hk,
                hv,
                q,
                k,
                v,
                p: pr,
                y,
                ho,
                x2,
                h2,
                r2,
                gp,
                hg,
                u,
                hu,
                m,
                hd,
            });
            x = x3;
        }
        x
    }

    /// La suma de las pérdidas de predecir `ids[1..]` desde `ids[..t-1]`, y sus
    /// gradientes multiplicados por `escala` (1 / cuántos tokens tiene el lote).
    fn secuencia(&self, ids: &[u32], tablas: &Tablas, escala: f32) -> (f32, Gradientes) {
        let f = self.forma;
        let p = &self.params;
        let entrada = &ids[..ids.len() - 1];
        let t = entrada.len();
        let mut g = Gradientes {
            params: p.gradientes(),
            entrada: if tablas.gradientes {
                vec![0f32; tablas.entrada.len()]
            } else {
                Vec::new()
            },
            salida: if tablas.gradientes {
                vec![0f32; tablas.salida.len()]
            } else {
                Vec::new()
            },
        };
        let mut guardado = Vec::new();
        let x = self.cuerpo(entrada, tablas, &mut guardado);
        let (xf, rf) = rms_ida(&x, &p.valores[self.norma_final], f.eps);
        let mut dl = vec![0f32; t * f.vocab];
        mm(
            t,
            f.vocab,
            f.dim,
            mat(&xf, f.dim, 1),
            mat(tablas.salida, 1, f.dim),
            &mut dl,
            f.vocab,
        );
        let mut perdida = 0f32;
        for (fila, &obj) in dl.chunks_mut(f.vocab).zip(&ids[1..]) {
            let max = fila.iter().cloned().fold(f32::MIN, f32::max);
            let suma: f32 = fila.iter().map(|v| (v - max).exp()).sum();
            let lse = max + suma.ln();
            perdida += lse - fila[obj as usize];
            for v in fila.iter_mut() {
                *v = (*v - lse).exp() * escala;
            }
            fila[obj as usize] -= escala;
        }
        // La salida: dTabla += dLᵀ xf, dxf = dL · Tabla.
        if tablas.gradientes {
            mm(
                f.vocab,
                f.dim,
                t,
                mat(&dl, 1, f.vocab),
                mat(&xf, f.dim, 1),
                &mut g.salida,
                f.dim,
            );
        }
        let mut dxf = vec![0f32; t * f.dim];
        mm(
            t,
            f.dim,
            f.vocab,
            mat(&dl, f.vocab, 1),
            mat(tablas.salida, f.dim, 1),
            &mut dxf,
            f.dim,
        );
        let mut dx = vec![0f32; t * f.dim];
        let dg = if p.entrena[self.norma_final] {
            Some(&mut g.params[self.norma_final][..])
        } else {
            None
        };
        rms_vuelta(&x, &p.valores[self.norma_final], &rf, &dxf, &mut dx, dg);

        for (c, s) in self.capas.iter().zip(&guardado).rev() {
            // x3 = x2 + down(m)
            let mut dm = vec![0f32; t * f.ffn];
            c.down
                .vuelta(p, &mut g.params, (&s.m, &s.hd), &dx, &mut dm, t);
            let mut dgp = vec![0f32; t * f.ffn];
            let mut du = vec![0f32; t * f.ffn];
            for i in 0..dm.len() {
                let z = s.gp[i];
                let sg = 1.0 / (1.0 + (-z).exp());
                du[i] = dm[i] * z * sg;
                dgp[i] = dm[i] * s.u[i] * sg * (1.0 + z * (1.0 - sg));
            }
            let mut dh2 = vec![0f32; t * f.dim];
            c.gate
                .vuelta(p, &mut g.params, (&s.h2, &s.hg), &dgp, &mut dh2, t);
            c.up.vuelta(p, &mut g.params, (&s.h2, &s.hu), &du, &mut dh2, t);
            let mut dx2 = dx.clone();
            let dgn = if p.entrena[c.norma_ffn] {
                Some(&mut g.params[c.norma_ffn][..])
            } else {
                None
            };
            rms_vuelta(&s.x2, &p.valores[c.norma_ffn], &s.r2, &dh2, &mut dx2, dgn);
            // x2 = x + o(atención)
            let mut dyat = vec![0f32; t * f.dim];
            c.o.vuelta(p, &mut g.params, (&s.y, &s.ho), &dx2, &mut dyat, t);
            let (mut dq, mut dk, dv) = self.atencion_vuelta(&s.q, &s.k, &s.v, &s.p, &dyat, t);
            self.rope(&mut dq, f.cabezas, true);
            self.rope(&mut dk, f.cabezas_kv, true);
            let mut dh1 = vec![0f32; t * f.dim];
            c.q.vuelta(p, &mut g.params, (&s.h1, &s.hq), &dq, &mut dh1, t);
            c.k.vuelta(p, &mut g.params, (&s.h1, &s.hk), &dk, &mut dh1, t);
            c.v.vuelta(p, &mut g.params, (&s.h1, &s.hv), &dv, &mut dh1, t);
            let mut dxa = dx2;
            let dgn = if p.entrena[c.norma_atencion] {
                Some(&mut g.params[c.norma_atencion][..])
            } else {
                None
            };
            rms_vuelta(
                &s.x,
                &p.valores[c.norma_atencion],
                &s.r1,
                &dh1,
                &mut dxa,
                dgn,
            );
            dx = dxa;
        }
        for (fila, &id) in dx.chunks(f.dim).zip(entrada).filter(|_| tablas.gradientes) {
            let d = &mut g.entrada[id as usize * f.dim..(id as usize + 1) * f.dim];
            for (a, b) in d.iter_mut().zip(fila) {
                *a += b;
            }
        }
        (perdida, g)
    }

    /// La pérdida media de un lote (cada secuencia predice su siguiente
    /// token) y sus gradientes, repartiendo las secuencias entre los núcleos.
    pub fn perdida_y_gradientes(&self, lote: &[Vec<u32>], tablas: &Tablas) -> (f32, Gradientes) {
        let n: usize = lote.iter().map(|s| s.len() - 1).sum();
        let escala = 1.0 / n as f32;
        let (suma, g) = lote
            .par_iter()
            .map(|s| self.secuencia(s, tablas, escala))
            .reduce_with(|(a, ga), (b, gb)| (a + b, ga.sumar(gb)))
            .expect("un lote no vacío");
        (suma / n as f32, g)
    }

    /// La pérdida media de un lote, sin gradientes.
    pub fn perdida(&self, lote: &[Vec<u32>], tablas: &Tablas) -> f32 {
        let n: usize = lote.iter().map(|s| s.len() - 1).sum();
        let suma: f32 = lote
            .par_iter()
            .map(|s| {
                let l = self.logits(&s[..s.len() - 1], tablas);
                l.chunks(self.forma.vocab)
                    .zip(&s[1..])
                    .map(|(fila, &obj)| {
                        let max = fila.iter().cloned().fold(f32::MIN, f32::max);
                        max + fila.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
                            - fila[obj as usize]
                    })
                    .sum::<f32>()
            })
            .sum();
        suma / n as f32
    }
}

/// AdamW, con los mismos valores por defecto que usaba el de candle:
/// β₁ = 0,9, β₂ = 0,999, ε = 1e-8 y decaimiento de 0,01.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub decaimiento: f32,
    paso: i32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn nuevo(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            decaimiento: 0.01,
            paso: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Un paso sobre los parámetros que entrenan.
    pub fn paso(&mut self, valores: &mut [Vec<f32>], grads: &[Vec<f32>]) {
        if self.m.is_empty() {
            self.m = grads.iter().map(|g| vec![0f32; g.len()]).collect();
            self.v = self.m.clone();
        }
        self.paso += 1;
        let (c1, c2) = (
            1.0 - self.beta1.powi(self.paso),
            1.0 - self.beta2.powi(self.paso),
        );
        let (lr, b1, b2, eps, wd) = (self.lr, self.beta1, self.beta2, self.eps, self.decaimiento);
        valores
            .par_iter_mut()
            .zip(grads.par_iter())
            .zip(self.m.par_iter_mut().zip(self.v.par_iter_mut()))
            .for_each(|((w, g), (m, v))| {
                for i in 0..g.len() {
                    m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                    w[i] *= 1.0 - lr * wd;
                    w[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + eps);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn azar(a: &mut crate::daiss::Azar, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| a.normales(s).0).collect()
    }

    /// Un modelo chico con todo: sesgos, cabezas agrupadas y LoRA en q y v.
    fn chico(lora: bool, entrena_base: bool) -> (Transformer, Vec<f32>, Vec<f32>) {
        let f = Forma {
            dim: 8,
            ffn: 12,
            capas: 2,
            cabezas: 4,
            cabezas_kv: 2,
            vocab: 11,
            eps: 1e-5,
            rope: 10000.0,
        };
        let mut a = crate::daiss::Azar::con_semilla(5);
        let mut p = Params::default();
        let mut capas = Vec::new();
        for i in 0..f.capas {
            let lin = |p: &mut Params,
                       a: &mut crate::daiss::Azar,
                       n: &str,
                       ent: usize,
                       sal: usize,
                       sesgo: bool,
                       con_lora: bool| {
                let w = p.agregar(
                    format!("{i}.{n}.w"),
                    vec![sal, ent],
                    azar(a, sal * ent, 0.3),
                    entrena_base,
                );
                let b = sesgo.then(|| {
                    p.agregar(
                        format!("{i}.{n}.b"),
                        vec![sal],
                        azar(a, sal, 0.1),
                        entrena_base,
                    )
                });
                let lora = (lora && con_lora).then(|| {
                    let r = 3;
                    Lora {
                        a: p.agregar(
                            format!("{i}.{n}.a"),
                            vec![r, ent],
                            azar(a, r * ent, 0.3),
                            true,
                        ),
                        b: p.agregar(
                            format!("{i}.{n}.lb"),
                            vec![sal, r],
                            azar(a, sal * r, 0.3),
                            true,
                        ),
                        r,
                        escala: 0.7,
                    }
                });
                Lineal {
                    w,
                    b,
                    lora,
                    ent,
                    sal,
                }
            };
            let na = p.agregar(
                format!("{i}.na"),
                vec![f.dim],
                azar(&mut a, f.dim, 0.2).iter().map(|v| 1.0 + v).collect(),
                entrena_base,
            );
            let q = lin(&mut p, &mut a, "q", f.dim, f.dim, true, true);
            let k = lin(&mut p, &mut a, "k", f.dim, f.dim_kv(), true, false);
            let v = lin(&mut p, &mut a, "v", f.dim, f.dim_kv(), true, true);
            let o = lin(&mut p, &mut a, "o", f.dim, f.dim, false, false);
            let nf = p.agregar(
                format!("{i}.nf"),
                vec![f.dim],
                azar(&mut a, f.dim, 0.2).iter().map(|v| 1.0 + v).collect(),
                entrena_base,
            );
            let gate = lin(&mut p, &mut a, "g", f.dim, f.ffn, false, false);
            let up = lin(&mut p, &mut a, "u", f.dim, f.ffn, false, false);
            let down = lin(&mut p, &mut a, "d", f.ffn, f.dim, false, false);
            capas.push(Capa {
                norma_atencion: na,
                q,
                k,
                v,
                o,
                norma_ffn: nf,
                gate,
                up,
                down,
            });
        }
        let nfin = p.agregar("nfin", vec![f.dim], vec![1.0; f.dim], entrena_base);
        let emb = azar(&mut a, f.vocab * f.dim, 0.5);
        let sal = azar(&mut a, f.vocab * f.dim, 0.5);
        (Transformer::nuevo(f, p, capas, nfin, 16), emb, sal)
    }

    /// Cada gradiente contra diferencias finitas centradas, en una muestra de
    /// valores de cada parámetro y de las dos tablas.
    fn comprobar(lora: bool, entrena_base: bool) {
        let (mut m, mut emb, mut sal) = chico(lora, entrena_base);
        let lote = vec![vec![1u32, 4, 7, 2, 9, 3], vec![5u32, 5, 0, 10, 2]];
        let (_, g) = m.perdida_y_gradientes(
            &lote,
            &Tablas {
                entrada: &emb,
                salida: &sal,
                gradientes: true,
            },
        );
        let h = 1e-3f32;
        let mut peor = 0f32;
        let mut probar = |analitico: f32, f: &mut dyn FnMut(f32) -> f32| {
            let (mas, menos) = (f(h), f(-h));
            let numerico = (mas - menos) / (2.0 * h);
            let err = (analitico - numerico).abs() / (analitico.abs() + numerico.abs()).max(1e-2);
            peor = peor.max(err);
            assert!(
                err < 2e-2,
                "analítico {analitico} contra numérico {numerico}"
            );
        };
        for pi in 0..m.params.valores.len() {
            if !m.params.entrena[pi] {
                assert!(
                    g.params[pi].is_empty(),
                    "{} no entrena y tiene gradiente",
                    m.params.nombres[pi]
                );
                continue;
            }
            let n = m.params.valores[pi].len();
            for &i in &[0, n / 2, n - 1] {
                let an = g.params[pi][i];
                probar(an, &mut |d| {
                    let orig = m.params.valores[pi][i];
                    m.params.valores[pi][i] = orig + d;
                    let l = m.perdida(
                        &lote,
                        &Tablas {
                            entrada: &emb,
                            salida: &sal,
                            gradientes: true,
                        },
                    );
                    m.params.valores[pi][i] = orig;
                    l
                });
            }
        }
        for &i in &[4 * 8 + 1, 9 * 8 + 3] {
            let an = g.entrada[i];
            probar(an, &mut |d| {
                let o = emb[i];
                emb[i] = o + d;
                let l = m.perdida(
                    &lote,
                    &Tablas {
                        entrada: &emb,
                        salida: &sal,
                        gradientes: true,
                    },
                );
                emb[i] = o;
                l
            });
            let an = g.salida[i];
            probar(an, &mut |d| {
                let o = sal[i];
                sal[i] = o + d;
                let l = m.perdida(
                    &lote,
                    &Tablas {
                        entrada: &emb,
                        salida: &sal,
                        gradientes: true,
                    },
                );
                sal[i] = o;
                l
            });
        }
        assert!(peor < 2e-2);
    }

    #[test]
    fn los_gradientes_dan_lo_que_dicen_las_diferencias_finitas() {
        comprobar(false, true);
    }

    #[test]
    fn con_lora_solo_entrenan_los_adaptadores_y_dan_lo_mismo() {
        comprobar(true, false);
    }

    /// La pérdida de `perdida_y_gradientes` es la misma que la de `perdida`.
    #[test]
    fn la_perdida_coincide_por_los_dos_caminos() {
        let (m, emb, sal) = chico(true, true);
        let lote = vec![vec![1u32, 4, 7, 2], vec![3u32, 3, 8, 1, 0, 6]];
        let t = Tablas {
            entrada: &emb,
            salida: &sal,
            gradientes: true,
        };
        let (a, _) = m.perdida_y_gradientes(&lote, &t);
        let b = m.perdida(&lote, &t);
        assert!((a - b).abs() < 1e-5, "{a} contra {b}");
    }

    /// Con AdamW, la pérdida de un lote chico baja.
    #[test]
    fn adamw_baja_la_perdida() {
        let (mut m, emb, sal) = chico(false, true);
        let lote = vec![vec![1u32, 4, 7, 2, 9, 3, 5, 5]];
        let mut opt = AdamW::nuevo(0.01);
        let inicial = m.perdida(
            &lote,
            &Tablas {
                entrada: &emb,
                salida: &sal,
                gradientes: true,
            },
        );
        for _ in 0..40 {
            let (_, g) = m.perdida_y_gradientes(
                &lote,
                &Tablas {
                    entrada: &emb,
                    salida: &sal,
                    gradientes: true,
                },
            );
            opt.paso(&mut m.params.valores, &g.params);
        }
        let fin = m.perdida(
            &lote,
            &Tablas {
                entrada: &emb,
                salida: &sal,
                gradientes: true,
            },
        );
        assert!(fin < inicial * 0.7, "{inicial} → {fin}");
    }
}
