//! El UNet de SD 2.x (`UNet2DConditionModel` de diffusers), sin candle.
//!
//! Bloques de 320, 640, 1280 y 1280 canales; cada uno con dos ResNet, y en los
//! tres primeros un Transformer2D tras cada ResNet (cabezas de 64: 5, 10 y 20);
//! en el medio ResNet, Transformer y ResNet; y la subida al revés, con tres
//! ResNet por bloque que concatenan las salidas guardadas de la bajada. El paso
//! de tiempo entra en cada ResNet por su `time_emb_proj`, y el texto en la
//! atención cruzada de cada Transformer.
//!
//! La configuración es la de `unet/config.json` de SD-Turbo (y de SD 2.1):
//! `attention_head_dim = [5, 10, 20, 20]` es, en esa versión de diffusers, la
//! cantidad de cabezas, no su dimensión.

use std::path::Path;
use std::sync::Arc;

use super::nn::{a_canales, a_filas, ampliar, atencion, gelu, silu, sumar, Conv, Lineal, Norma};
use super::safetensors::Safetensors;

type Resultado<T> = Result<T, String>;

const CANALES: [usize; 4] = [320, 640, 1280, 1280];
const CABEZAS: [usize; 4] = [5, 10, 20, 20];
const TIEMPO: usize = 1280;
const TEXTO: usize = 1024;
const GRUPOS: usize = 32;

struct Resnet {
    norma1: Norma,
    conv1: Conv,
    tiempo: Lineal,
    norma2: Norma,
    conv2: Conv,
    atajo: Option<Conv>,
}

impl Resnet {
    fn cargar(st: &Arc<Safetensors>, p: &str, ent: usize, sal: usize) -> Resultado<Self> {
        Ok(Self {
            norma1: Norma::cargar(st, &format!("{p}.norm1"), ent, 1e-5)?,
            conv1: Conv::cargar(st, &format!("{p}.conv1"), (ent, sal, 3, 1), true)?,
            tiempo: Lineal::cargar(st, &format!("{p}.time_emb_proj"), TIEMPO, sal, true)?,
            norma2: Norma::cargar(st, &format!("{p}.norm2"), sal, 1e-5)?,
            conv2: Conv::cargar(st, &format!("{p}.conv2"), (sal, sal, 3, 1), true)?,
            atajo: if ent != sal {
                Some(Conv::cargar(
                    st,
                    &format!("{p}.conv_shortcut"),
                    (ent, sal, 1, 1),
                    true,
                )?)
            } else {
                None
            },
        })
    }

    /// `temb` ya pasado por SiLU.
    fn aplicar(&self, x: &[f32], h: usize, w: usize, temb: &[f32]) -> Vec<f32> {
        let hw = h * w;
        let mut a = x.to_vec();
        self.norma1.grupos(&mut a, hw, GRUPOS);
        silu(&mut a);
        let (mut a, _, _) = self.conv1.aplicar(&a, h, w);
        let t = self.tiempo.aplicar(temb, 1);
        for (canal, tc) in a.chunks_mut(hw).zip(&t) {
            for v in canal {
                *v += tc;
            }
        }
        self.norma2.grupos(&mut a, hw, GRUPOS);
        silu(&mut a);
        let (mut a, _, _) = self.conv2.aplicar(&a, h, w);
        match &self.atajo {
            Some(c) => sumar(&mut a, &c.aplicar(x, h, w).0),
            None => sumar(&mut a, x),
        }
        a
    }
}

struct Transformer {
    norma: Norma,
    entrada: Lineal,
    n1: Norma,
    q1: Lineal,
    k1: Lineal,
    v1: Lineal,
    o1: Lineal,
    n2: Norma,
    q2: Lineal,
    k2: Lineal,
    v2: Lineal,
    o2: Lineal,
    n3: Norma,
    geglu: Lineal,
    ff: Lineal,
    salida: Lineal,
    c: usize,
    cabezas: usize,
}

impl Transformer {
    fn cargar(st: &Arc<Safetensors>, p: &str, c: usize, cabezas: usize) -> Resultado<Self> {
        let b = format!("{p}.transformer_blocks.0");
        let l = |n: &str, ent, sal, sesgo| Lineal::cargar(st, &format!("{b}.{n}"), ent, sal, sesgo);
        Ok(Self {
            norma: Norma::cargar(st, &format!("{p}.norm"), c, 1e-6)?,
            entrada: Lineal::cargar(st, &format!("{p}.proj_in"), c, c, true)?,
            n1: Norma::cargar(st, &format!("{b}.norm1"), c, 1e-5)?,
            q1: l("attn1.to_q", c, c, false)?,
            k1: l("attn1.to_k", c, c, false)?,
            v1: l("attn1.to_v", c, c, false)?,
            o1: l("attn1.to_out.0", c, c, true)?,
            n2: Norma::cargar(st, &format!("{b}.norm2"), c, 1e-5)?,
            q2: l("attn2.to_q", c, c, false)?,
            k2: l("attn2.to_k", TEXTO, c, false)?,
            v2: l("attn2.to_v", TEXTO, c, false)?,
            o2: l("attn2.to_out.0", c, c, true)?,
            n3: Norma::cargar(st, &format!("{b}.norm3"), c, 1e-5)?,
            geglu: l("ff.net.0.proj", c, 8 * c, true)?,
            ff: l("ff.net.2", 4 * c, c, true)?,
            salida: Lineal::cargar(st, &format!("{p}.proj_out"), c, c, true)?,
            c,
            cabezas,
        })
    }

    fn aplicar(&self, x: &[f32], hw: usize, texto: &[f32]) -> Vec<f32> {
        let (c, tt) = (self.c, texto.len() / TEXTO);
        let mut a = x.to_vec();
        self.norma.grupos(&mut a, hw, GRUPOS);
        let mut f = self.entrada.aplicar(&a_filas(&a, c, hw), hw);

        // Atención propia.
        let mut n = f.clone();
        self.n1.capa(&mut n);
        let at = atencion(
            &self.q1.aplicar(&n, hw),
            &self.k1.aplicar(&n, hw),
            &self.v1.aplicar(&n, hw),
            (hw, hw, c),
            self.cabezas,
            false,
        );
        sumar(&mut f, &self.o1.aplicar(&at, hw));

        // Atención cruzada, contra el texto.
        let mut n = f.clone();
        self.n2.capa(&mut n);
        let at = atencion(
            &self.q2.aplicar(&n, hw),
            &self.k2.aplicar(texto, tt),
            &self.v2.aplicar(texto, tt),
            (hw, tt, c),
            self.cabezas,
            false,
        );
        sumar(&mut f, &self.o2.aplicar(&at, hw));

        // GEGLU: la mitad de la proyección por GELU de la otra mitad.
        let mut n = f.clone();
        self.n3.capa(&mut n);
        let g = self.geglu.aplicar(&n, hw);
        let mut m = vec![0f32; hw * 4 * c];
        for (fila_m, fila_g) in m.chunks_mut(4 * c).zip(g.chunks(8 * c)) {
            let (valor, puerta) = fila_g.split_at(4 * c);
            for ((d, v), p) in fila_m.iter_mut().zip(valor).zip(puerta) {
                *d = v * gelu(*p);
            }
        }
        sumar(&mut f, &self.ff.aplicar(&m, hw));

        let mut y = a_canales(&self.salida.aplicar(&f, hw), c, hw);
        sumar(&mut y, x);
        y
    }
}

pub struct Unet {
    conv_in: Conv,
    t1: Lineal,
    t2: Lineal,
    bajada: Vec<(Vec<Resnet>, Vec<Transformer>, Option<Conv>)>,
    medio: (Resnet, Transformer, Resnet),
    subida: Vec<(Vec<Resnet>, Vec<Transformer>, Option<Conv>)>,
    norma_salida: Norma,
    conv_salida: Conv,
}

impl Unet {
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let st = Safetensors::abrir(ruta)?;
        let mut bajada = Vec::new();
        let mut ent = CANALES[0];
        for (i, (&c, &cab)) in CANALES.iter().zip(&CABEZAS).enumerate() {
            let p = format!("down_blocks.{i}");
            let mut res = Vec::new();
            let mut tr = Vec::new();
            for j in 0..2 {
                res.push(Resnet::cargar(
                    &st,
                    &format!("{p}.resnets.{j}"),
                    if j == 0 { ent } else { c },
                    c,
                )?);
                if i < 3 {
                    tr.push(Transformer::cargar(
                        &st,
                        &format!("{p}.attentions.{j}"),
                        c,
                        cab,
                    )?);
                }
            }
            let baja = if i < 3 {
                Some(Conv::cargar(
                    &st,
                    &format!("{p}.downsamplers.0.conv"),
                    (c, c, 3, 2),
                    true,
                )?)
            } else {
                None
            };
            bajada.push((res, tr, baja));
            ent = c;
        }
        let medio = (
            Resnet::cargar(&st, "mid_block.resnets.0", 1280, 1280)?,
            Transformer::cargar(&st, "mid_block.attentions.0", 1280, CABEZAS[3])?,
            Resnet::cargar(&st, "mid_block.resnets.1", 1280, 1280)?,
        );
        let rev = [CANALES[3], CANALES[2], CANALES[1], CANALES[0]];
        let cab_rev = [CABEZAS[3], CABEZAS[2], CABEZAS[1], CABEZAS[0]];
        let mut subida = Vec::new();
        let mut anterior = rev[0];
        for i in 0..4 {
            let (c, entrada) = (rev[i], rev[(i + 1).min(3)]);
            let p = format!("up_blocks.{i}");
            let mut res = Vec::new();
            let mut tr = Vec::new();
            for j in 0..3 {
                let salteada = if j == 2 { entrada } else { c };
                let base = if j == 0 { anterior } else { c };
                res.push(Resnet::cargar(
                    &st,
                    &format!("{p}.resnets.{j}"),
                    base + salteada,
                    c,
                )?);
                if i > 0 {
                    tr.push(Transformer::cargar(
                        &st,
                        &format!("{p}.attentions.{j}"),
                        c,
                        cab_rev[i],
                    )?);
                }
            }
            let sube = if i < 3 {
                Some(Conv::cargar(
                    &st,
                    &format!("{p}.upsamplers.0.conv"),
                    (c, c, 3, 1),
                    true,
                )?)
            } else {
                None
            };
            subida.push((res, tr, sube));
            anterior = c;
        }
        Ok(Self {
            conv_in: Conv::cargar(&st, "conv_in", (4, 320, 3, 1), true)?,
            t1: Lineal::cargar(&st, "time_embedding.linear_1", 320, TIEMPO, true)?,
            t2: Lineal::cargar(&st, "time_embedding.linear_2", TIEMPO, TIEMPO, true)?,
            bajada,
            medio,
            subida,
            norma_salida: Norma::cargar(&st, "conv_norm_out", 320, 1e-5)?,
            conv_salida: Conv::cargar(&st, "conv_out", (320, 4, 3, 1), true)?,
        })
    }

    /// El embedding sinusoidal del paso `t` (320 valores, coseno primero, como
    /// `flip_sin_to_cos`), por las dos capas, y ya con SiLU.
    fn tiempo(&self, t: f32) -> Vec<f32> {
        let mitad = 160;
        let mut e = vec![0f32; 320];
        for i in 0..mitad {
            let f = (-(10000f64.ln()) * i as f64 / mitad as f64).exp();
            let a = f64::from(t) * f;
            e[i] = a.cos() as f32;
            e[mitad + i] = a.sin() as f32;
        }
        let mut h = self.t1.aplicar(&e, 1);
        silu(&mut h);
        let mut h = self.t2.aplicar(&h, 1);
        silu(&mut h);
        h
    }

    /// El ruido que predice para los latentes `x` (4 × h × w) en el paso `t`,
    /// con el texto `texto` (77 × 1024).
    pub fn aplicar(&self, x: &[f32], (h, w): (usize, usize), t: f32, texto: &[f32]) -> Vec<f32> {
        let temb = self.tiempo(t);
        let (mut a, mut hh, mut ww) = self.conv_in.aplicar(x, h, w);
        let mut guardadas: Vec<(Vec<f32>, usize, usize)> = vec![(a.clone(), hh, ww)];
        for (res, tr, baja) in &self.bajada {
            for (j, r) in res.iter().enumerate() {
                a = r.aplicar(&a, hh, ww, &temb);
                if let Some(t) = tr.get(j) {
                    a = t.aplicar(&a, hh * ww, texto);
                }
                guardadas.push((a.clone(), hh, ww));
            }
            if let Some(b) = baja {
                let (y, nh, nw) = b.aplicar(&a, hh, ww);
                (a, hh, ww) = (y, nh, nw);
                guardadas.push((a.clone(), hh, ww));
            }
        }
        a = self.medio.0.aplicar(&a, hh, ww, &temb);
        a = self.medio.1.aplicar(&a, hh * ww, texto);
        a = self.medio.2.aplicar(&a, hh, ww, &temb);
        for (res, tr, sube) in &self.subida {
            for (j, r) in res.iter().enumerate() {
                let (s, _, _) = guardadas.pop().expect("una salida guardada por ResNet");
                a.extend_from_slice(&s);
                a = r.aplicar(&a, hh, ww, &temb);
                if let Some(t) = tr.get(j) {
                    a = t.aplicar(&a, hh * ww, texto);
                }
            }
            if let Some(s) = sube {
                let c = a.len() / (hh * ww);
                let grande = ampliar(&a, c, hh, ww);
                (hh, ww) = (2 * hh, 2 * ww);
                a = s.aplicar(&grande, hh, ww).0;
            }
        }
        self.norma_salida.grupos(&mut a, hh * ww, GRUPOS);
        silu(&mut a);
        self.conv_salida.aplicar(&a, hh, ww).0
    }
}
