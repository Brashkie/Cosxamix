//! El codificador de texto de SD 2.x: el transformer de texto de OpenCLIP
//! ViT-H (en formato `CLIPTextModel`), ya cortado en la penúltima capa —23 de
//! 24—, que es la que usa Stable Diffusion 2.
//!
//! Por capa: `x += atención(LN₁(x))` con máscara causal y 16 cabezas de 64, y
//! `x += fc₂(GELU(fc₁(LN₂(x))))`. Al final, una LayerNorm. Devuelve los 77
//! vectores de 1024 que el UNet atiende en su atención cruzada.

use std::path::Path;

use rayon::prelude::*;

use super::nn::{atencion, gelu, Lineal, Norma};
use super::safetensors::{Safetensors, P16};

type Resultado<T> = Result<T, String>;

pub const TOKENS: usize = 77;
const DIM: usize = 1024;
const CABEZAS: usize = 16;
const INTERMEDIA: usize = 4096;
const CAPAS: usize = 23;
const VOCAB: usize = 49408;

struct Capa {
    ln1: Norma,
    q: Lineal,
    k: Lineal,
    v: Lineal,
    salida: Lineal,
    ln2: Norma,
    fc1: Lineal,
    fc2: Lineal,
}

pub struct TextoClip {
    tokens: P16,
    posiciones: Vec<f32>,
    capas: Vec<Capa>,
    ln_final: Norma,
}

impl TextoClip {
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let st = Safetensors::abrir(ruta)?;
        let e = "text_model.embeddings";
        let mut capas = Vec::with_capacity(CAPAS);
        for i in 0..CAPAS {
            let p = format!("text_model.encoder.layers.{i}");
            let l = |n: &str, ent, sal| Lineal::cargar(&st, &format!("{p}.{n}"), ent, sal, true);
            capas.push(Capa {
                ln1: Norma::cargar(&st, &format!("{p}.layer_norm1"), DIM, 1e-5)?,
                q: l("self_attn.q_proj", DIM, DIM)?,
                k: l("self_attn.k_proj", DIM, DIM)?,
                v: l("self_attn.v_proj", DIM, DIM)?,
                salida: l("self_attn.out_proj", DIM, DIM)?,
                ln2: Norma::cargar(&st, &format!("{p}.layer_norm2"), DIM, 1e-5)?,
                fc1: l("mlp.fc1", DIM, INTERMEDIA)?,
                fc2: l("mlp.fc2", INTERMEDIA, DIM)?,
            });
        }
        Ok(Self {
            tokens: Safetensors::f16(&st, &format!("{e}.token_embedding.weight"), &[VOCAB, DIM])?,
            posiciones: st.f32(&format!("{e}.position_embedding.weight"), &[TOKENS, DIM])?,
            capas,
            ln_final: Norma::cargar(&st, "text_model.final_layer_norm", DIM, 1e-5)?,
        })
    }

    /// Los 77 ids (ver `Clip::codificar`) → 77 × 1024.
    pub fn codificar(&self, ids: &[u32]) -> Resultado<Vec<f32>> {
        if ids.len() != TOKENS || ids.iter().any(|&i| i as usize >= VOCAB) {
            return Err(format!("hacen falta {TOKENS} ids menores que {VOCAB}"));
        }
        let tabla = self.tokens.datos();
        let mut x = vec![0f32; TOKENS * DIM];
        for (t, (&id, fila)) in ids.iter().zip(x.chunks_mut(DIM)).enumerate() {
            let id = id as usize;
            crate::kernels::sgemm::f16_a_f32(&tabla[id * DIM..(id + 1) * DIM], fila);
            for (v, p) in fila
                .iter_mut()
                .zip(&self.posiciones[t * DIM..(t + 1) * DIM])
            {
                *v += p;
            }
        }
        for c in &self.capas {
            let mut h = x.clone();
            c.ln1.capa(&mut h);
            let (q, k, v) = (
                c.q.aplicar(&h, TOKENS),
                c.k.aplicar(&h, TOKENS),
                c.v.aplicar(&h, TOKENS),
            );
            let a = atencion(&q, &k, &v, (TOKENS, TOKENS, DIM), CABEZAS, true);
            let a = c.salida.aplicar(&a, TOKENS);
            for (u, w) in x.iter_mut().zip(&a) {
                *u += w;
            }
            let mut h = x.clone();
            c.ln2.capa(&mut h);
            let mut m = c.fc1.aplicar(&h, TOKENS);
            m.par_iter_mut().for_each(|v| *v = gelu(*v));
            let m = c.fc2.aplicar(&m, TOKENS);
            for (u, w) in x.iter_mut().zip(&m) {
                *u += w;
            }
        }
        self.ln_final.capa(&mut x);
        Ok(x)
    }
}
