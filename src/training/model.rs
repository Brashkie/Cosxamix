//! Qwen2 en f32, entrenable, construido descuantizando un GGUF.
//!
//! ## Por qué existe, si ya hay un Qwen2 en `src/model/`
//!
//! Aquel es el de inferencia y usa `QMatMul`, que **no implementa `bwd`**: el
//! gradiente no atraviesa una multiplicación cuantizada. El backward pide
//! `dL/dx = dL/dy · W` —multiplicar por `W` sin transponer— y candle solo sabe
//! hacer `x · Wᵀ`.
//!
//! Así que para entrenar hay que pasar la base a f32, y eso tiene un precio
//! medido que acota el alcance: el 0,5B son ~2 GB y entra en esta máquina; el
//! 3B serían ~12 GB y no. Entrenar el 3B pide el kernel cuantizado que falta,
//! y está anotado en el ROADMAP.
//!
//! ## Qué NO tiene, a propósito
//!
//! No hay caché de atención. En entrenamiento se procesa la secuencia entera
//! de una vez y cada paso empieza de cero, así que el caché —que es lo que
//! hace rápida la inferencia conversacional— aquí no tendría a quién servir.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::ops::softmax_last_dim;
use candle_transformers::utils::repeat_kv;

use super::lora::LinealLora;

/// Lo que define la forma del modelo. Sale del GGUF, no de constantes.
#[derive(Debug, Clone)]
pub struct Config {
    pub oculto: usize,
    pub intermedio: usize,
    pub capas: usize,
    pub cabezas: usize,
    pub cabezas_kv: usize,
    pub dim_cabeza: usize,
    pub eps: f64,
    pub rope_base: f32,
    pub vocabulario: usize,
}

/// Normalización RMS con el peso congelado.
struct Rms {
    peso: Tensor,
    eps: f64,
}

impl Rms {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let media = xf.sqr()?.mean_keepdim(x.rank() - 1)?;
        let norm = xf.broadcast_div(&(media + self.eps)?.sqrt()?)?;
        norm.broadcast_mul(&self.peso)
    }
}

struct Capa {
    norm_attn: Rms,
    q: LinealLora,
    k: LinealLora,
    v: LinealLora,
    o: LinealLora,
    norm_ffn: Rms,
    gate: LinealLora,
    up: LinealLora,
    down: LinealLora,
}

pub struct ModeloEntrenable {
    embeddings: Tensor,
    capas: Vec<Capa>,
    norm_final: Rms,
    salida: Tensor,
    cos: Tensor,
    sin: Tensor,
    cfg: Config,
}

fn rms(t: Tensor, eps: f64) -> Rms {
    Rms { peso: t, eps }
}

impl ModeloEntrenable {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Carga el GGUF y lo descuantiza entero a f32.
    ///
    /// `r` y `alfa` son los del LoRA. Los adaptadores van sobre **q y v**, que
    /// es la eleccion habitual: son las proyecciones donde un adaptador rinde
    /// mas por parametro, y poner uno en las siete multiplicaciones de cada
    /// capa multiplicaria el coste sin multiplicar el resultado.
    pub fn cargar(ruta: &Path, r: usize, alfa: f64, dev: &Device) -> Result<Self> {
        let mut f = std::fs::File::open(ruta)?;
        let ct = gguf_file::Content::read(&mut f)?;

        let leer_u32 = |k: &str| -> Result<usize> {
            Ok(ct
                .metadata
                .get(k)
                .ok_or_else(|| candle_core::Error::Msg(format!("falta {k} en el gguf")))?
                .to_u32()? as usize)
        };
        let oculto = leer_u32("qwen2.embedding_length")?;
        let cabezas = leer_u32("qwen2.attention.head_count")?;
        let cfg = Config {
            oculto,
            intermedio: leer_u32("qwen2.feed_forward_length")?,
            capas: leer_u32("qwen2.block_count")?,
            cabezas,
            cabezas_kv: leer_u32("qwen2.attention.head_count_kv")?,
            dim_cabeza: oculto / cabezas,
            eps: ct
                .metadata
                .get("qwen2.attention.layer_norm_rms_epsilon")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1e-6) as f64,
            rope_base: ct
                .metadata
                .get("qwen2.rope.freq_base")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(10000.0),
            vocabulario: 0,
        };

        let mut den =
            |nombre: &str| -> Result<Tensor> { ct.tensor(&mut f, nombre, dev)?.dequantize(dev) };

        let embeddings = den("token_embd.weight")?;
        let vocabulario = embeddings.dim(0)?;
        // El 0,5B ata los embeddings a la salida y no trae `output.weight`.
        let salida = den("output.weight").unwrap_or_else(|_| embeddings.clone());
        let norm_final = rms(den("output_norm.weight")?, cfg.eps);

        let mut capas = Vec::with_capacity(cfg.capas);
        for i in 0..cfg.capas {
            let p = format!("blk.{i}");
            let lin =
                |n: &str, den: &mut dyn FnMut(&str) -> Result<Tensor>| -> Result<LinealLora> {
                    let peso = den(&format!("{p}.{n}.weight"))?;
                    let sesgo = den(&format!("{p}.{n}.bias")).ok();
                    Ok(LinealLora::nueva(&format!("{p}.{n}"), peso, sesgo))
                };
            capas.push(Capa {
                norm_attn: rms(den(&format!("{p}.attn_norm.weight"))?, cfg.eps),
                q: lin("attn_q", &mut den)?.con_lora(r, alfa, dev)?,
                k: lin("attn_k", &mut den)?,
                v: lin("attn_v", &mut den)?.con_lora(r, alfa, dev)?,
                o: lin("attn_output", &mut den)?,
                norm_ffn: rms(den(&format!("{p}.ffn_norm.weight"))?, cfg.eps),
                gate: lin("ffn_gate", &mut den)?,
                up: lin("ffn_up", &mut den)?,
                down: lin("ffn_down", &mut den)?,
            });
        }

        let (cos, sin) = frecuencias(cfg.dim_cabeza, cfg.rope_base, 8192, dev)?;

        Ok(Self {
            embeddings,
            capas,
            norm_final,
            salida,
            cos,
            sin,
            cfg: Config { vocabulario, ..cfg },
        })
    }

    /// Todos los parámetros entrenables: solo los adaptadores.
    pub fn variables(&self) -> Vec<candle_core::Var> {
        self.capas
            .iter()
            .flat_map(|c| {
                c.q.variables()
                    .into_iter()
                    .chain(c.v.variables())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn adaptadores(&self) -> Vec<&LinealLora> {
        self.capas.iter().flat_map(|c| [&c.q, &c.v]).collect()
    }

    /// `ids` es una secuencia de `(t,)`. Devuelve los logits de `(t, vocab)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let t = ids.dim(0)?;
        let mut x = self.embeddings.index_select(ids, 0)?;

        let mascara = self.mascara(t, ids.device())?;
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;

        for capa in &self.capas {
            let h = capa.norm_attn.forward(&x)?;
            let a = self.atencion(capa, &h, t, &mascara, &cos, &sin)?;
            x = (x + a)?;

            let h = capa.norm_ffn.forward(&x)?;
            let g = capa.gate.forward(&h)?.silu()?;
            let u = capa.up.forward(&h)?;
            x = (x + capa.down.forward(&(g * u)?)?)?;
        }

        let x = self.norm_final.forward(&x)?;
        x.broadcast_matmul(&self.salida.t()?)
    }

    fn atencion(
        &self,
        capa: &Capa,
        h: &Tensor,
        t: usize,
        mascara: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let q = capa
            .q
            .forward(h)?
            .reshape((t, c.cabezas, c.dim_cabeza))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = capa
            .k
            .forward(h)?
            .reshape((t, c.cabezas_kv, c.dim_cabeza))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = capa
            .v
            .forward(h)?
            .reshape((t, c.cabezas_kv, c.dim_cabeza))?
            .transpose(0, 1)?
            .contiguous()?;

        // `rope` espera (b, cabezas, t, dim); se agrega el lote de 1.
        let q = candle_nn::rotary_emb::rope(&q.unsqueeze(0)?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.unsqueeze(0)?, cos, sin)?;

        let rep = c.cabezas / c.cabezas_kv;
        let k = repeat_kv(k, rep)?;
        let v = repeat_kv(v.unsqueeze(0)?, rep)?;

        let escala = 1f64 / (c.dim_cabeza as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * escala)?;
        let att = att.broadcast_add(mascara)?;
        let att = softmax_last_dim(&att)?;
        let salida = att.matmul(&v.contiguous()?)?;

        salida
            .i(0)?
            .transpose(0, 1)?
            .reshape((t, c.cabezas * c.dim_cabeza))?
            .apply(&|x: &Tensor| capa.o.forward(x))
    }

    /// Máscara causal: la posición `i` no puede mirar a `j > i`.
    fn mascara(&self, t: usize, dev: &Device) -> Result<Tensor> {
        let datos: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        Tensor::from_vec(datos, (t, t), dev)
    }
}

/// `cos` y `sin` de RoPE, precalculados una vez.
fn frecuencias(dim: usize, base: f32, largo: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..dim / 2)
        .map(|i| 1f32 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), dev)?;
    let idx = Tensor::arange(0u32, largo as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((largo, 1))?;
    let m = idx.matmul(&theta.reshape((1, dim / 2))?)?;
    Ok((m.cos()?, m.sin()?))
}
