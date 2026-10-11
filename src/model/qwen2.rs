//! Qwen2 cuantizado, con el caché de atención expuesto.
//!
//! ## De dónde sale esto
//!
//! Copiado de `candle-transformers` 0.9.2,
//! `src/models/quantized_qwen2.rs`, de huggingface/candle, bajo MIT OR
//! Apache-2.0. Los cambios respecto al original están marcados con
//! `// COSXAMIX:` y son los mínimos para poder guardar y restaurar el caché.
//!
//! ## Por qué está acá y no se usa el del crate
//!
//! El caché de atención vive dentro de `ModelWeights` y es privado, y el tipo
//! no es `Clone`. Con el crate tal cual hay **un solo caché**: mientras una
//! conversación avanza sola funciona muy bien, pero cuando varias llegan
//! intercaladas —lo normal en WhatsApp— cada una invalida la de la otra.
//!
//! Medido antes de decidir esto: dos conversaciones de tres turnos, las mismas
//! seis peticiones, cambiando solo el orden, pasaban de 5541 ms con 244 tokens
//! reaprovechados a 9770 ms con **cero**. Un 76% peor.
//!
//! La alternativa era cargar el modelo una vez por sesión, y también se midió:
//! **931 MB por instancia**, el doble del GGUF. Diez chats serían 9,1 GB.
//!
//! ## El coste de tener esto acá
//!
//! Son 338 líneas de código ajeno bajo nuestro mantenimiento, que no reciben
//! los arreglos de upstream. Al actualizar candle hay que comparar este archivo
//! con el del crate. El cambio propio es chico y está marcado, justamente para
//! que esa comparación sea posible.
//! Qwen2 model implementation with quantization support.
//!
//! Qwen2 is a chat-optimized language model that supports 8-bit quantization
//! for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group Query Attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - [Model Card](https://huggingface.co/Qwen/Qwen2)
//!

// COSXAMIX: los dos vienen del crate, que los expone como públicos; solo hacía
// falta traer el modelo.
use candle_core::quantized::{QStorage, QTensor};
use candle_core::{
    quantized::{gguf_file, QMatMul},
    DType, Device, IndexOp, Result, Tensor,
};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
// COSXAMIX: ya no se usa `Embedding`: la tabla queda cuantizada.
use candle_nn::Module;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_bq: Tensor,
    attention_bk: Tensor,
    attention_bv: Tensor,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // let (q, k) = self
        //     .rotary_embedding
        //     .apply_rotary_emb_qkv(&q, &k, index_pos)?;
        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for MQA, useful for 70B models and mistral.
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.attention_wo.forward(&y)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    // COSXAMIX: se guarda CUANTIZADA.
    //
    // El original hacía `dequantize()` sobre la tabla entera. Medido en
    // Qwen2.5-0.5B: 151936 x 896 en f32 son **519 MB residentes**, más que
    // el modelo completo (469 MB), y explican casi todo el exceso de los
    // 938 MB que ocupaba el proceso.
    //
    // Un forward solo necesita las filas de los tokens que entran: 3,5 KB
    // por token. Se descuantizan esas.
    tok_embeddings: QTensor,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    // COSXAMIX: la clave lleva también cuántos tokens hay en el caché, porque
    // la máscara ya no es cuadrada.
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let context_length = md_get("qwen2.context_length")?.to_u32()? as usize;
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let head_dim = embedding_length / head_count;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // COSXAMIX: sin `dequantize()`. Ver el comentario del campo.
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(v) => QMatMul::from_qtensor(v)?,
            _ => {
                // use tie_word_embeddings
                QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
            }
        };

        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp = {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                }
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: attention_bq.dequantize(device)?,
                attention_bk: attention_bk.dequantize(device)?,
                attention_bv: attention_bv.dequantize(device)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings,
            embedding_length,
            layers,
            norm,
            output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    // COSXAMIX: la máscara contempla lo que ya está en el caché.
    //
    // El original la armaba cuadrada, `t x t`, que solo es correcta con el
    // caché vacío. Con caché la atención es `t x (cacheados + t)` y las formas
    // no cuadran: `cannot broadcast [16, 16] to [1, 14, 16, 69]`. Por eso
    // candle solo sabe hacer el prefill entero o decodificar de a un token.
    //
    // Esto lo habilita, y de eso dependen dos cosas: alimentar el delta de una
    // conversación por lotes en vez de token a token —que costaba el doble por
    // token— y la decodificación especulativa, que necesita evaluar K tokens de
    // una pasada con el caché puesto.
    //
    // Con `cacheados = 0` da exactamente la matriz de antes.
    fn mask(&mut self, t: usize, cacheados: usize, device: &Device) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&(t, cacheados)) {
            return Ok(mask.clone());
        }

        // La posición `i` del bloque nuevo es la absoluta `cacheados + i`, y
        // puede mirar todo lo anterior o igual a ella. Se marca con 1 lo que
        // NO puede ver.
        let total = cacheados + t;
        let datos: Vec<u8> = (0..t)
            .flat_map(|i| (0..total).map(move |j| u8::from(j > cacheados + i)))
            .collect();

        let mask = Tensor::from_slice(&datos, (t, total), device)?;
        self.masks.insert((t, cacheados), mask.clone());
        Ok(mask)
    }

    // COSXAMIX: lo único que se agregó al original. Tres métodos para que el
    // caché de atención se pueda guardar, restaurar y vaciar desde fuera, que
    // es lo que permite tener uno por conversación con un solo modelo cargado.

    /// Se lleva el caché de atención, dejando el modelo limpio.
    ///
    /// Los tensores salen por valor y no clonados: devolverlos al modelo es
    /// mover punteros, no copiar los datos. Clonar aquí costaría una copia de
    /// todo el caché en cada cambio de conversación, que es justo lo que esto
    /// viene a evitar.
    pub fn tomar_cache(&mut self) -> CacheAtencion {
        CacheAtencion {
            capas: self.layers.iter_mut().map(|c| c.kv_cache.take()).collect(),
        }
    }

    /// Pone un caché guardado, descartando el que hubiera.
    ///
    /// Quien llama es responsable de que corresponda a la secuencia que va a
    /// procesar: el modelo no tiene forma de comprobarlo, y restaurar el caché
    /// equivocado da respuestas condicionadas por otra conversación, sin error.
    pub fn poner_cache(&mut self, cache: CacheAtencion) {
        for (capa, guardado) in self.layers.iter_mut().zip(cache.capas) {
            capa.kv_cache = guardado;
        }
    }

    /// Recorta el caché a los primeros `n` tokens.
    ///
    /// Hace falta para la decodificación especulativa, y es lo que candle no
    /// expone. Al verificar una ráfaga, el modelo procesa las K propuestas
    /// enteras y su caché crece con las K — pero si solo se aceptan 2, las
    /// otras no forman parte de la secuencia real y tienen que irse.
    ///
    /// Sin esto, el caché y la posición lógica se separan, y en la ronda
    /// siguiente la máscara causal se construye para un largo y la atención
    /// para otro: `cannot broadcast [5, 48] to [1, 16, 5, 51]`.
    ///
    /// La dimensión 2 es la de la secuencia en [lote, cabezas, tokens, dim].
    pub fn truncar_cache(&mut self, n: usize) -> Result<()> {
        for capa in self.layers.iter_mut() {
            if let Some((k, v)) = capa.kv_cache.take() {
                let largo = k.dims()[2];
                capa.kv_cache = if n >= largo {
                    Some((k, v))
                } else {
                    // `narrow` devuelve una vista; `contiguous` evita que el
                    // tensor original siga vivo detrás de ella y que la
                    // memoria no se libere nunca.
                    Some((
                        k.narrow(2, 0, n)?.contiguous()?,
                        v.narrow(2, 0, n)?.contiguous()?,
                    ))
                };
            }
        }
        Ok(())
    }

    /// Vacía el caché. Equivale a arrancar una conversación nueva.
    pub fn limpiar_cache(&mut self) {
        for capa in self.layers.iter_mut() {
            capa.kv_cache = None;
        }
    }

    /// COSXAMIX: busca las filas de la tabla de embeddings, sin descuantizarla entera.
    ///
    /// La tabla está guardada por filas y cada una ocupa un número entero de
    /// bloques de cuantización —896 elementos en bloques de 32 son 28 bloques—,
    /// así que los bytes de una fila se pueden recortar sin desarmar nada. Se
    /// arma un tensor chico con solo las filas pedidas y se descuantiza eso.
    ///
    /// Coste: 3,5 KB por token en vez de 519 MB residentes.
    fn embebidos(&self, ids: &Tensor) -> Result<Tensor> {
        let (lote, largo) = ids.dims2()?;
        let cols = self.embedding_length;

        let dtype = self.tok_embeddings.dtype();
        let bloque = dtype.block_size();
        if !cols.is_multiple_of(bloque) {
            candle_core::bail!(
                "la fila de embeddings ({cols}) no es multiplo del bloque de {dtype:?} ({bloque})"
            );
        }
        let bytes_fila = cols / bloque * dtype.type_size();

        let datos = self.tok_embeddings.data()?;
        let indices = ids.flatten_all()?.to_vec1::<u32>()?;

        let mut recorte = Vec::with_capacity(indices.len() * bytes_fila);
        for &i in &indices {
            let desde = i as usize * bytes_fila;
            let hasta = desde + bytes_fila;
            if hasta > datos.len() {
                candle_core::bail!("token {i} fuera de la tabla de embeddings");
            }
            recorte.extend_from_slice(&datos[desde..hasta]);
        }

        let storage = QStorage::Cpu(dtype.from_data(std::borrow::Cow::Borrowed(&recorte)));
        let chico = QTensor::new(storage, (indices.len(), cols))?;

        chico
            .dequantize(&self.tok_embeddings.device())?
            .reshape((lote, largo, cols))
    }

    /// COSXAMIX: como `forward`, pero devuelve los logits de TODAS las posiciones.
    ///
    /// `forward` se queda solo con la última (`x.i((.., seq_len - 1, ..))`),
    /// que es lo único que hace falta para generar. La decodificación
    /// especulativa necesita las demás: con ellas verifica K propuestas en una
    /// sola pasada, que es justo de dónde sale su ventaja.
    pub fn forward_todas(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.tronco(x, index_pos, false)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.tronco(x, index_pos, true)
    }

    /// El cuerpo compartido. `solo_ultima` decide si se recorta antes de la
    /// proyección de salida: recortar ahorra una multiplicación por el
    /// vocabulario entero —151936 columnas— en cada posición que no se use.
    fn tronco(&mut self, x: &Tensor, index_pos: usize, solo_ultima: bool) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        // Con un solo token no hace falta máscara: no hay futuro que tapar
        // dentro del bloque y todo lo cacheado es pasado.
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.embebidos(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        let x = self.norm.forward(&layer_in)?;
        let x = if solo_ultima {
            x.i((.., seq_len - 1, ..))?
        } else {
            x
        };
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }
}

// COSXAMIX: tipo propio, no está en el original.

/// El caché de atención de todas las capas, fuera del modelo.
///
/// Es opaco a propósito: nadie fuera de este módulo necesita saber que son
/// pares de tensores por capa, y exponerlo invitaría a manipularlo.
///
/// Pesa lo suyo —del orden de 24 KB por token en Qwen2-0.5B— así que quien lo
/// guarde por sesión tiene que ponerle un tope.
pub struct CacheAtencion {
    capas: Vec<Option<(Tensor, Tensor)>>,
}

impl CacheAtencion {
    /// COSXAMIX: cuántos bytes ocupa este caché.
    ///
    /// Hace falta para decidir con datos cuántas sesiones caben y si vale la
    /// pena persistirlas. La cuenta de servilleta se equivoca facil: depende
    /// de las capas, de las cabezas KV y de la dimension por cabeza, que
    /// cambian con el modelo.
    pub fn bytes(&self) -> usize {
        self.capas
            .iter()
            .flatten()
            .map(|(k, v)| {
                k.elem_count() * k.dtype().size_in_bytes()
                    + v.elem_count() * v.dtype().size_in_bytes()
            })
            .sum()
    }

    /// Un caché vacío: el estado de una conversación que todavía no empezó.
    pub fn vacio(capas: usize) -> Self {
        Self {
            capas: (0..capas).map(|_| None).collect(),
        }
    }

    /// Cuántos tokens tiene guardados.
    ///
    /// Se lee de la primera capa que tenga algo: todas avanzan juntas, así que
    /// alcanza con mirar una. La dimensión 2 es la de la secuencia en el
    /// tensor [lote, cabezas, tokens, dim].
    pub fn tokens(&self) -> usize {
        self.capas
            .iter()
            .flatten()
            .next()
            .and_then(|(k, _)| k.dims().get(2).copied())
            .unwrap_or(0)
    }

    pub fn esta_vacio(&self) -> bool {
        self.capas.iter().all(Option::is_none)
    }
}
