//! El forward de Qwen2 **sobre la pila propia**, sin candle.
//!
//! Es la fase 3 del motor propio, y el momento en que las piezas de las fases
//! anteriores dejan de ser ejercicios: lector de GGUF, descuantización,
//! `matmul` con despacho automático, RoPE, atención y softmax, todas nuestras,
//! encadenadas hasta producir logits.
//!
//! ## Cómo se verifica
//!
//! Comparando los **logits** contra candle sobre el mismo modelo y el mismo
//! prompt. No el texto generado: dos implementaciones que difieran en el
//! último bit pueden elegir tokens distintos y divergir, y entonces la
//! comparación no dice nada. Los logits sí se comparan número a número.
//!
//! ## Lo que no hace
//!
//! No muestrea ni genera: devuelve los logits del último token y ahí termina.
//! Elegir el siguiente token es otra cosa y ya existe más arriba.
//!
//! Los pesos se leen **mapeados y sin copiar** con [`crate::gguf::Pesos`], así
//! que el pico de memoria es el de las activaciones y no el del modelo.

use std::path::{Path, PathBuf};

use crate::gguf::{Gguf, InfoTensor, Pesos, TipoGgml};
use crate::ops;

type Resultado<T> = Result<T, String>;

/// La configuración, leída del archivo.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub capas: usize,
    pub dim: usize,
    pub cabezas: usize,
    pub cabezas_kv: usize,
    pub vocab: usize,
    pub contexto: usize,
    pub eps: f32,
    pub rope: f32,
}

impl Config {
    pub fn dim_cabeza(&self) -> usize {
        self.dim / self.cabezas
    }

    pub fn dim_kv(&self) -> usize {
        self.cabezas_kv * self.dim_cabeza()
    }
}

/// El caché de claves y valores de una conversación.
///
/// Crece por tokens. Es lo que hace que el segundo turno no reprocese el
/// primero, y lo que obliga a que la máscara causal use la posición absoluta
/// y no el índice dentro de la tanda.
pub struct Cache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    largo: usize,
}

impl Cache {
    pub fn nuevo(cfg: &Config) -> Self {
        Self {
            k: vec![Vec::new(); cfg.capas],
            v: vec![Vec::new(); cfg.capas],
            largo: 0,
        }
    }

    pub fn largo(&self) -> usize {
        self.largo
    }

    pub fn limpiar(&mut self) {
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            k.clear();
            v.clear();
        }
        self.largo = 0;
    }

    /// Se queda con los primeros `n` tokens.
    ///
    /// Es lo que permite reutilizar el prefijo común entre dos prompts: el
    /// segundo mensaje de WinsiBot comparte el *system* con el primero y
    /// después diverge, así que se conserva lo común y se tira el resto. Sin
    /// recortar, la atención miraría tokens que ya no están en el prompt.
    ///
    /// Cada capa guarda `largo * dim_kv` valores, así que el ancho por token
    /// sale de dividir.
    pub fn truncar(&mut self, n: usize) {
        if n >= self.largo {
            return;
        }
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            let ancho = k.len() / self.largo;
            k.truncate(n * ancho);
            v.truncate(n * ancho);
        }
        self.largo = n;
    }

    /// Lo que ocupa, en bytes.
    pub fn bytes(&self) -> usize {
        self.k
            .iter()
            .chain(&self.v)
            .map(|x| x.len() * 4)
            .sum::<usize>()
    }
}

/// Un tensor del modelo, ya localizado.
struct Peso {
    info: InfoTensor,
}

impl Peso {
    /// `(k, n)` en el orden que pide `matmul`: entradas y salidas.
    fn forma(&self) -> (usize, usize) {
        (self.info.dims[0] as usize, self.info.dims[1] as usize)
    }
}

/// Los pesos de una capa.
struct Capa {
    attn_norm: Peso,
    q: Peso,
    q_bias: Peso,
    k: Peso,
    k_bias: Peso,
    v: Peso,
    v_bias: Peso,
    salida: Peso,
    ffn_norm: Peso,
    gate: Peso,
    up: Peso,
    down: Peso,
}

/// El modelo, listo para hacer forward.
pub struct Modelo {
    pub cfg: Config,
    gguf: Gguf,
    mapa: Pesos,
    ruta: PathBuf,
    capas: Vec<Capa>,
    embd: Peso,
    norma_final: Peso,
    salida: Peso,
    /// La proyección de salida comparte los pesos con los embeddings cuando el
    /// archivo no trae `output.weight`, que es lo que hacen los modelos con
    /// *tied embeddings* —el 0,5B de Qwen2.5 entre ellos—.
    salida_atada: bool,
    cos: Vec<f32>,
    sin: Vec<f32>,
    forzar_f32: bool,
}

impl Modelo {
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let gguf = Gguf::abrir(ruta)?;
        let mapa = Pesos::abrir(&gguf, ruta)?;

        let entero = |clave: &str| -> Resultado<usize> {
            gguf.meta_entero(clave)
                .map(|v| v as usize)
                .ok_or_else(|| format!("falta el metadato {clave}"))
        };
        let cfg = Config {
            capas: entero("qwen2.block_count")?,
            dim: entero("qwen2.embedding_length")?,
            cabezas: entero("qwen2.attention.head_count")?,
            cabezas_kv: entero("qwen2.attention.head_count_kv")?,
            vocab: 0, // se completa con la forma de la tabla
            contexto: entero("qwen2.context_length")?,
            eps: gguf
                .meta_decimal("qwen2.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            rope: gguf
                .meta_decimal("qwen2.rope.freq_base")
                .unwrap_or(10_000.0) as f32,
        };

        let buscar = |nombre: &str| -> Resultado<Peso> {
            gguf.tensor(nombre)
                .map(|info| Peso { info: info.clone() })
                .ok_or_else(|| format!("falta el tensor {nombre}"))
        };

        let embd = buscar("token_embd.weight")?;
        let vocab = embd.info.dims[1] as usize;
        let (salida, salida_atada) = match gguf.tensor("output.weight") {
            Some(info) => (Peso { info: info.clone() }, false),
            None => (
                Peso {
                    info: embd.info.clone(),
                },
                true,
            ),
        };

        let norma_final = buscar("output_norm.weight")?;

        let mut capas = Vec::with_capacity(cfg.capas);
        for i in 0..cfg.capas {
            let p = format!("blk.{i}");
            capas.push(Capa {
                attn_norm: buscar(&format!("{p}.attn_norm.weight"))?,
                q: buscar(&format!("{p}.attn_q.weight"))?,
                q_bias: buscar(&format!("{p}.attn_q.bias"))?,
                k: buscar(&format!("{p}.attn_k.weight"))?,
                k_bias: buscar(&format!("{p}.attn_k.bias"))?,
                v: buscar(&format!("{p}.attn_v.weight"))?,
                v_bias: buscar(&format!("{p}.attn_v.bias"))?,
                salida: buscar(&format!("{p}.attn_output.weight"))?,
                ffn_norm: buscar(&format!("{p}.ffn_norm.weight"))?,
                gate: buscar(&format!("{p}.ffn_gate.weight"))?,
                up: buscar(&format!("{p}.ffn_up.weight"))?,
                down: buscar(&format!("{p}.ffn_down.weight"))?,
            });
        }

        let (cos, sin) = ops::frecuencias_rope(cfg.dim / cfg.cabezas, cfg.rope, cfg.contexto);

        Ok(Self {
            cfg: Config { vocab, ..cfg },
            gguf,
            mapa,
            ruta: ruta.to_path_buf(),
            capas,
            embd,
            norma_final,
            salida,
            salida_atada,
            cos,
            sin,
            forzar_f32: std::env::var("COSXAMIX_MATMUL").is_ok_and(|v| v == "f32"),
        })
    }

    pub fn ruta(&self) -> &Path {
        &self.ruta
    }

    pub fn salida_atada(&self) -> bool {
        self.salida_atada
    }

    /// Cuántos bytes de pesos tiene el modelo, por el archivo.
    pub fn bytes(&self) -> usize {
        self.gguf.tensores.iter().map(|t| t.bytes()).sum()
    }

    fn bytes_de(&self, p: &Peso) -> Resultado<&[u8]> {
        self.mapa.de(&p.info)
    }

    /// Descuantiza un vector de `f32` —normas y sesgos, que van sin
    /// cuantizar— a un buffer.
    fn vector(&self, p: &Peso, destino: &mut Vec<f32>) -> Resultado<()> {
        let n = p.info.elementos();
        destino.clear();
        destino.resize(n, 0.0);
        crate::quant::descuantizar(p.info.tipo, self.bytes_de(p)?, destino)
    }

    /// `destino = x · W^T + sesgo`, con el camino que corresponda.
    fn proyectar(
        &self,
        x: &[f32],
        tokens: usize,
        w: &Peso,
        sesgo: Option<&[f32]>,
        destino: &mut Vec<f32>,
    ) -> Resultado<()> {
        let (k, n) = w.forma();
        destino.clear();
        destino.resize(tokens * n, 0.0);
        let bytes = self.bytes_de(w)?;
        // `COSXAMIX_MATMUL=f32` fuerza el camino que NO cuantiza la entrada.
        // Existe para separar dos cosas que se confunden: si el forward propio
        // se aparta de candle por el error de cuantizar —esperable— o por un
        // error de implementación. Si forzando `f32` se acerca, es lo primero.
        if self.forzar_f32 {
            ops::matmul((tokens, k, n), x, bytes, w.info.tipo, destino)?;
        } else {
            ops::matmul_auto((tokens, k, n), x, bytes, w.info.tipo, destino)?;
        }
        if let Some(b) = sesgo {
            for fila in destino.chunks_mut(n) {
                for (d, s) in fila.iter_mut().zip(b) {
                    *d += s;
                }
            }
        }
        Ok(())
    }

    /// Los logits del **último** token de `tokens`.
    ///
    /// `cache` se extiende con los tokens nuevos, así que llamar dos veces
    /// seguidas continúa la secuencia en vez de reempezarla. Eso es lo que
    /// hace que el decode cueste un token y no todo el contexto.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Resultado<Vec<f32>> {
        if tokens.is_empty() {
            return Err("no hay tokens que procesar".into());
        }
        let cfg = &self.cfg;
        let t = tokens.len();
        let total = cache.largo + t;
        if total > cfg.contexto {
            return Err(format!(
                "{total} tokens pasan el contexto de {}",
                cfg.contexto
            ));
        }

        // ── embeddings ──────────────────────────────────────────────────────
        // La tabla es [dim, vocab]: cada fila es un token y mide `dim`.
        let bytes_embd = self.bytes_de(&self.embd)?;
        let por_bloque = self.embd.info.tipo.valores_por_bloque();
        let bytes_fila = cfg.dim / por_bloque * self.embd.info.tipo.bytes_por_bloque();
        let mut x = vec![0f32; t * cfg.dim];
        for (i, tok) in tokens.iter().enumerate() {
            if *tok as usize >= cfg.vocab {
                return Err(format!("el token {tok} no entra en {} palabras", cfg.vocab));
            }
            let desde = *tok as usize * bytes_fila;
            crate::quant::descuantizar(
                self.embd.info.tipo,
                &bytes_embd[desde..desde + bytes_fila],
                &mut x[i * cfg.dim..(i + 1) * cfg.dim],
            )?;
        }

        let dim_cabeza = cfg.dim_cabeza();

        // Buffers reusados por todas las capas: aca si vale la pena, porque
        // son 36 capas y no una llamada suelta.
        let mut xn = vec![0f32; t * cfg.dim];
        let mut peso = Vec::new();
        let mut sesgo = Vec::new();
        let (mut q, mut kk, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let mut att = vec![0f32; t * cfg.dim];
        let mut proy = Vec::new();
        let mut gate = Vec::new();
        let mut up = Vec::new();

        for (i, capa) in self.capas.iter().enumerate() {
            // ── atencion ────────────────────────────────────────────────────
            self.vector(&capa.attn_norm, &mut peso)?;
            for (dst, src) in xn.chunks_mut(cfg.dim).zip(x.chunks(cfg.dim)) {
                crate::kernels::rmsnorm(dst, src, &peso, cfg.eps);
            }

            self.vector(&capa.q_bias, &mut sesgo)?;
            self.proyectar(&xn, t, &capa.q, Some(&sesgo), &mut q)?;
            self.vector(&capa.k_bias, &mut sesgo)?;
            self.proyectar(&xn, t, &capa.k, Some(&sesgo), &mut kk)?;
            self.vector(&capa.v_bias, &mut sesgo)?;
            self.proyectar(&xn, t, &capa.v, Some(&sesgo), &mut vv)?;

            ops::rope(
                &mut q,
                (t, cfg.cabezas, dim_cabeza),
                &self.cos,
                &self.sin,
                cache.largo,
            )?;
            ops::rope(
                &mut kk,
                (t, cfg.cabezas_kv, dim_cabeza),
                &self.cos,
                &self.sin,
                cache.largo,
            )?;

            cache.k[i].extend_from_slice(&kk);
            cache.v[i].extend_from_slice(&vv);

            ops::atencion(
                &q,
                &cache.k[i],
                &cache.v[i],
                (t, total, cfg.cabezas, cfg.cabezas_kv, dim_cabeza),
                &mut att,
            )?;

            self.proyectar(&att, t, &capa.salida, None, &mut proy)?;
            for (a, b) in x.iter_mut().zip(&proy) {
                *a += b;
            }

            // ── FFN ─────────────────────────────────────────────────────────
            self.vector(&capa.ffn_norm, &mut peso)?;
            for (dst, src) in xn.chunks_mut(cfg.dim).zip(x.chunks(cfg.dim)) {
                crate::kernels::rmsnorm(dst, src, &peso, cfg.eps);
            }

            self.proyectar(&xn, t, &capa.gate, None, &mut gate)?;
            self.proyectar(&xn, t, &capa.up, None, &mut up)?;
            // SwiGLU: silu(gate) * up, en su sitio sobre `gate`.
            let mut activado = vec![0f32; gate.len()];
            crate::kernels::silu(&mut activado, &gate);
            for (a, u) in activado.iter_mut().zip(&up) {
                *a *= u;
            }
            self.proyectar(&activado, t, &capa.down, None, &mut proy)?;
            for (a, b) in x.iter_mut().zip(&proy) {
                *a += b;
            }
        }

        // ── norma final y logits, solo del ultimo token ─────────────────────
        self.vector(&self.norma_final, &mut peso)?;
        let ultimo = &x[(t - 1) * cfg.dim..];
        let mut final_ = vec![0f32; cfg.dim];
        crate::kernels::rmsnorm(&mut final_, ultimo, &peso, cfg.eps);

        let mut logits = Vec::new();
        self.proyectar(&final_, 1, &self.salida, None, &mut logits)?;

        cache.largo = total;
        Ok(logits)
    }
}

impl Modelo {
    /// El token de fin de texto, del archivo.
    ///
    /// En los Qwen2.5 de chat es `<|im_end|>`, 151645, que es donde el motor
    /// de candle también para. Se lee de `tokenizer.ggml.eos_token_id` en vez
    /// de fijarlo, porque un daiss con tokenizador propio tiene otro.
    pub fn fin(&self) -> Option<u32> {
        self.gguf
            .meta_entero("tokenizer.ggml.eos_token_id")
            .map(|v| v as u32)
    }

    /// Genera hasta `max` tokens a continuación de `prompt`.
    ///
    /// Llama a `al_token` con cada token apenas sale, que es lo que permite
    /// mandarlo en streaming. Si devuelve `false` se para en el acto: es el
    /// caso del cliente que cerró la conexión, y seguir generando para nadie
    /// es tirar segundos de CPU.
    ///
    /// Para también en el token de fin —que no se devuelve— y antes de pasarse
    /// del contexto, donde el forward fallaría.
    ///
    /// El prompt se procesa entero de una vez y después va de a un token: el
    /// prefill aprovecha el lote, que es donde la matmul propia le gana a
    /// candle, y el decode no tiene lote que aprovechar.
    pub fn generar(
        &self,
        prompt: &[u32],
        max: usize,
        muestreador: &mut crate::sampler::Muestreador,
        cache: &mut Cache,
        mut al_token: impl FnMut(u32) -> bool,
    ) -> Resultado<Vec<u32>> {
        let fin = self.fin();
        let mut salida = Vec::with_capacity(max);
        let mut logits = self.forward(prompt, cache)?;
        for _ in 0..max {
            let t = muestreador.elegir(&logits);
            if Some(t) == fin {
                break;
            }
            salida.push(t);
            if !al_token(t) {
                break;
            }
            if cache.largo() >= self.cfg.contexto {
                break;
            }
            logits = self.forward(&[t], cache)?;
        }
        Ok(salida)
    }
}

/// Dimensiones del caché por token, que es lo que decide cuánto contexto
/// entra en memoria.
pub fn bytes_de_cache_por_token(cfg: &Config) -> usize {
    2 * cfg.capas * cfg.dim_kv() * 4
}

/// El tipo de los pesos del modelo, para informar.
pub fn tipo_dominante(g: &Gguf) -> TipoGgml {
    let mut cuenta: std::collections::HashMap<TipoGgml, usize> = Default::default();
    for t in &g.tensores {
        *cuenta.entry(t.tipo).or_insert(0) += t.bytes();
    }
    cuenta
        .into_iter()
        .max_by_key(|(_, b)| *b)
        .map(|(t, _)| t)
        .unwrap_or(TipoGgml::F32)
}
