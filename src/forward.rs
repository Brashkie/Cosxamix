//! El forward de Qwen2 y Qwen3 **sobre la pila propia**, sin candle.
//!
//! Las dos arquitecturas comparten casi todo. Qwen3 cambia tres cosas, y las
//! tres se leen del archivo en vez de suponerse:
//!
//! - q, k y v **sin sesgo**;
//! - una RMSNorm **por cabeza** sobre q y k antes de RoPE (`attn_q_norm`,
//!   `attn_k_norm`), que es lo que estabiliza la atención sin el sesgo;
//! - el ancho de cabeza **no sale de dividir**: en el 4B son 32 cabezas de 128
//!   sobre un `dim` de 2560, no de 80. Lo dice `<arq>.attention.key_length`.
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

/// De a cuántos valores se aplica SiLU en su sitio: lo que entra en L1 junto
/// con el tramo de `gate`.
const TRAMO_SILU: usize = 1024;

/// Desde cuántos tokens por tanda la atención pasa el caché de la capa a f32
/// una vez en vez de leerlo en f16: con muchos tokens nuevos cada clave se lee
/// muchas veces, y convertirla cada vez costaría más que convertirla una.
const ATENCION_F32_DESDE: usize = 16;

/// Una proyección para [`Modelo::proyectar_varias`]: los pesos, el sesgo si
/// lo hay, y dónde va el resultado.
type Proyeccion<'a> = (&'a Peso, Option<&'a Vec<f32>>, &'a mut Vec<f32>);

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
    /// El ancho de cada cabeza de atención. En Qwen2 es `dim / cabezas`; en
    /// Qwen3 lo fija el archivo y puede ser otro.
    pub cabeza: usize,
}

impl Config {
    pub fn dim_cabeza(&self) -> usize {
        self.cabeza
    }

    /// El ancho de q y de la salida de la atención: `cabezas × dim_cabeza`,
    /// que en Qwen3 no es `dim`.
    pub fn dim_q(&self) -> usize {
        self.cabezas * self.cabeza
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
///
/// **En f16**, la mitad que en f32: en Qwen3-4B (8 cabezas de clave y valor de
/// 128) son 147 KB por token en vez de 294, ~600 MB menos en una charla de
/// 4.000 tokens. Cada capa pasa su tramo a f32 antes de la atención, en un
/// búfer que se reusa entre capas (ver [`Modelo::forward`]). Es lo que hace
/// llama.cpp por defecto.
pub struct Cache {
    k: Vec<Vec<u16>>,
    v: Vec<Vec<u16>>,
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

    /// Una copia de los primeros `n` tokens.
    ///
    /// Es lo que deja que una conversación nueva arranque con el prefijo de
    /// otra: el caché de un token depende solo de los tokens anteriores, así
    /// que si dos prompts empiezan igual, ese tramo es el mismo en las dos.
    pub fn prefijo(&self, n: usize) -> Cache {
        let n = n.min(self.largo);
        let ancho = if self.largo == 0 {
            0
        } else {
            self.k.first().map_or(0, |k| k.len() / self.largo)
        };
        Cache {
            k: self.k.iter().map(|k| k[..n * ancho].to_vec()).collect(),
            v: self.v.iter().map(|v| v[..n * ancho].to_vec()).collect(),
            largo: n,
        }
    }

    /// Lo que ocupa, en bytes.
    pub fn bytes(&self) -> usize {
        self.k
            .iter()
            .chain(&self.v)
            .map(|x| x.len() * 2)
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
    q_bias: Option<Peso>,
    k: Peso,
    k_bias: Option<Peso>,
    v: Peso,
    v_bias: Option<Peso>,
    /// Las normas por cabeza de Qwen3; `None` en Qwen2.
    q_norm: Option<Peso>,
    k_norm: Option<Peso>,
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
    forzar_f32: bool,
}

impl Modelo {
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let gguf = Gguf::abrir(ruta)?;
        let mapa = Pesos::abrir(&gguf, ruta)?;

        // Las claves llevan el nombre de la arquitectura delante: `qwen2.` o
        // `qwen3.`. Un daiss se escribe como `qwen2`.
        let arq = match gguf
            .metadatos
            .get("general.architecture")
            .and_then(|v| v.texto())
        {
            Some(a @ ("qwen2" | "qwen3")) => a.to_string(),
            Some(otra) => return Err(format!("arquitectura {otra}: se leen qwen2 y qwen3")),
            None => "qwen2".to_string(),
        };
        let entero = |clave: &str| -> Resultado<usize> {
            let clave = format!("{arq}.{clave}");
            gguf.meta_entero(&clave)
                .map(|v| v as usize)
                .ok_or_else(|| format!("falta el metadato {clave}"))
        };
        let dim = entero("embedding_length")?;
        let cabezas = entero("attention.head_count")?;
        let cfg = Config {
            capas: entero("block_count")?,
            dim,
            cabezas,
            cabezas_kv: entero("attention.head_count_kv")?,
            vocab: 0, // se completa con la forma de la tabla
            contexto: entero("context_length")?,
            eps: gguf
                .meta_decimal(&format!("{arq}.attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6) as f32,
            rope: gguf
                .meta_decimal(&format!("{arq}.rope.freq_base"))
                .unwrap_or(10_000.0) as f32,
            cabeza: entero("attention.key_length").unwrap_or(dim / cabezas),
        };

        let buscar = |nombre: &str| -> Resultado<Peso> {
            gguf.tensor(nombre)
                .map(|info| Peso { info: info.clone() })
                .ok_or_else(|| format!("falta el tensor {nombre}"))
        };

        let quizas = |nombre: &str| gguf.tensor(nombre).map(|info| Peso { info: info.clone() });

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
                q_bias: quizas(&format!("{p}.attn_q.bias")),
                k: buscar(&format!("{p}.attn_k.weight"))?,
                k_bias: quizas(&format!("{p}.attn_k.bias")),
                v: buscar(&format!("{p}.attn_v.weight"))?,
                v_bias: quizas(&format!("{p}.attn_v.bias")),
                q_norm: quizas(&format!("{p}.attn_q_norm.weight")),
                k_norm: quizas(&format!("{p}.attn_k_norm.weight")),
                salida: buscar(&format!("{p}.attn_output.weight"))?,
                ffn_norm: buscar(&format!("{p}.ffn_norm.weight"))?,
                gate: buscar(&format!("{p}.ffn_gate.weight"))?,
                up: buscar(&format!("{p}.ffn_up.weight"))?,
                down: buscar(&format!("{p}.ffn_down.weight"))?,
            });
        }

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

    /// Como [`Self::vector`], para lo que no todas las arquitecturas traen:
    /// sin el tensor, `destino` queda vacío y devuelve `false`.
    fn vector_si_hay(&self, p: Option<&Peso>, destino: &mut Vec<f32>) -> Resultado<bool> {
        match p {
            Some(p) => self.vector(p, destino).map(|_| true),
            None => {
                destino.clear();
                Ok(false)
            }
        }
    }

    /// La RMSNorm por cabeza de Qwen3: cada tramo de `dim_cabeza` se normaliza
    /// por separado, con el mismo peso para todas las cabezas.
    fn norma_por_cabeza(
        &self,
        x: &mut [f32],
        p: Option<&Peso>,
        peso: &mut Vec<f32>,
    ) -> Resultado<()> {
        if !self.vector_si_hay(p, peso)? {
            return Ok(());
        }
        let mut tramo = vec![0f32; self.cfg.cabeza];
        for cabeza in x.chunks_mut(self.cfg.cabeza) {
            tramo.copy_from_slice(cabeza);
            crate::kernels::rmsnorm(cabeza, &tramo, peso, self.cfg.eps);
        }
        Ok(())
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

    /// Varias proyecciones del mismo `x` —q, k y v, o gate y up—.
    ///
    /// Con un solo token van en **una** pasada paralela
    /// ([`ops::matmul_varias_una_fila`]): medido dentro del forward, q, k y v
    /// por separado iban a 12 GB/s, porque `attn_k` y `attn_v` son tan chicas
    /// que el costo fijo de cada llamada paralela era casi todo. Con más
    /// tokens, o con `COSXAMIX_MATMUL=f32`, una por una.
    fn proyectar_varias(
        &self,
        x: &[f32],
        tokens: usize,
        trabajos: &mut [Proyeccion<'_>],
    ) -> Resultado<()> {
        // `COSXAMIX_SIN_VARIAS` vuelve a una llamada por matriz, para comparar.
        static VARIAS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let varias = *VARIAS.get_or_init(|| std::env::var_os("COSXAMIX_SIN_VARIAS").is_none());
        if tokens != 1 || self.forzar_f32 || !varias {
            for (w, sesgo, destino) in trabajos.iter_mut() {
                self.proyectar(x, tokens, w, sesgo.map(|v| v.as_slice()), destino)?;
            }
            return Ok(());
        }
        let mut pesos = Vec::with_capacity(trabajos.len());
        for (w, _, destino) in trabajos.iter_mut() {
            let (_, n) = w.forma();
            destino.clear();
            destino.resize(n, 0.0);
            pesos.push((self.bytes_de(w)?, w.info.tipo));
        }
        let mut destinos: Vec<&mut [f32]> = trabajos
            .iter_mut()
            .map(|(_, _, d)| d.as_mut_slice())
            .collect();
        ops::matmul_varias_una_fila(x, &pesos, &mut destinos)?;
        for (_, sesgo, destino) in trabajos.iter_mut() {
            if let Some(b) = sesgo {
                for (d, s) in destino.iter_mut().zip(b.iter()) {
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
        // Los senos y cosenos de RoPE de las posiciones de esta tanda, una vez
        // para todas las capas (ver `ops::frecuencias_rope_desde`).
        let (cos, sin) = ops::frecuencias_rope_desde(dim_cabeza, cfg.rope, cache.largo, t);

        // Buffers reusados por todas las capas: aca si vale la pena, porque
        // son 36 capas y no una llamada suelta.
        let mut xn = vec![0f32; t * cfg.dim];
        let mut peso = Vec::new();
        let (mut sesgo_q, mut sesgo_k, mut sesgo_v) = (Vec::new(), Vec::new(), Vec::new());
        let (mut q, mut kk, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let mut att = vec![0f32; t * cfg.dim_q()];
        let mut proy = Vec::new();
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut tramo_silu = vec![0f32; TRAMO_SILU];
        // El caché de la capa en f32, para la atención.
        let (mut k32, mut v32) = (Vec::new(), Vec::new());

        for (i, capa) in self.capas.iter().enumerate() {
            // ── atencion ────────────────────────────────────────────────────
            self.vector(&capa.attn_norm, &mut peso)?;
            for (dst, src) in xn.chunks_mut(cfg.dim).zip(x.chunks(cfg.dim)) {
                crate::kernels::rmsnorm(dst, src, &peso, cfg.eps);
            }

            let hay_q = self.vector_si_hay(capa.q_bias.as_ref(), &mut sesgo_q)?;
            let hay_k = self.vector_si_hay(capa.k_bias.as_ref(), &mut sesgo_k)?;
            let hay_v = self.vector_si_hay(capa.v_bias.as_ref(), &mut sesgo_v)?;
            self.proyectar_varias(
                &xn,
                t,
                &mut [
                    (&capa.q, hay_q.then_some(&sesgo_q), &mut q),
                    (&capa.k, hay_k.then_some(&sesgo_k), &mut kk),
                    (&capa.v, hay_v.then_some(&sesgo_v), &mut vv),
                ],
            )?;
            self.norma_por_cabeza(&mut q, capa.q_norm.as_ref(), &mut peso)?;
            self.norma_por_cabeza(&mut kk, capa.k_norm.as_ref(), &mut peso)?;

            ops::rope(&mut q, (t, cfg.cabezas, dim_cabeza), &cos, &sin, 0)?;
            ops::rope(&mut kk, (t, cfg.cabezas_kv, dim_cabeza), &cos, &sin, 0)?;

            crate::kernels::sgemm::f32_a_f16_agregar(&kk, &mut cache.k[i]);
            crate::kernels::sgemm::f32_a_f16_agregar(&vv, &mut cache.v[i]);

            let dims = (t, total, cfg.cabezas, cfg.cabezas_kv, dim_cabeza);
            if t < ATENCION_F32_DESDE {
                // El decode: se lee el caché en f16 directo.
                ops::atencion_f16(&q, &cache.k[i], &cache.v[i], dims, &mut att)?;
            } else {
                // El prefill: cada clave se lee una vez por token nuevo, así
                // que conviene pasar la capa a f32 una sola vez.
                k32.resize(cache.k[i].len(), 0.0);
                v32.resize(cache.v[i].len(), 0.0);
                crate::kernels::sgemm::f16_a_f32(&cache.k[i], &mut k32);
                crate::kernels::sgemm::f16_a_f32(&cache.v[i], &mut v32);
                ops::atencion(&q, &k32, &v32, dims, &mut att)?;
            }

            self.proyectar(&att, t, &capa.salida, None, &mut proy)?;
            for (a, b) in x.iter_mut().zip(&proy) {
                *a += b;
            }

            // ── FFN ─────────────────────────────────────────────────────────
            self.vector(&capa.ffn_norm, &mut peso)?;
            for (dst, src) in xn.chunks_mut(cfg.dim).zip(x.chunks(cfg.dim)) {
                crate::kernels::rmsnorm(dst, src, &peso, cfg.eps);
            }

            self.proyectar_varias(
                &xn,
                t,
                &mut [(&capa.gate, None, &mut gate), (&capa.up, None, &mut up)],
            )?;
            // SwiGLU: silu(gate) * up, en su sitio sobre `gate`. Antes iba a un
            // búfer nuevo de `t × ffn` por capa: 14,7 MB con 333 tokens en el
            // 3B, pedidos 36 veces por forward.
            for (g, u) in gate.chunks_mut(TRAMO_SILU).zip(up.chunks(TRAMO_SILU)) {
                tramo_silu[..g.len()].copy_from_slice(g);
                crate::kernels::silu(g, &tramo_silu[..g.len()]);
                for (a, b) in g.iter_mut().zip(u) {
                    *a *= b;
                }
            }
            self.proyectar(&gate, t, &capa.down, None, &mut proy)?;
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
    // Claves y valores, en f16.
    2 * cfg.capas * cfg.dim_kv() * 2
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

#[cfg(test)]
mod tests {
    use super::*;

    /// El prefijo es una copia: los primeros `n` tokens de cada capa, sin
    /// tocar el original, y lo mismo que truncar.
    #[test]
    fn el_prefijo_copia_los_primeros_tokens_de_cada_capa() {
        // Dos capas, 4 tokens de 3 valores cada uno.
        let mut c = Cache {
            k: vec![(0..12).collect(); 2],
            v: vec![(100..112).collect(); 2],
            largo: 4,
        };
        let p = c.prefijo(2);
        assert_eq!(p.largo(), 2);
        assert_eq!(p.k[1], [0, 1, 2, 3, 4, 5]);
        assert_eq!(p.v[0], [100, 101, 102, 103, 104, 105]);
        assert_eq!(c.largo(), 4, "el original no se toca");
        assert_eq!(c.prefijo(10).largo(), 4, "pedir de más no rompe");
        c.truncar(2);
        assert_eq!((c.k, c.v), (p.k, p.v));
    }
}
