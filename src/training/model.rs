//! Qwen2 en f32, entrenable con LoRA, construido descuantizando un GGUF.
//!
//! La base se descuantiza a f32 y queda **congelada**: lo único que entrena son
//! los adaptadores de `q` y `v`. El backward ([`super::transformer`]) no calcula
//! nada que no vaya a usarse: ni el gradiente de los pesos congelados ni el de
//! las tablas de vocabulario, que en el 0,5B serían 545 MB por secuencia.
//!
//! En f32 el 0,5B son ~2 GB y entra en una máquina común; el 3B serían ~12 GB.
//! Entrenar sobre la base cuantizada pide multiplicar por `W` sin transponer
//! con los kernels cuantizados, y está anotado en el ROADMAP.
//!
//! No hay caché de atención: en entrenamiento se procesa la secuencia entera de
//! una vez y cada paso empieza de cero.

use std::path::Path;

use super::transformer::{AdamW, Capa, Forma, Lineal, Lora, Params, Tablas, Transformer};
use crate::daiss::Azar;
use crate::gguf::{Gguf, Pesos};

type Resultado<T> = Result<T, String>;

/// Lo que define la forma del modelo. Sale del GGUF, no de constantes.
#[derive(Debug, Clone)]
pub struct Config {
    pub oculto: usize,
    pub intermedio: usize,
    pub capas: usize,
    pub cabezas: usize,
    pub cabezas_kv: usize,
    pub dim_cabeza: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub vocabulario: usize,
}

pub struct ModeloEntrenable {
    cfg: Config,
    t: Transformer,
    embeddings: usize,
    salida: usize,
}

impl ModeloEntrenable {
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Carga el GGUF y lo descuantiza entero a f32.
    ///
    /// `r` y `alfa` son los del LoRA. Los adaptadores van sobre **q y v**, que
    /// es la elección habitual: son las proyecciones donde un adaptador rinde
    /// más por parámetro. `A` arranca con ruido y `B` en cero, así que al
    /// empezar el modelo da exactamente lo mismo que la base.
    pub fn cargar(ruta: &Path, r: usize, alfa: f32) -> Resultado<Self> {
        let g = Gguf::abrir(ruta)?;
        let pesos = Pesos::abrir(&g, ruta)?;
        let entero = |k: &str| -> Resultado<usize> {
            g.meta_entero(k)
                .map(|v| v as usize)
                .ok_or_else(|| format!("falta {k} en el gguf"))
        };
        let oculto = entero("qwen2.embedding_length")?;
        let cabezas = entero("qwen2.attention.head_count")?;
        let mut cfg = Config {
            oculto,
            intermedio: entero("qwen2.feed_forward_length")?,
            capas: entero("qwen2.block_count")?,
            cabezas,
            cabezas_kv: entero("qwen2.attention.head_count_kv")?,
            dim_cabeza: oculto / cabezas,
            eps: g
                .meta_decimal("qwen2.attention.layer_norm_rms_epsilon")
                .unwrap_or(1e-6) as f32,
            rope_base: g.meta_decimal("qwen2.rope.freq_base").unwrap_or(10000.0) as f32,
            vocabulario: 0,
        };

        let mut p = Params::default();
        // Descuantiza un tensor del GGUF a un parámetro congelado; `None` si no está.
        let den = |p: &mut Params, nombre: &str| -> Resultado<Option<(usize, usize, usize)>> {
            let Some(info) = g.tensor(nombre) else {
                return Ok(None);
            };
            let mut v = vec![0f32; info.elementos()];
            crate::quant::descuantizar(info.tipo, pesos.de(info)?, &mut v)?;
            let ent = info.dims[0] as usize;
            let sal = info.elementos() / ent;
            let forma = if info.dims.len() == 1 {
                vec![ent]
            } else {
                vec![sal, ent]
            };
            Ok(Some((p.agregar(nombre, forma, v, false), ent, sal)))
        };
        let falta = |n: &str| format!("el gguf no trae {n}");

        let (embeddings, _, vocab) =
            den(&mut p, "token_embd.weight")?.ok_or_else(|| falta("token_embd.weight"))?;
        cfg.vocabulario = vocab;
        // El 0,5B ata los embeddings a la salida y no trae `output.weight`.
        let salida = den(&mut p, "output.weight")?.map_or(embeddings, |(i, _, _)| i);
        let norma_final = den(&mut p, "output_norm.weight")?
            .ok_or_else(|| falta("output_norm.weight"))?
            .0;

        let mut azar = Azar::con_semilla(0x10_8A);
        let mut capas = Vec::with_capacity(cfg.capas);
        for i in 0..cfg.capas {
            let pre = format!("blk.{i}");
            let mut lin = |p: &mut Params, n: &str, con_lora: bool| -> Resultado<Lineal> {
                let nombre = format!("{pre}.{n}.weight");
                let (w, ent, sal) = den(p, &nombre)?.ok_or_else(|| falta(&nombre))?;
                let b = den(p, &format!("{pre}.{n}.bias"))?.map(|(b, _, _)| b);
                let lora = con_lora.then(|| {
                    let a: Vec<f32> = (0..r * ent).map(|_| azar.normales(0.02).0).collect();
                    Lora {
                        a: p.agregar(format!("{pre}.{n}.lora_a"), vec![r, ent], a, true),
                        b: p.agregar(
                            format!("{pre}.{n}.lora_b"),
                            vec![sal, r],
                            vec![0.0; sal * r],
                            true,
                        ),
                        r,
                        escala: alfa / r as f32,
                    }
                });
                Ok(Lineal {
                    w,
                    b,
                    lora,
                    ent,
                    sal,
                })
            };
            let q = lin(&mut p, "attn_q", true)?;
            let k = lin(&mut p, "attn_k", false)?;
            let v = lin(&mut p, "attn_v", true)?;
            let o = lin(&mut p, "attn_output", false)?;
            let gate = lin(&mut p, "ffn_gate", false)?;
            let up = lin(&mut p, "ffn_up", false)?;
            let down = lin(&mut p, "ffn_down", false)?;
            let norma = |p: &mut Params, n: &str| {
                let nombre = format!("{pre}.{n}.weight");
                den(p, &nombre)?.map(|x| x.0).ok_or_else(|| falta(&nombre))
            };
            let norma_atencion = norma(&mut p, "attn_norm")?;
            let norma_ffn = norma(&mut p, "ffn_norm")?;
            capas.push(Capa {
                norma_atencion,
                q,
                k,
                v,
                o,
                norma_ffn,
                gate,
                up,
                down,
            });
        }
        let forma = Forma {
            dim: cfg.oculto,
            ffn: cfg.intermedio,
            capas: cfg.capas,
            cabezas: cfg.cabezas,
            cabezas_kv: cfg.cabezas_kv,
            vocab: cfg.vocabulario,
            eps: cfg.eps,
            rope: cfg.rope_base,
        };
        let t = Transformer::nuevo(forma, p, capas, norma_final, 8192);
        Ok(Self {
            cfg,
            t,
            embeddings,
            salida,
        })
    }

    fn tablas(&self) -> Tablas<'_> {
        let v = &self.t.params.valores;
        Tablas {
            entrada: &v[self.embeddings],
            salida: &v[self.salida],
            gradientes: false,
        }
    }

    /// Los parámetros, para guardar o cargar los adaptadores ([`super::lora`]).
    pub fn params(&self) -> &Params {
        &self.t.params
    }

    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.t.params
    }

    /// Cuántos valores entrenan, y en cuántas matrices.
    pub fn entrenables(&self) -> (usize, usize) {
        let p = &self.t.params;
        (p.entrenables(), p.entrena.iter().filter(|e| **e).count())
    }

    /// Los logits de una secuencia: `t × vocab`.
    pub fn logits(&self, ids: &[u32]) -> Vec<f32> {
        self.t.logits(ids, &self.tablas())
    }

    pub fn perdida(&self, lote: &[Vec<u32>]) -> f32 {
        self.t.perdida(lote, &self.tablas())
    }

    /// La pérdida media y los gradientes de los adaptadores.
    pub fn perdida_y_gradientes(&self, lote: &[Vec<u32>]) -> (f32, Vec<Vec<f32>>) {
        let (l, g) = self.t.perdida_y_gradientes(lote, &self.tablas());
        (l, g.params)
    }

    pub fn aplicar(&mut self, opt: &mut AdamW, grads: &[Vec<f32>]) {
        opt.paso(&mut self.t.params.valores, grads);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::full::ModeloCompleto;

    /// Un daiss chico exportado con nombres de Qwen2, para cargarlo como base.
    fn base(nombre: &str) -> (ModeloCompleto, std::path::PathBuf) {
        let cfg = crate::daiss::Config {
            nombre: "daiss-prueba",
            dim: 32,
            capas: 2,
            cabezas: 4,
            cabezas_kv: 2,
            ffn: 64,
            vocab: 40,
            contexto: 64,
            eps: 1e-6,
            rope: 10000.0,
        };
        let m = ModeloCompleto::nuevo(&cfg, 9, 16);
        let ruta =
            std::env::temp_dir().join(format!("cosxamix-{nombre}-{}.gguf", std::process::id()));
        m.exportar(crate::gguf::TipoGgml::F32, None, &ruta).unwrap();
        (m, ruta)
    }

    /// Lo que hace que un LoRA se pueda enchufar sin romper nada: con `B` en
    /// cero, la salida es exactamente la de la base.
    #[test]
    fn al_empezar_no_cambia_la_salida() {
        let (m, ruta) = base("lora-igual");
        let l = ModeloEntrenable::cargar(&ruta, 2, 4.0).unwrap();
        std::fs::remove_file(&ruta).unwrap();
        let ids = [3u32, 17, 17, 0, 39, 8];
        assert_eq!(m.logits(&ids), l.logits(&ids));
        // Dos capas × (q y v) × (A y B).
        assert_eq!(l.entrenables().1, 8);
    }

    /// Todo adaptador aprende: `B` recibe gradiente en el primer paso, y `A`
    /// en cuanto `B` deja de ser cero.
    #[test]
    fn todos_los_adaptadores_reciben_gradiente() {
        let (_, ruta) = base("lora-grad");
        let mut l = ModeloEntrenable::cargar(&ruta, 2, 4.0).unwrap();
        std::fs::remove_file(&ruta).unwrap();
        let lote = vec![(0..17).map(|i| (i * i * 7 % 40) as u32).collect::<Vec<_>>()];
        let mut opt = AdamW::nuevo(1e-2);
        let (_, g) = l.perdida_y_gradientes(&lote);
        l.aplicar(&mut opt, &g);
        let (_, g) = l.perdida_y_gradientes(&lote);
        let p = l.params();
        let sin: Vec<&str> = (0..g.len())
            .filter(|&i| p.entrena[i] && g[i].iter().all(|v| *v == 0.0))
            .map(|i| p.nombres[i].as_str())
            .collect();
        assert!(sin.is_empty(), "sin gradiente: {sin:?}");
        assert!(g.iter().zip(&p.entrena).all(|(g, e)| *e || g.is_empty()));
    }
}
