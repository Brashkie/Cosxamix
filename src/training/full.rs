//! Entrenar un modelo desde cero: todos los parámetros, no un adaptador.
//!
//! El LoRA de `model.rs` ajusta el 0,1% de un modelo que ya sabe. Esto es lo otro:
//! arranca de los pesos al azar de un daiss y aprende de un corpus. El bloque es
//! el mismo que sirve el motor —RMSNorm, atención agrupada, SwiGLU y RoPE—, y
//! [`ModeloCompleto::exportar`] lo escribe en GGUF con los mismos nombres, así que
//! lo que sale de acá lo corre `forward.rs`. Hay un test que exige que los dos den
//! los mismos logits.
//!
//! Por lote: `(b, t)`, con todas las secuencias del mismo largo.

use std::path::Path;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::ops::softmax;

use crate::daiss::{Azar, Config};
use crate::gguf::TipoGgml;
use crate::tokenizer::Bpe;

/// Una proyección lineal: `y = x * Wᵀ + sesgo`, con `W` de `(salidas, entradas)`,
/// que es como las guarda el GGUF.
struct Lineal {
    nombre: String,
    peso: Var,
    sesgo: Option<(String, Var)>,
}

impl Lineal {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.peso.as_tensor().t()?)?;
        match &self.sesgo {
            Some((_, s)) => y.broadcast_add(s.as_tensor()),
            None => Ok(y),
        }
    }
}

struct Capa {
    norm_attn: (String, Var),
    q: Lineal,
    k: Lineal,
    v: Lineal,
    o: Lineal,
    norm_ffn: (String, Var),
    gate: Lineal,
    up: Lineal,
    down: Lineal,
}

/// Una tabla de vocabulario `(V, D)` guardada como una cadena de Tensor-Train:
/// cuatro núcleos `(r₋, mₖ·nₖ, r₊)`, con `V = m₁m₂m₃m₄` y `D = n₁n₂n₃n₄`. Es la
/// misma forma que mide `examples/measure_tensor_train.rs`, pero entrenada desde
/// el principio en vez de cortada después.
struct TablaTt {
    nucleos: Vec<Var>,
    filas: [usize; 4],
    columnas: [usize; 4],
}

/// `n = 2ᵏ` en cuatro factores potencia de dos, lo más parejos posible, con los
/// grandes al final: 8.192 da `8·8·8·16` y 256, `4·4·4·4`.
fn cuatro_factores(n: usize) -> Result<[usize; 4]> {
    if !n.is_power_of_two() || n < 16 {
        return Err(candle_core::Error::Msg(format!(
            "Tensor-Train pide una potencia de dos de al menos 16, y es {n}"
        )));
    }
    let k = n.trailing_zeros() as usize;
    let mut f = [1usize << (k / 4); 4];
    for x in f.iter_mut().skip(4 - k % 4) {
        *x *= 2;
    }
    Ok(f)
}

impl TablaTt {
    /// Núcleos al azar, con la escala que deja cada valor de la tabla con desvío
    /// `sigma`: un valor es una suma de `r₁r₂r₃` productos de cuatro factores.
    fn nueva(
        v: usize,
        d: usize,
        rango: usize,
        sigma: f32,
        azar: &mut Azar,
        dev: &Device,
    ) -> Result<Self> {
        let (filas, columnas) = (cuatro_factores(v)?, cuatro_factores(d)?);
        let s: Vec<usize> = (0..4).map(|k| filas[k] * columnas[k]).collect();
        let mut rangos = [1usize; 5];
        for k in 1..4 {
            let izquierda: usize = s[..k].iter().product();
            let derecha: usize = s[k..].iter().product();
            rangos[k] = rango.min(izquierda).min(derecha);
        }
        let caminos = (rangos[1] * rangos[2] * rangos[3]) as f32;
        let escala = (sigma * sigma / caminos).powf(1.0 / 8.0);
        let mut nucleos = Vec::with_capacity(4);
        for k in 0..4 {
            let plano = normal(azar, rangos[k] * s[k], rangos[k + 1], escala, dev)?;
            let forma = (rangos[k], s[k], rangos[k + 1]);
            nucleos.push(Var::from_tensor(&plano.as_tensor().reshape(forma)?)?);
        }
        Ok(Self {
            nucleos,
            filas,
            columnas,
        })
    }

    /// La tabla densa: contraer la cadena y reordenar los índices. Es
    /// diferenciable, así que el gradiente llega a los núcleos.
    fn densa(&self) -> Result<Tensor> {
        let (_, s0, r1) = self.nucleos[0].dims3()?;
        let mut x = self.nucleos[0].as_tensor().reshape((s0, r1))?;
        let mut p = s0;
        for g in &self.nucleos[1..] {
            let (r, sk, rn) = g.dims3()?;
            x = x
                .matmul(&g.as_tensor().reshape((r, sk * rn))?)?
                .reshape((p * sk, rn))?;
            p *= sk;
        }
        let (m, n) = (self.filas, self.columnas);
        let v: usize = m.iter().product();
        let d: usize = n.iter().product();
        x.reshape(vec![m[0], n[0], m[1], n[1], m[2], n[2], m[3], n[3]])?
            .permute(vec![0, 2, 4, 6, 1, 3, 5, 7])?
            .contiguous()?
            .reshape((v, d))
    }
}

/// Un daiss entrenable.
pub struct ModeloCompleto {
    cfg: Config,
    semilla: u64,
    embeddings: Var,
    capas: Vec<Capa>,
    norm_final: Var,
    salida: Var,
    /// Las dos tablas de vocabulario en Tensor-Train, si se pidió
    /// ([`Self::con_tensor_train`]). Entonces lo que entrena son los núcleos, y
    /// `embeddings` y `salida` solo se llenan para exportar.
    tt: Option<(TablaTt, TablaTt)>,
    cos: Tensor,
    sin: Tensor,
    largo_maximo: usize,
}

/// `normal(0, sigma)` con semilla, fila por fila, como `daiss::escribir`.
fn normal(azar: &mut Azar, filas: usize, ancho: usize, sigma: f32, dev: &Device) -> Result<Var> {
    let n = filas * ancho;
    let mut v = Vec::with_capacity(n + 1);
    while v.len() < n {
        let (a, b) = azar.normales(sigma);
        v.push(a);
        v.push(b);
    }
    v.truncate(n);
    Var::from_tensor(&Tensor::from_vec(v, (filas, ancho), dev)?)
}

fn rms(x: &Tensor, peso: &Var, eps: f64) -> Result<Tensor> {
    let media = x.sqr()?.mean_keepdim(x.rank() - 1)?;
    x.broadcast_div(&(media + eps)?.sqrt()?)?
        .broadcast_mul(peso.as_tensor())
}

impl ModeloCompleto {
    /// Inicializa como `daiss::escribir`: `0,02`, y las proyecciones que escriben
    /// en el residual divididas por `sqrt(2 * capas)`.
    pub fn nuevo(cfg: &Config, semilla: u64, largo_maximo: usize, dev: &Device) -> Result<Self> {
        let mut azar = Azar::con_semilla(semilla);
        let (d, kv, ffn, voc) = (cfg.dim, cfg.dim_kv(), cfg.ffn, cfg.vocab);
        let residual = 0.02 / (2.0 * cfg.capas as f32).sqrt();
        let ceros = |n: usize| -> Result<Var> { Var::zeros(n, DType::F32, dev) };
        let unos =
            |n: usize| -> Result<Var> { Var::from_tensor(&Tensor::ones(n, DType::F32, dev)?) };

        let embeddings = normal(&mut azar, voc, d, 0.02, dev)?;
        let mut capas = Vec::with_capacity(cfg.capas);
        for i in 0..cfg.capas {
            let p = format!("blk.{i}");
            let mut lineal = |nombre: &str, sal: usize, ent: usize, sigma: f32, sesgo: bool| {
                let peso = normal(&mut azar, sal, ent, sigma, dev)?;
                let sesgo = if sesgo {
                    Some((format!("{p}.{nombre}.bias"), ceros(sal)?))
                } else {
                    None
                };
                Ok::<_, candle_core::Error>(Lineal {
                    nombre: format!("{p}.{nombre}.weight"),
                    peso,
                    sesgo,
                })
            };
            let q = lineal("attn_q", d, d, 0.02, true)?;
            let k = lineal("attn_k", kv, d, 0.02, true)?;
            let v = lineal("attn_v", kv, d, 0.02, true)?;
            let o = lineal("attn_output", d, d, residual, false)?;
            let gate = lineal("ffn_gate", ffn, d, 0.02, false)?;
            let up = lineal("ffn_up", ffn, d, 0.02, false)?;
            let down = lineal("ffn_down", d, ffn, residual, false)?;
            capas.push(Capa {
                norm_attn: (format!("{p}.attn_norm.weight"), unos(d)?),
                q,
                k,
                v,
                o,
                norm_ffn: (format!("{p}.ffn_norm.weight"), unos(d)?),
                gate,
                up,
                down,
            });
        }
        let norm_final = unos(d)?;
        let salida = normal(&mut azar, voc, d, 0.02, dev)?;
        let (cos, sin) = frecuencias(cfg.dim_cabeza(), cfg.rope, largo_maximo, dev)?;
        Ok(Self {
            cfg: *cfg,
            semilla,
            embeddings,
            capas,
            norm_final,
            salida,
            tt: None,
            cos,
            sin,
            largo_maximo,
        })
    }

    /// Lo mismo, con las tablas de vocabulario en Tensor-Train de rango `rango`.
    /// El vocabulario y el ancho tienen que ser potencias de dos.
    pub fn con_tensor_train(
        cfg: &Config,
        semilla: u64,
        largo_maximo: usize,
        rango: usize,
        dev: &Device,
    ) -> Result<Self> {
        let mut m = Self::nuevo(cfg, semilla, largo_maximo, dev)?;
        let mut azar = Azar::con_semilla(semilla ^ 0x7474);
        let e = TablaTt::nueva(cfg.vocab, cfg.dim, rango, 0.02, &mut azar, dev)?;
        let s = TablaTt::nueva(cfg.vocab, cfg.dim, rango, 0.02, &mut azar, dev)?;
        m.tt = Some((e, s));
        Ok(m)
    }

    /// Escribe las tablas densas desde los núcleos: lo que lee `exportar`.
    fn materializar(&self) -> Result<()> {
        if let Some((e, s)) = &self.tt {
            self.embeddings.set(&e.densa()?)?;
            self.salida.set(&s.densa()?)?;
        }
        Ok(())
    }

    /// Todos los pesos con su nombre en el GGUF.
    fn con_nombre(&self) -> Vec<(&str, &Var)> {
        let mut v: Vec<(&str, &Var)> = vec![
            ("token_embd.weight", &self.embeddings),
            ("output_norm.weight", &self.norm_final),
            ("output.weight", &self.salida),
        ];
        for c in &self.capas {
            v.push((&c.norm_attn.0, &c.norm_attn.1));
            v.push((&c.norm_ffn.0, &c.norm_ffn.1));
            for l in [&c.q, &c.k, &c.v, &c.o, &c.gate, &c.up, &c.down] {
                v.push((&l.nombre, &l.peso));
                if let Some((n, s)) = &l.sesgo {
                    v.push((n, s));
                }
            }
        }
        v
    }

    /// Lo que entrena el optimizador: todo. Con Tensor-Train, los núcleos en
    /// vez de las dos tablas densas.
    pub fn variables(&self) -> Vec<Var> {
        let densas = ["token_embd.weight", "output.weight"];
        let mut v: Vec<Var> = self
            .con_nombre()
            .into_iter()
            .filter(|(n, _)| self.tt.is_none() || !densas.contains(n))
            .map(|(_, v)| v.clone())
            .collect();
        if let Some((e, s)) = &self.tt {
            v.extend(e.nucleos.iter().chain(&s.nucleos).cloned());
        }
        v
    }

    /// Los parámetros que se entrenan.
    pub fn parametros(&self) -> usize {
        self.variables().iter().map(|v| v.elem_count()).sum()
    }

    /// Los logits de un lote `(b, t)`: `(b, t, vocab)`.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        if t > self.largo_maximo {
            return Err(candle_core::Error::Msg(format!(
                "{t} tokens y el máximo es {}",
                self.largo_maximo
            )));
        }
        let dev = ids.device();
        let (embeddings, salida) = match &self.tt {
            Some((e, s)) => (e.densa()?, s.densa()?),
            None => (
                self.embeddings.as_tensor().clone(),
                self.salida.as_tensor().clone(),
            ),
        };
        let mut x =
            embeddings
                .index_select(&ids.flatten_all()?, 0)?
                .reshape((b, t, self.cfg.dim))?;

        let mascara: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mascara = Tensor::from_vec(mascara, (t, t), dev)?;
        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;

        for capa in &self.capas {
            let h = rms(&x, &capa.norm_attn.1, f64::from(self.cfg.eps))?;
            let a = self.atencion(capa, &h, (b, t), &mascara, &cos, &sin)?;
            x = (x + a)?;
            let h = rms(&x, &capa.norm_ffn.1, f64::from(self.cfg.eps))?;
            let g = capa.gate.forward(&h)?.silu()?;
            let u = capa.up.forward(&h)?;
            x = (x + capa.down.forward(&(g * u)?)?)?;
        }
        let x = rms(&x, &self.norm_final, f64::from(self.cfg.eps))?;
        x.broadcast_matmul(&salida.t()?)
    }

    fn atencion(
        &self,
        capa: &Capa,
        h: &Tensor,
        (b, t): (usize, usize),
        mascara: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let dc = c.dim_cabeza();
        let separar = |y: Tensor, cabezas: usize| -> Result<Tensor> {
            y.reshape((b, t, cabezas, dc))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = separar(capa.q.forward(h)?, c.cabezas)?;
        let k = separar(capa.k.forward(h)?, c.cabezas_kv)?;
        let v = separar(capa.v.forward(h)?, c.cabezas_kv)?;
        // `rope_slow` y no `rope`, por lo mismo que `softmax` más abajo: el rápido
        // no tiene camino de vuelta, y `q` y `k` no aprenderían nada.
        let q = candle_nn::rotary_emb::rope_slow(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_slow(&k, cos, sin)?;

        // Atención agrupada: cada cabeza de clave y valor la comparten varias
        // de consulta.
        let rep = c.cabezas / c.cabezas_kv;
        let repetir = |x: Tensor| -> Result<Tensor> {
            x.unsqueeze(2)?
                .expand((b, c.cabezas_kv, rep, t, dc))?
                .reshape((b, c.cabezas, t, dc))?
                .contiguous()
        };
        let (k, v) = (repetir(k)?, repetir(v)?);

        let escala = 1f64 / (dc as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * escala)?;
        // `softmax` y no `softmax_last_dim`: ese es más rápido pero **no tiene
        // camino de vuelta** (`apply_op1_no_bwd`), y corta el gradiente sin avisar:
        // las proyecciones de consulta y clave no aprenden nada.
        let att = softmax(&att.broadcast_add(mascara)?, D::Minus1)?;
        let y = att
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, t, c.cabezas * dc))?;
        capa.o.forward(&y)
    }

    /// La pérdida media de un lote: el objetivo de cada token es el siguiente.
    pub fn perdida(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let entrada = ids.narrow(1, 0, t - 1)?;
        let objetivo = ids.narrow(1, 1, t - 1)?.flatten_all()?;
        let logits = self.forward(&entrada)?;
        let logits = logits.reshape((b * (t - 1), self.cfg.vocab))?;
        candle_nn::loss::cross_entropy(&logits, &objetivo)
    }

    /// La pérdida media y los gradientes de un lote, repartido en `partes` tramos
    /// que corren en paralelo, cada uno con su grafo.
    ///
    /// candle no escala con el lote: medido con 6 o con 12 hilos, 16 secuencias
    /// tardan 2,5 s y una sola, 235 ms. Repartir las secuencias entre los núcleos
    /// y promediar los gradientes es la misma cuenta, y sí escala. Hay un test que
    /// compara el resultado con el del lote entero.
    pub fn perdida_y_gradientes(&self, ids: &Tensor, partes: usize) -> Result<(f32, GradStore)> {
        use rayon::prelude::*;
        let (b, _) = ids.dims2()?;
        let partes = partes.clamp(1, b);
        let tramos: Vec<(usize, usize)> = (0..partes)
            .map(|i| (i * b / partes, (i + 1) * b / partes))
            .filter(|(d, h)| h > d)
            .collect();
        let resultados: Vec<Result<(f32, usize, GradStore)>> = tramos
            .par_iter()
            .map(|(d, h)| {
                let trozo = ids.narrow(0, *d, h - d)?;
                let p = self.perdida(&trozo)?;
                let g = p.backward()?;
                Ok((p.to_scalar::<f32>()?, h - d, g))
            })
            .collect();

        // El promedio de cada gradiente, pesado por cuántas secuencias tenía su
        // tramo: los tramos pueden no medir lo mismo.
        let mut total = 0f32;
        let mut hechos = Vec::with_capacity(resultados.len());
        for r in resultados {
            let (p, n, g) = r?;
            total += p * n as f32 / b as f32;
            hechos.push((n, g));
        }
        let (n0, mut acumulado) = hechos.remove(0);
        let vars = self.variables();
        for v in &vars {
            if let Some(g) = acumulado.remove(v.as_tensor()) {
                let mut suma = (g * (n0 as f64 / b as f64))?;
                for (n, otro) in hechos.iter() {
                    if let Some(o) = otro.get(v.as_tensor()) {
                        suma = (suma + (o * (*n as f64 / b as f64))?)?;
                    }
                }
                acumulado.insert(v.as_tensor(), suma);
            }
        }
        Ok((total, acumulado))
    }

    /// Escribe el modelo en GGUF, en `tipo` lo que se pueda cuantizar: normas y
    /// sesgos quedan en `F32`, como en `daiss::escribir`.
    pub fn exportar(&self, tipo: TipoGgml, tok: Option<&Bpe>, ruta: &Path) -> Result<u64> {
        self.materializar()?;
        let escritor = crate::daiss::escritor_de(&self.cfg, tipo, self.semilla, tok);
        let pesos = self.con_nombre();
        escritor
            .escribir(ruta, |d, w| {
                let var = pesos
                    .iter()
                    .find(|(n, _)| *n == d.nombre)
                    .ok_or_else(|| format!("el modelo no tiene {}", d.nombre))?
                    .1;
                let datos = var
                    .as_tensor()
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| e.to_string())?;
                let ancho = d.dims[0] as usize;
                let mut b = Vec::with_capacity(d.bytes());
                for fila in datos.chunks(ancho) {
                    crate::quant::cuantizar_bloques(d.tipo, fila, &mut b)?;
                }
                w.write_all(&b).map_err(|e| e.to_string())
            })
            .map_err(candle_core::Error::Msg)
    }

    /// El camino de vuelta de [`Self::exportar`]: pisa cada peso con el del GGUF, en
    /// el tipo que esté. El modelo tiene que tener la misma forma.
    pub fn cargar_pesos(&self, ruta: &Path) -> Result<()> {
        let g = crate::gguf::Gguf::abrir(ruta).map_err(candle_core::Error::Msg)?;
        for (nombre, var) in self.con_nombre() {
            let info = g
                .tensor(nombre)
                .ok_or_else(|| candle_core::Error::Msg(format!("el GGUF no trae {nombre}")))?;
            if info.elementos() != var.elem_count() {
                return Err(candle_core::Error::Msg(format!(
                    "{nombre}: {} valores en el GGUF y {} en el modelo",
                    info.elementos(),
                    var.elem_count()
                )));
            }
            let bytes = g.datos_de(ruta, info).map_err(candle_core::Error::Msg)?;
            let mut v = vec![0f32; info.elementos()];
            crate::quant::descuantizar(info.tipo, &bytes, &mut v)
                .map_err(candle_core::Error::Msg)?;
            var.set(&Tensor::from_vec(v, var.shape(), var.device())?)?;
        }
        Ok(())
    }

    /// Un peso por su nombre en el GGUF, para leerlo o pisarlo.
    pub fn peso(&self, nombre: &str) -> Option<&Var> {
        self.con_nombre()
            .into_iter()
            .find(|(n, _)| *n == nombre)
            .map(|(_, v)| v)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    fn chico(vocab: usize) -> Config {
        Config {
            nombre: "daiss-prueba",
            dim: 64,
            capas: 2,
            cabezas: 4,
            cabezas_kv: 2,
            ffn: 128,
            vocab,
            contexto: 64,
            eps: 1e-6,
            rope: 10000.0,
        }
    }

    /// Lo que más importa: lo exportado, en el motor propio, da lo mismo que acá.
    #[test]
    fn el_motor_propio_da_los_mismos_logits() {
        let dev = Device::Cpu;
        let cfg = chico(50);
        let m = ModeloCompleto::nuevo(&cfg, 3, 32, &dev).unwrap();
        let ruta = std::env::temp_dir().join(format!("cosxamix-full-{}.gguf", std::process::id()));
        m.exportar(TipoGgml::F32, None, &ruta).unwrap();

        let tokens = [7u32, 3, 49, 0, 21, 21, 5];
        let ids = Tensor::new(&tokens[..], &dev)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let aca = m
            .forward(&ids)
            .unwrap()
            .narrow(1, tokens.len() - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let motor = crate::forward::Modelo::abrir(&ruta).unwrap();
        let mut cache = crate::forward::Cache::nuevo(&motor.cfg);
        let alla = motor.forward(&tokens, &mut cache).unwrap();
        std::fs::remove_file(&ruta).unwrap();

        // `forward` devuelve la última fila o todas: se comparan los últimos.
        let alla = &alla[alla.len() - cfg.vocab..];
        for (i, (a, b)) in aca.iter().zip(alla).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit {i}: {a} contra {b}");
        }
    }

    /// Con Tensor-Train: los factores salen como deben, todo núcleo recibe
    /// gradiente, la pérdida baja, y lo exportado da lo mismo en el motor.
    #[test]
    fn tensor_train_entrena_y_se_exporta() {
        assert_eq!(cuatro_factores(8192).unwrap(), [8, 8, 8, 16]);
        assert_eq!(cuatro_factores(256).unwrap(), [4, 4, 4, 4]);
        assert_eq!(cuatro_factores(64).unwrap(), [2, 2, 4, 4]);
        assert!(cuatro_factores(50).is_err());

        let dev = Device::Cpu;
        let cfg = chico(64);
        let m = ModeloCompleto::con_tensor_train(&cfg, 4, 16, 4, &dev).unwrap();
        let densos = ModeloCompleto::nuevo(&cfg, 4, 16, &dev).unwrap();
        assert!(m.parametros() < densos.parametros());

        let datos: Vec<u32> = (0..4 * 16).map(|i| (i * i * 7 % 64) as u32).collect();
        let ids = Tensor::from_vec(datos, (4, 16), &dev).unwrap();
        let g = m.perdida(&ids).unwrap().backward().unwrap();
        let sin = m
            .variables()
            .iter()
            .filter(|v| g.get(v.as_tensor()).is_none())
            .count();
        assert_eq!(sin, 0, "{sin} pesos sin gradiente");

        let mut opt = AdamW::new(
            m.variables(),
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();
        let antes = m.perdida(&ids).unwrap().to_scalar::<f32>().unwrap();
        for _ in 0..30 {
            let p = m.perdida(&ids).unwrap();
            opt.backward_step(&p).unwrap();
        }
        let despues = m.perdida(&ids).unwrap().to_scalar::<f32>().unwrap();
        assert!(despues < antes * 0.8, "{antes} -> {despues}");

        // La tabla densa que se exporta es la de los núcleos.
        let ruta = std::env::temp_dir().join(format!("cosxamix-tt-{}.gguf", std::process::id()));
        m.exportar(TipoGgml::F32, None, &ruta).unwrap();
        let otro = ModeloCompleto::nuevo(&cfg, 99, 16, &dev).unwrap();
        otro.cargar_pesos(&ruta).unwrap();
        std::fs::remove_file(&ruta).unwrap();
        let a = m.forward(&ids).unwrap().flatten_all().unwrap();
        let b = otro.forward(&ids).unwrap().flatten_all().unwrap();
        let d = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-4, "difieren en {d}");
    }

    /// Exportar y volver a cargar en otro modelo da los mismos logits.
    #[test]
    fn cargar_pesos_es_la_vuelta_de_exportar() {
        let dev = Device::Cpu;
        let cfg = chico(30);
        let a = ModeloCompleto::nuevo(&cfg, 5, 16, &dev).unwrap();
        let b = ModeloCompleto::nuevo(&cfg, 6, 16, &dev).unwrap();
        let ruta =
            std::env::temp_dir().join(format!("cosxamix-vuelta-{}.gguf", std::process::id()));
        a.exportar(TipoGgml::F32, None, &ruta).unwrap();
        b.cargar_pesos(&ruta).unwrap();
        std::fs::remove_file(&ruta).unwrap();

        let ids = Tensor::new(&[[1u32, 29, 4, 4, 17]], &dev).unwrap();
        let la = a.forward(&ids).unwrap().flatten_all().unwrap();
        let lb = b.forward(&ids).unwrap().flatten_all().unwrap();
        let d = (la - lb)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(d, 0.0);
        assert!(b.peso("output.weight").is_some());
        assert!(b.peso("no.existe").is_none());
    }

    /// Todo peso tiene que recibir gradiente: uno que no lo reciba no aprende nada,
    /// y el modelo sigue andando sin que se note.
    #[test]
    fn todos_los_pesos_reciben_gradiente() {
        let dev = Device::Cpu;
        let cfg = chico(40);
        let m = ModeloCompleto::nuevo(&cfg, 9, 16, &dev).unwrap();
        let datos: Vec<u32> = (0..4 * 16).map(|i| (i * i * 7 % 40) as u32).collect();
        let ids = Tensor::from_vec(datos, (4, 16), &dev).unwrap();
        let g = m.perdida(&ids).unwrap().backward().unwrap();
        let faltan: Vec<&str> = m
            .con_nombre()
            .into_iter()
            .filter(|(_, v)| g.get(v.as_tensor()).is_none())
            .map(|(n, _)| n)
            .collect();
        assert!(faltan.is_empty(), "sin gradiente: {faltan:?}");
    }

    /// El LoRA tampoco puede perder gradiente: hasta ahora el adaptador de `q` no
    /// aprendía nada, por el mismo `softmax_last_dim`.
    #[test]
    fn todos_los_adaptadores_del_lora_reciben_gradiente() {
        let dev = Device::Cpu;
        let cfg = chico(40);
        let m = ModeloCompleto::nuevo(&cfg, 9, 16, &dev).unwrap();
        let ruta = std::env::temp_dir().join(format!("cosxamix-lora-{}.gguf", std::process::id()));
        m.exportar(TipoGgml::F32, None, &ruta).unwrap();
        let lora = crate::training::ModeloEntrenable::cargar(&ruta, 2, 4.0, &dev).unwrap();
        std::fs::remove_file(&ruta).unwrap();

        let datos: Vec<u32> = (0..17).map(|i| (i * i * 7 % 40) as u32).collect();
        let entrada = Tensor::new(&datos[..16], &dev).unwrap();
        let objetivo = Tensor::new(&datos[1..], &dev).unwrap();
        let logits = lora.forward(&entrada).unwrap();
        let p = candle_nn::loss::cross_entropy(&logits, &objetivo).unwrap();
        let g = p.backward().unwrap();
        let vars = lora.variables();
        let sin = vars
            .iter()
            .filter(|v| g.get(v.as_tensor()).is_none())
            .count();
        assert_eq!(sin, 0, "{sin} de {} matrices sin gradiente", vars.len());
    }

    /// Repartir el lote entre los núcleos da la misma pérdida y los mismos
    /// gradientes que calcularlo entero.
    #[test]
    fn repartir_el_lote_da_el_mismo_gradiente() {
        let dev = Device::Cpu;
        let cfg = chico(40);
        let m = ModeloCompleto::nuevo(&cfg, 9, 16, &dev).unwrap();
        let datos: Vec<u32> = (0..7 * 16).map(|i| (i * i * 7 % 40) as u32).collect();
        let ids = Tensor::from_vec(datos, (7, 16), &dev).unwrap();

        let (p1, g1) = m.perdida_y_gradientes(&ids, 1).unwrap();
        for partes in [2, 3, 7, 50] {
            let (p, g) = m.perdida_y_gradientes(&ids, partes).unwrap();
            assert!((p - p1).abs() < 1e-4, "{partes} tramos: {p} contra {p1}");
            for v in m.variables() {
                let (a, b) = (
                    g1.get(v.as_tensor()).unwrap(),
                    g.get(v.as_tensor()).unwrap(),
                );
                let d = (a - b).unwrap().abs().unwrap().max_all().unwrap();
                let d = d.to_scalar::<f32>().unwrap();
                assert!(d < 1e-3, "{partes} tramos: el gradiente difiere en {d}");
            }
        }
    }

    /// Y aprende: con un texto que se repite, la pérdida tiene que bajar mucho.
    #[test]
    fn la_perdida_baja() {
        let dev = Device::Cpu;
        let cfg = chico(16);
        let m = ModeloCompleto::nuevo(&cfg, 5, 32, &dev).unwrap();
        let datos: Vec<u32> = (0..32).map(|i| (i % 4) as u32).collect();
        let lote: Vec<u32> = datos.iter().cycle().take(32 * 4).copied().collect();
        let ids = Tensor::from_vec(lote, (4, 32), &dev).unwrap();

        let mut opt = AdamW::new(
            m.variables(),
            ParamsAdamW {
                lr: 3e-3,
                ..Default::default()
            },
        )
        .unwrap();
        let inicial = m.perdida(&ids).unwrap().to_scalar::<f32>().unwrap();
        for _ in 0..40 {
            let p = m.perdida(&ids).unwrap();
            opt.backward_step(&p).unwrap();
        }
        let fin = m.perdida(&ids).unwrap().to_scalar::<f32>().unwrap();
        assert!(fin < inicial * 0.2, "de {inicial} a {fin}");
    }
}
