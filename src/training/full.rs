//! Entrenar un modelo desde cero: todos los parámetros, no un adaptador.
//!
//! El LoRA de `model.rs` ajusta el 0,1% de un modelo que ya sabe. Esto es lo otro:
//! arranca de los pesos al azar de un daiss y aprende de un corpus. El bloque es
//! el mismo que sirve el motor —RMSNorm, atención agrupada, SwiGLU y RoPE—, y
//! [`ModeloCompleto::exportar`] lo escribe en GGUF con los mismos nombres, así que
//! lo que sale de acá lo corre `forward.rs`. Hay un test que exige que los dos den
//! los mismos logits.
//!
//! El forward y el backward son los de [`super::transformer`]; acá están la
//! inicialización de daiss, las tablas de vocabulario en Tensor-Train (con su
//! propio backward) y la ida y vuelta al GGUF.

use std::path::Path;

use super::transformer::{AdamW, Capa, Forma, Lineal, Params, Tablas, Transformer};
use crate::daiss::{Azar, Config};
use crate::gguf::TipoGgml;
use crate::kernels::sgemm::{sgemm, Mat};
use crate::tokenizer::Bpe;

type Resultado<T> = Result<T, String>;

/// `n = 2ᵏ` en cuatro factores potencia de dos, lo más parejos posible, con los
/// grandes al final: 8.192 da `8·8·8·16` y 256, `4·4·4·4`.
fn cuatro_factores(n: usize) -> Resultado<[usize; 4]> {
    if !n.is_power_of_two() || n < 16 {
        return Err(format!(
            "Tensor-Train pide una potencia de dos de al menos 16, y es {n}"
        ));
    }
    let k = n.trailing_zeros() as usize;
    let mut f = [1usize << (k / 4); 4];
    for x in f.iter_mut().skip(4 - k % 4) {
        *x *= 2;
    }
    Ok(f)
}

/// `normal(0, sigma)` con semilla, de a pares, como `daiss::escribir`.
fn normal(azar: &mut Azar, n: usize, sigma: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n + 1);
    while v.len() < n {
        let (a, b) = azar.normales(sigma);
        v.push(a);
        v.push(b);
    }
    v.truncate(n);
    v
}

/// Una tabla de vocabulario `(V, D)` guardada como una cadena de Tensor-Train:
/// cuatro núcleos `(r₋, mₖ·nₖ, r₊)`, con `V = m₁m₂m₃m₄` y `D = n₁n₂n₃n₄`. Es la
/// misma forma que mide `examples/measure_tensor_train.rs`, pero entrenada desde
/// el principio en vez de cortada después.
struct TablaTt {
    /// Los núcleos, como índices en el registro de parámetros.
    nucleos: [usize; 4],
    rangos: [usize; 5],
    s: [usize; 4],
    /// Dónde cae en la tabla densa cada valor de la cadena contraída.
    destino: Vec<u32>,
}

impl TablaTt {
    /// Núcleos al azar, con la escala que deja cada valor de la tabla con desvío
    /// `sigma`: un valor es una suma de `r₁r₂r₃` productos de cuatro factores.
    fn nueva(
        p: &mut Params,
        nombre: &str,
        (v, d): (usize, usize),
        rango: usize,
        sigma: f32,
        azar: &mut Azar,
    ) -> Resultado<Self> {
        let (m, n) = (cuatro_factores(v)?, cuatro_factores(d)?);
        let s: [usize; 4] = std::array::from_fn(|k| m[k] * n[k]);
        let mut rangos = [1usize; 5];
        for k in 1..4 {
            let izquierda: usize = s[..k].iter().product();
            let derecha: usize = s[k..].iter().product();
            rangos[k] = rango.min(izquierda).min(derecha);
        }
        let caminos = (rangos[1] * rangos[2] * rangos[3]) as f32;
        let escala = (sigma * sigma / caminos).powf(1.0 / 8.0);
        let nucleos = std::array::from_fn(|k| {
            let forma = vec![rangos[k], s[k], rangos[k + 1]];
            let valores = normal(azar, forma.iter().product(), escala);
            p.agregar(format!("{nombre}.tt.{k}"), forma, valores, true)
        });
        // El índice de la cadena recorre (m₀n₀)(m₁n₁)(m₂n₂)(m₃n₃); la fila de la
        // tabla es (m₀m₁m₂m₃) y la columna, (n₀n₁n₂n₃).
        let mut destino = Vec::with_capacity(v * d);
        for i in 0..v * d {
            let mut resto = i;
            let mut digitos = [(0, 0); 4];
            for k in (0..4).rev() {
                let sk = resto % s[k];
                resto /= s[k];
                digitos[k] = (sk / n[k], sk % n[k]);
            }
            let (mut fila, mut col) = (0, 0);
            for k in 0..4 {
                fila = fila * m[k] + digitos[k].0;
                col = col * n[k] + digitos[k].1;
            }
            destino.push((fila * d + col) as u32);
        }
        Ok(Self {
            nucleos,
            rangos,
            s,
            destino,
        })
    }

    /// La cadena contraída, paso por paso: `X₀ = G₀` y `Xₖ = Xₖ₋₁ · Gₖ`.
    fn cadena(&self, p: &Params) -> Vec<Vec<f32>> {
        let mut xs = vec![p.valores[self.nucleos[0]].clone()];
        let mut filas = self.s[0];
        for k in 1..4 {
            let (r, sr) = (self.rangos[k], self.s[k] * self.rangos[k + 1]);
            let mut y = vec![0f32; filas * sr];
            let a = &xs[k - 1];
            let g = &p.valores[self.nucleos[k]];
            sgemm(
                (filas, sr, r),
                &Mat {
                    datos: a,
                    fila: r,
                    col: 1,
                },
                &Mat {
                    datos: g,
                    fila: sr,
                    col: 1,
                },
                &mut y,
                sr,
            );
            xs.push(y);
            filas *= self.s[k];
        }
        xs
    }

    fn densa(&self, p: &Params, salida: &mut [f32]) {
        let x = self.cadena(p).pop().expect("cuatro núcleos");
        for (v, &i) in x.iter().zip(&self.destino) {
            salida[i as usize] = *v;
        }
    }

    /// Lleva el gradiente de la tabla densa a los núcleos, por la cadena al revés.
    fn vuelta(&self, p: &Params, d_densa: &[f32], grads: &mut [Vec<f32>]) {
        let xs = self.cadena(p);
        let mut dx: Vec<f32> = self.destino.iter().map(|&i| d_densa[i as usize]).collect();
        let mut filas: usize = self.s.iter().product();
        for k in (1..4).rev() {
            filas /= self.s[k];
            let (r, sr) = (self.rangos[k], self.s[k] * self.rangos[k + 1]);
            let (a, g) = (&xs[k - 1], &p.valores[self.nucleos[k]]);
            // dG = Aᵀ dX y dA = dX Gᵀ
            sgemm(
                (r, sr, filas),
                &Mat {
                    datos: a,
                    fila: 1,
                    col: r,
                },
                &Mat {
                    datos: &dx,
                    fila: sr,
                    col: 1,
                },
                &mut grads[self.nucleos[k]],
                sr,
            );
            let mut da = vec![0f32; filas * r];
            sgemm(
                (filas, r, sr),
                &Mat {
                    datos: &dx,
                    fila: sr,
                    col: 1,
                },
                &Mat {
                    datos: g,
                    fila: 1,
                    col: sr,
                },
                &mut da,
                r,
            );
            dx = da;
        }
        for (a, b) in grads[self.nucleos[0]].iter_mut().zip(&dx) {
            *a += b;
        }
    }
}

/// Un daiss entrenable.
pub struct ModeloCompleto {
    cfg: Config,
    semilla: u64,
    t: Transformer,
    embeddings: usize,
    salida: usize,
    /// Las dos tablas de vocabulario en Tensor-Train, si se pidió
    /// ([`Self::con_tensor_train`]). Entonces lo que entrena son los núcleos, y
    /// las tablas densas se rehacen de ellos después de cada paso.
    tt: Option<(TablaTt, TablaTt)>,
}

impl ModeloCompleto {
    /// Inicializa como `daiss::escribir`: `0,02`, y las proyecciones que escriben
    /// en el residual divididas por `sqrt(2 * capas)`.
    pub fn nuevo(cfg: &Config, semilla: u64, largo_maximo: usize) -> Self {
        let mut azar = Azar::con_semilla(semilla);
        let (d, kv, ffn, voc) = (cfg.dim, cfg.dim_kv(), cfg.ffn, cfg.vocab);
        let residual = 0.02 / (2.0 * cfg.capas as f32).sqrt();
        let mut p = Params::default();

        let embeddings = p.agregar(
            "token_embd.weight",
            vec![voc, d],
            normal(&mut azar, voc * d, 0.02),
            true,
        );
        let mut capas = Vec::with_capacity(cfg.capas);
        for i in 0..cfg.capas {
            let pre = format!("blk.{i}");
            let mut lineal =
                |p: &mut Params, nombre: &str, sal: usize, ent: usize, sigma: f32, sesgo: bool| {
                    let w = p.agregar(
                        format!("{pre}.{nombre}.weight"),
                        vec![sal, ent],
                        normal(&mut azar, sal * ent, sigma),
                        true,
                    );
                    let b = sesgo.then(|| {
                        p.agregar(
                            format!("{pre}.{nombre}.bias"),
                            vec![sal],
                            vec![0.0; sal],
                            true,
                        )
                    });
                    Lineal {
                        w,
                        b,
                        lora: None,
                        ent,
                        sal,
                    }
                };
            let q = lineal(&mut p, "attn_q", d, d, 0.02, true);
            let k = lineal(&mut p, "attn_k", kv, d, 0.02, true);
            let v = lineal(&mut p, "attn_v", kv, d, 0.02, true);
            let o = lineal(&mut p, "attn_output", d, d, residual, false);
            let gate = lineal(&mut p, "ffn_gate", ffn, d, 0.02, false);
            let up = lineal(&mut p, "ffn_up", ffn, d, 0.02, false);
            let down = lineal(&mut p, "ffn_down", d, ffn, residual, false);
            let norma_atencion = p.agregar(
                format!("{pre}.attn_norm.weight"),
                vec![d],
                vec![1.0; d],
                true,
            );
            let norma_ffn = p.agregar(
                format!("{pre}.ffn_norm.weight"),
                vec![d],
                vec![1.0; d],
                true,
            );
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
        let norma_final = p.agregar("output_norm.weight", vec![d], vec![1.0; d], true);
        let salida = p.agregar(
            "output.weight",
            vec![voc, d],
            normal(&mut azar, voc * d, 0.02),
            true,
        );
        let forma = Forma {
            dim: d,
            ffn,
            capas: cfg.capas,
            cabezas: cfg.cabezas,
            cabezas_kv: cfg.cabezas_kv,
            vocab: voc,
            eps: cfg.eps,
            rope: cfg.rope,
        };
        Self {
            cfg: *cfg,
            semilla,
            t: Transformer::nuevo(forma, p, capas, norma_final, largo_maximo),
            embeddings,
            salida,
            tt: None,
        }
    }

    /// Lo mismo, con las tablas de vocabulario en Tensor-Train de rango `rango`.
    /// El vocabulario y el ancho tienen que ser potencias de dos.
    pub fn con_tensor_train(
        cfg: &Config,
        semilla: u64,
        largo_maximo: usize,
        rango: usize,
    ) -> Resultado<Self> {
        let mut m = Self::nuevo(cfg, semilla, largo_maximo);
        let mut azar = Azar::con_semilla(semilla ^ 0x7474);
        let p = &mut m.t.params;
        let e = TablaTt::nueva(
            p,
            "token_embd",
            (cfg.vocab, cfg.dim),
            rango,
            0.02,
            &mut azar,
        )?;
        let s = TablaTt::nueva(p, "output", (cfg.vocab, cfg.dim), rango, 0.02, &mut azar)?;
        p.entrena[m.embeddings] = false;
        p.entrena[m.salida] = false;
        m.tt = Some((e, s));
        m.materializar();
        Ok(m)
    }

    /// Escribe las tablas densas desde los núcleos.
    fn materializar(&mut self) {
        if let Some((e, s)) = &self.tt {
            let p = &mut self.t.params;
            let mut te = std::mem::take(&mut p.valores[self.embeddings]);
            let mut ts = std::mem::take(&mut p.valores[self.salida]);
            e.densa(p, &mut te);
            s.densa(p, &mut ts);
            p.valores[self.embeddings] = te;
            p.valores[self.salida] = ts;
        }
    }

    fn tablas(&self, gradientes: bool) -> Tablas<'_> {
        let v = &self.t.params.valores;
        Tablas {
            entrada: &v[self.embeddings],
            salida: &v[self.salida],
            gradientes,
        }
    }

    /// Los parámetros que se entrenan.
    pub fn parametros(&self) -> usize {
        self.t.params.entrenables()
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Los logits de una secuencia: `t × vocab`.
    pub fn logits(&self, ids: &[u32]) -> Vec<f32> {
        self.t.logits(ids, &self.tablas(false))
    }

    /// La pérdida media de un lote: el objetivo de cada token es el siguiente.
    pub fn perdida(&self, lote: &[Vec<u32>]) -> f32 {
        self.t.perdida(lote, &self.tablas(false))
    }

    /// La pérdida media y los gradientes de un lote, uno por parámetro (vacío
    /// en los que no entrenan). Las secuencias se reparten entre los núcleos.
    pub fn perdida_y_gradientes(&self, lote: &[Vec<u32>]) -> (f32, Vec<Vec<f32>>) {
        let (l, mut g) = self.t.perdida_y_gradientes(lote, &self.tablas(true));
        match &self.tt {
            None => {
                g.params[self.embeddings] = std::mem::take(&mut g.entrada);
                g.params[self.salida] = std::mem::take(&mut g.salida);
            }
            Some((e, s)) => {
                e.vuelta(&self.t.params, &g.entrada, &mut g.params);
                s.vuelta(&self.t.params, &g.salida, &mut g.params);
            }
        }
        (l, g.params)
    }

    /// Un paso del optimizador con los gradientes de [`Self::perdida_y_gradientes`].
    pub fn aplicar(&mut self, opt: &mut AdamW, grads: &[Vec<f32>]) {
        opt.paso(&mut self.t.params.valores, grads);
        self.materializar();
    }

    /// Escribe el modelo en GGUF, en `tipo` lo que se pueda cuantizar: normas y
    /// sesgos quedan en `F32`, como en `daiss::escribir`.
    pub fn exportar(&self, tipo: TipoGgml, tok: Option<&Bpe>, ruta: &Path) -> Resultado<u64> {
        let escritor = crate::daiss::escritor_de(&self.cfg, tipo, self.semilla, tok);
        escritor.escribir(ruta, |d, w| {
            let datos = self
                .peso(&d.nombre)
                .ok_or_else(|| format!("el modelo no tiene {}", d.nombre))?;
            let ancho = d.dims[0] as usize;
            let mut b = Vec::with_capacity(d.bytes());
            for fila in datos.chunks(ancho) {
                crate::quant::cuantizar_bloques(d.tipo, fila, &mut b)?;
            }
            w.write_all(&b).map_err(|e| e.to_string())
        })
    }

    /// El camino de vuelta de [`Self::exportar`]: pisa cada peso con el del GGUF, en
    /// el tipo que esté. El modelo tiene que tener la misma forma.
    pub fn cargar_pesos(&mut self, ruta: &Path) -> Resultado<()> {
        let g = crate::gguf::Gguf::abrir(ruta)?;
        let p = &mut self.t.params;
        for (nombre, valores) in p.nombres.iter().zip(&mut p.valores) {
            if nombre.contains(".tt.") {
                continue;
            }
            let info = g
                .tensor(nombre)
                .ok_or_else(|| format!("el GGUF no trae {nombre}"))?;
            if info.elementos() != valores.len() {
                return Err(format!(
                    "{nombre}: {} valores en el GGUF y {} en el modelo",
                    info.elementos(),
                    valores.len()
                ));
            }
            let bytes = g.datos_de(ruta, info)?;
            crate::quant::descuantizar(info.tipo, &bytes, valores)?;
        }
        Ok(())
    }

    /// Un peso por su nombre en el GGUF.
    pub fn peso(&self, nombre: &str) -> Option<&[f32]> {
        self.t
            .params
            .indice(nombre)
            .map(|i| &self.t.params.valores[i][..])
    }

    /// Un peso por su nombre en el GGUF, para pisarlo.
    pub fn peso_mut(&mut self, nombre: &str) -> Option<&mut [f32]> {
        self.t
            .params
            .indice(nombre)
            .map(|i| &mut self.t.params.valores[i][..])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn lote(n: usize, t: usize, vocab: usize) -> Vec<Vec<u32>> {
        (0..n)
            .map(|s| {
                (0..t)
                    .map(|i| ((s * t + i).pow(2) * 7 % vocab) as u32)
                    .collect()
            })
            .collect()
    }

    fn ruta(nombre: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("cosxamix-{nombre}-{}.gguf", std::process::id()))
    }

    /// Lo que más importa: lo exportado, en el motor propio, da lo mismo que acá.
    #[test]
    fn el_motor_propio_da_los_mismos_logits() {
        let cfg = chico(50);
        let m = ModeloCompleto::nuevo(&cfg, 3, 32);
        let r = ruta("full");
        m.exportar(TipoGgml::F32, None, &r).unwrap();

        let tokens = [7u32, 3, 49, 0, 21, 21, 5];
        let aca = m.logits(&tokens);
        let aca = &aca[aca.len() - cfg.vocab..];

        let motor = crate::forward::Modelo::abrir(&r).unwrap();
        let mut cache = crate::forward::Cache::nuevo(&motor.cfg);
        let alla = motor.forward(&tokens, &mut cache).unwrap();
        std::fs::remove_file(&r).unwrap();

        // `forward` devuelve la última fila o todas: se comparan los últimos.
        let alla = &alla[alla.len() - cfg.vocab..];
        for (i, (a, b)) in aca.iter().zip(alla).enumerate() {
            assert!((a - b).abs() < 1e-3, "logit {i}: {a} contra {b}");
        }
    }

    /// El backward de la cadena de Tensor-Train contra diferencias finitas.
    #[test]
    fn el_gradiente_de_los_nucleos_da_lo_de_las_diferencias_finitas() {
        let cfg = Config {
            dim: 16,
            ffn: 32,
            ..chico(16)
        };
        let mut m = ModeloCompleto::con_tensor_train(&cfg, 8, 8, 3).unwrap();
        let l = lote(2, 8, 16);
        let (_, g) = m.perdida_y_gradientes(&l);
        let (e, s) = m.tt.as_ref().map(|(e, s)| (e.nucleos, s.nucleos)).unwrap();
        let h = 1e-3;
        for pi in e.into_iter().chain(s) {
            let n = m.t.params.valores[pi].len();
            for i in [0, n / 3, n - 1] {
                let orig = m.t.params.valores[pi][i];
                let mut f = |d: f32| {
                    m.t.params.valores[pi][i] = orig + d;
                    m.materializar();
                    let v = m.perdida(&l);
                    m.t.params.valores[pi][i] = orig;
                    m.materializar();
                    v
                };
                let num = (f(h) - f(-h)) / (2.0 * h);
                let an = g[pi][i];
                // Piso de 1e-2: la pérdida es f32 y con h = 1e-3 la diferencia finita
                // solo resuelve de a ~1e-4, que en un gradiente de 1e-3 es el 10%.
                let err = (an - num).abs() / (an.abs() + num.abs()).max(1e-2);
                assert!(
                    err < 3e-2,
                    "{}[{i}]: {an} contra {num}",
                    m.t.params.nombres[pi]
                );
            }
        }
    }

    /// Con Tensor-Train: los factores salen como deben, todo núcleo recibe
    /// gradiente, la pérdida baja, y lo exportado da lo mismo al volver a cargarlo.
    #[test]
    fn tensor_train_entrena_y_se_exporta() {
        assert_eq!(cuatro_factores(8192).unwrap(), [8, 8, 8, 16]);
        assert_eq!(cuatro_factores(256).unwrap(), [4, 4, 4, 4]);
        assert_eq!(cuatro_factores(64).unwrap(), [2, 2, 4, 4]);
        assert!(cuatro_factores(50).is_err());

        let cfg = chico(64);
        let mut m = ModeloCompleto::con_tensor_train(&cfg, 4, 16, 4).unwrap();
        assert!(m.parametros() < ModeloCompleto::nuevo(&cfg, 4, 16).parametros());

        let l = lote(4, 16, 64);
        let (_, g) = m.perdida_y_gradientes(&l);
        let p = &m.t.params;
        let sin = (0..p.valores.len())
            .filter(|&i| p.entrena[i] && g[i].iter().all(|v| *v == 0.0))
            .count();
        assert_eq!(sin, 0, "{sin} pesos sin gradiente");

        let mut opt = AdamW::nuevo(1e-2);
        let antes = m.perdida(&l);
        for _ in 0..30 {
            let (_, g) = m.perdida_y_gradientes(&l);
            m.aplicar(&mut opt, &g);
        }
        let despues = m.perdida(&l);
        assert!(despues < antes * 0.8, "{antes} -> {despues}");

        // La tabla densa que se exporta es la de los núcleos.
        let r = ruta("tt");
        m.exportar(TipoGgml::F32, None, &r).unwrap();
        let mut otro = ModeloCompleto::nuevo(&cfg, 99, 16);
        otro.cargar_pesos(&r).unwrap();
        std::fs::remove_file(&r).unwrap();
        let d = m
            .logits(&l[0])
            .iter()
            .zip(otro.logits(&l[0]))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-4, "difieren en {d}");
    }

    /// Exportar y volver a cargar en otro modelo da los mismos logits.
    #[test]
    fn cargar_pesos_es_la_vuelta_de_exportar() {
        let cfg = chico(30);
        let a = ModeloCompleto::nuevo(&cfg, 5, 16);
        let mut b = ModeloCompleto::nuevo(&cfg, 6, 16);
        let r = ruta("vuelta");
        a.exportar(TipoGgml::F32, None, &r).unwrap();
        b.cargar_pesos(&r).unwrap();
        std::fs::remove_file(&r).unwrap();

        let ids = [1u32, 29, 4, 4, 17];
        assert_eq!(a.logits(&ids), b.logits(&ids));
        assert!(b.peso("output.weight").is_some());
        assert!(b.peso("no.existe").is_none());
    }

    /// Todo peso tiene que recibir gradiente: uno que no lo reciba no aprende nada,
    /// y el modelo sigue andando sin que se note.
    #[test]
    fn todos_los_pesos_reciben_gradiente() {
        let cfg = chico(40);
        let m = ModeloCompleto::nuevo(&cfg, 9, 16);
        let (_, g) = m.perdida_y_gradientes(&lote(4, 16, 40));
        let p = &m.t.params;
        let faltan: Vec<&str> = (0..p.valores.len())
            .filter(|&i| g[i].len() != p.valores[i].len() || g[i].iter().all(|v| *v == 0.0))
            .map(|i| p.nombres[i].as_str())
            .collect();
        assert!(faltan.is_empty(), "sin gradiente: {faltan:?}");
    }

    /// Y aprende: con un texto que se repite, la pérdida tiene que bajar mucho.
    #[test]
    fn la_perdida_baja() {
        let cfg = chico(16);
        let mut m = ModeloCompleto::nuevo(&cfg, 5, 32);
        let sec: Vec<u32> = (0..32).map(|i| (i % 4) as u32).collect();
        let l = vec![sec; 4];
        let mut opt = AdamW::nuevo(3e-3);
        let inicial = m.perdida(&l);
        for _ in 0..40 {
            let (_, g) = m.perdida_y_gradientes(&l);
            m.aplicar(&mut opt, &g);
        }
        let fin = m.perdida(&l);
        assert!(fin < inicial * 0.2, "de {inicial} a {fin}");
    }
}
