//! El motor propio, detrás del mismo trait que el de candle.
//!
//! [`crate::forward::Modelo`] ya generaba texto —idéntico a candle token por
//! token, con el prefill 2,8x más rápido (ROADMAP §9)— pero solo desde
//! `examples/generate`. Esto es lo que faltaba para que lo use el servidor:
//! sesiones con su caché de atención, el prefijo común reaprovechado, el texto
//! saliendo en partes, y el corte en `<|im_end|>`.
//!
//! Copia a propósito el contrato de [`crate::engine_candle::MotorCandle`]
//! donde decide el resultado —la misma plantilla de chat ([`formatear`]), el
//! mismo muestreo (semilla 42, top-p 0,9), el mismo tope de sesiones y de
//! tokens por sesión—, porque elegir entre los dos tiene que ser cambiar
//! `COSXAMIX_MOTOR`, no cambiar de comportamiento.
//!
//! ## Lo que es más simple que en candle
//!
//! El modelo de candle tiene **un** caché de atención adentro, y cada sesión
//! se lo turna: guardar el de la que estaba, poner el de la que llega. Acá el
//! caché es un valor aparte ([`Cache`]) que `forward` recibe, así que cada
//! sesión tiene el suyo y entrar a una conversación es sacarlo del mapa.
//!
//! El tokenizador es el de Qwen —el crate `tokenizers`, que no es candle
//! aunque viaje con la misma feature—: estos modelos se entrenaron con él.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tokenizers::Tokenizer;

use crate::engine::{formatear, ErrorMotor, ModeloDisponible, Motor, Peticion, Respuesta};
use crate::forward::{Cache, Modelo};
use crate::sampler::Muestreador;

/// Los mismos topes que el motor de candle, por lo mismo: ver allí.
const MAX_CACHE_TOKENS: usize = 4096;
const MAX_SESIONES: usize = 16;

struct Sesion {
    cache: Cache,
    /// La secuencia exacta de tokens que el caché contiene.
    tokens: Vec<u32>,
    usada: u64,
}

struct Cargado {
    modelo: Modelo,
    tokenizer: Tokenizer,
    bytes: u64,
    sesiones: HashMap<String, Sesion>,
    reloj: u64,
    /// `<|im_end|>`, donde termina el turno del asistente. Se busca en el
    /// tokenizador, como el motor de candle: el `eos` del archivo puede ser
    /// `<|endoftext|>`, que en un chat llega tarde.
    fin_turno: Option<u32>,
}

pub struct MotorPropio {
    nombre: String,
    gguf: PathBuf,
    tok: PathBuf,
    estado: Mutex<Option<Cargado>>,
}

impl MotorPropio {
    /// No carga nada todavía: solo comprueba que los archivos existan.
    pub fn nuevo(nombre: impl Into<String>, gguf: &Path, tok: &Path) -> Result<Self, ErrorMotor> {
        for p in [gguf, tok] {
            if !p.exists() {
                return Err(ErrorMotor::NoCargado(format!("no existe {}", p.display())));
            }
        }
        Ok(Self {
            nombre: nombre.into(),
            gguf: gguf.to_path_buf(),
            tok: tok.to_path_buf(),
            estado: Mutex::new(None),
        })
    }

    fn cargar(&self) -> Result<Cargado, ErrorMotor> {
        let interno = |e: String| ErrorMotor::Interno(e);
        let bytes = std::fs::metadata(&self.gguf)
            .map_err(|e| interno(format!("no pude medir el gguf: {e}")))?
            .len();
        let modelo = Modelo::abrir(&self.gguf).map_err(|e| interno(format!("gguf: {e}")))?;
        let tokenizer = Tokenizer::from_file(&self.tok)
            .map_err(|e| interno(format!("tokenizer inválido: {e}")))?;
        let fin_turno = tokenizer.token_to_id("<|im_end|>");

        // Los pesos están mapeados, no leídos: sin esto, "cargado" querría
        // decir "abierto", y la primera pregunta pagaría traer el modelo
        // entero del disco —1,6 GB a los 41 MB/s de esta máquina son 40 s—.
        // Un forward de un token toca todas las matrices, que es lo que hace
        // el `from_gguf` de candle al leerlas.
        let mut cache = Cache::nuevo(&modelo.cfg);
        modelo
            .forward(&[fin_turno.unwrap_or(0)], &mut cache)
            .map_err(|e| interno(format!("calentar: {e}")))?;

        Ok(Cargado {
            modelo,
            tokenizer,
            bytes,
            sesiones: HashMap::new(),
            reloj: 0,
            fin_turno,
        })
    }

    pub fn precalentar(&self) -> Result<(), ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        Ok(())
    }

    pub fn descargar(&self) {
        *self.estado.lock().expect("mutex envenenado") = None;
    }

    pub fn esta_caliente(&self) -> bool {
        self.estado.lock().expect("mutex envenenado").is_some()
    }
}

/// Guarda la sesión y desaloja las menos usadas que sobren.
fn guardar(c: &mut Cargado, clave: String, cache: Cache, tokens: Vec<u32>) {
    // Una conversación muy larga se descarta en vez de arrastrarse: el caché
    // crece con cada token y el modelo tampoco atiende más allá de su ventana.
    if tokens.len() > MAX_CACHE_TOKENS {
        return;
    }
    c.reloj += 1;
    let usada = c.reloj;
    c.sesiones.insert(
        clave,
        Sesion {
            cache,
            tokens,
            usada,
        },
    );
    while c.sesiones.len() > MAX_SESIONES {
        let Some(victima) = c
            .sesiones
            .iter()
            .min_by_key(|(_, s)| s.usada)
            .map(|(k, _)| k.clone())
        else {
            break;
        };
        c.sesiones.remove(&victima);
    }
}

impl Motor for MotorPropio {
    fn modelos(&self) -> Vec<ModeloDisponible> {
        let g = self.estado.lock().expect("mutex envenenado");
        vec![ModeloDisponible {
            nombre: self.nombre.clone(),
            bytes: g
                .as_ref()
                .map(|c| c.bytes)
                .unwrap_or_else(|| std::fs::metadata(&self.gguf).map(|m| m.len()).unwrap_or(0)),
            en_ram: g.is_some(),
        }]
    }

    fn generar(&self, p: &Peticion) -> Result<Respuesta, ErrorMotor> {
        self.generar_en_partes(p, &mut |_| true)
    }

    fn generar_en_partes(
        &self,
        p: &Peticion,
        emitir: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Respuesta, ErrorMotor> {
        if !p.modelo.is_empty() && p.modelo != self.nombre {
            return Err(ErrorMotor::ModeloDesconocido(p.modelo.clone()));
        }
        let interno = |e: String| ErrorMotor::Interno(e);

        let mut g = self.estado.lock().expect("mutex envenenado");
        let estaba_caliente = g.is_some();
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_mut().expect("recién cargado");

        let entrada: Vec<u32> = c
            .tokenizer
            .encode(formatear(&p.mensajes), true)
            .map_err(|e| interno(format!("no pude tokenizar: {e}")))?
            .get_ids()
            .to_vec();
        if entrada.is_empty() {
            return Err(interno("el prompt quedó vacío tras tokenizar".into()));
        }

        // La sesión, con su caché. Sin sesión se arranca de cero y no se
        // guarda nada: una petición suelta no debe pisar una conversación.
        let con_sesion = !p.sesion.is_empty();
        let (mut cache, cacheados) =
            match con_sesion.then(|| c.sesiones.remove(&p.sesion)).flatten() {
                Some(s) => (s.cache, s.tokens),
                None => (Cache::nuevo(&c.modelo.cfg), Vec::new()),
            };

        // El prefijo común más largo, dejando al menos un token por procesar:
        // sin uno no hay logits. Es lo que reaprovecha el *system* de WinsiBot
        // —cientos de tokens— entre un mensaje y el siguiente.
        let comun = cacheados
            .iter()
            .zip(&entrada)
            .take_while(|(a, b)| a == b)
            .count()
            .min(entrada.len() - 1);
        cache.truncar(comun);
        if comun == 0 {
            cache.limpiar();
        }

        let mut muestreador = Muestreador::nuevo(42, p.temperatura, 0.9);
        let mut logits = c
            .modelo
            .forward(&entrada[comun..], &mut cache)
            .map_err(|e| interno(format!("prefill: {e}")))?;

        let eos = c.modelo.fin();
        let contexto = c.modelo.cfg.contexto;
        let mut salida: Vec<u32> = Vec::new();
        let mut emitido = String::new();
        for _ in 0..p.max_tokens {
            let t = muestreador.elegir(&logits);
            if Some(t) == c.fin_turno || Some(t) == eos {
                break;
            }
            salida.push(t);

            // Un token no es un trozo de texto: puede ser media palabra o
            // medio carácter UTF-8. Se decodifica todo lo generado y se emite
            // lo que creció (ver el motor de candle).
            let mut cortado = false;
            if let Ok(completo) = c.tokenizer.decode(&salida, true) {
                if let Some(nuevo) = completo.strip_prefix(emitido.as_str()) {
                    if !nuevo.is_empty() {
                        cortado = !emitir(nuevo);
                        emitido = completo.clone();
                    }
                }
            }
            if cortado || cache.largo() >= contexto {
                break;
            }
            logits = c
                .modelo
                .forward(&[t], &mut cache)
                .map_err(|e| interno(format!("paso {}: {e}", salida.len())))?;
        }

        let texto = c
            .tokenizer
            .decode(&salida, true)
            .map_err(|e| interno(format!("no pude decodificar: {e}")))?;

        // El caché tiene ahora el prompt y lo generado menos el último token
        // —ese salió de los logits y no pasó por `forward`—. Se guarda la
        // secuencia que de verdad contiene.
        if con_sesion {
            let mut vistos = entrada.clone();
            vistos.extend_from_slice(&salida);
            vistos.truncate(cache.largo());
            guardar(c, p.sesion.clone(), cache, vistos);
        }

        Ok(Respuesta {
            texto: texto.trim().to_string(),
            tokens_prompt: entrada.len() as u32,
            tokens_generados: salida.len() as u32,
            tokens_cacheados: comun as u32,
            estaba_caliente,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_archivos_falla_al_construir_y_no_al_generar() {
        let r = MotorPropio::nuevo(
            "x",
            Path::new("no/existe.gguf"),
            Path::new("no/existe.json"),
        );
        assert!(matches!(r, Err(ErrorMotor::NoCargado(_))));
    }
}
