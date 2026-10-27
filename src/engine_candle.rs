//! Motor de inferencia real, sobre `candle`.
//!
//! ## Por qué candle y no llama.cpp
//!
//! Medido antes de escribir una línea, que es lo que pide el §6 de la guía:
//!
//! | | candle | llama.cpp |
//! |---|---|---|
//! | dependencias | 155 crates, Rust puro | + toolchain de C++ por plataforma |
//! | build release | 84 s | minutos, y hay que tener compilador |
//! | cruza a `aarch64-linux-android` | **sí** | con NDK y pelea |
//!
//! Lo decisivo es la última fila. Termux es un objetivo del proyecto, y un
//! motor que necesita un compilador de C++ en el teléfono reintroduce el
//! requisito que se estaba quitando — movido de sitio, pero el mismo.
//!
//! Apareció una trampa que solo se ve probando: el crate `tokenizers` trae por
//! defecto `onig`, que es **Oniguruma, una librería de C**, y rompe el cruce a
//! Android con `failed to run custom build command for onig_sys`. Se desactiva
//! con `default-features = false`; el camino alternativo usa `fancy-regex`, que
//! es Rust puro. Está anotado en `Cargo.toml` porque es justo la clase de
//! dependencia que se vuelve a colar sin que nadie la vea.
//!
//! ## Qué hace y qué no
//!
//! Carga **un** modelo GGUF y genera. No gestiona varios, ni los desaloja, ni
//! comparte caché entre peticiones: eso es el gestor de modelos, que va aparte
//! porque es trabajo de Cosxamix y no del motor. Mezclarlos sería volver a tener
//! dos dueños para una responsabilidad.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
// El modelo viene de `src/model/`, no del crate: ahí está la versión con el
// caché de atención expuesto. El porqué está en la cabecera de ese archivo.
use crate::model::qwen2::{CacheAtencion, ModelWeights};
use tokenizers::Tokenizer;

use crate::engine::{formatear, ErrorMotor, ModeloDisponible, Motor, Peticion, Respuesta};

/// Tope de tokens que se conservan en el caché de atención entre peticiones.
///
/// No es el límite del modelo: es el punto donde conviene más reprocesar que
/// seguir arrastrando. Sin tope, una conversación larga haría crecer la memoria
/// y el coste de cada paso de atención sin límite.
const MAX_CACHE_TOKENS: usize = 4096;

/// Cuántas conversaciones se recuerdan a la vez.
///
/// Cada una guarda su caché de atención, del orden de 24 KB por token en
/// Qwen2-0.5B. Con 16 sesiones de 1000 tokens son unos 380 MB, que es un
/// precio razonable al lado de los 931 MB que costaba **cada** instancia del
/// modelo. Pasado el tope se desaloja la menos usada.
const MAX_SESIONES: usize = 16;

/// Los pesos y el tokenizador, ya en memoria.
///
/// Va detrás de un `Mutex` porque `forward()` necesita `&mut self`: el modelo
/// arrastra el caché de atención entre llamadas, así que dos generaciones a la
/// vez sobre el mismo estado se pisarían. Serializar es correcto y es lo que
/// hace falta hoy; cuando haya varias peticiones concurrentes lo resolverá el
/// planificador con un estado por sesión, no quitando este candado.
/// Lo que se guarda de una conversación entre peticiones.
struct Sesion {
    cache: CacheAtencion,
    /// La secuencia EXACTA de tokens que ese caché contiene.
    ///
    /// Sin esto no se puede reutilizar nada: el caché guarda claves y valores
    /// por capa, pero no qué tokens los produjeron, y restaurar uno sin saber
    /// a qué corresponde da respuestas condicionadas por otra conversación,
    /// sin ningún error de por medio.
    tokens: Vec<u32>,
    /// Para desalojar la menos usada cuando sobran.
    usada: u64,
}

struct Cargado {
    pesos: ModelWeights,
    tokenizer: Tokenizer,
    /// Para `/api/tags` y para informar `estaba_caliente`.
    bytes: u64,
    capas: usize,

    /// Una conversación guardada por clave de sesión.
    ///
    /// El modelo tiene un solo caché activo; acá se guardan los de las demás y
    /// se intercambian al entrar. Eso es lo que permite que varias
    /// conversaciones avancen intercaladas sin pisarse, con **un** modelo
    /// cargado: medido, una instancia por sesión costaba 931 MB cada una.
    sesiones: HashMap<String, Sesion>,

    /// Qué sesión tiene su caché puesto en el modelo ahora mismo.
    activa: Option<String>,

    /// Reloj lógico para el desalojo. No se usa la hora: dos peticiones en el
    /// mismo milisegundo empatarían, y el orden de llegada es lo que importa.
    reloj: u64,
}

pub struct MotorCandle {
    nombre: String,
    gguf: PathBuf,
    tok: PathBuf,
    /// `None` mientras el modelo no se cargó. Esto es lo que hace posible
    /// distinguir frío de caliente, que es el dato que ordena toda la
    /// optimización de latencia.
    estado: Mutex<Option<Cargado>>,
    device: Device,
}

impl MotorCandle {
    /// No carga nada todavía: solo comprueba que los archivos existan.
    ///
    /// Arrancar sin cargar es deliberado. Cargar en el constructor haría que
    /// el servidor tardara en levantar y que un modelo roto impidiera arrancar,
    /// cuando lo correcto es responder un error en esa petición y seguir vivo.
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
            // CPU por ahora. El día que haya GPU, esto es lo único que cambia
            // acá: candle enruta los tensores solo.
            device: Device::Cpu,
        })
    }

    /// Deja el modelo en memoria sin generar nada.
    ///
    /// Esto es el `keep warm`: el motivo número uno por el que Cosxamix existe.
    /// Ollama descarga el modelo a los 5 minutos de inactividad, y un bot con
    /// mensajes espaciados paga esa recarga casi siempre. Acá el modelo entra
    /// y **se queda** hasta que alguien decida lo contrario.
    pub fn precalentar(&self) -> Result<(), ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        Ok(())
    }

    /// Saca el modelo de la RAM.
    ///
    /// Se lleva por delante los caches de sesion de ese modelo, y tiene que
    /// ser asi: un cache de atencion son estados internos de unos pesos
    /// concretos que ya no estan. La siguiente peticion reprocesa el prompt,
    /// que es el precio de haber desalojado.
    pub fn descargar(&self) {
        *self.estado.lock().expect("mutex envenenado") = None;
    }

    /// Cuantos bytes ocupa el cache de atencion activo. Solo diagnostico.
    pub fn bytes_de_cache(&self) -> usize {
        let mut g = self.estado.lock().expect("mutex envenenado");
        match g.as_mut() {
            Some(c) => {
                let cache = c.pesos.tomar_cache();
                let n = cache.bytes();
                c.pesos.poner_cache(cache);
                n
            }
            None => 0,
        }
    }

    pub fn esta_caliente(&self) -> bool {
        self.estado.lock().expect("mutex envenenado").is_some()
    }

    /// Tokeniza un texto. Solo diagnóstico.
    ///
    /// Hace falta para que las pruebas usen un prompt REAL: con ids inventados,
    /// la distribución de salida queda plana y el token más probable lo decide
    /// el ruido, así que comparar argmax sobre basura no dice nada.
    pub fn tokenizar_diagnostico(&self, texto: &str) -> Result<Vec<u32>, ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_ref().expect("cargado");
        c.tokenizer
            .encode(texto, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ErrorMotor::Interno(format!("tokenizar: {e}")))
    }

    /// Los logits del siguiente token para una secuencia dada. Solo diagnóstico.
    ///
    /// Existe para poder comparar NUMÉRICAMENTE el camino con caché contra el
    /// camino sin él. Comparar los textos generados no sirve: una diferencia de
    /// un bit cambia un token y de ahí todo diverge, así que dos salidas
    /// distintas no distinguen un caché roto de ruido de punto flotante.
    ///
    /// `usar_cache` decide si se reaprovecha lo que ya está cargado o se
    /// reprocesa la secuencia entera.
    pub fn logits_diagnostico(
        &self,
        tokens: &[u32],
        usar_cache: bool,
    ) -> Result<Vec<f32>, ErrorMotor> {
        let interno = |e: String| ErrorMotor::Interno(e);
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_mut().expect("cargado");

        // Usa una sesión fija propia para no interferir con las de verdad: esto
        // es diagnóstico y no debería dejar rastro en una conversación real.
        const DIAG: &str = "\u{0}diagnostico";
        let cacheados = Self::activar(c, DIAG);

        let desde = if usar_cache
            && !cacheados.is_empty()
            && tokens.len() > cacheados.len()
            && tokens.starts_with(&cacheados)
        {
            cacheados.len()
        } else {
            c.pesos.limpiar_cache();
            0
        };

        let logits = if desde == 0 {
            let t = Tensor::new(tokens, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| interno(format!("tensor: {e}")))?;
            c.pesos
                .forward(&t, 0)
                .map_err(|e| interno(format!("forward: {e}")))?
        } else {
            let mut u = None;
            for (i, &tk) in tokens[desde..].iter().enumerate() {
                let t = Tensor::new(&[tk], &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| interno(format!("tensor: {e}")))?;
                u = Some(
                    c.pesos
                        .forward(&t, desde + i)
                        .map_err(|e| interno(format!("forward: {e}")))?,
                );
            }
            u.expect("al menos uno")
        };

        Self::guardar(c, DIAG, tokens.to_vec());

        logits
            .squeeze(0)
            .and_then(|l| {
                if l.dims().len() == 2 {
                    l.get(l.dim(0)? - 1)
                } else {
                    Ok(l)
                }
            })
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(|e| interno(format!("logits: {e}")))
    }

    /// Deja el caché de `sesion` puesto en el modelo y devuelve sus tokens.
    ///
    /// El modelo tiene un solo caché activo. Entrar a una conversación es
    /// guardar el de la que estaba y poner el de la que llega — mover punteros,
    /// no copiar datos.
    fn activar(c: &mut Cargado, sesion: &str) -> Vec<u32> {
        if c.activa.as_deref() == Some(sesion) {
            c.reloj += 1;
            if let Some(s) = c.sesiones.get_mut(sesion) {
                s.usada = c.reloj;
                return s.tokens.clone();
            }
        }

        // Guardar el caché de la sesión que estaba puesta. Si no se hace, se
        // pierde y esa conversación vuelve a arrancar de cero la próxima vez.
        if let Some(anterior) = c.activa.take() {
            let cache = c.pesos.tomar_cache();
            if let Some(s) = c.sesiones.get_mut(&anterior) {
                s.cache = cache;
            }
        } else {
            c.pesos.limpiar_cache();
        }

        c.reloj += 1;
        let reloj = c.reloj;
        let capas = c.capas;

        let s = c
            .sesiones
            .entry(sesion.to_string())
            .or_insert_with(|| Sesion {
                cache: CacheAtencion::vacio(capas),
                tokens: Vec::new(),
                usada: reloj,
            });
        s.usada = reloj;

        let cache = std::mem::replace(&mut s.cache, CacheAtencion::vacio(capas));
        let tokens = s.tokens.clone();

        c.pesos.poner_cache(cache);
        c.activa = Some(sesion.to_string());
        tokens
    }

    /// Guarda el caché activo y desaloja las sesiones que sobren.
    fn guardar(c: &mut Cargado, sesion: &str, tokens: Vec<u32>) {
        let cache = c.pesos.tomar_cache();
        c.activa = None;

        // Una conversación muy larga se descarta en vez de arrastrarse: el
        // caché crece ~24 KB por token y el modelo tampoco atiende más allá de
        // su ventana. Cuesta un prefill y evita que la memoria crezca sin tope.
        if tokens.len() > MAX_CACHE_TOKENS {
            c.sesiones.remove(sesion);
            return;
        }

        if let Some(s) = c.sesiones.get_mut(sesion) {
            s.cache = cache;
            s.tokens = tokens;
        }

        // Desalojo por menos usada recientemente. Sin esto, cada chat nuevo
        // suma un caché que nadie libera.
        while c.sesiones.len() > MAX_SESIONES {
            let victima = c
                .sesiones
                .iter()
                .min_by_key(|(_, s)| s.usada)
                .map(|(k, _)| k.clone());
            match victima {
                Some(k) => {
                    c.sesiones.remove(&k);
                }
                None => break,
            }
        }
    }

    // ── Primitivas para la decodificación especulativa ───────────────────
    //
    // `generar()` hace el camino completo: formatea, tokeniza, genera y
    // decodifica. La especulativa necesita menos y más fino — alimentar
    // tokens sueltos y leer logits— así que estas dos quedan aparte en vez de
    // llenar `generar()` de banderas.

    /// Tokeniza un texto ya formateado.
    pub fn tokenizar(&self, texto: &str) -> Result<Vec<u32>, ErrorMotor> {
        self.tokenizar_diagnostico(texto)
    }

    pub fn destokenizar(&self, ids: &[u32]) -> Result<String, ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        g.as_ref()
            .expect("cargado")
            .tokenizer
            .decode(ids, true)
            .map_err(|e| ErrorMotor::Interno(format!("decodificar: {e}")))
    }

    /// Procesa `tokens` desde `desde` y devuelve el token más probable de cada
    /// posición.
    ///
    /// Devuelve `n` predicciones para `n` tokens de entrada: la de la posición
    /// `i` es lo que el modelo pondría DESPUÉS de haber visto hasta `i`. Eso es
    /// lo que permite verificar K propuestas en una sola pasada.
    ///
    /// Ojo: avanza el caché del modelo. Quien llama lleva la cuenta de qué hay
    /// dentro.
    pub fn evaluar(&self, tokens: &[u32], desde: usize) -> Result<Vec<u32>, ErrorMotor> {
        let interno = |e: String| ErrorMotor::Interno(e);
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_mut().expect("cargado");

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let t = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| interno(format!("tensor: {e}")))?;

        let logits = c
            .pesos
            .forward_todas(&t, desde)
            .map_err(|e| interno(format!("forward: {e}")))?;

        // [1, n, vocab] -> un argmax por posición.
        logits
            .squeeze(0)
            .and_then(|l| l.argmax(candle_core::D::Minus1))
            .and_then(|a| a.to_vec1::<u32>())
            .map_err(|e| interno(format!("argmax: {e}")))
    }

    /// Recorta el caché de atención a `n` tokens.
    ///
    /// Para la especulativa: después de verificar una ráfaga hay que sacar las
    /// propuestas que no se aceptaron, o el caché queda más largo que la
    /// secuencia real.
    pub fn truncar(&self, n: usize) -> Result<(), ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        let c = g
            .as_mut()
            .ok_or_else(|| ErrorMotor::NoCargado("sin cargar".into()))?;
        c.pesos
            .truncar_cache(n)
            .map_err(|e| ErrorMotor::Interno(format!("truncar: {e}")))
    }

    /// Vacía el caché de la sesión activa. Para arrancar una especulación.
    pub fn reiniciar(&self) -> Result<(), ErrorMotor> {
        let mut g = self.estado.lock().expect("mutex envenenado");
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_mut().expect("cargado");
        c.pesos.limpiar_cache();
        c.activa = None;
        c.sesiones.clear();
        Ok(())
    }

    fn cargar(&self) -> Result<Cargado, ErrorMotor> {
        let interno = |e: String| ErrorMotor::Interno(e);

        let bytes = std::fs::metadata(&self.gguf)
            .map_err(|e| interno(format!("no pude medir el gguf: {e}")))?
            .len();

        let mut f =
            File::open(&self.gguf).map_err(|e| interno(format!("no pude abrir el gguf: {e}")))?;

        let contenido =
            gguf_file::Content::read(&mut f).map_err(|e| interno(format!("gguf inválido: {e}")))?;

        // Cuántas capas tiene, leído de la metadata ANTES de que `from_gguf`
        // consuma el contenido. Hace falta para construir cachés vacíos del
        // tamaño correcto.
        let capas = contenido
            .metadata
            .get("qwen2.block_count")
            .and_then(|v| v.to_u32().ok())
            .map(|n| n as usize)
            .ok_or_else(|| interno("el gguf no declara qwen2.block_count".into()))?;

        let pesos = ModelWeights::from_gguf(contenido, &mut f, &self.device)
            .map_err(|e| interno(format!("no pude construir el modelo: {e}")))?;

        let tokenizer = Tokenizer::from_file(&self.tok)
            .map_err(|e| interno(format!("tokenizer inválido: {e}")))?;

        Ok(Cargado {
            pesos,
            tokenizer,
            bytes,
            capas,
            sesiones: HashMap::new(),
            activa: None,
            reloj: 0,
        })
    }
}

impl Motor for MotorCandle {
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

        let mut g = self.estado.lock().expect("mutex envenenado");

        // Se mide ANTES de cargar: es el dato que distingue una respuesta de
        // medio segundo de una de diez, y sin él las dos latencias se mezclan
        // en un promedio que no dice nada.
        let estaba_caliente = g.is_some();
        if g.is_none() {
            *g = Some(self.cargar()?);
        }
        let c = g.as_mut().expect("recién cargado");

        let interno = |e: String| ErrorMotor::Interno(e);

        let prompt = formatear(&p.mensajes);
        let codificado = c
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| interno(format!("no pude tokenizar: {e}")))?;
        let entrada: Vec<u32> = codificado.get_ids().to_vec();

        if entrada.is_empty() {
            return Err(interno("el prompt quedó vacío tras tokenizar".into()));
        }

        let mut muestreador = LogitsProcessor::new(
            // Semilla fija: dos peticiones iguales dan la misma respuesta, que
            // es lo que hace posible comparar cambios. La variedad sale de la
            // temperatura, no del reloj.
            42,
            Some(p.temperatura as f64),
            Some(0.9),
        );

        // ── Prefill, reutilizando el caché cuando se puede ───────────────────
        //
        // El prefill domina la latencia: medido, generar UN token costaba 857 ms
        // de los cuales ~817 eran procesar el prompt. Y en una conversación ese
        // trabajo se repite casi entero en cada turno, porque el prompt del
        // turno N empieza exactamente igual que el del turno N-1.
        //
        // Se reutiliza **solo si el prompt nuevo EXTIENDE al que está en el
        // caché**. No se intenta nada más fino —ni prefijos parciales, ni
        // recortar el caché— porque candle no expone forma de truncarlo, y
        // reutilizar de más daría respuestas condicionadas por tokens que ya no
        // están en el prompt. Preferible reprocesar que contestar mal.
        //
        // El caso de una conversación cae justo en el camino bueno: tras
        // generar, el caché contiene prompt + respuesta, y el turno siguiente
        // empieza precisamente con eso.
        // Entrar a la conversación: se guarda el caché de la que estuviera
        // puesta y se pone el de esta. Sin sesión no se toca nada y el caché
        // queda como esté — una petición suelta no debe pisar una conversación
        // en curso.
        let con_sesion = !p.sesion.is_empty();
        let cacheados = if con_sesion {
            Self::activar(c, &p.sesion)
        } else {
            c.pesos.limpiar_cache();
            c.activa = None;
            Vec::new()
        };

        // Prefijo común más largo, no "solo si extiende".
        //
        // La primera versión exigía que el prompt nuevo empezara exactamente
        // con todo lo cacheado. Eso funciona en una conversación que crece,
        // pero no en el caso real de WinsiBot: manda `system + pregunta` sin
        // historial, así que el turno 2 comparte el system con el 1 y después
        // diverge — y no se reaprovechaba nada. Medido: `cacheados=0` siempre.
        //
        // Con `truncar_cache` se puede quedar con la parte común y tirar el
        // resto. El system de WinsiBot lleva los perfiles de estilo y el
        // catálogo de comandos: son cientos de tokens que antes se
        // reprocesaban en cada mensaje.
        //
        // Se deja un token sin reutilizar a propósito: el modelo necesita al
        // menos uno que procesar para producir logits, y si el prompt nuevo
        // fuera idéntico al cacheado no quedaría ninguno.
        let comun = cacheados
            .iter()
            .zip(&entrada)
            .take_while(|(a, b)| a == b)
            .count()
            .min(entrada.len().saturating_sub(1));

        let reutilizados = if comun == 0 {
            // Nada en común: hay que vaciar, porque `forward` con index_pos 0
            // ignora el caché pero lo deja puesto, y el token siguiente se
            // concatenaría sobre lo que quedó.
            c.pesos.limpiar_cache();
            0
        } else {
            // Recortar el caché a la parte común. Sin esto, el caché tendría
            // tokens que ya no pertenecen a la secuencia y la atención los
            // miraría igual.
            c.pesos
                .truncar_cache(comun)
                .map_err(|e| interno(format!("truncar: {e}")))?;
            comun
        };

        let nuevos = &entrada[reutilizados..];

        // El delta va en UN lote, no de a un token.
        //
        // Antes iba de a uno porque la máscara causal del modelo era cuadrada
        // y solo valía con el caché vacío. Arreglada esa máscara en
        // `src/model/qwen2.rs`, el bloque entra de una: la atención se
        // calcula igual y se ahorran N-1 invocaciones.
        let t = Tensor::new(nuevos, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| interno(format!("tensor de entrada: {e}")))?;

        let mut logits = c
            .pesos
            .forward(&t, reutilizados)
            .map_err(|e| interno(format!("prefill: {e}")))?;
        // ── Generación, token a token ────────────────────────────────────────
        let fin = c.tokenizer.token_to_id("<|im_end|>");
        let mut salida: Vec<u32> = Vec::new();
        // Lo ya emitido, para poder mandar solo la diferencia.
        let mut emitido = String::new();
        let mut cortado = false;
        // `pos` es la posición absoluta en la secuencia, que es lo que el
        // modelo necesita para ubicar el token en su caché de atención — no
        // un contador del bucle. Por eso arranca en el largo del prompt.
        for pos in (entrada.len()..).take(p.max_tokens as usize) {
            let ultimo = logits
                .squeeze(0)
                .and_then(|l| {
                    if l.dims().len() == 2 {
                        l.get(l.dim(0)? - 1)
                    } else {
                        Ok(l)
                    }
                })
                .map_err(|e| interno(format!("logits: {e}")))?;

            let siguiente = muestreador
                .sample(&ultimo)
                .map_err(|e| interno(format!("muestreo: {e}")))?;

            if Some(siguiente) == fin {
                break;
            }
            salida.push(siguiente);

            // Un token NO es un trozo de texto: puede ser media palabra o
            // medio caracter UTF-8. Lo unico seguro es decodificar todo lo
            // generado y emitir lo que crecio desde la vez anterior.
            //
            // Cuesta O(n^2) en el largo de la respuesta, y no importa:
            // decodificar 200 tokens son microsegundos al lado de los ~100 ms
            // que cuesta el forward que produce cada uno.
            if let Ok(completo) = c.tokenizer.decode(&salida, true) {
                // Si el decodificado nuevo no empieza por el anterior, el
                // tokenizador reescribio algo que ya habia salido. Es raro,
                // pero emitir un trozo mal cortado partiria un caracter, asi
                // que en ese caso se espera al token siguiente.
                if let Some(nuevo) = completo.strip_prefix(emitido.as_str()) {
                    if !nuevo.is_empty() {
                        if !emitir(nuevo) {
                            cortado = true;
                        }
                        emitido = completo.clone();
                    }
                }
            }
            if cortado {
                break;
            }

            let t = Tensor::new(&[siguiente], &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| interno(format!("tensor del paso: {e}")))?;

            logits = c
                .pesos
                .forward(&t, pos)
                .map_err(|e| interno(format!("paso {pos}: {e}")))?;
        }

        let texto = c
            .tokenizer
            .decode(&salida, true)
            .map_err(|e| interno(format!("no pude decodificar: {e}")))?;

        // El caché contiene ahora el prompt MÁS lo generado: todo eso pasó por
        // `forward`. Registrarlo completo es lo que hace que el turno siguiente
        // —que empieza justo así— no tenga que reprocesar nada.
        //
        // Si se cortó por `max_tokens` sin llegar a `<|im_end|>`, da igual: lo
        // que importa es qué tokens vio el modelo, no si la frase terminó.
        // El caché contiene ahora el prompt MÁS lo generado: todo eso pasó por
        // `forward`. Guardarlo completo es lo que hace que el turno siguiente
        // —que empieza justo así— no tenga que reprocesar nada.
        if con_sesion {
            let mut vistos = entrada.clone();
            vistos.extend_from_slice(&salida);
            Self::guardar(c, &p.sesion, vistos);
        }

        Ok(Respuesta {
            texto: texto.trim().to_string(),
            tokens_prompt: entrada.len() as u32,
            tokens_generados: salida.len() as u32,
            tokens_cacheados: reutilizados as u32,
            estaba_caliente,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Mensaje;

    #[test]
    fn el_formato_lleva_las_marcas_de_turno_de_qwen() {
        let m = vec![
            Mensaje {
                rol: "system".into(),
                contenido: "sos un bot".into(),
            },
            Mensaje {
                rol: "user".into(),
                contenido: "hola".into(),
            },
        ];
        let p = formatear(&m);
        assert!(
            p.contains("<|im_start|>system\nsos un bot<|im_end|>"),
            "{p}"
        );
        assert!(p.contains("<|im_start|>user\nhola<|im_end|>"), "{p}");
        // Tiene que TERMINAR abriendo el turno del asistente, o el modelo
        // seguiría escribiendo el turno del usuario.
        assert!(p.ends_with("<|im_start|>assistant\n"), "{p}");
    }

    #[test]
    fn un_rol_desconocido_cae_a_user() {
        // Un cliente puede mandar "tool" o cualquier cosa. Mejor tratarlo como
        // usuario que romper la petición.
        let m = vec![Mensaje {
            rol: "marciano".into(),
            contenido: "x".into(),
        }];
        assert!(formatear(&m).contains("<|im_start|>user\nx<|im_end|>"));
    }

    #[test]
    fn sin_archivos_falla_al_construir_y_no_al_generar() {
        // Mejor enterarse al arrancar que en la primera petición de un usuario.
        let r = MotorCandle::nuevo(
            "x",
            Path::new("no/existe.gguf"),
            Path::new("no/existe.json"),
        );
        assert!(matches!(r, Err(ErrorMotor::NoCargado(_))));
    }
}
