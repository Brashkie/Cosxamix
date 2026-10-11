//! Varios modelos a la vez, con un presupuesto de memoria.
//!
//! Hasta la 0.2.0 Cosxamix servía **uno**, fijado por `COSXAMIX_GGUF`. Pedir
//! otro devolvía `ModeloDesconocido`, aunque el archivo estuviera ahí al lado.
//!
//! ## Cómo está hecho
//!
//! No reescribe nada: compone. El gestor sostiene un [`MotorCandle`] por cada
//! GGUF que encuentra —cada uno con su tokenizador, su caché de atención y sus
//! sesiones— y decide cuáles están en RAM. La inferencia y el caché por sesión,
//! que es la parte difícil y la que está medida, no se tocan.
//!
//! Que las sesiones vivan **dentro** de cada motor es lo correcto y no un
//! accidente: un caché de atención es una secuencia de estados internos de un
//! modelo concreto. Restaurárselo a otro no daría un error, daría basura.
//!
//! ## El presupuesto, y por qué en bytes y no en cantidad
//!
//! Los modelos no pesan parecido: en este directorio van de 468 MB a 2007 MB.
//! Un tope de "tres modelos" dejaría entrar 6 GB o 1,4 GB según cuáles sean,
//! que es lo mismo que no tener tope. El presupuesto va en bytes, con
//! `COSXAMIX_RAM_MB`.
//!
//! Se mide por el **tamaño del archivo**, que es una buena aproximación desde
//! que los embeddings dejaron de descuantizarse: la relación RAM/disco quedó
//! en x1,02-x1,08, medida. Antes era x2 y esta cuenta habría mentido por el
//! doble.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::engine::{ErrorMotor, ModeloDisponible, Motor, Peticion, Respuesta};
use crate::engine_candle::MotorCandle;

/// Presupuesto por defecto si no se fija `COSXAMIX_RAM_MB`.
///
/// 4 GB entra en una máquina de 8 y deja sitio para el resto. No se detecta la
/// RAM de la máquina a propósito: haría falta una dependencia para leer algo
/// que quien despliega sabe mejor que nosotros.
pub const PRESUPUESTO_MB_POR_DEFECTO: u64 = 4096;

struct Entrada {
    motor: Arc<MotorCandle>,
    bytes: u64,
}

pub struct GestorModelos {
    entradas: HashMap<String, Entrada>,
    /// Reloj lógico del último uso, para el desalojo. No se usa la hora: dos
    /// peticiones en el mismo milisegundo empatarían, y lo que importa es el
    /// orden de llegada.
    usos: Mutex<HashMap<String, u64>>,
    reloj: AtomicU64,
    presupuesto: u64,
    /// Serializa las cargas. Que dos modelos grandes entren a la vez es
    /// justamente lo que el presupuesto quiere evitar, y sin este candado el
    /// cálculo de "cuánto hay cargado" se haría sobre un estado que está
    /// cambiando.
    carga: Mutex<()>,
}

/// Empareja cada GGUF con su tokenizador.
///
/// Un tokenizador equivocado **no da error**: da texto sin sentido, porque los
/// ids existen igual y el modelo los procesa. Por eso el emparejamiento se
/// registra en el log al arrancar, para que se vea.
fn tokenizador_de(gguf: &Path, dir: &Path) -> Option<PathBuf> {
    let nombre = gguf.file_stem()?.to_string_lossy().to_lowercase();

    // Primero el específico, `modelo.tokenizer.json`, que es lo que haría
    // quien quiere mandar sin ambigüedad.
    let propio = dir.join(format!("{nombre}.tokenizer.json"));
    if propio.exists() {
        return Some(propio);
    }

    // Después por la etiqueta de tamaño del nombre: `...-3b-...` busca
    // `tokenizer-3b.json`.
    for etiqueta in ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"] {
        if nombre.contains(&format!("-{etiqueta}-")) || nombre.contains(&format!("-{etiqueta}.")) {
            let p = dir.join(format!("tokenizer-{etiqueta}.json"));
            if p.exists() {
                return Some(p);
            }
        }
    }

    let general = dir.join("tokenizer.json");
    general.exists().then_some(general)
}

impl GestorModelos {
    /// Recorre el directorio y registra un motor por cada GGUF emparejado.
    ///
    /// No carga ninguno: igual que `MotorCandle`, descubrir es barato y cargar
    /// no.
    pub fn descubrir(dir: &Path, presupuesto_mb: u64) -> Result<Self, ErrorMotor> {
        let mut entradas = HashMap::new();

        let leidos = std::fs::read_dir(dir)
            .map_err(|e| ErrorMotor::NoCargado(format!("no pude leer {}: {e}", dir.display())))?;

        for item in leidos.flatten() {
            let ruta = item.path();
            if ruta.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            let Some(nombre) = ruta.file_stem().map(|s| s.to_string_lossy().into_owned()) else {
                continue;
            };
            let Some(tok) = tokenizador_de(&ruta, dir) else {
                tracing::warn!(modelo = %nombre, "sin tokenizador, se omite");
                continue;
            };
            let bytes = item.metadata().map(|m| m.len()).unwrap_or(0);

            match MotorCandle::nuevo(&nombre, &ruta, &tok) {
                Ok(m) => {
                    tracing::info!(
                        modelo = %nombre,
                        mb = bytes / 1_048_576,
                        tokenizador = %tok.file_name().unwrap_or_default().to_string_lossy(),
                        "modelo registrado"
                    );
                    entradas.insert(
                        nombre,
                        Entrada {
                            motor: Arc::new(m),
                            bytes,
                        },
                    );
                }
                Err(e) => tracing::warn!(modelo = %nombre, error = %e, "no se pudo registrar"),
            }
        }

        if entradas.is_empty() {
            return Err(ErrorMotor::NoCargado(format!(
                "no encontré ningún .gguf con tokenizador en {}",
                dir.display()
            )));
        }

        Ok(Self {
            entradas,
            usos: Mutex::new(HashMap::new()),
            reloj: AtomicU64::new(0),
            presupuesto: presupuesto_mb * 1_048_576,
            carga: Mutex::new(()),
        })
    }

    /// El modelo que se usa cuando la petición no nombra ninguno.
    pub fn por_defecto(&self) -> Option<&str> {
        // El más chico: es el que menos tarda en cargar si nadie pidió nada en
        // concreto, y el que menos ocupa si se queda ahí.
        self.entradas
            .iter()
            .min_by_key(|(_, e)| e.bytes)
            .map(|(n, _)| n.as_str())
    }

    pub fn precalentar(&self, nombre: &str) -> Result<(), ErrorMotor> {
        self.asegurar(nombre)?;
        Ok(())
    }

    fn bytes_cargados(&self) -> u64 {
        self.entradas
            .values()
            .filter(|e| e.motor.esta_caliente())
            .map(|e| e.bytes)
            .sum()
    }

    /// Deja el modelo en RAM, desalojando los menos usados si hace falta.
    fn asegurar(&self, nombre: &str) -> Result<Arc<MotorCandle>, ErrorMotor> {
        let entrada = self
            .entradas
            .get(nombre)
            .ok_or_else(|| ErrorMotor::ModeloDesconocido(nombre.to_string()))?;

        let ahora = self.reloj.fetch_add(1, Ordering::Relaxed);
        self.usos
            .lock()
            .expect("mutex envenenado")
            .insert(nombre.to_string(), ahora);

        if entrada.motor.esta_caliente() {
            return Ok(entrada.motor.clone());
        }

        let _carga = self.carga.lock().expect("mutex envenenado");
        // Otra petición pudo cargarlo mientras se esperaba el candado.
        if entrada.motor.esta_caliente() {
            return Ok(entrada.motor.clone());
        }

        while self.bytes_cargados() + entrada.bytes > self.presupuesto {
            let Some(victima) = self.menos_usado_excepto(nombre) else {
                // No queda nada que desalojar. Se carga igual: negarse sería
                // peor que pasarse del presupuesto, que es una guía y no un
                // límite del sistema operativo.
                tracing::warn!(
                    modelo = %nombre,
                    mb = entrada.bytes / 1_048_576,
                    presupuesto_mb = self.presupuesto / 1_048_576,
                    "no entra en el presupuesto y no hay a quien desalojar"
                );
                break;
            };
            if let Some(e) = self.entradas.get(&victima) {
                tracing::info!(modelo = %victima, mb = e.bytes / 1_048_576, "desalojado");
                e.motor.descargar();
            }
        }

        entrada.motor.precalentar()?;
        Ok(entrada.motor.clone())
    }

    fn menos_usado_excepto(&self, excluido: &str) -> Option<String> {
        let usos = self.usos.lock().expect("mutex envenenado");
        self.entradas
            .iter()
            .filter(|(n, e)| n.as_str() != excluido && e.motor.esta_caliente())
            .min_by_key(|(n, _)| usos.get(n.as_str()).copied().unwrap_or(0))
            .map(|(n, _)| n.clone())
    }

    fn elegir(&self, pedido: &str) -> Result<Arc<MotorCandle>, ErrorMotor> {
        let nombre = if pedido.is_empty() {
            self.por_defecto()
                .ok_or_else(|| ErrorMotor::NoCargado("no hay modelos".into()))?
                .to_string()
        } else {
            pedido.to_string()
        };
        self.asegurar(&nombre)
    }
}

impl Motor for GestorModelos {
    fn modelos(&self) -> Vec<ModeloDisponible> {
        let mut v: Vec<_> = self
            .entradas
            .values()
            .flat_map(|e| e.motor.modelos())
            .collect();
        // Orden estable: un catálogo que cambia de orden entre llamadas es
        // molesto de leer y rompe cualquier prueba que lo compare.
        v.sort_by(|a, b| a.nombre.cmp(&b.nombre));
        v
    }

    fn generar(&self, p: &Peticion) -> Result<Respuesta, ErrorMotor> {
        let motor = self.elegir(&p.modelo)?;
        let mut propia = p.clone();
        propia.modelo = String::new(); // ya está elegido; no revalidar el nombre
        motor.generar(&propia)
    }

    fn generar_en_partes(
        &self,
        p: &Peticion,
        emitir: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Respuesta, ErrorMotor> {
        let motor = self.elegir(&p.modelo)?;
        let mut propia = p.clone();
        propia.modelo = String::new();
        motor.generar_en_partes(&propia, emitir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Un directorio limpio y propio de cada test.
    ///
    /// Se borra **al empezar** y no solo al terminar: si un test falla a mitad
    /// no llega al borrado final, y la corrida siguiente arranca con basura
    /// que la hace fallar en otro sitio. Paso de verdad acá, y lo peor no fue
    /// el fallo sino que senalaba la asercion equivocada.
    fn dir_limpio(nombre: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cosxamix_test_{nombre}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("crear el directorio del test");
        d
    }

    /// El nombre del archivo, que es lo que se quiere comparar.
    ///
    /// No se usa `Path::ends_with`: compara **componentes** enteros, no
    /// sufijos de texto, asi que `ends_with("q3_k_m.tokenizer.json")` sobre
    /// `.../qwen2.5-3b-instruct-q3_k_m.tokenizer.json` es siempre falso. Es
    /// facil de leer como si funcionara.
    fn archivo(p: &Path) -> String {
        p.file_name().unwrap().to_string_lossy().into_owned()
    }

    #[test]
    fn empareja_el_tokenizador_por_tamano() {
        let dir = dir_limpio("tok_tamano");
        for n in ["tokenizer.json", "tokenizer-3b.json"] {
            std::fs::write(dir.join(n), "{}").unwrap();
        }

        let tres = dir.join("qwen2.5-3b-instruct-q3_k_m.gguf");
        assert_eq!(
            archivo(&tokenizador_de(&tres, &dir).unwrap()),
            "tokenizer-3b.json"
        );

        // Sin tokenizador propio ni por tamano, cae al general.
        let medio = dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
        assert_eq!(
            archivo(&tokenizador_de(&medio, &dir).unwrap()),
            "tokenizer.json"
        );
    }

    #[test]
    fn el_tokenizador_propio_gana() {
        let dir = dir_limpio("tok_propio");
        for n in [
            "tokenizer.json",
            "tokenizer-3b.json",
            "qwen2.5-3b-instruct-q3_k_m.tokenizer.json",
        ] {
            std::fs::write(dir.join(n), "{}").unwrap();
        }
        let tres = dir.join("qwen2.5-3b-instruct-q3_k_m.gguf");
        assert_eq!(
            archivo(&tokenizador_de(&tres, &dir).unwrap()),
            "qwen2.5-3b-instruct-q3_k_m.tokenizer.json"
        );
    }

    #[test]
    fn sin_tokenizador_no_hay_emparejamiento() {
        let dir = dir_limpio("tok_vacio");
        assert!(tokenizador_de(&dir.join("algo-3b-x.gguf"), &dir).is_none());
    }

    /// Un directorio sin ningun gguf no da un gestor vacio: da error.
    ///
    /// Arrancar con un catalogo vacio dejaria el servidor en pie respondiendo
    /// `ModeloDesconocido` a todo, que es la clase de fallo que se descubre
    /// tarde.
    #[test]
    fn sin_modelos_falla_al_descubrir() {
        let dir = dir_limpio("sin_modelos");
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        assert!(GestorModelos::descubrir(&dir, 4096).is_err());
    }
}
