//! El motor de inferencia, detrás de un trait.
//!
//! Esto existe para que la decisión de **qué motor usar** no quede soldada al
//! resto del programa. Hoy hay tres candidatos razonables —`llama.cpp` por FFI,
//! `candle` (Rust puro, sin toolchain de C++) y ONNX Runtime— y la forma
//! honesta de elegir es medir arranque, latencia y tamaño del binario en la
//! máquina real, no leer comparativas. Con el trait, probar uno es escribir una
//! implementación más y cambiar una línea en `main`.
//!
//! El trait es deliberadamente chico. Cada método que se agregue acá hay que
//! implementarlo en TODOS los motores, así que solo entra lo que de verdad se
//! necesita.

use std::fmt;

/// Un turno de conversación, tal como llega por la API.
#[derive(Debug, Clone)]
pub struct Mensaje {
    pub rol: String,
    pub contenido: String,
}

/// Lo que se le pide a un motor para una generación.
#[derive(Debug, Clone)]
pub struct Peticion {
    pub modelo: String,
    pub mensajes: Vec<Mensaje>,
    pub temperatura: f32,
    pub max_tokens: u32,

    /// De qué conversación es esta petición.
    ///
    /// El motor guarda un caché de atención por sesión, así que esto decide
    /// cuál se reaprovecha. Vacío significa "sin sesión": se procesa el
    /// prompt entero y no se guarda nada, que es lo correcto para una
    /// petición suelta.
    ///
    /// Es `String` y no un identificador tipado porque viene de fuera —de un
    /// JID de WhatsApp, de una cabecera HTTP— y el motor no tiene por qué
    /// saber de dónde.
    pub sesion: String,
}

/// Lo que devuelve, con lo necesario para medir.
#[derive(Debug, Clone)]
pub struct Respuesta {
    pub texto: String,
    pub tokens_prompt: u32,
    pub tokens_generados: u32,
    /// Cuántos tokens del prompt NO hubo que reprocesar porque ya estaban en el
    /// caché de atención.
    ///
    /// Se informa porque es la única forma de ver si la reutilización está
    /// sirviendo: si siempre vale 0, el caché no acierta nunca y hay que
    /// entender por qué en vez de suponer que funciona.
    pub tokens_cacheados: u32,
    /// Si el modelo ya estaba cargado en memoria al recibir la petición.
    ///
    /// Se informa porque es **la** diferencia entre una respuesta de medio
    /// segundo y una de diez: con el modelo frío hay que leerlo del disco
    /// antes de generar nada. Sin este dato, las dos latencias se mezclan en
    /// un promedio que no dice nada.
    pub estaba_caliente: bool,
}

#[derive(Debug)]
// `NoCargado` e `Interno` todavía no los construye nadie: son los dos fallos
// que solo puede tener un motor que carga pesos de disco, y el de prueba no
// carga nada. Se declaran igual porque forman parte del contrato que tendrá
// que cumplir quien implemente el trait, y descubrirlos recién al escribir el
// primer motor real obligaría a tocar a todos los que ya manejen el error.
#[allow(dead_code)]
pub enum ErrorMotor {
    ModeloDesconocido(String),
    NoCargado(String),
    Interno(String),
}

impl fmt::Display for ErrorMotor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModeloDesconocido(m) => write!(f, "no conozco el modelo '{m}'"),
            Self::NoCargado(m) => write!(f, "el modelo '{m}' no se pudo cargar"),
            Self::Interno(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ErrorMotor {}

/// Un modelo disponible, para `/api/tags`.
#[derive(Debug, Clone)]
pub struct ModeloDisponible {
    pub nombre: String,
    pub bytes: u64,
    /// Cargado en memoria ahora mismo.
    pub en_ram: bool,
}

pub trait Motor: Send + Sync {
    /// Qué modelos puede servir. No implica que estén cargados.
    fn modelos(&self) -> Vec<ModeloDisponible>;

    /// Genera una respuesta. Bloqueante a propósito: la inferencia satura un
    /// hilo haciendo cuentas, no esperando E/S, así que `async` acá solo
    /// serviría para bloquear el ejecutor de tokio. Quien llame lo mete en un
    /// `spawn_blocking`.
    fn generar(&self, p: &Peticion) -> Result<Respuesta, ErrorMotor>;

    /// Genera emitiendo cada trozo de texto a medida que sale.
    ///
    /// `emitir` recibe el texto **nuevo** desde la llamada anterior, no el
    /// acumulado, y devuelve si hay que seguir. Devolver `false` corta la
    /// generación: con respuestas de varios segundos, que un cliente se
    /// desconecte y el servidor siga calculando para nadie no es un detalle.
    ///
    /// La implementación por defecto genera todo y emite una sola vez, para
    /// que un motor que no sepa transmitir siga cumpliendo el contrato. No
    /// transmite de verdad, pero responde.
    fn generar_en_partes(
        &self,
        p: &Peticion,
        emitir: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Respuesta, ErrorMotor> {
        let r = self.generar(p)?;
        emitir(&r.texto);
        Ok(r)
    }
}

// ─── Motor de prueba ─────────────────────────────────────────────────────────

/// Devuelve texto fijo, sin cargar nada.
///
/// No es un relleno: sirve para cerrar el contrato HTTP de punta a punta
/// —WinsiBot apuntando su `OLLAMA_URL` acá y recibiendo una respuesta válida—
/// **antes** de meter un motor de verdad. Si la integración falla, se sabe que
/// el problema está en el contrato y no en la inferencia, que es justo lo que
/// cuesta distinguir cuando se construyen las dos cosas a la vez.
pub struct MotorDemo {
    nombre: String,
}

impl MotorDemo {
    pub fn nuevo(nombre: impl Into<String>) -> Self {
        Self {
            nombre: nombre.into(),
        }
    }
}

impl Motor for MotorDemo {
    fn modelos(&self) -> Vec<ModeloDisponible> {
        vec![ModeloDisponible {
            nombre: self.nombre.clone(),
            bytes: 0,
            en_ram: true,
        }]
    }

    fn generar(&self, p: &Peticion) -> Result<Respuesta, ErrorMotor> {
        // El demo respeta el contrato en serio en vez de ignorar los campos
        // que todavía no "usa": así el camino de error y el recorte por
        // longitud quedan probados ANTES de que llegue el motor real, que es
        // cuando salen caros de depurar.
        if !p.modelo.is_empty() && p.modelo != self.nombre {
            return Err(ErrorMotor::ModeloDesconocido(p.modelo.clone()));
        }

        let ultimo = p
            .mensajes
            .iter()
            .rev()
            .find(|m| m.rol == "user")
            .map(|m| m.contenido.as_str())
            .unwrap_or("");

        // Se cuenta por palabras y no por tokens de verdad: un tokenizador
        // real llega con el motor real, y poner uno aproximado acá daría una
        // cifra que parece exacta y no lo es.
        let palabras = |s: &str| s.split_whitespace().count() as u32;

        let completo = format!(
            "[cosxamix demo · temp {:.2}] Todavía no hay motor de inferencia conectado. \
             Recibí {} caracteres y el último mensaje del usuario fue: \"{}\"",
            p.temperatura,
            ultimo.len(),
            ultimo.chars().take(60).collect::<String>(),
        );

        // Recortar a `max_tokens` es lo que hará el motor real, y conviene que
        // el cliente vea ese comportamiento desde ahora: un cliente que asuma
        // respuestas enteras se rompería recién al cambiar de motor.
        let texto: String = completo
            .split_whitespace()
            .take(p.max_tokens.max(1) as usize)
            .collect::<Vec<_>>()
            .join(" ");

        Ok(Respuesta {
            tokens_prompt: p.mensajes.iter().map(|m| palabras(&m.contenido)).sum(),
            tokens_generados: palabras(&texto),
            tokens_cacheados: 0,
            texto,
            estaba_caliente: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peticion(textos: &[(&str, &str)]) -> Peticion {
        Peticion {
            modelo: "cosxamix-demo".into(),
            mensajes: textos
                .iter()
                .map(|(r, c)| Mensaje {
                    rol: (*r).into(),
                    contenido: (*c).into(),
                })
                .collect(),
            temperatura: 0.8,
            max_tokens: 300,
            sesion: String::new(),
        }
    }

    #[test]
    fn el_demo_contesta_siempre_algo() {
        let m = MotorDemo::nuevo("cosxamix-demo");
        let r = m.generar(&peticion(&[("user", "hola")])).unwrap();
        assert!(!r.texto.is_empty());
    }

    #[test]
    fn toma_el_ultimo_mensaje_del_usuario_no_el_primero() {
        // Importa: el prompt lleva un `system` adelante y puede llevar varios
        // turnos. Confundirse de mensaje haría que el bot respondiera a algo
        // que se dijo hace rato.
        let m = MotorDemo::nuevo("cosxamix-demo");
        let r = m
            .generar(&peticion(&[
                ("system", "sos un bot"),
                ("user", "primera"),
                ("assistant", "ok"),
                ("user", "segunda"),
            ]))
            .unwrap();
        assert!(r.texto.contains("segunda"), "{}", r.texto);
        assert!(!r.texto.contains("primera"), "{}", r.texto);
    }

    #[test]
    fn sin_mensajes_de_usuario_no_revienta() {
        // Un cliente puede mandar solo `system`. Devolver un error acá sería
        // más molesto que contestar algo vacío pero válido.
        let m = MotorDemo::nuevo("cosxamix-demo");
        assert!(m.generar(&peticion(&[("system", "hola")])).is_ok());
    }

    #[test]
    fn un_modelo_que_no_sirve_da_error_y_no_una_respuesta_falsa() {
        // Si contestara igual, un cliente mal configurado creería que está
        // hablando con el modelo que pidió.
        let m = MotorDemo::nuevo("cosxamix-demo");
        let mut p = peticion(&[("user", "hola")]);
        p.modelo = "llama3.2:3b".into();
        assert!(matches!(
            m.generar(&p),
            Err(ErrorMotor::ModeloDesconocido(_))
        ));
    }

    #[test]
    fn respeta_max_tokens() {
        let m = MotorDemo::nuevo("cosxamix-demo");
        let mut p = peticion(&[("user", "hola")]);
        p.max_tokens = 3;
        let r = m.generar(&p).unwrap();
        assert_eq!(r.texto.split_whitespace().count(), 3, "{}", r.texto);
    }

    #[test]
    fn sin_modelo_pedido_contesta_igual() {
        // El endpoint rellena el modelo cuando viene vacío, pero el motor no
        // debe depender de que alguien lo haya hecho.
        let m = MotorDemo::nuevo("cosxamix-demo");
        let mut p = peticion(&[("user", "hola")]);
        p.modelo = String::new();
        assert!(m.generar(&p).is_ok());
    }

    #[test]
    fn se_anuncia_en_la_lista_de_modelos() {
        let m = MotorDemo::nuevo("cosxamix-demo");
        let ms = m.modelos();
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].nombre, "cosxamix-demo");
    }
}
