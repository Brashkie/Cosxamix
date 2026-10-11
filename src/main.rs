//! Cosxamix — runtime de IA que habla el dialecto de Ollama.
//!
//! El objetivo del primer hito es muy concreto: que WinsiBot funcione contra
//! esto **sin cambiar una línea de su código**, solo apuntando su `OLLAMA_URL`
//! acá. Para eso hacen falta exactamente dos endpoints, que son los únicos que
//! su `ai_chat.rs` llama:
//!
//!   GET  /api/tags              para decidir si hay IA local disponible
//!   POST /v1/chat/completions   para generar, en formato OpenAI
//!
//! Imitar un dialecto ajeno en vez de inventar uno propio es deliberado:
//! permite cambiar el motor de abajo, o volver a Ollama, sin tocar a quien
//! consume. Un formato propio obligaría a modificar cada cliente y a
//! mantenerlo para siempre, a cambio de nada.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use cosxamix::engine::{Mensaje, Motor, MotorDemo, Peticion};
use cosxamix::kernels;

#[derive(Clone)]
struct Estado {
    motor: Arc<dyn Motor>,
}

fn env(clave: &str, por_defecto: &str) -> String {
    std::env::var(clave).unwrap_or_else(|_| por_defecto.into())
}

// ─── /api/tags ───────────────────────────────────────────────────────────────
//
// WinsiBot no mira el detalle: pide la lista y comprueba que algún `name`
// CONTENGA la parte del modelo anterior a los dos puntos ("llama3.2" de
// "llama3.2:3b"). Se devuelven igual los campos que Ollama pone, porque otros
// clientes sí los leen y costaba lo mismo.

#[derive(Serialize)]
struct Etiqueta {
    name: String,
    model: String,
    size: u64,
    /// Fuera del dialecto de Ollama, pero es la cifra que de verdad importa
    /// para la latencia y no hay forma de saberla desde fuera.
    #[serde(rename = "cosxamix_en_ram")]
    en_ram: bool,
}

#[derive(Serialize)]
struct RespuestaTags {
    models: Vec<Etiqueta>,
}

async fn tags(State(e): State<Estado>) -> Json<RespuestaTags> {
    Json(RespuestaTags {
        models: e
            .motor
            .modelos()
            .into_iter()
            .map(|m| Etiqueta {
                model: m.nombre.clone(),
                name: m.nombre,
                size: m.bytes,
                en_ram: m.en_ram,
            })
            .collect(),
    })
}

// ─── /v1/chat/completions ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MensajeEntrada {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct PeticionChat {
    #[serde(default)]
    model: String,
    messages: Vec<MensajeEntrada>,
    #[serde(default = "temperatura_por_defecto")]
    temperature: f32,
    #[serde(default = "max_tokens_por_defecto")]
    max_tokens: u32,
    /// Se acepta y se ignora: todavía no hay streaming. Se declara para que un
    /// cliente que lo mande reciba una respuesta válida en vez de un 422 por
    /// un campo que no esperábamos.
    #[serde(default)]
    #[allow(dead_code)]
    stream: bool,

    /// De qué conversación es esta petición. Fuera del dialecto de OpenAI,
    /// que no tiene el concepto: ahí cada llamada es independiente.
    ///
    /// Es lo que permite reaprovechar el caché de atención entre turnos. Sin
    /// esto, cada petición reprocesa el prompt entero — medido, un 76% más
    /// de tiempo cuando varias conversaciones llegan intercaladas.
    ///
    /// Un cliente que no lo mande funciona igual, solo que sin esa ventaja.
    #[serde(default, rename = "cosxamix_sesion")]
    sesion: String,
}

fn temperatura_por_defecto() -> f32 {
    0.8
}
fn max_tokens_por_defecto() -> u32 {
    300
}

#[derive(Serialize)]
struct MensajeSalida {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct Eleccion {
    index: u32,
    message: MensajeSalida,
    finish_reason: String,
}

#[derive(Serialize)]
struct Uso {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct RespuestaChat {
    id: String,
    object: String,
    model: String,
    choices: Vec<Eleccion>,
    usage: Uso,
    /// Propio, igual que `cosxamix_en_ram`: sin esto, una respuesta de medio
    /// segundo y una de diez se mezclan en el mismo promedio y no hay forma
    /// de saber si el problema es generar o cargar.
    #[serde(rename = "cosxamix")]
    extra: Extra,
}

#[derive(Serialize)]
struct Extra {
    ms: u128,
    estaba_caliente: bool,
    /// Tokens del prompt que no hubo que reprocesar.
    tokens_cacheados: u32,
}

async fn chat(
    State(e): State<Estado>,
    Json(req): Json<PeticionChat>,
) -> Result<Json<RespuestaChat>, (StatusCode, Json<serde_json::Value>)> {
    if req.messages.is_empty() {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "messages no puede venir vacío",
        ));
    }

    let modelo = if req.model.is_empty() {
        e.motor
            .modelos()
            .first()
            .map(|m| m.nombre.clone())
            .unwrap_or_default()
    } else {
        req.model.clone()
    };

    let peticion = Peticion {
        modelo: modelo.clone(),
        mensajes: req
            .messages
            .into_iter()
            .map(|m| Mensaje {
                rol: m.role,
                contenido: m.content,
            })
            .collect(),
        temperatura: req.temperature,
        max_tokens: req.max_tokens,
        sesion: req.sesion.clone(),
    };

    // La inferencia satura un hilo haciendo cuentas, no esperando E/S: correrla
    // en el ejecutor de tokio dejaría al servidor sin atender nada mientras
    // genera.
    let motor = e.motor.clone();
    let t0 = Instant::now();
    let salida = tokio::task::spawn_blocking(move || motor.generar(&peticion))
        .await
        .map_err(|err| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("tarea caída: {err}"),
            )
        })?;

    let r = match salida {
        Ok(r) => r,
        Err(e) => return Err(error(StatusCode::BAD_REQUEST, &e.to_string())),
    };

    let ms = t0.elapsed().as_millis();
    tracing::info!(
        modelo = %modelo, ms, caliente = r.estaba_caliente,
        tokens = r.tokens_generados, cacheados = r.tokens_cacheados, "generado"
    );

    Ok(Json(RespuestaChat {
        id: format!("cosxamix-{}", t0.elapsed().as_nanos()),
        object: "chat.completion".into(),
        model: modelo,
        usage: Uso {
            prompt_tokens: r.tokens_prompt,
            completion_tokens: r.tokens_generados,
            total_tokens: r.tokens_prompt + r.tokens_generados,
        },
        choices: vec![Eleccion {
            index: 0,
            message: MensajeSalida {
                role: "assistant".into(),
                content: r.texto,
            },
            finish_reason: "stop".into(),
        }],
        extra: Extra {
            ms,
            estaba_caliente: r.estaba_caliente,
            tokens_cacheados: r.tokens_cacheados,
        },
    }))
}

fn error(c: StatusCode, msg: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        c,
        Json(serde_json::json!({ "error": { "message": msg, "type": "cosxamix" } })),
    )
}

async fn salud() -> impl IntoResponse {
    Json(serde_json::json!({ "ok": true, "servicio": "cosxamix" }))
}

/// Decide con qué motor arrancar.
#[cfg(feature = "candle")]
fn elegir_motor() -> (Arc<dyn Motor>, bool) {
    use cosxamix::engine_candle::MotorCandle;
    use std::path::PathBuf;

    let dir = PathBuf::from(env("COSXAMIX_MODELOS", "./modelos"));
    // El 3B por defecto, decidido midiendo los tres de la familia con las
    // mismas 15 preguntas de respuesta verificable:
    //
    //   0,5B   504 MB   ~32 tok/s    7/15 aciertos
    //   1,5B  1108 MB    19 tok/s    8/15
    //   3B    2051 MB    10 tok/s   10/15
    //
    // El 1,5B quedó descartado: cuesta 1,7x de velocidad y gana una pregunta
    // de quince, que es ruido. Entre los otros dos pesa más acertar: en
    // WhatsApp nadie espera respuesta instantánea de un bot, pero una
    // respuesta rápida que dice que la capital de Australia es Sydney no
    // sirve para nada.
    // Y Q3_K_M y no Q4_K_M, también medido: mismo modelo, misma prueba.
    //
    //   Q2_K    1313 MB  13,7 tok/s   6/15   <- se rompe
    //   Q3_K_M  1644 MB  11,9 tok/s  10/15   <- acá
    //   Q4_K_M  2007 MB  10,8 tok/s  10/15
    //
    // Q3 acierta lo mismo que Q4 con 363 MB menos y un 10% más rápido: menos
    // bytes por peso son menos bytes que leer de la RAM en cada token. Q2 ya
    // no: ahí el daño se ve a simple vista —dice "canossa" por Canberra y
    // 1972 por 1969—, que es como se rompe un modelo sobrecuantizado.
    //
    // Los formatos IQ (IQ3_M, IQ2_XXS) habrían sido mejores por bit, pero
    // candle no los soporta: su enumeración llega hasta los k-quants.
    let gguf = env("COSXAMIX_GGUF", "qwen2.5-3b-instruct-q3_k_m.gguf");
    let tok = env("COSXAMIX_TOKENIZER", "tokenizer-3b.json");
    let nombre = gguf.trim_end_matches(".gguf").to_string();

    match MotorCandle::nuevo(&nombre, &dir.join(&gguf), &dir.join(&tok)) {
        Ok(m) => {
            // Precalentar al arrancar es la razón de ser de Cosxamix: el usuario
            // no debe pagar los ~2 s de carga en su primera pregunta.
            // Medido: frío 3024 ms contra caliente 1088 ms.
            tracing::info!("cargando {nombre}...");
            let t0 = std::time::Instant::now();
            match m.precalentar() {
                Ok(()) => tracing::info!(ms = t0.elapsed().as_millis(), "modelo caliente"),
                Err(e) => tracing::warn!("no pude precalentar: {e}"),
            }
            (Arc::new(m), false)
        }
        Err(e) => {
            tracing::warn!("sin modelo ({e}) — arranco con el motor de prueba");
            (Arc::new(MotorDemo::nuevo("cosxamix-demo")), true)
        }
    }
}

#[cfg(not(feature = "candle"))]
fn elegir_motor() -> (Arc<dyn Motor>, bool) {
    tracing::warn!("compilado sin la feature `candle`: solo el motor de prueba");
    (Arc::new(MotorDemo::nuevo("cosxamix-demo")), true)
}

/// Fija cuántos hilos usa el cálculo matricial.
///
/// Por defecto rayon toma los procesadores **lógicos**, que en un i5-10400 son
/// 12 por hyperthreading. Medido en este modelo, pasar de 6 a 12 hilos:
///
/// | hilos | decode | CPU consumida |
/// |---|---|---|
/// | 6 | 25,58 tok/s | 53,2 s |
/// | 12 | 26,08 tok/s | 81,8 s |
///
/// **2% más rápido por 54% más de CPU.** Los dos hilos de un mismo núcleo
/// compiten por la misma unidad vectorial, así que en un trabajo que es casi
/// todo AVX2 el segundo aporta casi nada y gasta igual. En un teléfono o un
/// portátil eso es calor y batería a cambio de nada.
///
/// `RAYON_NUM_THREADS` sigue mandando: si alguien lo fija, se respeta.
#[cfg(feature = "candle")]
fn configurar_hilos() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        tracing::info!("hilos: los que fije RAYON_NUM_THREADS");
        return;
    }

    let fisicos = num_cpus::get_physical().max(1);
    match rayon::ThreadPoolBuilder::new()
        .num_threads(fisicos)
        .build_global()
    {
        Ok(()) => tracing::info!(
            hilos = fisicos,
            logicos = num_cpus::get(),
            "pool de calculo"
        ),
        // Si ya estaba construido no es un error que valga la pena: se sigue
        // con el que haya.
        Err(e) => tracing::warn!("no pude fijar los hilos ({e}); sigo con el pool por defecto"),
    }
}

#[cfg(not(feature = "candle"))]
fn configurar_hilos() {}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // El motor real si esta compilado Y el modelo esta en su sitio; el de
    // prueba si no. Nunca se cae por falta de modelo: con el demo el contrato
    // HTTP sigue en pie y se puede trabajar en la capa de arriba.
    configurar_hilos();
    let (motor, estado_demo) = elegir_motor();
    let estado = Estado { motor };

    let app = Router::new()
        .route("/api/tags", get(tags))
        .route("/v1/chat/completions", post(chat))
        .route("/health", get(salud))
        .with_state(estado);

    // 11435 y no 11434: así Cosxamix y Ollama pueden correr a la vez y se pueden
    // comparar sin apagar uno.
    let puerto = env("COSXAMIX_PORT", "11435");
    let dir = format!("127.0.0.1:{puerto}");

    let l = tokio::net::TcpListener::bind(&dir)
        .await
        .unwrap_or_else(|e| panic!("no pude escuchar en {dir}: {e}"));

    tracing::info!(
        kernel = %kernels::elegido(),
        disponibles = ?kernels::disponibles(),
        "backend de kernels",
    );
    tracing::info!("cosxamix escuchando en http://{dir}");
    // El aviso tiene que depender del motor que de verdad se cargó, no de una
    // constante: decía "motor de prueba" con el modelo real cargado, que es
    // justo el tipo de log que hace perder una tarde.
    if estado_demo {
        tracing::warn!("motor de prueba: devuelve texto fijo, no hay inferencia real");
    }

    axum::serve(l, app).await.unwrap();
}
