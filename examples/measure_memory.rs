//! ¿Cuánto cuesta tener más de una instancia del modelo cargada?
//!
//! Es la pregunta que decide si se puede dar un caché de atención por sesión.
//! `ModelWeights` no es `Clone` y su `kv_cache` es privado, así que la única
//! forma de tener dos cachés independientes sin parchear candle es cargar el
//! modelo dos veces.
//!
//! Si cada instancia cuesta el tamaño completo del GGUF, la idea no escala más
//! allá de dos o tres sesiones. Si candle comparte los pesos, cambia todo. Se
//! mide en vez de suponerlo.

use cosxamix::engine_candle::MotorCandle;

// Sin este `cfg` las dos versiones quedan definidas a la vez fuera de Windows y
// el ejemplo no compila. En Windows el error NO aparece, porque alli la que
// queda fuera es la otra: compilar en una sola plataforma no puede encontrarlo.
#[cfg(windows)]
fn memoria_mb() -> f64 {
    // Se lee con `tasklist` en vez de sumar una dependencia nativa: esto es un
    // ejemplo de diagnóstico, no código de producción.
    let pid = std::process::id();
    let salida = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .expect("tasklist");
    let texto = String::from_utf8_lossy(&salida.stdout);

    // El formato real es:  "powershell.exe","17904","Console","1","72,788 KB"
    //
    // Dos trampas que costaron una medición en cero, reportada como si fuera un
    // hallazgo: el número lleva **coma de miles**, así que cortar por comas lo
    // parte al medio; y la unidad es "KB", no "K", así que quitar una 'K' final
    // deja una 'B' pegada y el parseo falla.
    //
    // Se corta por comillas y se queda solo con los dígitos, que es lo único
    // que no depende del idioma ni del separador.
    let ultimo = texto
        .split('"')
        .rfind(|c| !c.trim().is_empty() && c.trim() != ",")
        .unwrap_or("");

    let digitos: String = ultimo.chars().filter(char::is_ascii_digit).collect();
    digitos.parse::<f64>().map(|kb| kb / 1024.0).unwrap_or(0.0)
}

#[cfg(not(windows))]
fn memoria_mb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

fn main() {
    // Un modelo por corrida: cargar tres a la vez no mide lo que ocupa cada
    // uno, mide la suma.
    let gguf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "modelos/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());
    let tok = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "modelos/tokenizer.json".into());

    let disco = std::fs::metadata(&gguf)
        .map(|m| m.len() as f64 / 1048576.0)
        .unwrap_or(0.0);
    let base = memoria_mb();

    let m = MotorCandle::nuevo("x", std::path::Path::new(&gguf), std::path::Path::new(&tok))
        .expect("construir");
    m.precalentar().expect("precalentar");
    let cargado = memoria_mb();

    println!(
        "{:<34} disco {disco:>7.1} MB  residente {:>7.1} MB  (x{:.2})",
        gguf.rsplit('/').next().unwrap_or(&gguf),
        cargado - base,
        (cargado - base) / disco,
    );
}
