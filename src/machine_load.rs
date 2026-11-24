//! ¿Está libre la máquina? Antes de medir hay que saberlo.
//!
//! Una medición de velocidad hecha con otro programa usando los núcleos no
//! mide Cosxamix: mide el reparto. Pasó de verdad: el 23-11-2026 la batería de
//! `compare_models` dio 0,9 tok/s en vez de ~14,7 porque en paralelo corría
//! `gnatprove -j0` —las demostraciones SPARK de Fabianus, que el build de ese
//! proyecto lanza con un hilo por núcleo— con la CPU al 100%. Los números
//! parecían de un motor roto y eran de una máquina ocupada.
//!
//! Por eso los ejemplos que miden llaman a [`antes_de_medir`] al empezar:
//!
//! - mira un segundo cuánto de la CPU usa **todo el sistema**;
//! - si pasa del 15%, avisa que lo que sigue no sirve para comparar;
//! - con `COSXAMIX_ESPERAR_CPU=<segundos>`, espera hasta ese tiempo a que la
//!   máquina se libere antes de seguir.
//!
//! En Windows lo da `GetSystemTimes` (kernel32, que ya viene enlazado) y en
//! Linux y Android `/proc/stat`. En macOS no se mide y no se avisa.

use std::time::Duration;

/// Qué fracción de la CPU, de 0 a 1, estuvo ocupada durante `ventana`.
/// `None` donde no se sabe medir.
pub fn ocupacion(ventana: Duration) -> Option<f32> {
    let (ocupado_0, total_0) = tiempos()?;
    std::thread::sleep(ventana);
    let (ocupado_1, total_1) = tiempos()?;
    let total = total_1.saturating_sub(total_0);
    if total == 0 {
        return None;
    }
    Some(ocupado_1.saturating_sub(ocupado_0) as f32 / total as f32)
}

/// Por encima de esto, una medición de velocidad no se puede comparar con
/// otra. Una máquina de escritorio en reposo, con navegador y editor
/// abiertos, anda entre el 1% y el 8%.
pub const LIMITE: f32 = 0.15;

/// Lo que se llama al empezar un ejemplo que mide. Devuelve si la máquina
/// estaba libre; además lo dice en la salida, para que quede escrito junto a
/// los números.
pub fn antes_de_medir() -> bool {
    let espera: u64 = std::env::var("COSXAMIX_ESPERAR_CPU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut esperado = 0;
    loop {
        let Some(o) = ocupacion(Duration::from_secs(1)) else {
            println!("  máquina: no sé medir la ocupación en esta plataforma");
            return true;
        };
        if o <= LIMITE {
            println!("  máquina: libre ({:.0}% de CPU ocupada)", o * 100.0);
            return true;
        }
        if esperado >= espera {
            println!(
                "  AVISO: la CPU está ocupada al {:.0}% por otros programas.",
                o * 100.0
            );
            println!("  Las velocidades que siguen no sirven para comparar. Puede ser un build en");
            println!(
                "  paralelo (gnatprove de SPARK usa todos los núcleos). Para esperar a que se"
            );
            println!("  libere: COSXAMIX_ESPERAR_CPU=<segundos>.");
            return false;
        }
        if esperado == 0 {
            println!(
                "  máquina: ocupada al {:.0}%, espero hasta {espera} s a que se libere",
                o * 100.0
            );
        }
        std::thread::sleep(Duration::from_secs(4));
        esperado += 5;
    }
}

/// La memoria de este proceso, en bytes.
///
/// Se separan dos cosas que el administrador de tareas suma: la **privada**
/// (activaciones, búferes, caché de atención: lo que el proceso pidió y nadie
/// más puede usar) y la **residente**, que además cuenta las páginas del
/// modelo mapeado que se tocaron. Esas son del archivo: el sistema las suelta
/// si necesita la RAM y las vuelve a leer del disco, así que no son memoria
/// que el motor ocupe en el mismo sentido.
#[derive(Debug, Clone, Copy)]
pub struct Memoria {
    pub residente: u64,
    pub residente_pico: u64,
    pub privada: u64,
    /// En Linux no hay pico de la privada: es la actual.
    pub privada_pico: u64,
}

/// `None` donde no se sabe medir.
#[cfg(windows)]
pub fn memoria() -> Option<Memoria> {
    // PROCESS_MEMORY_COUNTERS.
    #[repr(C)]
    #[derive(Default)]
    struct Contadores {
        cb: u32,
        fallos: u32,
        residente_pico: usize,
        residente: usize,
        cuota_paginada_pico: usize,
        cuota_paginada: usize,
        cuota_no_paginada_pico: usize,
        cuota_no_paginada: usize,
        privada: usize,
        privada_pico: usize,
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(proceso: isize, c: *mut Contadores, cb: u32) -> i32;
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
    }
    let mut c = Contadores {
        cb: std::mem::size_of::<Contadores>() as u32,
        ..Default::default()
    };
    // SAFETY: una estructura del tamaño que declara `cb`.
    if unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return None;
    }
    Some(Memoria {
        residente: c.residente as u64,
        residente_pico: c.residente_pico as u64,
        privada: c.privada as u64,
        privada_pico: c.privada_pico as u64,
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn memoria() -> Option<Memoria> {
    let estado = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb = |clave: &str| -> Option<u64> {
        let linea = estado.lines().find(|l| l.starts_with(clave))?;
        linea
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()
            .map(|v| v * 1024)
    };
    let privada = kb("RssAnon:")?;
    Some(Memoria {
        residente: kb("VmRSS:")?,
        residente_pico: kb("VmHWM:")?,
        privada,
        privada_pico: privada,
    })
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
pub fn memoria() -> Option<Memoria> {
    None
}

/// `(ocupado, total)` acumulados desde el arranque, en la unidad que dé el
/// sistema; solo sirven restados.
#[cfg(windows)]
fn tiempos() -> Option<(u64, u64)> {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        bajo: u32,
        alto: u32,
    }
    extern "system" {
        fn GetSystemTimes(
            ocioso: *mut FileTime,
            nucleo: *mut FileTime,
            usuario: *mut FileTime,
        ) -> i32;
    }
    let (mut ocioso, mut nucleo, mut usuario) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    // SAFETY: tres punteros a estructuras válidas del tamaño de FILETIME.
    let ok = unsafe { GetSystemTimes(&mut ocioso, &mut nucleo, &mut usuario) };
    if ok == 0 {
        return None;
    }
    let v = |f: &FileTime| (u64::from(f.alto) << 32) | u64::from(f.bajo);
    // El tiempo de núcleo **incluye** el ocioso.
    let total = v(&nucleo) + v(&usuario);
    Some((total.saturating_sub(v(&ocioso)), total))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn tiempos() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let linea = stat.lines().next()?;
    let v: Vec<u64> = linea
        .split_whitespace()
        .skip(1)
        .filter_map(|x| x.parse().ok())
        .collect();
    if v.len() < 4 {
        return None;
    }
    // user nice system idle iowait irq softirq steal...: idle e iowait son
    // tiempo en que la CPU no hizo nada.
    let total: u64 = v.iter().sum();
    let ocioso = v[3] + v.get(4).copied().unwrap_or(0);
    Some((total - ocioso, total))
}

#[cfg(not(any(windows, target_os = "linux", target_os = "android")))]
fn tiempos() -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Donde se sabe medir, la ocupación es una fracción entre 0 y 1.
    #[test]
    fn la_memoria_privada_no_pasa_de_la_residente_pico() {
        let Some(m) = memoria() else {
            return;
        };
        assert!(m.residente > 0 && m.residente <= m.residente_pico);
        assert!(m.privada > 0 && m.privada <= m.privada_pico);
    }

    #[test]
    fn la_ocupacion_es_una_fraccion() {
        let Some(o) = ocupacion(Duration::from_millis(200)) else {
            return;
        };
        assert!((0.0..=1.0).contains(&o), "{o}");
    }
}
