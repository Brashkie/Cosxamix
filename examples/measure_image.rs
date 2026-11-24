//! Una imagen con SD-Turbo, sin candle, medida por etapas: cargar, el texto
//! (CLIP), el UNet y TAESD, y la memoria pico.
//!
//!     cargo run --release --features zig,c --example measure_image -- "un gato astronauta"
//!
//! El modelo va en `modelos/sd-turbo/` y `modelos/taesd/` (ver `npm run
//! modelo:imagen` en WinsiBot). `COSXAMIX_LADO` (512) y `COSXAMIX_SEMILLA` (42).
//! Escribe `imagen.ppm`.

use std::path::Path;
use std::time::Instant;

use cosxamix::image_gen::{clip_text::TextoClip, sigma, taesd::Taesd, unet::Unet, Clip};

fn env<T: std::str::FromStr>(k: &str, defecto: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(defecto)
}

fn rss_pico_mb() -> f64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Pmc {
            cb: u32,
            fallos: u32,
            pico: usize,
            actual: usize,
            otros: [usize; 6],
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(p: isize, c: *mut Pmc, cb: u32) -> i32;
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        let mut c = Pmc {
            cb: std::mem::size_of::<Pmc>() as u32,
            ..Default::default()
        };
        // SAFETY: estructura del tamaño que declara `cb`.
        unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        c.pico as f64 / 1e6
    }
    #[cfg(not(windows))]
    0.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cosxamix::machine_load::antes_de_medir();
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a cat astronaut floating in space, digital art".into());
    let lado: usize = env("COSXAMIX_LADO", 512);
    let semilla: u64 = env("COSXAMIX_SEMILLA", 42);
    let sd = Path::new("modelos/sd-turbo");

    let t = Instant::now();
    let tok = Clip::abrir(&sd.join("tokenizer"))?;
    let texto = TextoClip::abrir(&sd.join("text_encoder/model.fp16.safetensors"))?;
    let unet = Unet::abrir(&sd.join("unet/diffusion_pytorch_model.fp16.safetensors"))?;
    let taesd = Taesd::abrir(Path::new("modelos/taesd/taesd_decoder.safetensors"))?;
    let cargar = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let emb = texto.codificar(&tok.codificar(&prompt, "!")?)?;
    let t_texto = t.elapsed().as_secs_f64();

    let l = lado / 8;
    let s = sigma(999) as f32;
    let mut azar = cosxamix::daiss::Azar::con_semilla(semilla);
    let x: Vec<f32> = (0..4 * l * l).map(|_| azar.normales(1.0).0 * s).collect();
    let escala = 1.0 / (s * s + 1.0).sqrt();
    let entrada: Vec<f32> = x.iter().map(|v| v * escala).collect();
    let t = Instant::now();
    let ruido = unet.aplicar(&entrada, (l, l), 999.0, &emb);
    let t_unet = t.elapsed().as_secs_f64();
    let lat: Vec<f32> = x.iter().zip(&ruido).map(|(v, e)| v - s * e).collect();

    let t = Instant::now();
    let (img, alto, ancho) = taesd.decodificar(&lat, (l, l));
    let t_taesd = t.elapsed().as_secs_f64();

    let hw = alto * ancho;
    let mut ppm = format!("P6\n{ancho} {alto}\n255\n").into_bytes();
    for p in 0..hw {
        for c in 0..3 {
            ppm.push((img[c * hw + p] * 255.0).round() as u8);
        }
    }
    std::fs::write("imagen.ppm", ppm)?;
    println!(
        "{prompt:?}, {lado}x{lado}, 1 paso\n  cargar {cargar:.1} s, texto {t_texto:.2} s, UNet {t_unet:.2} s, \
         TAESD {t_taesd:.2} s — imagen {:.2} s (sin cargar), memoria pico {:.0} MB",
        t_texto + t_unet + t_taesd,
        rss_pico_mb()
    );
    Ok(())
}
