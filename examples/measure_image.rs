//! ¿Se puede crear una imagen en esta CPU, sin GPU y sin API? SD-Turbo con
//! candle, medido por etapas: cargar, el texto (CLIP), el paso del UNet y el
//! VAE, y la memoria.
//!
//!     cargo run --release --features candle --example measure_image -- "un gato astronauta"
//!
//! El modelo va en `modelos/sd-turbo/` (de huggingface.co/stabilityai/sd-turbo:
//! los `.fp16.safetensors` del UNet, el codificador de texto y el VAE, y el
//! tokenizador). Es SD 2.1 destilado para dar la imagen en **un paso**, con el
//! scheduler Euler, espaciado `trailing` y predicción `epsilon` (su
//! `scheduler_config.json`). Licencia: Stability AI Community License.
//!
//! `COSXAMIX_PASOS` (1), `COSXAMIX_LADO` (512), `COSXAMIX_SEMILLA` (42) y
//! `COSXAMIX_TIPO` (`f32` o `f16`). Escribe `imagen.ppm`.

use std::collections::HashMap;
use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_transformers::models::stable_diffusion::{
    self,
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{PredictionType, SchedulerConfig, TimestepSpacing},
};

fn env<T: std::str::FromStr>(k: &str, defecto: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(defecto)
}

/// El tokenizador BPE de CLIP: minúsculas, palabras con `</w>` al final y los
/// bytes pasados a caracteres como en GPT-2. Lo justo para un prompt.
struct Clip {
    vocab: HashMap<String, u32>,
    rangos: HashMap<(String, String), usize>,
    byte_a_char: [char; 256],
}

impl Clip {
    fn abrir(dir: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json"))?)?;
        let merges = std::fs::read_to_string(dir.join("merges.txt"))?;
        let mut rangos = HashMap::new();
        for (i, l) in merges.lines().skip(1).enumerate() {
            if let Some((a, b)) = l.split_once(' ') {
                rangos.insert((a.to_string(), b.to_string()), i);
            }
        }
        // bytes_to_unicode de GPT-2.
        let mut byte_a_char = ['\0'; 256];
        let mut n = 0u32;
        for b in 0..256u32 {
            let directo =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            byte_a_char[b as usize] = if directo {
                char::from_u32(b).unwrap()
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap()
            };
        }
        Ok(Self {
            vocab,
            rangos,
            byte_a_char,
        })
    }

    fn bpe(&self, palabra: &str) -> Vec<String> {
        let mut partes: Vec<String> = palabra
            .bytes()
            .map(|b| self.byte_a_char[b as usize].to_string())
            .collect();
        if let Some(u) = partes.last_mut() {
            u.push_str("</w>");
        }
        loop {
            let mejor = partes
                .windows(2)
                .enumerate()
                .filter_map(|(i, p)| {
                    self.rangos
                        .get(&(p[0].clone(), p[1].clone()))
                        .map(|r| (*r, i))
                })
                .min();
            let Some((_, i)) = mejor else { break };
            let unido = format!("{}{}", partes[i], partes[i + 1]);
            partes.splice(i..i + 2, [unido]);
        }
        partes
    }

    /// Los 77 ids: inicio, el texto, fin, y relleno con `relleno`.
    fn codificar(&self, texto: &str, relleno: u32) -> Vec<u32> {
        let texto = texto.to_lowercase();
        let mut palabras = Vec::new();
        let mut actual = String::new();
        let mut clase = 0u8; // 1 letras, 3 otros
        let empujar = |p: &mut Vec<String>, a: &mut String| {
            if !a.is_empty() {
                p.push(std::mem::take(a));
            }
        };
        for c in texto.chars() {
            let k = if c.is_whitespace() {
                0
            } else if c.is_alphabetic() {
                1
            } else if c.is_numeric() {
                2
            } else {
                3
            };
            if k == 0 || k == 2 || k != clase {
                empujar(&mut palabras, &mut actual);
            }
            if k == 2 {
                palabras.push(c.to_string());
            } else if k != 0 {
                actual.push(c);
            }
            clase = k;
        }
        empujar(&mut palabras, &mut actual);
        let mut ids = vec![self.vocab["<|startoftext|>"]];
        for p in palabras {
            for t in self.bpe(&p) {
                if let Some(&id) = self.vocab.get(&t) {
                    ids.push(id);
                }
            }
        }
        ids.truncate(76);
        ids.push(self.vocab["<|endoftext|>"]);
        while ids.len() < 77 {
            ids.push(relleno);
        }
        ids
    }
}

/// TAESD (github.com/madebyollin/taesd, MIT): un decodificador chico, 4,9 MB,
/// que lee los mismos latentes que el VAE de SD 1.x y 2.x. Convoluciones de 64
/// canales, tres bloques por escala y tres ampliaciones ×2; los nombres de los
/// tensores son los índices de su `nn.Sequential`.
struct Taesd {
    capas: Vec<Capa>,
}

enum Capa {
    Conv(candle_nn::Conv2d),
    Relu,
    Bloque(candle_nn::Conv2d, candle_nn::Conv2d, candle_nn::Conv2d),
    Ampliar,
}

impl Taesd {
    fn abrir(ruta: &std::path::Path, dev: &Device) -> candle_core::Result<Self> {
        use candle_nn::{conv2d, conv2d_no_bias, Conv2dConfig, VarBuilder};
        // SAFETY: el archivo no cambia mientras se usa.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ruta], DType::F32, dev)? };
        let c = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let bloque = |i: usize| -> candle_core::Result<Capa> {
            let v = vb.pp(i.to_string()).pp("conv");
            Ok(Capa::Bloque(
                conv2d(64, 64, 3, c, v.pp("0"))?,
                conv2d(64, 64, 3, c, v.pp("2"))?,
                conv2d(64, 64, 3, c, v.pp("4"))?,
            ))
        };
        let mut capas = vec![Capa::Conv(conv2d(4, 64, 3, c, vb.pp("1"))?), Capa::Relu];
        let mut i = 3;
        for _ in 0..3 {
            for _ in 0..3 {
                capas.push(bloque(i)?);
                i += 1;
            }
            capas.push(Capa::Ampliar);
            capas.push(Capa::Conv(conv2d_no_bias(
                64,
                64,
                3,
                c,
                vb.pp((i + 1).to_string()),
            )?));
            i += 2;
        }
        capas.push(bloque(18)?);
        capas.push(Capa::Conv(conv2d(64, 3, 3, c, vb.pp("19"))?));
        Ok(Self { capas })
    }

    /// Los latentes del UNet a una imagen en [0, 1].
    fn decodificar(&self, lat: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = ((lat / 3.)?.tanh()? * 3.)?;
        for capa in &self.capas {
            x = match capa {
                Capa::Conv(c) => c.forward(&x)?,
                Capa::Relu => x.relu()?,
                Capa::Ampliar => {
                    // El vecino más cercano ×2, con broadcast.
                    let (b, c, h, w) = x.dims4()?;
                    x.reshape((b, c, h, 1, w, 1))?
                        .broadcast_as((b, c, h, 2, w, 2))?
                        .contiguous()?
                        .reshape((b, c, h * 2, w * 2))?
                }
                Capa::Bloque(a, b, c) => {
                    let y = c.forward(&b.forward(&a.forward(&x)?.relu()?)?.relu()?)?;
                    (y + &x)?.relu()?
                }
            };
        }
        x.clamp(0f32, 1f32)
    }
}

fn rss_mb() -> f64 {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct Pmc {
            cb: u32,
            page_fault_count: u32,
            peak_working_set: usize,
            working_set: usize,
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
        c.peak_working_set as f64 / 1e6
    }
    #[cfg(not(windows))]
    0.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a cat astronaut, digital art".into());
    let pasos: usize = env("COSXAMIX_PASOS", 1);
    let lado: usize = env("COSXAMIX_LADO", 512);
    let semilla: u64 = env("COSXAMIX_SEMILLA", 42);
    let tipo = if std::env::var("COSXAMIX_TIPO").as_deref() == Ok("f16") {
        DType::F16
    } else {
        DType::F32
    };
    let dir = std::path::Path::new("modelos/sd-turbo");
    let dev = Device::Cpu;

    let t0 = Instant::now();
    let cfg = stable_diffusion::StableDiffusionConfig::v2_1(None, Some(lado), Some(lado));
    let tok = Clip::abrir(&dir.join("tokenizer"))?;
    let relleno = tok.vocab[cfg.clip.pad_with.as_deref().unwrap_or("<|endoftext|>")];
    let clip = stable_diffusion::build_clip_transformer(
        &cfg.clip,
        dir.join("text_encoder/model.fp16.safetensors"),
        &dev,
        tipo,
    )?;
    let unet = cfg.build_unet(
        dir.join("unet/diffusion_pytorch_model.fp16.safetensors"),
        &dev,
        4,
        false,
        tipo,
    )?;
    let taesd = std::env::var("COSXAMIX_VAE").as_deref() != Ok("completo");
    let vae = if taesd {
        None
    } else {
        Some(cfg.build_vae(
            dir.join("vae/diffusion_pytorch_model.fp16.safetensors"),
            &dev,
            tipo,
        )?)
    };
    let chico = Taesd::abrir(
        std::path::Path::new("modelos/taesd/taesd_decoder.safetensors"),
        &dev,
    )?;
    let cargar = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let ids = tok.codificar(&prompt, relleno);
    let ids = Tensor::new(ids.as_slice(), &dev)?.unsqueeze(0)?;
    let texto = clip.forward(&ids)?;
    let t_texto = t1.elapsed().as_secs_f64();

    let mut sched = EulerAncestralDiscreteSchedulerConfig {
        prediction_type: PredictionType::Epsilon,
        timestep_spacing: TimestepSpacing::Trailing,
        ..Default::default()
    }
    .build(pasos)?;
    let t2 = Instant::now();
    // El ruido inicial, con la semilla: la misma petición da la misma imagen.
    let n = 4 * (lado / 8) * (lado / 8);
    let mut azar = cosxamix::daiss::Azar::con_semilla(semilla);
    let ruido: Vec<f32> = (0..n).map(|_| azar.normales(1.0).0).collect();
    let mut lat = (Tensor::from_vec(ruido, (1, 4, lado / 8, lado / 8), &dev)?
        * sched.init_noise_sigma())?
    .to_dtype(tipo)?;
    for &t in sched.timesteps().to_vec().iter() {
        let entrada = sched.scale_model_input(lat.clone(), t)?;
        let ruido = unet.forward(&entrada, t as f64, &texto)?;
        lat = sched.step(&ruido, t, &lat)?;
    }
    let t_unet = t2.elapsed().as_secs_f64();
    {
        let v = lat.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = v.len() as f32;
        let media = v.iter().sum::<f32>() / n;
        let desvio = (v.iter().map(|x| (x - media).powi(2)).sum::<f32>() / n).sqrt();
        let (mn, mx) = v
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        let fuera = v.iter().filter(|x| x.abs() > 3.0).count();
        eprintln!("latentes: media {media:.3}, desvío {desvio:.3}, min {mn:.2}, max {mx:.2}, |x|>3: {fuera} de {n}");
    }

    let t3 = Instant::now();
    let img = match &vae {
        Some(vae) => ((vae.decode(&(&lat / 0.18215)?)? / 2.)? + 0.5)?.clamp(0f32, 1f32)?,
        None => chico.decodificar(&lat.to_dtype(DType::F32)?)?,
    };
    // Para comparar contra una referencia: los latentes y la imagen en f32.
    if let Ok(d) = std::env::var("COSXAMIX_VOLCAR") {
        let f = |t: &Tensor| -> candle_core::Result<Vec<u8>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect())
        };
        std::fs::write(format!("{d}/lat.f32"), f(&lat)?)?;
        std::fs::write(format!("{d}/img.f32"), f(&img)?)?;
    }
    let img = (img * 255.)?.to_dtype(DType::U8)?.squeeze(0)?;
    let t_vae = t3.elapsed().as_secs_f64();

    let (c, h, w) = img.dims3()?;
    let px = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend_from_slice(&px);
    std::fs::write("imagen.ppm", ppm)?;
    let _ = (c, D::Minus1);

    println!(
        "{prompt:?}, {lado}x{lado}, {pasos} paso(s), {tipo:?}, VAE {}\n  cargar {cargar:.1} s, texto {t_texto:.2} s, \
         UNet {t_unet:.1} s, VAE {t_vae:.1} s — imagen {:.1} s (sin cargar), memoria pico {:.0} MB",
        if taesd { "TAESD" } else { "completo" },
        t_texto + t_unet + t_vae,
        rss_mb()
    );
    Ok(())
}
