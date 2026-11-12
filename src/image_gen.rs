//! Crear imágenes en la CPU, sin GPU y sin API: SD-Turbo con candle.
//!
//! SD-Turbo (huggingface.co/stabilityai/sd-turbo) es Stable Diffusion 2.1
//! destilado para dar la imagen en **un paso**: Euler, espaciado `trailing`,
//! predicción `epsilon` (su `scheduler_config.json`). Medido en un i5-10400 a
//! 512×512 (`examples/measure_image.rs`): el texto 0,5 s, el UNet ~10 s en f32 y
//! ~12,6 en f16, y el decodificado 3 s con TAESD contra 27 con el VAE completo.
//! En f16 la memoria pico baja de 6,6 a 4,2 GB, y es lo que se usa: tiene que
//! convivir con el modelo de texto.
//!
//! - El texto se tokeniza con el BPE de CLIP, escrito acá ([`Clip`]): el modelo
//!   trae `vocab.json` y `merges.txt`, no un `tokenizer.json`.
//! - El decodificado es **TAESD** ([`Taesd`], github.com/madebyollin/taesd,
//!   licencia MIT): 4,9 MB que leen los mismos latentes que el VAE de SD 2.x, y
//!   nueve veces más rápido. El VAE completo queda con `COSXAMIX_IMAGEN_VAE=completo`.
//!
//! **Licencia de SD-Turbo**: Stability AI Community License. Para un uso no
//! comercial es gratis; pide mostrar "Powered by Stability AI", seguir su
//! política de uso y no usar lo que genera para entrenar otro modelo generativo.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion::{
    self,
    clip::ClipTextTransformer,
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{PredictionType, SchedulerConfig, TimestepSpacing},
    unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
};

type Resultado<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

// ─── El tokenizador de CLIP ──────────────────────────────────────────────────

/// El BPE de CLIP: minúsculas, la última pieza de cada palabra con `</w>`, y
/// los bytes pasados a caracteres como en GPT-2. Lo justo para un prompt.
pub struct Clip {
    vocab: HashMap<String, u32>,
    rangos: HashMap<(String, String), usize>,
    byte_a_char: [char; 256],
}

impl Clip {
    pub fn abrir(dir: &Path) -> Resultado<Self> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).map_err(err)?)
                .map_err(err)?;
        let merges = std::fs::read_to_string(dir.join("merges.txt")).map_err(err)?;
        Ok(Self::nuevo(vocab, &merges))
    }

    fn nuevo(vocab: HashMap<String, u32>, merges: &str) -> Self {
        let mut rangos = HashMap::new();
        for (i, l) in merges
            .lines()
            .filter(|l| !l.starts_with("#version"))
            .enumerate()
        {
            if let Some((a, b)) = l.split_once(' ') {
                rangos.insert((a.to_string(), b.to_string()), i);
            }
        }
        // bytes_to_unicode de GPT-2: los imprimibles quedan, los demás pasan a
        // 256 en adelante.
        let mut byte_a_char = ['\0'; 256];
        let mut n = 0u32;
        for b in 0..256u32 {
            let directo =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            byte_a_char[b as usize] = if directo {
                char::from_u32(b).unwrap_or('?')
            } else {
                n += 1;
                char::from_u32(255 + n).unwrap_or('?')
            };
        }
        Self {
            vocab,
            rangos,
            byte_a_char,
        }
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

    /// Las palabras como las corta CLIP: letras seguidas, cada dígito suelto y
    /// lo demás seguido, separados por espacios.
    fn palabras(texto: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut actual = String::new();
        let mut clase = 0u8;
        for c in texto.to_lowercase().chars() {
            let k = if c.is_whitespace() {
                0
            } else if c.is_alphabetic() {
                1
            } else if c.is_numeric() {
                2
            } else {
                3
            };
            if (k == 0 || k == 2 || k != clase) && !actual.is_empty() {
                out.push(std::mem::take(&mut actual));
            }
            if k == 2 {
                out.push(c.to_string());
            } else if k != 0 {
                actual.push(c);
            }
            clase = k;
        }
        if !actual.is_empty() {
            out.push(actual);
        }
        out
    }

    /// Los 77 ids: inicio, el texto, fin, y `relleno` hasta completar.
    pub fn codificar(&self, texto: &str, relleno: &str) -> Resultado<Vec<u32>> {
        let id = |t: &str| {
            self.vocab
                .get(t)
                .copied()
                .ok_or_else(|| format!("el vocabulario no trae {t}"))
        };
        let (inicio, fin, relleno) = (id("<|startoftext|>")?, id("<|endoftext|>")?, id(relleno)?);
        let mut ids = vec![inicio];
        for p in Self::palabras(texto) {
            for t in self.bpe(&p) {
                if let Some(&i) = self.vocab.get(&t) {
                    ids.push(i);
                }
            }
        }
        ids.truncate(76);
        ids.push(fin);
        ids.resize(77, relleno);
        Ok(ids)
    }
}

// ─── TAESD ───────────────────────────────────────────────────────────────────

/// `conv.forward(x)`, esquivando un error de candle 0.9.2.
///
/// `conv2d_tiled` decide si la entrada ya está en NHWC comparando sus strides
/// con los de NHWC, y los de un NCHW contiguo coinciden cuando canales = alto =
/// ancho. Ahí la toma como NHWC sin copiarla, y lee los píxeles en otro orden:
/// TAESD tiene justo ese caso, 64 canales sobre latentes de 64×64, y salían
/// franjas de colores. Con otros strides —una vista una columna más ancha— copia
/// bien. Lo comprueba `convolucion_cuadrada_como_la_directa`.
fn conv(c: &candle_nn::Conv2d, x: &Tensor) -> candle_core::Result<Tensor> {
    let (_, ch, h, w) = x.dims4()?;
    if ch == h && h == w {
        let ancho = x.pad_with_zeros(3, 0, 1)?;
        return c.forward(&ancho.narrow(3, 0, w)?);
    }
    c.forward(x)
}

/// TAESD: convoluciones de 64 canales, tres bloques por escala y tres
/// ampliaciones ×2. Los tensores se llaman por su índice en el
/// `nn.Sequential` del original (`taesd.py`).
pub struct Taesd {
    capas: Vec<Capa>,
}

enum Capa {
    Conv(candle_nn::Conv2d),
    Relu,
    Bloque(candle_nn::Conv2d, candle_nn::Conv2d, candle_nn::Conv2d),
    Ampliar,
}

impl Taesd {
    pub fn abrir(ruta: &Path, dev: &Device) -> Resultado<Self> {
        use candle_nn::{conv2d, conv2d_no_bias, Conv2dConfig, VarBuilder};
        // SAFETY: el archivo de pesos no cambia mientras se usa.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ruta], DType::F32, dev) }
            .map_err(err)?;
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
        let armar = || -> candle_core::Result<Vec<Capa>> {
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
            Ok(capas)
        };
        Ok(Self {
            capas: armar().map_err(err)?,
        })
    }

    /// Los latentes del UNet a una imagen en [0, 1].
    pub fn decodificar(&self, lat: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = ((lat / 3.)?.tanh()? * 3.)?;
        for capa in &self.capas {
            x = match capa {
                Capa::Conv(c) => conv(c, &x)?,
                Capa::Relu => x.relu()?,
                Capa::Ampliar => {
                    let (_, _, h, w) = x.dims4()?;
                    x.upsample_nearest2d(h * 2, w * 2)?
                }
                Capa::Bloque(a, b, c) => {
                    let y = conv(c, &conv(b, &conv(a, &x)?.relu()?)?.relu()?)?;
                    (y + &x)?.relu()?
                }
            };
        }
        x.clamp(0f32, 1f32)
    }
}

// ─── El generador ────────────────────────────────────────────────────────────

/// SD-Turbo cargado.
pub struct Generador {
    clip_tok: Clip,
    relleno: String,
    clip: ClipTextTransformer,
    unet: UNet2DConditionModel,
    decodificador: Decodificador,
    tipo: DType,
    dev: Device,
}

enum Decodificador {
    Taesd(Taesd),
    Completo(Box<AutoEncoderKL>),
}

/// Una imagen en RGB, fila por fila.
pub struct Imagen {
    pub ancho: usize,
    pub alto: usize,
    pub rgb: Vec<u8>,
}

/// Si en `dir_modelos` están los archivos que hacen falta.
pub fn disponible(dir_modelos: &Path) -> bool {
    archivos(dir_modelos).iter().all(|a| a.exists())
}

fn archivos(dir: &Path) -> Vec<PathBuf> {
    let sd = dir.join("sd-turbo");
    vec![
        sd.join("text_encoder/model.fp16.safetensors"),
        sd.join("unet/diffusion_pytorch_model.fp16.safetensors"),
        sd.join("tokenizer/vocab.json"),
        sd.join("tokenizer/merges.txt"),
        dir.join("taesd/taesd_decoder.safetensors"),
    ]
}

impl Generador {
    /// Carga todo de `dir_modelos/sd-turbo` y `dir_modelos/taesd`. En f16
    /// salvo `COSXAMIX_IMAGEN_TIPO=f32` (+25% de velocidad, +2,4 GB).
    pub fn abrir(dir_modelos: &Path) -> Resultado<Self> {
        for a in archivos(dir_modelos) {
            if !a.exists() {
                return Err(format!("falta {}", a.display()));
            }
        }
        let tipo = match std::env::var("COSXAMIX_IMAGEN_TIPO").as_deref() {
            Ok("f32") => DType::F32,
            _ => DType::F16,
        };
        let dev = Device::Cpu;
        let sd = dir_modelos.join("sd-turbo");
        let cfg = stable_diffusion::StableDiffusionConfig::v2_1(None, Some(512), Some(512));
        let relleno = cfg
            .clip
            .pad_with
            .clone()
            .unwrap_or_else(|| "<|endoftext|>".into());
        let clip = stable_diffusion::build_clip_transformer(
            &cfg.clip,
            sd.join("text_encoder/model.fp16.safetensors"),
            &dev,
            tipo,
        )
        .map_err(err)?;
        let unet = cfg
            .build_unet(
                sd.join("unet/diffusion_pytorch_model.fp16.safetensors"),
                &dev,
                4,
                false,
                tipo,
            )
            .map_err(err)?;
        let decodificador = if std::env::var("COSXAMIX_IMAGEN_VAE").as_deref() == Ok("completo") {
            Decodificador::Completo(Box::new(
                cfg.build_vae(
                    sd.join("vae/diffusion_pytorch_model.fp16.safetensors"),
                    &dev,
                    tipo,
                )
                .map_err(err)?,
            ))
        } else {
            Decodificador::Taesd(Taesd::abrir(
                &dir_modelos.join("taesd/taesd_decoder.safetensors"),
                &dev,
            )?)
        };
        Ok(Self {
            clip_tok: Clip::abrir(&sd.join("tokenizer"))?,
            relleno,
            clip,
            unet,
            decodificador,
            tipo,
            dev,
        })
    }

    /// Una imagen de `lado`×`lado` (múltiplo de 8) para `prompt`, en un paso.
    /// La misma semilla da la misma imagen.
    pub fn generar(&self, prompt: &str, semilla: u64, lado: usize) -> Resultado<Imagen> {
        if !(64..=768).contains(&lado) || !lado.is_multiple_of(8) {
            return Err(format!(
                "lado {lado}: tiene que ser múltiplo de 8, de 64 a 768"
            ));
        }
        let ids = self.clip_tok.codificar(prompt, &self.relleno)?;
        let ids = Tensor::new(ids.as_slice(), &self.dev)
            .and_then(|t| t.unsqueeze(0))
            .map_err(err)?;
        let texto = self.clip.forward(&ids).map_err(err)?;

        let mut sched = EulerAncestralDiscreteSchedulerConfig {
            prediction_type: PredictionType::Epsilon,
            timestep_spacing: TimestepSpacing::Trailing,
            ..Default::default()
        }
        .build(1)
        .map_err(err)?;

        // El ruido inicial, con la semilla.
        let l = lado / 8;
        let mut azar = crate::daiss::Azar::con_semilla(semilla);
        let ruido: Vec<f32> = (0..4 * l * l).map(|_| azar.normales(1.0).0).collect();
        let mut lat = (Tensor::from_vec(ruido, (1, 4, l, l), &self.dev).map_err(err)?
            * sched.init_noise_sigma())
        .and_then(|t| t.to_dtype(self.tipo))
        .map_err(err)?;
        for t in sched.timesteps().to_vec() {
            let entrada = sched.scale_model_input(lat.clone(), t).map_err(err)?;
            let ruido = self.unet.forward(&entrada, t as f64, &texto).map_err(err)?;
            lat = sched.step(&ruido, t, &lat).map_err(err)?;
        }

        let img = match &self.decodificador {
            Decodificador::Taesd(d) => lat.to_dtype(DType::F32).and_then(|l| d.decodificar(&l)),
            Decodificador::Completo(vae) => (&lat / 0.18215)
                .and_then(|l| vae.decode(&l))
                .and_then(|i| (i / 2.)? + 0.5)
                .and_then(|i| i.clamp(0f32, 1f32)),
        }
        .and_then(|i| (i * 255.)?.to_dtype(DType::U8)?.squeeze(0))
        .map_err(err)?;
        let (_, alto, ancho) = img.dims3().map_err(err)?;
        let rgb = img
            .permute((1, 2, 0))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<u8>())
            .map_err(err)?;
        Ok(Imagen { ancho, alto, rgb })
    }
}

// ─── PNG ─────────────────────────────────────────────────────────────────────

fn crc32(datos: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in datos {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// Un PNG RGB sin comprimir (bloques "stored" de deflate): quien lo recibe lo
/// pasa a JPEG, así que comprimirlo acá sería trabajo tirado.
pub fn png(img: &Imagen) -> Vec<u8> {
    let fila = img.ancho * 3;
    let mut crudo = Vec::with_capacity((fila + 1) * img.alto);
    for y in 0..img.alto {
        crudo.push(0); // sin filtro
        crudo.extend_from_slice(&img.rgb[y * fila..(y + 1) * fila]);
    }
    let mut z = vec![0x78, 0x01];
    let trozos: Vec<&[u8]> = crudo.chunks(65_535).collect();
    for (i, t) in trozos.iter().enumerate() {
        z.push(u8::from(i + 1 == trozos.len()));
        let n = t.len() as u16;
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(!n).to_le_bytes());
        z.extend_from_slice(t);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &crudo {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut trozo = |tipo: &[u8; 4], datos: &[u8]| {
        out.extend_from_slice(&(datos.len() as u32).to_be_bytes());
        let mut c = tipo.to_vec();
        c.extend_from_slice(datos);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(img.ancho as u32).to_be_bytes());
    ihdr.extend_from_slice(&(img.alto as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8 bits, RGB
    trozo(b"IHDR", &ihdr);
    trozo(b"IDAT", &z);
    trozo(b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_y_png_conocidos() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
        let img = Imagen {
            ancho: 2,
            alto: 1,
            rgb: vec![255, 0, 0, 0, 0, 255],
        };
        let p = png(&img);
        assert_eq!(&p[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&p[p.len() - 12..], b"\0\0\0\0IEND\xaeB`\x82");
    }

    #[test]
    fn el_tokenizador_corta_como_clip() {
        assert_eq!(
            Clip::palabras("A Cat, 42 astronauts!!"),
            ["a", "cat", ",", "4", "2", "astronauts", "!!"]
        );
        let vocab: HashMap<String, u32> = [
            ("<|startoftext|>", 0),
            ("<|endoftext|>", 1),
            ("!", 2),
            ("c", 3),
            ("a", 4),
            ("t</w>", 5),
            ("ca", 6),
            ("cat</w>", 7),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let tok = Clip::nuevo(vocab, "#version: 0.2\nc a\nca t</w>\n");
        let ids = tok.codificar("Cat", "!").unwrap();
        assert_eq!(&ids[..3], &[0, 7, 1]);
        assert_eq!(ids.len(), 77);
        assert!(ids[3..].iter().all(|&i| i == 2));
    }

    /// El error de candle que esquiva [`conv`]: con canales = alto = ancho, la
    /// convolución tiene que dar lo mismo que la definición, a mano.
    #[test]
    fn convolucion_cuadrada_como_la_directa() {
        let dev = Device::Cpu;
        let n = 4usize;
        let mut azar = crate::daiss::Azar::con_semilla(7);
        let x: Vec<f32> = (0..n * n * n).map(|_| azar.normales(1.0).0).collect();
        let w: Vec<f32> = (0..n * n * 9).map(|_| azar.normales(0.5).0).collect();
        let xt = Tensor::from_vec(x.clone(), (1, n, n, n), &dev).unwrap();
        let wt = Tensor::from_vec(w.clone(), (n, n, 3, 3), &dev).unwrap();
        let c = candle_nn::Conv2d::new(
            wt,
            None,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let y = conv(&c, &xt)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for o in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0f32;
                    for ci in 0..n {
                        for dy in 0..3 {
                            for dx in 0..3 {
                                let (yy, xx) = (i as isize + dy - 1, j as isize + dx - 1);
                                if yy < 0 || xx < 0 || yy >= n as isize || xx >= n as isize {
                                    continue;
                                }
                                s += x[(ci * n + yy as usize) * n + xx as usize]
                                    * w[((o * n + ci) * 3 + dy as usize) * 3 + dx as usize];
                            }
                        }
                    }
                    let got = y[(o * n + i) * n + j];
                    assert!((got - s).abs() < 1e-4, "({o},{i},{j}): {got} contra {s}");
                }
            }
        }
    }
}
