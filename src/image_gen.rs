//! Crear imágenes en la CPU, sin GPU, sin API y sin candle: SD-Turbo.
//!
//! SD-Turbo (huggingface.co/stabilityai/sd-turbo) es Stable Diffusion 2.1
//! destilado para dar la imagen en **un paso**: Euler, espaciado `trailing`,
//! predicción `epsilon` (su `scheduler_config.json`). Todo es propio desde la
//! 0.11.0, sobre la multiplicación de matrices de [`crate::kernels::sgemm`]:
//!
//! - [`Clip`]: el tokenizador BPE de CLIP (el modelo trae `vocab.json` y
//!   `merges.txt`, no un `tokenizer.json`);
//! - [`clip_text`]: el codificador de texto, 23 capas;
//! - [`unet`]: el UNet, ~865 millones de parámetros en f16, leídos directo del
//!   mapeo del archivo;
//! - [`taesd`]: el decodificador, TAESD (github.com/madebyollin/taesd, MIT), 4,9
//!   MB y nueve veces más rápido que el VAE completo de SD.
//!
//! **Licencia de SD-Turbo**: Stability AI Community License. Para un uso no
//! comercial es gratis; pide mostrar "Powered by Stability AI", seguir su
//! política de uso y no usar lo que genera para entrenar otro modelo generativo.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub mod clip_text;
pub mod nn;
pub mod safetensors;
pub mod taesd;
pub mod unet;

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

// ─── El generador ────────────────────────────────────────────────────────────

/// SD-Turbo cargado: el texto, el UNet y el decodificador.
pub struct Generador {
    clip_tok: Clip,
    texto: clip_text::TextoClip,
    unet: unet::Unet,
    taesd: taesd::Taesd,
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

/// El σ de Euler en el paso `t`, con los betas de SD (`scaled_linear`, de
/// 0,00085 a 0,012 en 1000 pasos): `σ = √((1 − ᾱ) / ᾱ)`.
pub fn sigma(t: usize) -> f64 {
    let (b0, b1) = (0.00085f64.sqrt(), 0.012f64.sqrt());
    let mut acumulado = 1.0;
    for i in 0..=t {
        let beta = (b0 + (b1 - b0) * i as f64 / 999.0).powi(2);
        acumulado *= 1.0 - beta;
    }
    ((1.0 - acumulado) / acumulado).sqrt()
}

impl Generador {
    /// Carga todo de `dir_modelos/sd-turbo` y `dir_modelos/taesd`. Los pesos
    /// del UNet y del texto quedan en f16 en el mapeo de los archivos.
    pub fn abrir(dir_modelos: &Path) -> Resultado<Self> {
        for a in archivos(dir_modelos) {
            if !a.exists() {
                return Err(format!("falta {}", a.display()));
            }
        }
        let sd = dir_modelos.join("sd-turbo");
        Ok(Self {
            clip_tok: Clip::abrir(&sd.join("tokenizer"))?,
            texto: clip_text::TextoClip::abrir(&sd.join("text_encoder/model.fp16.safetensors"))?,
            unet: unet::Unet::abrir(&sd.join("unet/diffusion_pytorch_model.fp16.safetensors"))?,
            taesd: taesd::Taesd::abrir(&dir_modelos.join("taesd/taesd_decoder.safetensors"))?,
        })
    }

    /// Los latentes de un prompt, sin decodificar: el texto, el ruido de la
    /// semilla y un paso de Euler.
    ///
    /// Un paso con espaciado `trailing` es `t = 999`. El ruido inicial se
    /// multiplica por `σ(999)` (Euler, `trailing`: `init_noise_sigma` es el σ
    /// máximo), la entrada del UNet se divide por `√(σ² + 1)`, y con predicción
    /// `epsilon` el paso a `σ = 0` deja `x₀ = x − σ·ε`.
    pub fn latentes(&self, prompt: &str, semilla: u64, lado: usize) -> Resultado<Vec<f32>> {
        if !(64..=768).contains(&lado) || !lado.is_multiple_of(8) {
            return Err(format!(
                "lado {lado}: tiene que ser múltiplo de 8, de 64 a 768"
            ));
        }
        let ids = self.clip_tok.codificar(prompt, "!")?;
        let texto = self.texto.codificar(&ids)?;
        let l = lado / 8;
        let s = sigma(999) as f32;
        let mut azar = crate::daiss::Azar::con_semilla(semilla);
        let x: Vec<f32> = (0..4 * l * l).map(|_| azar.normales(1.0).0 * s).collect();
        let escala = 1.0 / (s * s + 1.0).sqrt();
        let entrada: Vec<f32> = x.iter().map(|v| v * escala).collect();
        let ruido = self.unet.aplicar(&entrada, (l, l), 999.0, &texto);
        Ok(x.iter().zip(&ruido).map(|(v, e)| v - s * e).collect())
    }

    /// Una imagen de `lado`×`lado` (múltiplo de 8) para `prompt`, en un paso.
    /// La misma semilla da la misma imagen.
    pub fn generar(&self, prompt: &str, semilla: u64, lado: usize) -> Resultado<Imagen> {
        let lat = self.latentes(prompt, semilla, lado)?;
        let l = lado / 8;
        let (img, alto, ancho) = self.taesd.decodificar(&lat, (l, l));
        let hw = alto * ancho;
        let mut rgb = vec![0u8; hw * 3];
        for (p, px) in rgb.chunks_mut(3).enumerate() {
            for (c, v) in px.iter_mut().enumerate() {
                *v = (img[c * hw + p] * 255.0).round() as u8;
            }
        }
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
    fn sigma_de_sd() {
        // El σ máximo de SD con estos betas: ~14,6.
        assert!((sigma(999) - 14.6146).abs() < 1e-3, "{}", sigma(999));
        assert!(sigma(0) < 0.03);
    }

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
}
