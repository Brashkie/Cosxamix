//! Leer un `tokenizer.json` de la forma de GPT-2 y Qwen, sin el crate
//! `tokenizers`.
//!
//! Ese crate es lo que sirve hoy a los modelos de Qwen, y es una librería para
//! todos los tokenizadores del mundo: normalizadores, recortes, relleno,
//! expresiones regulares en tiempo de ejecución. De todo eso, un modelo de esta
//! forma usa cuatro piezas fijas, y son las que están acá:
//!
//! 1. **Tokens especiales** (`<|im_start|>`): se buscan primero, en el texto
//!    crudo, y nunca pasan por lo demás.
//! 2. **NFC** sobre lo que queda.
//! 3. **La expresión regular de GPT-2**, escrita a mano en [`fin_de_pieza`].
//! 4. **BPE sobre bytes**, con el motor de [`crate::tokenizer::Bpe`].
//!
//! Si el archivo pide cualquier otra cosa, [`TokenizadorJson::desde_archivo`]
//! devuelve un error que lo dice, en vez de tokenizar distinto en silencio: un
//! tokenizador equivocado no falla, da texto sin sentido.
//!
//! **Da exactamente los mismos tokens que el crate**, y eso no es una
//! afirmación: `tests/tokenizer_json.rs` lo compara sobre los documentos del
//! proyecto, sobre todo Unicode carácter por carácter y sobre texto al azar.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use unicode_normalization::{is_nfc_quick, IsNormalized, UnicodeNormalization};

use crate::tokenizer::{Bpe, BASE};
use crate::unicode_classes::{en_tabla, ESPACIOS, LETRAS, NUMEROS};

type Resultado<T> = Result<T, String>;

/// La expresión regular que se sabe hacer a mano. Otra cualquiera es un error.
const PATRON_GPT2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// ─── El archivo ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Archivo<'a> {
    #[serde(borrow)]
    added_tokens: Vec<Agregado<'a>>,
    normalizer: Option<Tipo<'a>>,
    pre_tokenizer: serde_json::Value,
    #[serde(borrow)]
    decoder: Option<Tipo<'a>>,
    #[serde(borrow)]
    model: Modelo<'a>,
}

#[derive(Deserialize)]
struct Tipo<'a> {
    #[serde(rename = "type", borrow)]
    tipo: std::borrow::Cow<'a, str>,
}

#[derive(Deserialize)]
struct Agregado<'a> {
    id: u32,
    #[serde(borrow)]
    content: std::borrow::Cow<'a, str>,
    special: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
}

#[derive(Deserialize)]
struct Modelo<'a> {
    #[serde(rename = "type", borrow)]
    tipo: std::borrow::Cow<'a, str>,
    // Las claves llevan caracteres de fuera de ASCII en forma de escape en
    // algunos archivos, y ahí no se pueden prestar: `Cow` copia solo esas.
    #[serde(borrow)]
    vocab: HashMap<std::borrow::Cow<'a, str>, u32>,
    #[serde(borrow)]
    merges: Vec<std::borrow::Cow<'a, str>>,
}

/// La cabecera del archivo compilado, con su versión.
const COMPILADO: &[u8] = b"COSXTOK1";

/// FNV-1a de 64 bits, de a 8 bytes para que 7 MB tarden milisegundos. No es
/// criptográfico ni hace falta: detecta que el JSON cambió, y nadie tiene motivo
/// para falsificarlo en su propia carpeta de modelos.
fn huella_de(b: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ b.len() as u64;
    for t in b.chunks(8) {
        let mut v = [0u8; 8];
        v[..t.len()].copy_from_slice(t);
        h = (h ^ u64::from_le_bytes(v)).wrapping_mul(0x0000_0100_0000_01B3);
        h ^= h >> 29;
    }
    h
}

/// La tabla de GPT-2: a qué carácter corresponde cada byte, para que todos sean
/// imprimibles dentro del JSON. Los imprimibles quedan como están y el resto se
/// corre a partir de U+0100, en orden.
fn caracter_a_byte() -> HashMap<char, u8> {
    let mut m = HashMap::with_capacity(BASE);
    let mut extra = 0u32;
    for b in 0..=255u8 {
        let imprimible = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        let c = if imprimible {
            char::from(b)
        } else {
            let c = char::from_u32(256 + extra).expect("está en el BMP");
            extra += 1;
            c
        };
        m.insert(c, b);
    }
    m
}

fn bytes_de(texto: &str, tabla: &HashMap<char, u8>) -> Resultado<Vec<u8>> {
    texto
        .chars()
        .map(|c| {
            tabla
                .get(&c)
                .copied()
                .ok_or_else(|| format!("el carácter {c:?} del vocabulario no sale de ningún byte"))
        })
        .collect()
}

// ─── El tokenizador ─────────────────────────────────────────────────────────

/// Un token especial. Se busca tal cual en el texto.
#[derive(Debug, Clone)]
struct Especial {
    texto: Vec<u8>,
    id: u32,
    /// `special: true`: al decodificar pidiendo saltarlos, no aparece.
    saltable: bool,
}

/// Un tokenizador de la forma de GPT-2 y Qwen.
#[derive(Debug, Clone)]
pub struct TokenizadorJson {
    bpe: Bpe,
    /// Del más largo al más corto: gana la coincidencia más larga.
    especiales: Vec<Especial>,
    /// Marca los ids de `especiales` que se saltan al decodificar.
    saltables: Vec<bool>,
    /// Un vocabulario entrenado acá ([`Self::de_bpe`]): se codifica como se
    /// entrenó, con el pretokenizador de `tokenizer.rs` y sin NFC, para que el
    /// modelo reciba exactamente los tokens que vio.
    propio: bool,
}

impl TokenizadorJson {
    /// Lee un `tokenizer.json`.
    ///
    /// Parsear los 7 MB de JSON de Qwen es casi todo el costo de cargar —250 ms,
    /// y todo ese texto en memoria—, y lo que sale es siempre lo mismo. Así que
    /// el resultado se guarda compilado al lado (`tokenizer.json.cosxbin`) y las
    /// veces siguientes se lee de ahí. Lleva el hash del JSON del que salió: si
    /// el JSON cambia, no se usa. Y si no se puede escribir —una carpeta de solo
    /// lectura— no pasa nada, se parsea cada vez.
    pub fn desde_archivo(ruta: &Path) -> Resultado<Self> {
        let crudo =
            std::fs::read(ruta).map_err(|e| format!("no pude leer {}: {e}", ruta.display()))?;
        let huella = huella_de(&crudo);
        let compilado = ruta.with_extension("json.cosxbin");
        if let Ok(b) = std::fs::read(&compilado) {
            if let Some(t) = Self::de_compilado(&b, huella) {
                return Ok(t);
            }
        }
        let texto = std::str::from_utf8(&crudo)
            .map_err(|e| format!("{}: no es UTF-8: {e}", ruta.display()))?;
        let t = Self::desde_json(texto).map_err(|e| format!("{}: {e}", ruta.display()))?;
        // A un temporal y después renombrar: un corte a mitad no deja un
        // archivo roto. Igual, uno roto no pasaría `de_compilado`.
        let temporal = compilado.with_extension("cosxbin.parcial");
        if std::fs::write(&temporal, t.a_compilado(huella)).is_ok()
            && std::fs::rename(&temporal, &compilado).is_err()
        {
            let _ = std::fs::remove_file(&temporal);
        }
        Ok(t)
    }

    /// El tokenizador de un modelo, en este orden:
    ///
    /// 1. `<modelo>.tokenizer.json`, si `tok` es ese: quien lo pone ahí quiere
    ///    mandar sin ambigüedad;
    /// 2. el que viene dentro del GGUF, que no pide leer ningún JSON;
    /// 3. el JSON que se haya emparejado con el modelo.
    pub fn para_modelo(gguf: &Path, tok: &Path) -> Resultado<Self> {
        let especifico = gguf
            .file_stem()
            .map(|s| format!("{}.tokenizer.json", s.to_string_lossy()));
        let es_especifico = especifico.is_some()
            && tok.file_name().map(|n| n.to_string_lossy().into_owned()) == especifico;
        if !es_especifico {
            let g = crate::gguf::Gguf::abrir(gguf)?;
            if let Some(t) = Self::del_gguf(&g)? {
                return Ok(t);
            }
        }
        Self::desde_archivo(tok)
    }

    /// Si el GGUF trae su vocabulario adentro: el de ggml o uno entrenado acá.
    pub fn gguf_lo_trae(gguf: &Path) -> bool {
        crate::gguf::Gguf::abrir(gguf).is_ok_and(|g| {
            g.metadatos.contains_key("tokenizer.ggml.tokens")
                || g.metadatos.contains_key("cosxamix.tokenizer.merges")
        })
    }

    /// Un vocabulario entrenado con [`Bpe::entrenar`], el de un daiss propio. No
    /// tiene palabras especiales: el modelo no sabe de turnos de chat, y por eso
    /// [`Self::es_de_chat`] da `false`.
    pub fn de_bpe(bpe: Bpe) -> Self {
        let saltables = vec![false; bpe.tamano()];
        Self {
            bpe,
            especiales: Vec::new(),
            saltables,
            propio: true,
        }
    }

    /// Si conoce la plantilla de chat (`<|im_start|>`). Un modelo sin ella solo
    /// sabe continuar un texto.
    pub fn es_de_chat(&self) -> bool {
        self.token_a_id("<|im_start|>").is_some()
    }

    /// Saca el tokenizador del propio GGUF, si lo trae.
    ///
    /// Los GGUF oficiales llevan el vocabulario adentro
    /// (`tokenizer.ggml.tokens`, `merges` y `token_type`), así que para ellos
    /// no hace falta ningún JSON: ni parsearlo, ni leerlo, ni bajarlo. Da lo
    /// mismo que el `tokenizer.json` del modelo, y hay un test que lo compara.
    ///
    /// Un daiss entrenado acá trae el suyo en `cosxamix.tokenizer.merges`, y se
    /// lee con [`Self::de_bpe`].
    ///
    /// `Ok(None)` si no lo trae —un daiss o un Chavín con vocabulario prestado—;
    /// error si trae uno de otra forma: `gpt2`, `qwen2` es lo que se sabe leer.
    pub fn del_gguf(g: &crate::gguf::Gguf) -> Resultado<Option<Self>> {
        use crate::gguf::Valor;
        let m = &g.metadatos;
        let (Some(Valor::Lista(tokens)), Some(Valor::Lista(merges))) = (
            m.get("tokenizer.ggml.tokens"),
            m.get("tokenizer.ggml.merges"),
        ) else {
            return Ok(crate::daiss::tokenizador_de(g)?.map(Self::de_bpe));
        };
        let texto = |clave: &str| m.get(clave).and_then(|v| v.texto());
        let (modelo, pre) = (texto("tokenizer.ggml.model"), texto("tokenizer.ggml.pre"));
        if modelo != Some("gpt2") || pre != Some("qwen2") {
            return Err(format!(
                "el tokenizador del GGUF es {modelo:?}/{pre:?}: solo se lee gpt2/qwen2"
            ));
        }
        let Some(Valor::Lista(tipos)) = m.get("tokenizer.ggml.token_type") else {
            return Err("el GGUF no trae tokenizer.ggml.token_type".into());
        };
        if tipos.len() != tokens.len() {
            return Err("token_type no mide lo mismo que tokens".into());
        }

        // Los tipos de ggml: 1 normal, 3 de control (los especiales) y 4
        // definido por el usuario (`<tool_call>`, que se busca igual pero se ve
        // al decodificar). Los `[PAD151700]` de relleno son ids que el modelo
        // tiene y el vocabulario no: no aportan nada, como en el JSON.
        let tabla = caracter_a_byte();
        let mut piezas: Vec<Vec<u8>> = vec![Vec::new(); tokens.len()];
        let mut por_bytes: HashMap<Vec<u8>, u32> = HashMap::with_capacity(tokens.len());
        let mut saltables = vec![false; tokens.len()];
        let mut especiales = Vec::new();
        for (id, (t, tipo)) in tokens.iter().zip(tipos).enumerate() {
            let t = t.texto().ok_or("un token no es texto")?;
            if t.starts_with("[PAD") && t.ends_with(']') {
                continue;
            }
            match tipo.entero() {
                Some(3) | Some(4) => {
                    let especial = tipo.entero() == Some(3);
                    piezas[id] = t.as_bytes().to_vec();
                    saltables[id] = especial;
                    especiales.push(Especial {
                        texto: t.as_bytes().to_vec(),
                        id: id as u32,
                        saltable: especial,
                    });
                }
                _ => {
                    let b = bytes_de(t, &tabla)?;
                    piezas[id].clone_from(&b);
                    por_bytes.insert(b, id as u32);
                }
            }
        }
        especiales.sort_by_key(|e| std::cmp::Reverse(e.texto.len()));

        let mut base = [0u32; BASE];
        for (b, lugar) in base.iter_mut().enumerate() {
            *lugar = *por_bytes
                .get(&vec![b as u8][..])
                .ok_or_else(|| format!("el byte {b} no tiene token"))?;
        }
        let mut pares = Vec::with_capacity(merges.len());
        for linea in merges {
            let linea = linea.texto().ok_or("una fusión no es texto")?;
            let (i, d) = linea
                .split_once(' ')
                .ok_or_else(|| format!("fusión mal formada: {linea:?}"))?;
            let (bi, bd) = (bytes_de(i, &tabla)?, bytes_de(d, &tabla)?);
            let buscar = |b: &[u8]| {
                por_bytes
                    .get(b)
                    .copied()
                    .ok_or_else(|| format!("la fusión {linea:?} usa un token que no existe"))
            };
            let mut unida = bi.clone();
            unida.extend_from_slice(&bd);
            pares.push((buscar(&bi)?, buscar(&bd)?, buscar(&unida)?));
        }
        Ok(Some(Self {
            bpe: Bpe::de_partes(base, pares, &piezas)?,
            especiales,
            saltables,
            propio: false,
        }))
    }

    /// El tokenizador compilado: todo en `u32` little-endian y bytes crudos.
    fn a_compilado(&self, huella: u64) -> Vec<u8> {
        let (base, pares, inicios, bytes) = self.bpe.partes_planas();
        let mut b =
            Vec::with_capacity(64 + BASE * 4 + pares.len() * 12 + inicios.len() * 4 + bytes.len());
        let u32_ = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        b.extend_from_slice(COMPILADO);
        b.extend_from_slice(&huella.to_le_bytes());
        for v in base {
            u32_(&mut b, *v);
        }
        u32_(&mut b, pares.len() as u32);
        for (a, d, n) in pares {
            u32_(&mut b, *a);
            u32_(&mut b, *d);
            u32_(&mut b, *n);
        }
        u32_(&mut b, inicios.len() as u32);
        for v in inicios {
            u32_(&mut b, *v);
        }
        u32_(&mut b, bytes.len() as u32);
        b.extend_from_slice(bytes);
        u32_(&mut b, self.especiales.len() as u32);
        for e in &self.especiales {
            u32_(&mut b, e.id);
            u32_(&mut b, u32::from(e.saltable));
            u32_(&mut b, e.texto.len() as u32);
            b.extend_from_slice(&e.texto);
        }
        b
    }

    /// `None` si no es de esta versión, es de otro JSON o está roto.
    fn de_compilado(b: &[u8], huella: u64) -> Option<Self> {
        struct Lector<'a>(&'a [u8]);
        impl<'a> Lector<'a> {
            fn tomar(&mut self, n: usize) -> Option<&'a [u8]> {
                if n > self.0.len() {
                    return None;
                }
                let (a, r) = self.0.split_at(n);
                self.0 = r;
                Some(a)
            }
            fn u32_(&mut self) -> Option<u32> {
                Some(u32::from_le_bytes(self.tomar(4)?.try_into().ok()?))
            }
            fn lista(&mut self, n: usize) -> Option<Vec<u32>> {
                let t = self.tomar(n.checked_mul(4)?)?;
                Some(
                    t.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| u32::from_le_bytes(*c))
                        .collect(),
                )
            }
        }
        let mut l = Lector(b);
        if l.tomar(COMPILADO.len())? != COMPILADO {
            return None;
        }
        if u64::from_le_bytes(l.tomar(8)?.try_into().ok()?) != huella {
            return None;
        }
        let base: [u32; BASE] = l.lista(BASE)?.try_into().ok()?;
        let n_pares = l.u32_()? as usize;
        let planos = l.lista(n_pares.checked_mul(3)?)?;
        let pares = planos
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| (c[0], c[1], c[2]))
            .collect();
        let n_inicios = l.u32_()? as usize;
        let inicios = l.lista(n_inicios)?;
        let n_bytes = l.u32_()? as usize;
        let bytes = l.tomar(n_bytes)?.to_vec();
        let bpe = Bpe::de_partes_planas(base, pares, inicios, bytes).ok()?;

        let n_esp = l.u32_()? as usize;
        let mut especiales = Vec::with_capacity(n_esp.min(1024));
        let mut saltables = vec![false; bpe.tamano()];
        for _ in 0..n_esp {
            let id = l.u32_()?;
            let saltable = l.u32_()? != 0;
            let n = l.u32_()? as usize;
            *saltables.get_mut(id as usize)? = saltable;
            especiales.push(Especial {
                texto: l.tomar(n)?.to_vec(),
                id,
                saltable,
            });
        }
        if !l.0.is_empty() {
            return None;
        }
        Some(Self {
            bpe,
            especiales,
            saltables,
            propio: false,
        })
    }

    pub fn desde_json(texto: &str) -> Resultado<Self> {
        let a: Archivo =
            serde_json::from_str(texto).map_err(|e| format!("no es un tokenizer.json: {e}"))?;
        comprobar_forma(&a)?;

        let tabla = caracter_a_byte();
        let m = &a.model;
        let mayor = m
            .vocab
            .values()
            .copied()
            .chain(a.added_tokens.iter().map(|t| t.id))
            .max()
            .ok_or("el vocabulario está vacío")?;

        // Las piezas, en bytes y por id. Un id que nadie usa queda vacío, y
        // decodificarlo no da nada: es lo que hace el crate.
        let mut piezas: Vec<Vec<u8>> = vec![Vec::new(); mayor as usize + 1];
        let mut por_bytes: HashMap<Vec<u8>, u32> = HashMap::with_capacity(m.vocab.len());
        for (k, id) in &m.vocab {
            let b = bytes_de(k, &tabla)?;
            piezas[*id as usize].clone_from(&b);
            por_bytes.insert(b, *id);
        }

        let mut base = [0u32; BASE];
        for (b, lugar) in base.iter_mut().enumerate() {
            *lugar = *por_bytes
                .get(&vec![b as u8][..])
                .ok_or_else(|| format!("el byte {b} no tiene token"))?;
        }

        let mut pares = Vec::with_capacity(m.merges.len());
        for linea in &m.merges {
            let (i, d) = linea
                .split_once(' ')
                .ok_or_else(|| format!("fusión mal formada: {linea:?}"))?;
            let (bi, bd) = (bytes_de(i, &tabla)?, bytes_de(d, &tabla)?);
            let buscar = |b: &[u8]| {
                por_bytes
                    .get(b)
                    .copied()
                    .ok_or_else(|| format!("la fusión {linea:?} usa un token que no existe"))
            };
            let mut unida = bi.clone();
            unida.extend_from_slice(&bd);
            pares.push((buscar(&bi)?, buscar(&bd)?, buscar(&unida)?));
        }

        let mut saltables = vec![false; mayor as usize + 1];
        let mut especiales = Vec::with_capacity(a.added_tokens.len());
        for t in &a.added_tokens {
            let b = t.content.as_bytes().to_vec();
            piezas[t.id as usize].clone_from(&b);
            saltables[t.id as usize] = t.special;
            especiales.push(Especial {
                texto: b,
                id: t.id,
                saltable: t.special,
            });
        }
        especiales.sort_by_key(|e| std::cmp::Reverse(e.texto.len()));

        Ok(Self {
            bpe: Bpe::de_partes(base, pares, &piezas)?,
            especiales,
            saltables,
            propio: false,
        })
    }

    /// Cuántos ids tiene, contando los especiales.
    pub fn tamano(&self) -> usize {
        self.bpe.tamano()
    }

    /// El id de un token, por su texto. Es para los especiales —`<|im_end|>`—,
    /// que es lo único que el motor pregunta.
    pub fn token_a_id(&self, token: &str) -> Option<u32> {
        self.especiales
            .iter()
            .find(|e| e.texto == token.as_bytes())
            .map(|e| e.id)
    }

    /// Codifica un texto.
    pub fn codificar(&self, texto: &str) -> Vec<u32> {
        let mut salida = Vec::with_capacity(texto.len() / 3 + 8);
        let b = texto.as_bytes();
        let mut desde = 0;
        let mut i = 0;
        while let Some(p) = b[i..].iter().position(|x| *x == b'<') {
            i += p;
            match self.especial_en(&b[i..]) {
                Some(e) => {
                    self.codificar_normal(&texto[desde..i], &mut salida);
                    salida.push(e.id);
                    i += e.texto.len();
                    desde = i;
                }
                None => i += 1,
            }
        }
        self.codificar_normal(&texto[desde..], &mut salida);
        salida
    }

    /// El especial más largo que empieza acá.
    #[inline]
    fn especial_en(&self, b: &[u8]) -> Option<&Especial> {
        self.especiales.iter().find(|e| b.starts_with(&e.texto))
    }

    fn codificar_normal(&self, texto: &str, salida: &mut Vec<u32>) {
        if texto.is_empty() {
            return;
        }
        if self.propio {
            salida.extend_from_slice(&self.bpe.codificar(texto));
            return;
        }
        // NFC. Casi todo el texto ya lo está, y comprobarlo no pide memoria.
        if texto.is_ascii() || is_nfc_quick(texto.chars()) == IsNormalized::Yes {
            self.codificar_piezas(texto, salida);
        } else {
            let normal: String = texto.nfc().collect();
            self.codificar_piezas(&normal, salida);
        }
    }

    fn codificar_piezas(&self, texto: &str, salida: &mut Vec<u32>) {
        let mut i = 0;
        while i < texto.len() {
            let fin = fin_de_pieza(texto, i);
            self.bpe.codificar_trozo(&texto.as_bytes()[i..fin], salida);
            i = fin;
        }
    }

    /// Los bytes de una secuencia de tokens. Un id desconocido no aporta nada,
    /// igual que en el crate: el modelo tiene más salidas que tokens el
    /// vocabulario, y puede elegir una.
    pub fn decodificar_bytes(&self, ids: &[u32], saltar_especiales: bool) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for id in ids {
            let i = *id as usize;
            if saltar_especiales && self.saltables.get(i).copied().unwrap_or(false) {
                continue;
            }
            if let Some(p) = self.bpe.pieza(*id) {
                out.extend_from_slice(p);
            }
        }
        out
    }

    /// El texto de una secuencia de tokens. En generación, los tokens llegan de
    /// a uno y un carácter puede quedar a mitad: esos bytes salen como el
    /// carácter de reemplazo, y el que llama espera al token siguiente.
    pub fn decodificar(&self, ids: &[u32], saltar_especiales: bool) -> String {
        String::from_utf8_lossy(&self.decodificar_bytes(ids, saltar_especiales)).into_owned()
    }

    /// Las palabras especiales, para mostrarlas o comprobarlas.
    pub fn especiales(&self) -> impl Iterator<Item = (&[u8], u32, bool)> {
        self.especiales
            .iter()
            .map(|e| (e.texto.as_slice(), e.id, e.saltable))
    }
}

/// Que el archivo pida exactamente lo que se sabe hacer.
fn comprobar_forma(a: &Archivo) -> Resultado<()> {
    if a.model.tipo != "BPE" {
        return Err(format!("modelo {:?}: solo se lee BPE", a.model.tipo));
    }
    match &a.normalizer {
        Some(n) if n.tipo == "NFC" => {}
        Some(n) => return Err(format!("normalizador {:?}: solo se lee NFC", n.tipo)),
        None => return Err("sin normalizador: solo se lee NFC".into()),
    }
    match &a.decoder {
        Some(d) if d.tipo == "ByteLevel" => {}
        _ => return Err("el decodificador no es ByteLevel".into()),
    }
    let patron = a.pre_tokenizer["pretokenizers"][0]["pattern"]["Regex"].as_str();
    let byte_level = a.pre_tokenizer["pretokenizers"][1]["type"].as_str();
    if patron != Some(PATRON_GPT2) || byte_level != Some("ByteLevel") {
        return Err(format!(
            "la pre-tokenización no es la de GPT-2 ({patron:?}, {byte_level:?})"
        ));
    }
    for t in &a.added_tokens {
        if t.normalized || t.lstrip || t.rstrip || t.single_word {
            return Err(format!(
                "el token especial {:?} pide opciones que no se leen",
                t.content
            ));
        }
    }
    Ok(())
}

// ─── La expresión regular, a mano ───────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Clase {
    Letra,
    Numero,
    Espacio,
    Otro,
}

#[inline]
fn clase_de(c: char) -> Clase {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Clase::Letra,
            '0'..='9' => Clase::Numero,
            ' ' | '\t' | '\n' | '\r' | '\x0B' | '\x0C' => Clase::Espacio,
            _ => Clase::Otro,
        };
    }
    if en_tabla(LETRAS, c) {
        Clase::Letra
    } else if en_tabla(NUMEROS, c) {
        Clase::Numero
    } else if en_tabla(ESPACIOS, c) {
        Clase::Espacio
    } else {
        Clase::Otro
    }
}

#[inline]
fn caracter_en(t: &str, i: usize) -> Option<char> {
    let b = *t.as_bytes().get(i)?;
    if b < 0x80 {
        return Some(char::from(b));
    }
    t[i..].chars().next()
}

/// Dónde termina una corrida de `clase` que empieza en `i`.
#[inline]
fn fin_de_corrida(t: &str, mut i: usize, clase: Clase) -> usize {
    while let Some(c) = caracter_en(t, i) {
        if clase_de(c) != clase {
            break;
        }
        i += c.len_utf8();
    }
    i
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`, después del apóstrofo en `i`. La `ſ` (s
/// larga) cuenta: con mayúsculas y minúsculas indistintas, Unicode la pliega a
/// la `s`, y es la única de estas letras con una forma así.
fn fin_de_contraccion(t: &str, i: usize) -> Option<usize> {
    let d = i + 1;
    let igual = |pos: usize, minuscula: char| {
        caracter_en(t, pos).is_some_and(|c| {
            c.to_ascii_lowercase() == minuscula || (minuscula == 's' && c == '\u{17F}')
        })
    };
    let largo = |pos: usize| caracter_en(t, pos).map_or(0, char::len_utf8);
    if igual(d, 's') {
        return Some(d + largo(d));
    }
    if igual(d, 't') {
        return Some(d + 1);
    }
    if igual(d, 'r') && igual(d + 1, 'e') {
        return Some(d + 2);
    }
    if igual(d, 'v') && igual(d + 1, 'e') {
        return Some(d + 2);
    }
    if igual(d, 'm') {
        return Some(d + 1);
    }
    if igual(d, 'l') && igual(d + 1, 'l') {
        return Some(d + 2);
    }
    if igual(d, 'd') {
        return Some(d + 1);
    }
    None
}

/// Dónde termina la pieza que empieza en `i`: lo que encontraría, en orden,
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
pub fn fin_de_pieza(t: &str, i: usize) -> usize {
    let c = caracter_en(t, i).expect("i está dentro del texto");
    let tras_c = i + c.len_utf8();
    let clase = clase_de(c);

    if c == '\'' {
        if let Some(fin) = fin_de_contraccion(t, i) {
            return fin;
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    if clase == Clase::Letra {
        return fin_de_corrida(t, tras_c, Clase::Letra);
    }
    if clase != Clase::Numero && c != '\r' && c != '\n' {
        if let Some(sig) = caracter_en(t, tras_c) {
            if clase_de(sig) == Clase::Letra {
                return fin_de_corrida(t, tras_c, Clase::Letra);
            }
        }
    }

    // \p{N}: de a uno.
    if clase == Clase::Numero {
        return tras_c;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    let desde = if c == ' ' { tras_c } else { i };
    let hasta = fin_de_corrida(t, desde, Clase::Otro);
    if hasta > desde {
        let mut fin = hasta;
        while matches!(t.as_bytes().get(fin), Some(b'\r' | b'\n')) {
            fin += 1;
        }
        return fin;
    }

    // Lo que queda es una corrida de espacios en blanco.
    let fin = fin_de_corrida(t, i, Clase::Espacio);

    // \s*[\r\n]+: `\s*` traga todo y devuelve hasta dar con un salto de línea,
    // así que termina justo después del último de la corrida.
    if let Some(p) = t.as_bytes()[i..fin]
        .iter()
        .rposition(|b| matches!(b, b'\r' | b'\n'))
    {
        return i + p + 1;
    }

    // \s+(?!\S): al final del texto, toda la corrida. Si sigue algo, tiene que
    // devolver el último espacio para que lo que sigue no sea un no-espacio, y
    // si la corrida era de uno solo no alcanza.
    if fin == t.len() {
        return fin;
    }
    let ultimo = t[i..fin]
        .char_indices()
        .next_back()
        .map(|(k, _)| i + k)
        .expect("la corrida no está vacía");
    if ultimo > i {
        return ultimo;
    }

    // \s+
    fin
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piezas(t: &str) -> Vec<&str> {
        let mut v = Vec::new();
        let mut i = 0;
        while i < t.len() {
            let f = fin_de_pieza(t, i);
            v.push(&t[i..f]);
            i = f;
        }
        v
    }

    #[test]
    fn las_piezas_de_la_expresion_regular() {
        assert_eq!(piezas("Hola mundo"), ["Hola", " mundo"]);
        assert_eq!(piezas("don't"), ["don", "'t"]);
        assert_eq!(piezas("I'LL"), ["I", "'LL"]);
        assert_eq!(piezas("a'x"), ["a", "'x"]);
        assert_eq!(piezas("123"), ["1", "2", "3"]);
        assert_eq!(piezas(" 12"), [" ", "1", "2"]);
        assert_eq!(piezas("hola, chau"), ["hola", ",", " chau"]);
        assert_eq!(piezas("a  b"), ["a", " ", " b"]);
        assert_eq!(piezas("a   "), ["a", "   "]);
        assert_eq!(piezas("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(piezas("a \n b"), ["a", " \n", " b"]);
        assert_eq!(piezas("!!\n"), ["!!\n"]);
        assert_eq!(piezas(" ?!"), [" ?!"]);
        assert_eq!(piezas("\tx"), ["\tx"]);
        assert_eq!(piezas("año 🙂"), ["año", " 🙂"]);
    }

    #[test]
    fn un_archivo_de_otra_forma_es_un_error_que_lo_dice() {
        let e = TokenizadorJson::desde_json("no es json").unwrap_err();
        assert!(e.contains("tokenizer.json"), "{e}");
        let j = r#"{"added_tokens":[],"normalizer":null,"pre_tokenizer":{},
            "decoder":{"type":"ByteLevel"},"model":{"type":"WordPiece","vocab":{},"merges":[]}}"#;
        assert!(TokenizadorJson::desde_json(j)
            .unwrap_err()
            .contains("WordPiece"));
    }

    /// Un tokenizador chico, de verdad y con la forma completa.
    fn json_chico() -> String {
        let mut vocab = serde_json::Map::new();
        let tabla = caracter_a_byte();
        let mut por_byte: Vec<(u8, char)> = tabla.iter().map(|(c, b)| (*b, *c)).collect();
        por_byte.sort_unstable();
        for (i, (_, c)) in por_byte.iter().enumerate() {
            vocab.insert(c.to_string(), serde_json::json!(i));
        }
        vocab.insert("ho".to_string(), serde_json::json!(256));
        serde_json::json!({
            "added_tokens": [{"id": 257, "content": "<|fin|>", "special": true}],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PATRON_GPT2}},
                {"type": "ByteLevel"},
            ]},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": vocab, "merges": ["h o"]},
        })
        .to_string()
    }

    #[test]
    fn el_compilado_da_lo_mismo_y_se_descarta_si_no_corresponde() {
        let dir = std::env::temp_dir().join(format!("cosxamix-tok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("tokenizer.json");
        let compilado = dir.join("tokenizer.json.cosxbin");
        let _ = std::fs::remove_file(&compilado);
        let json = json_chico();
        std::fs::write(&ruta, &json).unwrap();

        let texto = "hola<|fin|> año";
        let del_json = TokenizadorJson::desde_archivo(&ruta).unwrap();
        assert_eq!(del_json.codificar("ho"), [256]);
        assert!(compilado.exists(), "no guardó el compilado");

        // Lo compilado da lo mismo, y de verdad se usa: corrupto el JSON con el
        // mismo largo cambia la huella, pero acá se prueba leyendo directo.
        let b = std::fs::read(&compilado).unwrap();
        let de_bin = TokenizadorJson::de_compilado(&b, huella_de(json.as_bytes())).unwrap();
        let ids = del_json.codificar(texto);
        assert_eq!(de_bin.codificar(texto), ids);
        assert_eq!(
            de_bin.decodificar(&ids, true),
            del_json.decodificar(&ids, true)
        );
        assert_eq!(
            de_bin.decodificar(&ids, false),
            del_json.decodificar(&ids, false)
        );
        assert_eq!(de_bin.token_a_id("<|fin|>"), Some(257));

        // De otro JSON, truncado o con basura: no se usa.
        assert!(TokenizadorJson::de_compilado(&b, 1).is_none());
        let h = huella_de(json.as_bytes());
        for n in [0, 4, 8, 20, 100, b.len() / 2, b.len() - 1] {
            assert!(
                TokenizadorJson::de_compilado(&b[..n], h).is_none(),
                "truncado a {n}"
            );
        }
        let mut sobra = b.clone();
        sobra.push(0);
        assert!(TokenizadorJson::de_compilado(&sobra, h).is_none());

        // Si el JSON cambia, manda el JSON aunque el compilado siga ahí.
        let cambiado = json
            .replace("\"h o\"", "\"o h\"")
            .replace("\"ho\":256", "\"oh\":256");
        std::fs::write(&ruta, &cambiado).unwrap();
        let nuevo = TokenizadorJson::desde_archivo(&ruta).unwrap();
        assert_eq!(nuevo.codificar("oh"), [256]);
        assert_ne!(nuevo.codificar("ho"), [256]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Un GGUF sin tensores con el vocabulario adentro: `ho` es una fusión.
    fn gguf_chico(ruta: &Path) {
        use crate::gguf::Valor;
        use crate::gguf_write::Escritor;
        let tabla = caracter_a_byte();
        let mut por_byte: Vec<(u8, char)> = tabla.iter().map(|(c, b)| (*b, *c)).collect();
        por_byte.sort_unstable();
        let mut tokens: Vec<Valor> = por_byte
            .iter()
            .map(|(_, c)| Valor::Texto(c.to_string()))
            .collect();
        let mut tipos = vec![Valor::I32(1); tokens.len()];
        tokens.push(Valor::Texto("ho".into()));
        tipos.push(Valor::I32(1));
        tokens.push(Valor::Texto("<|fin|>".into()));
        tipos.push(Valor::I32(3));
        tokens.push(Valor::Texto("[PAD258]".into()));
        tipos.push(Valor::I32(1));
        let mut e = Escritor::nuevo();
        e.meta_texto("tokenizer.ggml.model", "gpt2")
            .meta_texto("tokenizer.ggml.pre", "qwen2")
            .meta("tokenizer.ggml.tokens", Valor::Lista(tokens))
            .meta("tokenizer.ggml.token_type", Valor::Lista(tipos))
            .meta(
                "tokenizer.ggml.merges",
                Valor::Lista(vec![Valor::Texto("h o".into())]),
            );
        // Un GGUF sin tensores no se escribe: con uno alcanza.
        e.tensor(crate::gguf_write::Declaracion::nueva(
            "x",
            &[1],
            crate::gguf::TipoGgml::F32,
        ));
        e.escribir(ruta, |_, w| w.write_all(&[0; 4]).map_err(|x| x.to_string()))
            .unwrap();
    }

    /// Sin ningún JSON en la carpeta el modelo se tokeniza igual, y un
    /// `<modelo>.tokenizer.json` manda sobre lo que trae el GGUF.
    #[test]
    fn el_orden_de_donde_sale_el_tokenizador() {
        let dir = std::env::temp_dir().join(format!("cosxamix-tok-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("modelo.gguf");
        gguf_chico(&gguf);
        assert!(TokenizadorJson::gguf_lo_trae(&gguf));

        // Con un archivo que no existe: no lo necesita.
        let t = TokenizadorJson::para_modelo(&gguf, &dir.join("no-existe.json")).unwrap();
        assert_eq!(t.codificar("ho"), [256]);
        assert_eq!(t.codificar("<|fin|>"), [257]);
        assert_eq!(t.decodificar(&[256, 257, 258], false), "ho<|fin|>");
        assert_eq!(t.decodificar(&[256, 257, 258], true), "ho");

        // El específico manda: acá `ho` no es una fusión.
        let especifico = dir.join("modelo.tokenizer.json");
        std::fs::write(
            &especifico,
            json_chico().replace("\"merges\":[\"h o\"]", "\"merges\":[]"),
        )
        .unwrap();
        let t = TokenizadorJson::para_modelo(&gguf, &especifico).unwrap();
        assert_ne!(t.codificar("ho"), [256]);

        // Uno general no: sigue mandando lo que trae el GGUF.
        let general = dir.join("tokenizer.json");
        std::fs::write(&general, json_chico()).unwrap();
        let t = TokenizadorJson::para_modelo(&gguf, &general).unwrap();
        assert_eq!(t.token_a_id("<|fin|>"), Some(257));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn la_tabla_de_gpt2_cubre_los_256_bytes() {
        let t = caracter_a_byte();
        assert_eq!(t.len(), 256);
        assert_eq!(t[&'Ġ'], b' ');
        assert_eq!(t[&'!'], b'!');
        assert_eq!(t[&'Ċ'], b'\n');
    }
}
