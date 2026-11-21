//! Leer un `.safetensors` sin copiarlo: una cabecera JSON y los tensores
//! seguidos, mapeados en memoria.
//!
//! El formato es simple a propósito: 8 bytes con el largo de la cabecera, la
//! cabecera (JSON: por tensor su tipo, su forma y su tramo de bytes) y los datos.
//! Los pesos en f16 se leen **directo del mapeo**, sin copiarlos: el UNet pesa
//! 1,7 GB y en memoria queda eso, en las páginas del archivo.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

type Resultado<T> = Result<T, String>;

struct Info {
    tipo: String,
    forma: Vec<usize>,
    desde: usize,
    hasta: usize,
}

pub struct Safetensors {
    mapa: memmap2::Mmap,
    base: usize,
    tensores: HashMap<String, Info>,
}

/// Un tensor f16 dentro del mapeo: se lee con [`P16::datos`].
#[derive(Clone)]
pub struct P16 {
    st: Arc<Safetensors>,
    desde: usize,
    n: usize,
}

impl P16 {
    pub fn datos(&self) -> &[u16] {
        let b = &self.st.mapa[self.desde..self.desde + 2 * self.n];
        // SAFETY: `abrir_f16` comprobó que el tramo está alineado a 2 bytes y
        // que mide 2·n; el mapeo vive mientras viva `st`.
        unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u16>(), self.n) }
    }
}

impl Safetensors {
    pub fn abrir(ruta: &Path) -> Resultado<Arc<Self>> {
        let f = std::fs::File::open(ruta).map_err(|e| format!("{}: {e}", ruta.display()))?;
        // SAFETY: los pesos no se modifican mientras se usan.
        let mapa =
            unsafe { memmap2::Mmap::map(&f) }.map_err(|e| format!("{}: {e}", ruta.display()))?;
        if mapa.len() < 8 {
            return Err(format!("{}: muy corto", ruta.display()));
        }
        let n = u64::from_le_bytes(mapa[..8].try_into().expect("8 bytes")) as usize;
        if n > 100 << 20 || 8 + n > mapa.len() {
            return Err(format!(
                "{}: cabecera de {n} bytes, no cuadra",
                ruta.display()
            ));
        }
        let cab: serde_json::Value = serde_json::from_slice(&mapa[8..8 + n])
            .map_err(|e| format!("{}: cabecera: {e}", ruta.display()))?;
        let base = 8 + n;
        let mut tensores = HashMap::new();
        for (nombre, v) in cab.as_object().ok_or("la cabecera no es un objeto")? {
            if nombre == "__metadata__" {
                continue;
            }
            let tipo = v["dtype"].as_str().ok_or("sin dtype")?.to_string();
            let forma: Vec<usize> = v["shape"]
                .as_array()
                .ok_or("sin shape")?
                .iter()
                .map(|x| x.as_u64().map(|x| x as usize).ok_or("forma rara"))
                .collect::<Result<_, _>>()?;
            let o = v["data_offsets"].as_array().ok_or("sin data_offsets")?;
            let (desde, hasta) = (
                o[0].as_u64().ok_or("offset raro")? as usize,
                o[1].as_u64().ok_or("offset raro")? as usize,
            );
            if desde > hasta || base + hasta > mapa.len() {
                return Err(format!("{nombre}: el tramo se sale del archivo"));
            }
            tensores.insert(
                nombre.clone(),
                Info {
                    tipo,
                    forma,
                    desde,
                    hasta,
                },
            );
        }
        Ok(Arc::new(Self {
            mapa,
            base,
            tensores,
        }))
    }

    fn info(&self, nombre: &str) -> Resultado<&Info> {
        self.tensores
            .get(nombre)
            .ok_or_else(|| format!("no está el tensor {nombre}"))
    }

    pub fn forma(&self, nombre: &str) -> Resultado<Vec<usize>> {
        Ok(self.info(nombre)?.forma.clone())
    }

    pub fn tiene(&self, nombre: &str) -> bool {
        self.tensores.contains_key(nombre)
    }

    /// Un tensor en f16, sin copiarlo, con la forma que se espera.
    pub fn f16(st: &Arc<Self>, nombre: &str, forma: &[usize]) -> Resultado<P16> {
        let i = st.info(nombre)?;
        if i.tipo != "F16" {
            return Err(format!("{nombre}: es {} y se esperaba F16", i.tipo));
        }
        if i.forma != forma {
            return Err(format!(
                "{nombre}: forma {:?} y se esperaba {forma:?}",
                i.forma
            ));
        }
        let n: usize = forma.iter().product();
        let desde = st.base + i.desde;
        if i.hasta - i.desde != 2 * n || !desde.is_multiple_of(2) {
            return Err(format!("{nombre}: tramo de bytes que no cuadra"));
        }
        Ok(P16 {
            st: st.clone(),
            desde,
            n,
        })
    }

    /// Un tensor en f32 (de F16 o F32), copiado: para lo chico —sesgos,
    /// normas— y para los pesos que se guardaron en F32.
    pub fn f32(&self, nombre: &str, forma: &[usize]) -> Resultado<Vec<f32>> {
        let i = self.info(nombre)?;
        if i.forma != forma {
            return Err(format!(
                "{nombre}: forma {:?} y se esperaba {forma:?}",
                i.forma
            ));
        }
        let b = &self.mapa[self.base + i.desde..self.base + i.hasta];
        let n: usize = forma.iter().product();
        match i.tipo.as_str() {
            "F32" if b.len() == 4 * n => Ok(b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()),
            "F16" if b.len() == 2 * n => {
                let h: Vec<u16> = b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c))
                    .collect();
                let mut v = vec![0f32; n];
                crate::kernels::sgemm::f16_a_f32(&h, &mut v);
                Ok(v)
            }
            otro => Err(format!("{nombre}: tipo {otro} o largo que no cuadra")),
        }
    }
}
