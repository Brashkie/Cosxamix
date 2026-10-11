//! Adaptadores LoRA: entrenar el 0,1% de un modelo en vez del 100%.
//!
//! Un LoRA no toca los pesos originales. Les suma el producto de dos matrices
//! flacas:
//!
//! ```text
//!     y = W·x  +  (B·A·x) · (alfa/r)
//! ```
//!
//! con `A` de `(r, entrada)` y `B` de `(salida, r)`. Con `r = 8` sobre una
//! matriz de 896×896 eso son 14.336 parámetros en vez de 802.816: el 1,8%. En
//! el 0,5B entero, poniendo adaptadores en `q` y `v`, son **540.672 contra 494
//! millones**.
//!
//! Eso es lo que lo hace viable en una máquina sin GPU, y también lo que hace
//! útil al resultado: el adaptador entrenado pesa ~2 MB y se distribuye solo,
//! sin arrastrar el modelo base.
//!
//! `A` arranca con ruido y `B` en **cero**, que no es un detalle: así el
//! producto `B·A` vale cero en el primer paso y el modelo adaptado empieza
//! exactamente donde estaba el original. Si los dos arrancaran con ruido, el
//! primer paso movería la salida a un sitio arbitrario y habría que recuperarse
//! de eso antes de empezar a aprender.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor, Var};

/// Las dos matrices flacas de un adaptador.
pub struct Lora {
    a: Var,
    b: Var,
    escala: f64,
}

impl Lora {
    pub fn nuevo(entrada: usize, salida: usize, r: usize, alfa: f64, dev: &Device) -> Result<Self> {
        Ok(Self {
            a: Var::from_tensor(&Tensor::randn(0f32, 0.02f32, (r, entrada), dev)?)?,
            b: Var::from_tensor(&Tensor::zeros((salida, r), DType::F32, dev)?)?,
            escala: alfa / r as f64,
        })
    }

    fn delta(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.broadcast_matmul(&self.a.t()?)?;
        h.broadcast_matmul(&self.b.t()?)? * self.escala
    }
}

/// Una proyección lineal con los pesos congelados y, opcionalmente, un
/// adaptador encima.
///
/// Los pesos van como `Tensor` y no como `Var` a propósito: un `Var` es lo que
/// candle considera entrenable, y la base no lo es. Lo único que devuelve
/// [`LinealLora::variables`] son las matrices del adaptador.
pub struct LinealLora {
    peso: Tensor,
    sesgo: Option<Tensor>,
    lora: Option<Lora>,
    nombre: String,
}

impl LinealLora {
    /// `peso` viene como `(salida, entrada)`, que es como lo devuelve candle al
    /// descuantizar un tensor del GGUF.
    pub fn nueva(nombre: &str, peso: Tensor, sesgo: Option<Tensor>) -> Self {
        Self {
            peso,
            sesgo,
            lora: None,
            nombre: nombre.to_string(),
        }
    }

    pub fn con_lora(mut self, r: usize, alfa: f64, dev: &Device) -> Result<Self> {
        let (salida, entrada) = self.peso.dims2()?;
        self.lora = Some(Lora::nuevo(entrada, salida, r, alfa, dev)?);
        Ok(self)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.broadcast_matmul(&self.peso.t()?)?;
        if let Some(s) = &self.sesgo {
            y = y.broadcast_add(s)?;
        }
        match &self.lora {
            Some(l) => y + l.delta(x)?,
            None => Ok(y),
        }
    }

    pub fn variables(&self) -> Vec<Var> {
        match &self.lora {
            Some(l) => vec![l.a.clone(), l.b.clone()],
            None => vec![],
        }
    }

    fn tensores(&self) -> Vec<(String, Tensor)> {
        match &self.lora {
            Some(l) => vec![
                (format!("{}.lora_a", self.nombre), l.a.as_tensor().clone()),
                (format!("{}.lora_b", self.nombre), l.b.as_tensor().clone()),
            ],
            None => vec![],
        }
    }

    fn cargar(&mut self, de: &HashMap<String, Tensor>) -> Result<bool> {
        let Some(l) = &self.lora else {
            return Ok(false);
        };
        let (ka, kb) = (
            format!("{}.lora_a", self.nombre),
            format!("{}.lora_b", self.nombre),
        );
        match (de.get(&ka), de.get(&kb)) {
            (Some(a), Some(b)) => {
                l.a.set(a)?;
                l.b.set(b)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

/// Guarda solo los adaptadores, que es lo que los hace útiles: ~2 MB en vez de
/// los 2 GB del modelo base en f32.
pub fn guardar(capas: &[&LinealLora], ruta: &Path) -> Result<usize> {
    let mut mapa = HashMap::new();
    for c in capas {
        for (k, v) in c.tensores() {
            mapa.insert(k, v);
        }
    }
    let n = mapa.len();
    candle_core::safetensors::save(&mapa, ruta)?;
    Ok(n)
}

/// Devuelve cuántas capas encontraron su adaptador en el archivo.
pub fn cargar(capas: &mut [&mut LinealLora], ruta: &Path, dev: &Device) -> Result<usize> {
    let mapa = candle_core::safetensors::load(ruta, dev)?;
    let mut n = 0;
    for c in capas.iter_mut() {
        if c.cargar(&mapa)? {
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lo que hace que un LoRA se pueda enchufar sin romper nada: con `B` en
    /// cero, la salida es exactamente la de la capa original.
    #[test]
    fn al_empezar_no_cambia_la_salida() -> Result<()> {
        let dev = Device::Cpu;
        let peso = Tensor::randn(0f32, 0.5f32, (6, 4), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (3, 4), &dev)?;

        let sin = LinealLora::nueva("p", peso.clone(), None);
        let con = LinealLora::nueva("p", peso, None).con_lora(2, 16.0, &dev)?;

        let a = sin.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let b = con.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a, b, "el adaptador recien creado movio la salida");
        Ok(())
    }

    #[test]
    fn solo_el_adaptador_es_entrenable() -> Result<()> {
        let dev = Device::Cpu;
        let peso = Tensor::randn(0f32, 0.5f32, (6, 4), &dev)?;
        let sin = LinealLora::nueva("p", peso.clone(), None);
        let con = LinealLora::nueva("p", peso, None).con_lora(2, 16.0, &dev)?;

        assert!(sin.variables().is_empty());
        // Dos matrices: 2x4 y 6x2 = 20 parametros, contra los 24 de la base.
        let v = con.variables();
        assert_eq!(v.len(), 2);
        assert_eq!(v.iter().map(|t| t.elem_count()).sum::<usize>(), 20);
        Ok(())
    }

    #[test]
    fn el_adaptador_va_y_vuelve_del_disco() -> Result<()> {
        let dev = Device::Cpu;
        let peso = Tensor::randn(0f32, 0.5f32, (6, 4), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (3, 4), &dev)?;

        let uno = LinealLora::nueva("p", peso.clone(), None).con_lora(2, 16.0, &dev)?;
        // Se mueve `B` fuera de cero, que si no el adaptador no cambia nada y
        // el test pasaria aunque guardar no hiciera nada.
        if let Some(l) = &uno.lora {
            l.b.set(&Tensor::randn(0f32, 0.3f32, (6, 2), &dev)?)?;
        }
        let esperado = uno.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;

        let dir = std::env::temp_dir().join("cosxamix_lora_test.safetensors");
        guardar(&[&uno], &dir)?;

        let mut otro = LinealLora::nueva("p", peso, None).con_lora(2, 16.0, &dev)?;
        let n = cargar(&mut [&mut otro], &dir, &dev)?;
        assert_eq!(n, 1, "no encontro el adaptador en el archivo");

        let obtenido = otro.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(esperado, obtenido);
        let _ = std::fs::remove_file(&dir);
        Ok(())
    }
}
