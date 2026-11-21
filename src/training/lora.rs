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
//! exactamente donde estaba el original.
//!
//! Lo de acá es guardarlos y cargarlos: un `.safetensors` con `{capa}.lora_a` y
//! `{capa}.lora_b` en F32, que es lo que escribía la versión con candle, así que
//! los adaptadores viejos se siguen leyendo.

use std::io::Write;
use std::path::Path;

use super::transformer::Params;
use crate::image_gen::safetensors::Safetensors;

type Resultado<T> = Result<T, String>;

fn es_adaptador(nombre: &str) -> bool {
    nombre.ends_with(".lora_a") || nombre.ends_with(".lora_b")
}

/// Guarda solo los adaptadores: ~2 MB en vez de los 2 GB de la base en f32.
/// Devuelve cuántos tensores escribió.
pub fn guardar(p: &Params, ruta: &Path) -> Resultado<usize> {
    let indices: Vec<usize> = (0..p.nombres.len())
        .filter(|&i| es_adaptador(&p.nombres[i]))
        .collect();
    let mut cabecera = serde_json::Map::new();
    let mut desde = 0usize;
    for &i in &indices {
        let hasta = desde + 4 * p.valores[i].len();
        cabecera.insert(
            p.nombres[i].clone(),
            serde_json::json!({ "dtype": "F32", "shape": p.formas[i], "data_offsets": [desde, hasta] }),
        );
        desde = hasta;
    }
    let mut cab = serde_json::to_vec(&cabecera).map_err(|e| e.to_string())?;
    // Los datos empiezan alineados a 8, como pide el formato.
    while (8 + cab.len()) % 8 != 0 {
        cab.push(b' ');
    }
    let error = |e: std::io::Error| format!("{}: {e}", ruta.display());
    let mut f = std::io::BufWriter::new(std::fs::File::create(ruta).map_err(error)?);
    f.write_all(&(cab.len() as u64).to_le_bytes())
        .map_err(error)?;
    f.write_all(&cab).map_err(error)?;
    for &i in &indices {
        for v in &p.valores[i] {
            f.write_all(&v.to_le_bytes()).map_err(error)?;
        }
    }
    f.flush().map_err(error)?;
    Ok(indices.len())
}

/// Pisa los adaptadores con los del archivo. Devuelve cuántas matrices
/// encontró; las que no estén quedan como estaban.
pub fn cargar(p: &mut Params, ruta: &Path) -> Resultado<usize> {
    let st = Safetensors::abrir(ruta)?;
    let mut n = 0;
    for i in 0..p.nombres.len() {
        if es_adaptador(&p.nombres[i]) && st.tiene(&p.nombres[i]) {
            p.valores[i] = st.f32(&p.nombres[i], &p.formas[i])?;
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_adaptador_va_y_vuelve_del_disco() {
        let mut p = Params::default();
        p.agregar("blk.0.attn_q.weight", vec![3, 2], vec![9.0; 6], false);
        p.agregar(
            "blk.0.attn_q.lora_a",
            vec![2, 2],
            vec![1.0, -2.5, 3.25, 0.5],
            true,
        );
        p.agregar(
            "blk.0.attn_q.lora_b",
            vec![3, 2],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            true,
        );
        let ruta =
            std::env::temp_dir().join(format!("cosxamix-lora-{}.safetensors", std::process::id()));
        assert_eq!(guardar(&p, &ruta).unwrap(), 2);

        let mut otro = Params::default();
        otro.agregar("blk.0.attn_q.weight", vec![3, 2], vec![7.0; 6], false);
        otro.agregar("blk.0.attn_q.lora_a", vec![2, 2], vec![0.0; 4], true);
        otro.agregar("blk.0.attn_q.lora_b", vec![3, 2], vec![0.0; 6], true);
        assert_eq!(cargar(&mut otro, &ruta).unwrap(), 2);
        std::fs::remove_file(&ruta).unwrap();
        assert_eq!(otro.valores[1], p.valores[1]);
        assert_eq!(otro.valores[2], p.valores[2]);
        // La base no se toca: no va en el archivo.
        assert_eq!(otro.valores[0], vec![7.0; 6]);
    }
}
