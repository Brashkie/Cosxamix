//! Etapa 12 del ROADMAP, primer experimento: ¿la red E8 cuantiza mejor que
//! los k-quants a 2 bits por peso?
//!
//! Sobre matrices reales en F16 (sin cuantizar: medir sobre un modelo que ya
//! viene en Q3_K favorece a cualquier esquema parecido a Q3_K), se compara el
//! error relativo `‖W − Ŵ‖² / ‖W‖²` de:
//!
//! - **Q2_K, Q3_K y Q4_K**, con el cuantizador propio (`quant_kquants`);
//! - **E8**: cada grupo de 8 pesos va al punto más cercano de la red E8 —la de
//!   empaquetado de esferas más densa en 8 dimensiones— dentro de la bola de
//!   norma² ≤ 10, que tiene 56.881 puntos y entra en un índice de 16 bits: 2
//!   bits por peso, más una escala f16 por bloque de 256 (2,0625);
//! - **E8 con rotación**: lo mismo después de girar cada bloque de 256 con una
//!   transformada de Hadamard con signos al azar, la idea de QuIP#. Gira sin
//!   cambiar las distancias, así que el error se mide igual en el espacio
//!   girado, y deja los pesos más parecidos a una gaussiana: los valores
//!   extremos se reparten entre los 256. Al usar el modelo, la entrada se gira
//!   con la misma transformada, que cuesta `256 · log2(256)` sumas por bloque.
//!
//! Es el primer filtro: si E8 no le gana a Q2_K acá, no vale la pena escribir
//! el kernel. Si gana, lo que decide es la batería y la KL del modelo entero.
//!
//!     cargo run --release --example measure_lattice
//!     cargo run --release --example measure_lattice -- modelo-f16.gguf
//!
//! Y el modelo entero, que es lo que decide:
//!
//!     measure_lattice -- simular origen-f16.gguf destino.gguf (e8|q2k|q3k)
//!     measure_lattice -- kl original.gguf otro.gguf
//!
//! `simular` escribe una copia en F16 donde cada matriz de las capas pasó por
//! el esquema y volvió: el motor la corre tal cual, sin kernel nuevo, y se
//! puede medir la inteligencia que le queda (batería de `compare_models
//! -- --e8`). Los embeddings quedan en F16 en todos, para comparar parejo.
//! `kl` mide la divergencia KL media, token por token, de `otro` contra
//! `original` sobre un texto en español.

use std::path::Path;

use cosxamix::gguf::{Gguf, Pesos, TipoGgml, Valor};
use cosxamix::gguf_write::{Declaracion, Escritor};
use cosxamix::quant::{cuantizar_bloques, descuantizar};

/// El punto de D8 más cercano a `x`: redondear cada coordenada y, si la suma
/// queda impar, redondear para el otro lado la que más perdió (Conway y
/// Sloane, cap. 20).
fn d8(x: &[f32; 8]) -> [f32; 8] {
    let mut r = x.map(f32::round);
    let suma: f32 = r.iter().sum();
    if (suma as i64).rem_euclid(2) != 0 {
        let (i, _) = x
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        r[i] += if x[i] > r[i] { 1.0 } else { -1.0 };
    }
    r
}

fn dist2(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn norma2(a: &[f32; 8]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

/// E8 = D8 ∪ (D8 + ½): el más cercano de los dos.
fn e8(x: &[f32; 8]) -> [f32; 8] {
    let a = d8(x);
    let medio = x.map(|v| v - 0.5);
    let b = d8(&medio).map(|v| v + 0.5);
    if dist2(x, &a) <= dist2(x, &b) {
        a
    } else {
        b
    }
}

/// Norma² máxima del código: los puntos de E8 con norma² ≤ 10 son
/// 1 + 240 + 2160 + 6720 + 17520 + 30240 = 56.881, que entran en 16 bits.
const BOLA: f32 = 10.0;

/// El punto del código más cercano a `x`. Si el más cercano de la red se sale
/// de la bola, se prueba con `x` encogido hasta que caiga adentro, y se queda
/// el que menos se aleja del `x` original.
fn codigo_e8(x: &[f32; 8]) -> [f32; 8] {
    let p = e8(x);
    if norma2(&p) <= BOLA {
        return p;
    }
    let mut mejor = [0f32; 8];
    let mut mejor_d = dist2(x, &mejor);
    let mut a = 0.97f32;
    while a > 0.2 {
        let q = e8(&x.map(|v| v * a));
        if norma2(&q) <= BOLA {
            let d = dist2(x, &q);
            if d < mejor_d {
                mejor = q;
                mejor_d = d;
            }
        }
        a -= 0.03;
    }
    mejor
}

/// Cuantiza un bloque de 256 con E8 y devuelve el error cuadrático. La escala
/// se busca en una grilla alrededor del RMS y se redondea a f16, como quedaría
/// guardada.
fn bloque_e8(x: &[f32]) -> f64 {
    bloque_e8_vuelta(x).0
}

/// Lo mismo, y además el bloque reconstruido: lo que daría el modelo.
fn bloque_e8_vuelta(x: &[f32]) -> (f64, Vec<f32>) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    if rms == 0.0 {
        return (0.0, vec![0.0; x.len()]);
    }
    let mut mejor = f64::INFINITY;
    let mut mejor_s = 0f32;
    for paso in 0..24 {
        // El RMS de un punto de la bola anda por 1; se prueba la escala que
        // lleva el bloque a ese tamaño y alrededores.
        let s = rms * (0.45 + 0.05 * paso as f32);
        let s = cosxamix::quant::f16_a_f32(cosxamix::quant::f32_a_f16(s));
        let mut err = 0f64;
        let (grupos, _) = x.as_chunks::<8>();
        for g in grupos {
            let v: [f32; 8] = g.map(|w| w / s);
            let q = codigo_e8(&v);
            for i in 0..8 {
                let d = (g[i] - q[i] * s) as f64;
                err += d * d;
            }
            if err >= mejor {
                break;
            }
        }
        if err < mejor {
            mejor = err;
            mejor_s = s;
        }
    }
    let (grupos, _) = x.as_chunks::<8>();
    let vuelta = grupos
        .iter()
        .flat_map(|g| codigo_e8(&g.map(|w| w / mejor_s)).map(|q| q * mejor_s))
        .collect();
    (mejor, vuelta)
}

/// Un bloque de 256 por E8 girado, y de vuelta al espacio original: girar,
/// cuantizar, reconstruir y deshacer el giro (Hadamard normalizada es su
/// propia inversa; después, los mismos signos).
fn vuelta_e8_girado(b: &[f32], s: &[f32]) -> Vec<f32> {
    let mut r: Vec<f32> = b.iter().zip(s).map(|(v, g)| v * g).collect();
    hadamard(&mut r);
    let (_, mut q) = bloque_e8_vuelta(&r);
    hadamard(&mut q);
    q.iter().zip(s).map(|(v, g)| v * g).collect()
}

/// `simular`: el modelo con cada matriz de las capas por el esquema y de
/// vuelta, en F16.
fn simular(origen: &Path, destino: &Path, esquema: &str) -> Result<(), String> {
    use rayon::prelude::*;
    let g = Gguf::abrir(origen)?;
    let mapa = Pesos::abrir(&g, origen)?;
    let entra = |t: &cosxamix::gguf::InfoTensor| {
        t.nombre.starts_with("blk.") && t.dims.len() == 2 && t.dims[0].is_multiple_of(256)
    };
    let mut e = Escritor::nuevo();
    let mut claves: Vec<&String> = g.metadatos.keys().collect();
    claves.sort();
    for clave in claves {
        if clave != "general.alignment" {
            e.meta(clave.clone(), g.metadatos[clave].clone());
        }
    }
    e.meta("cosxamix.simulado", Valor::Texto(esquema.to_string()));
    for t in &g.tensores {
        let tipo = if entra(t) { TipoGgml::F16 } else { t.tipo };
        e.tensor(Declaracion::nueva(t.nombre.clone(), &t.dims, tipo));
    }
    let signos = signos(256);
    let mut hechos = 0;
    e.escribir(destino, |d, w| {
        let info = g.tensor(&d.nombre).ok_or("se perdió un tensor")?;
        let datos = mapa.de(info)?;
        if !entra(info) {
            return w.write_all(datos).map_err(|x| x.to_string());
        }
        let mut valores = vec![0f32; info.elementos()];
        descuantizar(info.tipo, datos, &mut valores)?;
        let vuelta: Vec<f32> = match esquema {
            "e8" => valores
                .par_chunks_exact(256)
                .flat_map_iter(|b| vuelta_e8_girado(b, &signos))
                .collect(),
            "q2k" | "q3k" => {
                let tipo = if esquema == "q2k" {
                    TipoGgml::Q2K
                } else {
                    TipoGgml::Q3K
                };
                let mut bytes = Vec::new();
                cuantizar_bloques(tipo, &valores, &mut bytes)?;
                let mut v = vec![0f32; valores.len()];
                descuantizar(tipo, &bytes, &mut v)?;
                v
            }
            otro => return Err(format!("esquema {otro}: e8, q2k o q3k")),
        };
        let mut bytes = Vec::with_capacity(vuelta.len() * 2);
        for v in vuelta {
            bytes.extend_from_slice(&cosxamix::quant::f32_a_f16(v).to_le_bytes());
        }
        hechos += 1;
        w.write_all(&bytes).map_err(|x| x.to_string())
    })?;
    println!(
        "  {hechos} matrices por {esquema}, en {}",
        destino.display()
    );
    Ok(())
}

/// El texto de la KL: español común, como el que lee el bot.
const TEXTO_KL: &str = "La fotosíntesis es el proceso por el cual las plantas \
    transforman la luz del sol en energía química. En las hojas, la clorofila \
    capta la luz y, con el agua que suben las raíces y el dióxido de carbono \
    del aire, la planta fabrica glucosa y libera oxígeno. Sin este proceso no \
    habría alimento para casi ningún ser vivo, porque las plantas están en la \
    base de casi todas las cadenas alimentarias. Para hacer un sticker de un \
    video en WhatsApp, primero se recorta el video a pocos segundos, después se \
    lo convierte a un formato animado y por último se lo envía como sticker.";

/// `kl`: la divergencia KL media de `otro` contra `original`, token por token.
fn kl(original: &Path, otro: &Path) -> Result<(), String> {
    use cosxamix::forward::{Cache, Modelo};
    use cosxamix::tokenizer_json::TokenizadorJson;
    let tok = TokenizadorJson::para_modelo(original, Path::new("modelos/tokenizer.json"))?;
    let ids = tok.codificar(TEXTO_KL);
    let a = Modelo::abrir(original)?;
    let b = Modelo::abrir(otro)?;
    let (mut ca, mut cb) = (Cache::nuevo(&a.cfg), Cache::nuevo(&b.cfg));
    let lsm = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().fold(f32::MIN, |x, y| x.max(*y)) as f64;
        let z: f64 = l.iter().map(|v| (*v as f64 - m).exp()).sum::<f64>().ln() + m;
        l.iter().map(|v| *v as f64 - z).collect()
    };
    let (mut suma, mut iguales) = (0f64, 0usize);
    for id in &ids {
        let la = lsm(&a.forward(&[*id], &mut ca)?);
        let lb = lsm(&b.forward(&[*id], &mut cb)?);
        suma += la
            .iter()
            .zip(&lb)
            .map(|(p, q)| p.exp() * (p - q))
            .sum::<f64>();
        let arg = |l: &[f64]| (0..l.len()).max_by(|i, j| l[*i].total_cmp(&l[*j])).unwrap();
        iguales += usize::from(arg(&la) == arg(&lb));
    }
    println!(
        "  {}: KL media {:.4} en {} tokens; el siguiente token más probable coincide en el {:.0}%",
        otro.file_name().unwrap().to_string_lossy(),
        suma / ids.len() as f64,
        ids.len(),
        iguales as f64 * 100.0 / ids.len() as f64
    );
    Ok(())
}

/// Transformada de Walsh-Hadamard rápida, normalizada: ortogonal.
fn hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let f = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= f);
}

/// Signos al azar fijos para la rotación: los mismos en todos los bloques, que
/// es lo que permite girar la entrada una sola vez al usar el modelo.
fn signos(n: usize) -> Vec<f32> {
    let mut a = cosxamix::daiss::Azar::con_semilla(0xE8);
    (0..n)
        .map(|_| if a.siguiente() & 1 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Error relativo de un k-quant: cuantizar, descuantizar y comparar.
fn error_kquant(tipo: TipoGgml, w: &[f32]) -> f64 {
    let mut bytes = Vec::new();
    cuantizar_bloques(tipo, w, &mut bytes).expect("cuantizar");
    let mut vuelta = vec![0f32; w.len()];
    descuantizar(tipo, &bytes, &mut vuelta).expect("descuantizar");
    let err: f64 = w
        .iter()
        .zip(&vuelta)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    err / energia(w)
}

fn energia(w: &[f32]) -> f64 {
    w.iter().map(|v| (*v as f64).powi(2)).sum()
}

fn error_e8(w: &[f32], rotar: bool) -> f64 {
    use rayon::prelude::*;
    let s = signos(256);
    let err: f64 = w
        .par_chunks_exact(256)
        .map(|b| {
            if rotar {
                let mut r: Vec<f32> = b.iter().zip(&s).map(|(v, g)| v * g).collect();
                hadamard(&mut r);
                bloque_e8(&r)
            } else {
                bloque_e8(b)
            }
        })
        .sum();
    err / energia(w)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("simular") if a.len() == 4 => {
            return simular(Path::new(&a[1]), Path::new(&a[2]), &a[3]);
        }
        Some("kl") if a.len() == 3 => return kl(Path::new(&a[1]), Path::new(&a[2])),
        _ => {}
    }
    cosxamix::machine_load::antes_de_medir();
    let ruta = a
        .first()
        .cloned()
        .unwrap_or_else(|| "modelos/qwen2.5-0.5b-instruct-fp16.gguf".into());
    let ruta = Path::new(&ruta);
    let g = Gguf::abrir(ruta)?;
    let pesos = Pesos::abrir(&g, ruta)?;

    let capas = g.meta_entero("qwen2.block_count").unwrap_or(24) as usize;
    let nombres: Vec<String> = [0, capas / 2, capas - 1]
        .iter()
        .flat_map(|i| {
            ["attn_q", "attn_output", "ffn_gate", "ffn_down"].map(|t| format!("blk.{i}.{t}.weight"))
        })
        .collect();

    println!();
    println!(
        "  error relativo ‖W − Ŵ‖² / ‖W‖², en {}",
        ruta.file_name().unwrap().to_string_lossy()
    );
    println!();
    println!(
        "  {:<26}{:>10}{:>10}{:>10}{:>10}{:>12}",
        "matriz", "Q4_K", "Q3_K", "Q2_K", "E8", "E8 girado"
    );
    println!(
        "  {:<26}{:>10}{:>10}{:>10}{:>10}{:>12}",
        "bits por peso", "4,5", "3,44", "2,63", "2,06", "2,06"
    );
    let mut sumas = [0f64; 5];
    for nombre in &nombres {
        let Some(info) = g.tensor(nombre) else {
            continue;
        };
        let mut w = vec![0f32; info.elementos()];
        descuantizar(info.tipo, pesos.de(info)?, &mut w)?;
        let e = [
            error_kquant(TipoGgml::Q4K, &w),
            error_kquant(TipoGgml::Q3K, &w),
            error_kquant(TipoGgml::Q2K, &w),
            error_e8(&w, false),
            error_e8(&w, true),
        ];
        for (s, v) in sumas.iter_mut().zip(e) {
            *s += v;
        }
        println!(
            "  {:<26}{:>10.5}{:>10.5}{:>10.5}{:>10.5}{:>12.5}",
            nombre, e[0], e[1], e[2], e[3], e[4]
        );
    }
    let n = nombres.len() as f64;
    println!(
        "  {:<26}{:>10.5}{:>10.5}{:>10.5}{:>10.5}{:>12.5}",
        "media",
        sumas[0] / n,
        sumas[1] / n,
        sumas[2] / n,
        sumas[3] / n,
        sumas[4] / n
    );
    Ok(())
}
