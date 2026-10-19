//! **daiss**: la familia de modelos propia, de 20 millones a 14 mil millones
//! de parámetros.
//!
//! Este módulo define la arquitectura, calcula exactamente cuánto pesa cada
//! tamaño, e **inicializa y escribe el GGUF** — en flujo, fila por fila, así
//! que escribir un daiss de 14B no pide 14 GB de RAM sino los 20 KB de una
//! fila.
//!
//! ## Qué es nuestro y qué no
//!
//! Conviene decirlo sin adornos, porque la diferencia importa.
//!
//! **Nuestro**: los tamaños, la inicialización, el código que lo genera, el
//! escritor de GGUF, y —cuando se entrene— los pesos, que son el modelo.
//!
//! **No nuestro**: la forma del bloque. daiss es un decoder con RMSNorm,
//! atención agrupada (GQA), SwiGLU y RoPE, que es la misma forma que usan
//! Llama, Qwen, Mistral y casi todo lo demás desde 2023. Inventar una forma
//! distinta sin tener con qué entrenarla sería inventar un problema.
//!
//! **Prestado a propósito**: el vocabulario. `tokenizer.ggml.*` sale por ahora
//! del tokenizador de Qwen2.5, 151.936 piezas. Un tokenizador propio está en
//! el ROADMAP §8 y no está escrito; hasta que lo esté, reusar uno que ya
//! funciona permite evaluar el modelo en vez de esperarlo.
//!
//! ## Por qué `general.architecture` dice `qwen2`
//!
//! Porque esa etiqueta no nombra a un modelo sino a una **forma de bloque**, y
//! es la que daiss tiene. Poner `daiss` ahí haría que ningún runtime del mundo
//! lo cargara —ni el nuestro, que lee esa clave en `model/qwen2.rs`— a cambio
//! de nada. El nombre va en `general.name`, que es donde se mira.
//!
//! ## Lo que este módulo NO hace
//!
//! No entrena. Un daiss recién escrito tiene pesos aleatorios y genera ruido.
//! Lo que da es que el camino completo exista y esté verificado —definir,
//! inicializar, escribir, leer, servir—, que es lo que hay que tener antes de
//! poder entrenar algo.

use std::path::Path;

use rayon::prelude::*;

use crate::gguf::{TipoGgml, Valor};
use crate::gguf_write::{Declaracion, Escritor};
use crate::tokenizer::Bpe;

type Resultado<T> = Result<T, String>;

/// Vocabulario de Qwen2.5, que es el que daiss usa prestado por ahora.
pub const VOCAB: usize = 151_936;

/// Un tamaño de la familia.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub nombre: &'static str,
    /// Ancho del modelo.
    pub dim: usize,
    pub capas: usize,
    pub cabezas: usize,
    /// Cabezas de clave y valor. Menos que `cabezas` es atención agrupada, y
    /// es lo que hace que el caché por token no crezca con el ancho.
    pub cabezas_kv: usize,
    /// Ancho interno del FFN.
    pub ffn: usize,
    pub vocab: usize,
    pub contexto: usize,
    pub eps: f32,
    pub rope: f32,
}

/// La familia completa.
///
/// Las formas siguen a Qwen2.5 a propósito: son las únicas con las que hoy se
/// puede comparar cara a cara contra un modelo entrenado del mismo tamaño, y
/// sin esa comparación cualquier medición nuestra no tendría contra qué.
pub const FAMILIA: &[Config] = &[
    // Para probar el camino entero en segundos, no para usar.
    Config {
        nombre: "daiss-mini",
        dim: 256,
        capas: 6,
        cabezas: 4,
        cabezas_kv: 2,
        ffn: 768,
        vocab: VOCAB,
        contexto: 2048,
        eps: 1e-6,
        rope: 10000.0,
    },
    Config {
        nombre: "daiss-0.5b",
        dim: 896,
        capas: 24,
        cabezas: 14,
        cabezas_kv: 2,
        ffn: 4864,
        vocab: VOCAB,
        contexto: 32768,
        eps: 1e-6,
        rope: 1_000_000.0,
    },
    Config {
        nombre: "daiss-3b",
        dim: 2048,
        capas: 36,
        cabezas: 16,
        cabezas_kv: 2,
        ffn: 11008,
        vocab: VOCAB,
        contexto: 32768,
        eps: 1e-6,
        rope: 1_000_000.0,
    },
    Config {
        nombre: "daiss-7b",
        dim: 3584,
        capas: 28,
        cabezas: 28,
        cabezas_kv: 4,
        ffn: 18944,
        vocab: VOCAB,
        contexto: 32768,
        eps: 1e-6,
        rope: 1_000_000.0,
    },
    Config {
        nombre: "daiss-14b",
        dim: 5120,
        capas: 48,
        cabezas: 40,
        cabezas_kv: 8,
        ffn: 13824,
        vocab: VOCAB,
        contexto: 32768,
        eps: 1e-6,
        rope: 1_000_000.0,
    },
];

impl Config {
    pub fn de_nombre(n: &str) -> Option<&'static Config> {
        FAMILIA.iter().find(|c| c.nombre == n)
    }

    pub fn dim_cabeza(&self) -> usize {
        self.dim / self.cabezas
    }

    /// Ancho de las proyecciones de clave y valor.
    pub fn dim_kv(&self) -> usize {
        self.cabezas_kv * self.dim_cabeza()
    }

    /// Cuántos parámetros tiene, contados de las declaraciones y no a mano:
    /// si las dos cuentas pudieran separarse, se separarían.
    pub fn parametros(&self) -> u64 {
        self.declaraciones(TipoGgml::F32)
            .iter()
            .map(|d| d.elementos() as u64)
            .sum()
    }

    /// Lo que pesa el archivo con los pesos en `tipo`. Las normas y los
    /// sesgos van siempre en `F32`, así que no es `parametros * bytes`.
    pub fn bytes_en(&self, tipo: TipoGgml) -> u64 {
        self.declaraciones(tipo)
            .iter()
            .map(|d| d.bytes() as u64)
            .sum()
    }

    /// Si los k-quants —Q2_K a Q6_K— se le pueden aplicar a todos los
    /// tensores.
    ///
    /// Piden filas de 256 valores, y `daiss-0.5b` tiene `dim` 896, que es
    /// 3,5 bloques. No es un descuido de la tabla: es la razón por la que los
    /// GGUF de Qwen2.5-0.5B que circulan **mezclan formatos**, con las
    /// matrices de atención en Q8_0 o Q5_0 y solo el FFN en k-quant. Mientras
    /// el escritor solo produzca Q8_0 da igual, pero decide qué se puede
    /// recuantizar después.
    pub fn admite_kquants(&self) -> bool {
        self.dim.is_multiple_of(256) && self.ffn.is_multiple_of(256)
    }

    /// Bytes del caché de atención por token, que es lo que decide cuánto
    /// contexto entra en memoria.
    ///
    /// Dos tensores —clave y valor— por capa, en `f32`.
    pub fn bytes_de_cache_por_token(&self) -> u64 {
        2 * self.capas as u64 * self.dim_kv() as u64 * 4
    }

    /// Las declaraciones de todos los tensores, en el orden en que se
    /// escriben.
    ///
    /// Las normas y los sesgos quedan en `F32` aunque los pesos vayan
    /// cuantizados: son menos del 0,1% del archivo y son justo los valores
    /// donde un error de cuantización se propaga a toda la capa.
    pub fn declaraciones(&self, pesos: TipoGgml) -> Vec<Declaracion> {
        let (dim, kv, ffn) = (self.dim as u64, self.dim_kv() as u64, self.ffn as u64);
        let vocab = self.vocab as u64;
        let f32_ = TipoGgml::F32;

        let mut v = Vec::with_capacity(self.capas * 12 + 3);
        v.push(Declaracion::nueva(
            "token_embd.weight",
            &[dim, vocab],
            pesos,
        ));

        for i in 0..self.capas {
            let p = format!("blk.{i}");
            v.push(Declaracion::nueva(
                format!("{p}.attn_norm.weight"),
                &[dim],
                f32_,
            ));
            v.push(Declaracion::nueva(
                format!("{p}.attn_q.weight"),
                &[dim, dim],
                pesos,
            ));
            v.push(Declaracion::nueva(format!("{p}.attn_q.bias"), &[dim], f32_));
            v.push(Declaracion::nueva(
                format!("{p}.attn_k.weight"),
                &[dim, kv],
                pesos,
            ));
            v.push(Declaracion::nueva(format!("{p}.attn_k.bias"), &[kv], f32_));
            v.push(Declaracion::nueva(
                format!("{p}.attn_v.weight"),
                &[dim, kv],
                pesos,
            ));
            v.push(Declaracion::nueva(format!("{p}.attn_v.bias"), &[kv], f32_));
            v.push(Declaracion::nueva(
                format!("{p}.attn_output.weight"),
                &[dim, dim],
                pesos,
            ));
            v.push(Declaracion::nueva(
                format!("{p}.ffn_norm.weight"),
                &[dim],
                f32_,
            ));
            v.push(Declaracion::nueva(
                format!("{p}.ffn_gate.weight"),
                &[dim, ffn],
                pesos,
            ));
            v.push(Declaracion::nueva(
                format!("{p}.ffn_up.weight"),
                &[dim, ffn],
                pesos,
            ));
            v.push(Declaracion::nueva(
                format!("{p}.ffn_down.weight"),
                &[ffn, dim],
                pesos,
            ));
        }

        v.push(Declaracion::nueva("output_norm.weight", &[dim], f32_));
        v.push(Declaracion::nueva("output.weight", &[dim, vocab], pesos));
        v
    }
}

// ─── El generador de números ────────────────────────────────────────────────

/// xorshift64*, con semilla.
///
/// Con semilla y no con la hora, porque un modelo que no se puede volver a
/// generar bit por bit no es un modelo: es un archivo. Dos corridas con la
/// misma semilla tienen que dar el mismo SHA-256.
///
/// No se suma el crate `rand` por esto: son tres líneas, y la calidad que
/// pide inicializar pesos —que no haya correlación visible entre valores
/// consecutivos— la da de sobra.
pub struct Azar {
    estado: u64,
}

impl Azar {
    pub fn con_semilla(s: u64) -> Self {
        // El estado cero es un punto fijo de xorshift: se queda en cero para
        // siempre y todos los pesos saldrian iguales.
        Self {
            estado: if s == 0 { 0x9E37_79B9_7F4A_7C15 } else { s },
        }
    }

    pub fn siguiente(&mut self) -> u64 {
        let mut x = self.estado;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.estado = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniforme en (0, 1]. Nunca devuelve cero, porque Box-Muller le toma el
    /// logaritmo y `ln(0)` es infinito.
    fn uniforme(&mut self) -> f32 {
        let bits = self.siguiente() >> 40; // 24 bits
        (bits as f32 + 1.0) / 16_777_216.0
    }

    /// Dos normales de media cero y desviación `sigma`, por Box-Muller.
    ///
    /// Salen de a dos porque el método las produce de a dos; devolver una y
    /// tirar la otra duplicaría el trabajo.
    pub fn normales(&mut self, sigma: f32) -> (f32, f32) {
        let u1 = self.uniforme();
        let u2 = self.uniforme();
        let r = sigma * (-2.0 * u1.ln()).sqrt();
        let a = std::f32::consts::TAU * u2;
        (r * a.cos(), r * a.sin())
    }
}

/// Semilla de un tensor: la del modelo mezclada con su nombre.
///
/// Que cada tensor tenga la suya, en vez de ir todos de un mismo flujo, hace
/// que regenerar uno solo dé exactamente lo mismo sin tener que repetir los
/// anteriores — y que agregar un tensor no corra los valores de los demás.
fn semilla_de(modelo: u64, nombre: &str, fila: usize) -> u64 {
    // FNV-1a de 64 bits.
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in nombre.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    // La fila entra en la semilla, y eso es lo que permite generar las filas
    // EN PARALELO sin que el resultado dependa de en que orden las termine
    // cada hilo. Con un solo flujo por tensor, el archivo cambiaria segun
    // como el planificador repartiera el trabajo, y un modelo que no se
    // puede reproducir no sirve.
    h ^= fila as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01B3);
    h ^ modelo.rotate_left(17)
}

/// Con qué se inicializa cada tensor.
enum Inicio {
    /// Normal de media cero y esta desviación.
    Normal(f32),
    Unos,
    Ceros,
}

/// La desviación con la que arranca cada tensor.
///
/// `0,02` fijo para casi todo, que es lo que usa GPT-2 y lo que heredó el
/// resto. Las dos proyecciones que **escriben en el residual** —`attn_output`
/// y `ffn_down`— arrancan divididas por `sqrt(2 * capas)`: por el residual
/// pasan `2 * capas` sumas, y sin esa división la varianza crece con la
/// profundidad y un modelo hondo diverge en los primeros pasos.
fn inicio_de(nombre: &str, capas: usize) -> Inicio {
    if nombre.ends_with(".bias") {
        return Inicio::Ceros;
    }
    if nombre.ends_with("_norm.weight") {
        return Inicio::Unos;
    }
    if nombre.ends_with("attn_output.weight") || nombre.ends_with("ffn_down.weight") {
        return Inicio::Normal(0.02 / (2.0 * capas as f32).sqrt());
    }
    Inicio::Normal(0.02)
}

/// Lo que salió de escribir un modelo.
#[derive(Debug, Clone)]
pub struct Resumen {
    pub nombre: String,
    pub parametros: u64,
    pub bytes: u64,
    pub tensores: usize,
    pub segundos: f64,
}

/// Inicializa un daiss y lo escribe como GGUF.
///
/// Genera **fila por fila**: el pico de memoria es una fila del tensor más
/// ancho, no el tensor y mucho menos el modelo. Escribir un daiss-14b pide
/// unos 20 KB de memoria de trabajo.
pub fn escribir(cfg: &Config, pesos: TipoGgml, semilla: u64, ruta: &Path) -> Resultado<Resumen> {
    escribir_con(cfg, pesos, semilla, None, ruta)
}

/// Igual, incrustando el tokenizador propio en el archivo.
///
/// Cuando se le pasa un [`Bpe`], el vocabulario de la configuración se ignora
/// y se usa el del tokenizador: un modelo cuya tabla de embeddings no coincide
/// con su tokenizador no sirve para nada, y que las dos cifras puedan
/// discrepar sería una forma de romperlo en silencio.
///
/// Las fusiones van bajo `cosxamix.tokenizer.*` y **no** bajo
/// `tokenizer.ggml.*`, que es la convención de ggml, y es a propósito: esa
/// convención guarda las piezas como cadenas y las nuestras son secuencias de
/// bytes, que para representarse como texto necesitan la tabla de
/// byte-a-unicode de GPT-2. Esa tabla no está escrita ni verificada, y poner
/// las fusiones bajo la clave estándar sin ella haría que otro runtime leyera
/// un vocabulario equivocado creyendo que lo entiende. Con nuestra clave, un
/// runtime ajeno la ignora y lo dice.
pub fn escribir_con(
    cfg: &Config,
    pesos: TipoGgml,
    semilla: u64,
    tokenizador: Option<&Bpe>,
    ruta: &Path,
) -> Resultado<Resumen> {
    let t0 = std::time::Instant::now();
    let mut propio = *cfg;
    if let Some(t) = tokenizador {
        propio.vocab = t.tamano();
    }
    let cfg = &propio;
    let mut e = Escritor::nuevo();

    e.meta_texto("general.architecture", "qwen2")
        .meta_texto("general.name", cfg.nombre)
        .meta_texto("general.basename", "daiss")
        .meta_texto(
            "general.size_label",
            cfg.nombre.trim_start_matches("daiss-"),
        )
        .meta_texto("general.license", "mit")
        .meta_u32("general.file_type", pesos as u32)
        .meta_u32("qwen2.block_count", cfg.capas as u32)
        .meta_u32("qwen2.context_length", cfg.contexto as u32)
        .meta_u32("qwen2.embedding_length", cfg.dim as u32)
        .meta_u32("qwen2.feed_forward_length", cfg.ffn as u32)
        .meta_u32("qwen2.attention.head_count", cfg.cabezas as u32)
        .meta_u32("qwen2.attention.head_count_kv", cfg.cabezas_kv as u32)
        .meta_f32("qwen2.attention.layer_norm_rms_epsilon", cfg.eps)
        .meta_f32("qwen2.rope.freq_base", cfg.rope)
        .meta_texto("cosxamix.init", "normal(0, 0.02), residual / sqrt(2*capas)")
        .meta("cosxamix.seed", Valor::U64(semilla));

    if let Some(t) = tokenizador {
        // Los pares se guardan aplanados en una lista de U32: GGUF no tiene
        // listas de tuplas, y una lista de listas obligaría a quien lee a
        // reconstruir la forma igual.
        let mut planos = Vec::with_capacity(t.fusiones() * 2);
        for (a, d) in t.pares() {
            planos.push(Valor::U32(a));
            planos.push(Valor::U32(d));
        }
        e.meta_texto("cosxamix.tokenizer.model", "bpe-bytes")
            .meta_u32("cosxamix.tokenizer.size", t.tamano() as u32)
            .meta("cosxamix.tokenizer.merges", Valor::Lista(planos));
    }

    for d in cfg.declaraciones(pesos) {
        e.tensor(d);
    }

    let capas = cfg.capas;

    let total = e.escribir(ruta, |d, w| {
        let ancho = d.dims[0] as usize;
        let filas = d.elementos() / ancho;
        let inicio = inicio_de(&d.nombre, capas);

        // Las filas se generan en paralelo y de a tandas. La tanda acota la
        // memoria: con 16 MB de tope, escribir el `token_embd` de un
        // daiss-14b usa 16 MB y no los 3,1 GB que ocuparia el tensor entero.
        //
        // Cada tarea se lleva un GRUPO de filas y no una sola. Con una fila
        // por tarea habia dos reservas de memoria y una escritura al disco
        // por fila —316.000 de cada una solo en el `token_embd` del mini— y
        // eso costaba mas que generar los numeros: medido, 34 MB/s. Asi las
        // reservas se amortizan y la tanda entera sale en UNA escritura.
        const TOPE: usize = 16 << 20;
        const GRUPO: usize = 64;
        let bytes_fila = d.tipo.bytes_de(ancho);
        let por_tanda = (TOPE / bytes_fila.max(1))
            .max(GRUPO)
            .min(filas.max(1))
            .next_multiple_of(GRUPO);

        let mut buffer = vec![0u8; por_tanda * bytes_fila];
        let mut desde = 0usize;
        while desde < filas {
            let cuantas = por_tanda.min(filas - desde);
            let destino = &mut buffer[..cuantas * bytes_fila];

            destino
                .par_chunks_mut(bytes_fila * GRUPO)
                .enumerate()
                .try_for_each(|(g, trozo)| -> Resultado<()> {
                    // Dos reservas por grupo de 64 filas, no por fila.
                    let mut fila = vec![0f32; ancho];
                    let mut bytes = Vec::with_capacity(bytes_fila);
                    for (j, sub) in trozo.chunks_mut(bytes_fila).enumerate() {
                        let f = desde + g * GRUPO + j;
                        match inicio {
                            Inicio::Unos => fila.fill(1.0),
                            Inicio::Ceros => fila.fill(0.0),
                            Inicio::Normal(sigma) => {
                                let mut azar = Azar::con_semilla(semilla_de(semilla, &d.nombre, f));
                                // De a dos, que es como Box-Muller los da.
                                let mut i = 0;
                                while i + 1 < ancho {
                                    let (a, b) = azar.normales(sigma);
                                    fila[i] = a;
                                    fila[i + 1] = b;
                                    i += 2;
                                }
                                if i < ancho {
                                    fila[i] = azar.normales(sigma).0;
                                }
                            }
                        }
                        bytes.clear();
                        crate::quant::cuantizar_bloques(d.tipo, &fila, &mut bytes)?;
                        sub.copy_from_slice(&bytes);
                    }
                    Ok(())
                })?;

            w.write_all(destino).map_err(|x| x.to_string())?;
            desde += cuantas;
        }
        Ok(())
    })?;

    Ok(Resumen {
        nombre: cfg.nombre.to_string(),
        parametros: cfg.parametros(),
        bytes: total,
        tensores: cfg.declaraciones(pesos).len(),
        segundos: t0.elapsed().as_secs_f64(),
    })
}

/// Saca el tokenizador de un GGUF escrito por [`escribir_con`].
///
/// Devuelve `Ok(None)` cuando el archivo no lo trae, que es el caso de un
/// daiss escrito con el vocabulario prestado de Qwen.
pub fn tokenizador_de(g: &crate::gguf::Gguf) -> Resultado<Option<Bpe>> {
    let Some(Valor::Lista(planos)) = g.metadatos.get("cosxamix.tokenizer.merges") else {
        return Ok(None);
    };
    if !planos.len().is_multiple_of(2) {
        return Err(format!(
            "la lista de fusiones tiene {} valores y tiene que ser par",
            planos.len()
        ));
    }
    let mut pares = Vec::with_capacity(planos.len() / 2);
    for par in planos.chunks(2) {
        let (Some(a), Some(d)) = (par[0].entero(), par[1].entero()) else {
            return Err("una fusión no es un entero".into());
        };
        pares.push((a as u32, d as u32));
    }
    let bpe = Bpe::de_pares(&pares)?;

    // El tamaño anunciado y el reconstruido tienen que coincidir: si no, el
    // archivo dice una cosa y trae otra, y los embeddings no cuadrarían.
    if let Some(dice) = g.meta_entero("cosxamix.tokenizer.size") {
        if dice != bpe.tamano() as u64 {
            return Err(format!(
                "el archivo dice {dice} tokens y las fusiones dan {}",
                bpe.tamano()
            ));
        }
    }
    Ok(Some(bpe))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Las formas tienen que cuadrar con lo que el formato y la atención
    /// piden, en los cinco tamaños. Un error acá produce un archivo que falla
    /// al cargarse, mucho después y lejos.
    #[test]
    fn las_cinco_formas_cierran() {
        for c in FAMILIA {
            assert_eq!(c.dim % c.cabezas, 0, "{}: dim entre cabezas", c.nombre);
            assert_eq!(
                c.cabezas % c.cabezas_kv,
                0,
                "{}: las cabezas se reparten entre los grupos de kv",
                c.nombre
            );
            // Lo que el escritor necesita de verdad es el bloque de 32, que
            // es el de Q8_0. Pedir 256 aca seria pedir de mas: `daiss-0.5b`
            // tiene dim 896, que son 3,5 bloques de 256, igual que el
            // Qwen2.5-0.5B real — y por eso sus GGUF mezclan formatos.
            for (que, v) in [("dim", c.dim), ("ffn", c.ffn), ("kv", c.dim_kv())] {
                assert_eq!(v % 32, 0, "{}: {que} = {v} no cuadra con Q8_0", c.nombre);
            }
            assert_eq!(
                c.admite_kquants(),
                c.dim.is_multiple_of(256) && c.ffn.is_multiple_of(256),
                "{}: admite_kquants no coincide con las formas",
                c.nombre
            );
            assert_eq!(
                c.declaraciones(TipoGgml::Q8_0).len(),
                c.capas * 12 + 3,
                "{}: cantidad de tensores",
                c.nombre
            );
        }

        // El unico de la familia que no admite k-quants es el 0.5b. Si algun
        // dia deja de ser cierto, que lo diga el test y no una sorpresa.
        let sin: Vec<&str> = FAMILIA
            .iter()
            .filter(|c| !c.admite_kquants())
            .map(|c| c.nombre)
            .collect();
        assert_eq!(sin, ["daiss-0.5b"], "quien no admite k-quants cambio");
    }

    /// La cuenta de parámetros tiene que dar lo que dice el nombre, con el
    /// margen de que el nombre redondea.
    #[test]
    fn los_parametros_cuadran_con_el_nombre() {
        let esperado = [
            ("daiss-mini", 80_000_000u64, 90_000_000u64),
            ("daiss-0.5b", 450_000_000, 650_000_000),
            ("daiss-3b", 2_900_000_000, 3_400_000_000),
            ("daiss-7b", 7_000_000_000, 8_000_000_000),
            ("daiss-14b", 13_000_000_000, 15_500_000_000),
        ];
        for (nombre, min, max) in esperado {
            let c = Config::de_nombre(nombre).unwrap();
            let p = c.parametros();
            assert!(
                (min..=max).contains(&p),
                "{nombre}: {p} parametros, se esperaba entre {min} y {max}"
            );
        }
    }

    /// La misma semilla tiene que dar el mismo modelo, y una distinta uno
    /// distinto. Sin esto, un daiss no se podría reproducir.
    #[test]
    fn la_semilla_manda() {
        let mut a = Azar::con_semilla(7);
        let mut b = Azar::con_semilla(7);
        let mut c = Azar::con_semilla(8);
        let xa: Vec<u64> = (0..8).map(|_| a.siguiente()).collect();
        let xb: Vec<u64> = (0..8).map(|_| b.siguiente()).collect();
        let xc: Vec<u64> = (0..8).map(|_| c.siguiente()).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);

        // Y el cero no puede quedarse clavado, que es el punto fijo de
        // xorshift.
        let mut z = Azar::con_semilla(0);
        assert_ne!(z.siguiente(), 0);
        assert_ne!(z.siguiente(), 0);
    }

    /// La inicialización tiene que tener la desviación que dice tener, y las
    /// dos proyecciones del residual la suya, más chica.
    #[test]
    fn la_inicializacion_tiene_la_desviacion_que_dice() {
        let mut azar = Azar::con_semilla(1234);
        let n = 20_000;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n / 2 {
            let (a, b) = azar.normales(0.02);
            v.push(a);
            v.push(b);
        }
        let media = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|x| (x - media).powi(2)).sum::<f32>() / n as f32;
        assert!(media.abs() < 0.001, "la media dio {media}");
        assert!(
            (var.sqrt() - 0.02).abs() < 0.001,
            "la desviacion dio {}",
            var.sqrt()
        );

        // Y el reparto por nombre.
        assert!(matches!(inicio_de("blk.0.attn_q.bias", 36), Inicio::Ceros));
        assert!(matches!(
            inicio_de("blk.0.attn_norm.weight", 36),
            Inicio::Unos
        ));
        assert!(matches!(inicio_de("output_norm.weight", 36), Inicio::Unos));
        match inicio_de("blk.0.ffn_down.weight", 36) {
            Inicio::Normal(s) => assert!((s - 0.02 / 72f32.sqrt()).abs() < 1e-9, "{s}"),
            otro => panic!("ffn_down salio como {:?}", std::mem::discriminant(&otro)),
        }
        match inicio_de("blk.0.ffn_gate.weight", 36) {
            Inicio::Normal(s) => assert_eq!(s, 0.02),
            otro => panic!("ffn_gate salio como {:?}", std::mem::discriminant(&otro)),
        }
    }

    /// El modelo más chico se escribe entero y se vuelve a leer, con las
    /// formas y los metadatos en su sitio.
    ///
    /// Corre en CI sin descargar nada: daiss-mini se genera acá mismo.
    #[test]
    fn el_mini_se_escribe_y_se_lee() {
        let dir = std::env::temp_dir().join("cosxamix_daiss_mini");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("daiss-mini.gguf");

        let cfg = Config::de_nombre("daiss-mini").unwrap();
        let r = escribir(cfg, TipoGgml::Q8_0, 42, &ruta).unwrap();
        assert_eq!(r.tensores, cfg.capas * 12 + 3);
        assert_eq!(r.bytes, std::fs::metadata(&ruta).unwrap().len());

        let g = crate::gguf::Gguf::abrir(&ruta).unwrap();
        assert_eq!(g.tensores.len(), r.tensores);
        assert_eq!(g.meta_entero("qwen2.block_count"), Some(6));
        assert_eq!(g.meta_entero("qwen2.embedding_length"), Some(256));
        assert_eq!(
            g.metadatos.get("general.name").and_then(|v| v.texto()),
            Some("daiss-mini")
        );

        // Las normas tienen que haber salido en unos de verdad, no en algo
        // que se le parezca: van en F32 y no pasan por cuantizador.
        let t = g.tensor("blk.0.attn_norm.weight").unwrap();
        assert_eq!(t.tipo, TipoGgml::F32);
        let bytes = g.datos_de(&ruta, t).unwrap();
        let mut v = vec![0f32; t.elementos()];
        crate::quant::descuantizar(t.tipo, &bytes, &mut v).unwrap();
        assert!(v.iter().all(|x| *x == 1.0), "las normas no son unos");

        // Y un peso cuantizado tiene que tener la desviacion con la que se
        // inicializo, dentro de lo que Q8_0 conserva.
        let t = g.tensor("blk.0.ffn_gate.weight").unwrap();
        let bytes = g.datos_de(&ruta, t).unwrap();
        let mut v = vec![0f32; t.elementos()];
        crate::quant::descuantizar(t.tipo, &bytes, &mut v).unwrap();
        let media = v.iter().sum::<f32>() / v.len() as f32;
        let sigma = (v.iter().map(|x| (x - media).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        assert!((sigma - 0.02).abs() < 0.002, "la desviacion dio {sigma}");

        // La misma semilla, el mismo archivo byte por byte.
        let otra = dir.join("otra.gguf");
        escribir(cfg, TipoGgml::Q8_0, 42, &otra).unwrap();
        assert_eq!(
            std::fs::read(&ruta).unwrap(),
            std::fs::read(&otra).unwrap(),
            "la misma semilla tiene que dar el mismo archivo"
        );
    }

    /// Un daiss escrito con el tokenizador propio tiene que traerlo dentro, y
    /// el que sale del archivo tiene que codificar **igual** que el original.
    ///
    /// Y el vocabulario del modelo tiene que ser el del tokenizador y no el de
    /// la configuración: un modelo cuya tabla de embeddings no coincide con su
    /// tokenizador está roto, y en silencio.
    #[test]
    fn el_tokenizador_viaja_dentro_del_modelo() {
        use crate::tokenizer::Bpe;

        let dir = std::env::temp_dir().join("cosxamix_daiss_tok");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("daiss-propio.gguf");

        let texto = "la multiplicacion fusionada gana con una fila. \
            La multiplicacion fusionada pierde con treinta y dos.";
        let tok = Bpe::entrenar(texto, 400);
        assert!(tok.tamano() > 256, "el entrenamiento no aprendio nada");

        // Un mini de verdad seria 151.936 de vocabulario; aca el tokenizador
        // manda y el modelo queda chiquito, que es justo el punto.
        let cfg = Config::de_nombre("daiss-mini").unwrap();
        let r = escribir_con(cfg, TipoGgml::Q8_0, 7, Some(&tok), &ruta).unwrap();
        assert!(
            r.bytes < 40 << 20,
            "con un vocabulario de {} el archivo salio de {} bytes",
            tok.tamano(),
            r.bytes
        );

        let g = crate::gguf::Gguf::abrir(&ruta).unwrap();
        // La tabla de embeddings tiene que tener el vocabulario del
        // tokenizador.
        let t = g.tensor("token_embd.weight").unwrap();
        assert_eq!(t.dims[1], tok.tamano() as u64, "el vocabulario no cuadra");
        assert_eq!(
            g.meta_entero("cosxamix.tokenizer.size"),
            Some(tok.tamano() as u64)
        );

        let vuelta = tokenizador_de(&g).unwrap().expect("no trajo tokenizador");
        assert_eq!(vuelta.tamano(), tok.tamano());
        assert_eq!(vuelta.codificar(texto), tok.codificar(texto));
        assert_eq!(vuelta.decodificar(&vuelta.codificar(texto)).unwrap(), texto);

        // Y un daiss sin tokenizador propio tiene que decir que no lo trae, no
        // fallar.
        let otro = dir.join("prestado.gguf");
        escribir(cfg, TipoGgml::Q8_0, 7, &otro).unwrap();
        let g2 = crate::gguf::Gguf::abrir(&otro).unwrap();
        assert!(tokenizador_de(&g2).unwrap().is_none());
    }
}
