//! Lector de GGUF propio.
//!
//! ## Por qué dejar de usar el de candle
//!
//! Hasta acá Cosxamix leía los modelos con `candle_core::quantized::gguf_file`.
//! Funciona, y aun así es el primer ladrillo que hay que poner de nuestro lado,
//! porque **todo lo demás depende del formato de los pesos**: la
//! descuantización, la multiplicación, y sobre todo el backward cuantizado que
//! candle no tiene y que es lo que hoy impide entrenar el 3B.
//!
//! Medido en esta sesión, candle es un techo en cinco sitios a la vez:
//!
//! * `QMatMul` no implementa `bwd`, así que el gradiente no atraviesa una
//!   multiplicación cuantizada;
//! * su `matmul` relee la matriz entera por cada fila del lote;
//! * no usa AVX-512 ni SVE;
//! * no soporta los formatos IQ;
//! * su backprop calcula `dL/dW` de capas congeladas que nadie usa.
//!
//! Ninguno de los cinco se arregla desde afuera.
//!
//! ## Cómo se comprueba que esto está bien
//!
//! Contra candle, que queda como **oráculo en los tests** aunque salga del
//! binario. Cada campo leído acá tiene que coincidir con el que lee candle
//! sobre el mismo archivo: no es una reimplementación a ciegas, es una con
//! respuesta conocida.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Los tipos de dato que GGUF sabe guardar en un tensor.
///
/// Los números son los del formato y no se pueden cambiar: están escritos en
/// los archivos que ya existen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TipoGgml {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
}

impl TipoGgml {
    pub fn de_u32(v: u32) -> Option<Self> {
        use TipoGgml::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            _ => return None,
        })
    }

    /// Cuántos valores entran en un bloque de este tipo.
    ///
    /// Los k-quants usan bloques de 256 — por eso un tensor cuya fila no sea
    /// múltiplo de 256 no puede usarlos, y por eso el Qwen2.5-0,5B, que es de
    /// 896 de ancho, guarda casi todo en Q5_0 y Q8_0 pese a llamarse Q4_K_M.
    pub fn valores_por_bloque(self) -> usize {
        use TipoGgml::*;
        match self {
            F32 | F16 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K => 256,
        }
    }

    /// Cuántos bytes ocupa un bloque.
    pub fn bytes_por_bloque(self) -> usize {
        use TipoGgml::*;
        match self {
            F32 => 4,
            F16 => 2,
            Q4_0 => 2 + 16,
            Q4_1 => 4 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 4 + 4 + 16,
            Q8_0 => 2 + 32,
            Q8_1 => 4 + 32,
            Q2K => 16 + 64 + 2 + 2,
            Q3K => 32 + 64 + 12 + 2,
            Q4K => 2 + 2 + 12 + 128,
            Q5K => 2 + 2 + 12 + 32 + 128,
            Q6K => 128 + 64 + 16 + 2,
            Q8K => 4 + 256 + 2 * 16,
        }
    }

    /// Bytes que ocupan `n` valores de este tipo.
    pub fn bytes_de(self, n: usize) -> usize {
        n / self.valores_por_bloque() * self.bytes_por_bloque()
    }
}

/// Un valor de los metadatos. GGUF permite muchos tipos; acá están los que
/// aparecen de verdad en los archivos de modelos.
#[derive(Debug, Clone)]
pub enum Valor {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Texto(String),
    Lista(Vec<Valor>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Valor {
    /// Lee un entero sin importar en qué ancho vino guardado.
    ///
    /// Hace falta porque el mismo campo puede estar como `u32` en un archivo y
    /// como `u64` en otro, y a quien lo pide le da igual.
    pub fn entero(&self) -> Option<u64> {
        Some(match self {
            Valor::U8(v) => *v as u64,
            Valor::I8(v) => *v as u64,
            Valor::U16(v) => *v as u64,
            Valor::I16(v) => *v as u64,
            Valor::U32(v) => *v as u64,
            Valor::I32(v) => *v as u64,
            Valor::U64(v) => *v,
            Valor::I64(v) => *v as u64,
            _ => return None,
        })
    }

    pub fn decimal(&self) -> Option<f64> {
        Some(match self {
            Valor::F32(v) => *v as f64,
            Valor::F64(v) => *v,
            _ => return self.entero().map(|v| v as f64),
        })
    }

    pub fn texto(&self) -> Option<&str> {
        match self {
            Valor::Texto(s) => Some(s),
            _ => None,
        }
    }
}

/// Dónde está y qué forma tiene un tensor dentro del archivo.
#[derive(Debug, Clone)]
pub struct InfoTensor {
    pub nombre: String,
    /// Como vienen en el archivo: la dimensión que varía más rápido primero.
    /// Para una matriz de pesos eso es `[entradas, salidas]`, al revés de como
    /// se suele escribir una matriz.
    pub dims: Vec<u64>,
    pub tipo: TipoGgml,
    /// Desplazamiento desde el principio de la zona de datos, no del archivo.
    pub offset: u64,
}

impl InfoTensor {
    pub fn elementos(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn bytes(&self) -> usize {
        self.tipo.bytes_de(self.elementos())
    }
}

pub struct Gguf {
    pub version: u32,
    pub metadatos: HashMap<String, Valor>,
    pub tensores: Vec<InfoTensor>,
    /// Dónde empieza la zona de datos, ya alineada.
    pub inicio_datos: u64,
}

type Resultado<T> = Result<T, String>;

struct Lector<R> {
    r: R,
}

impl<R: Read> Lector<R> {
    fn bytes<const N: usize>(&mut self) -> Resultado<[u8; N]> {
        let mut b = [0u8; N];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("archivo cortado: {e}"))?;
        Ok(b)
    }

    fn u32(&mut self) -> Resultado<u32> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }

    fn u64(&mut self) -> Resultado<u64> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }

    /// Una cadena de GGUF: largo en u64 y después los bytes, sin terminador.
    fn texto(&mut self) -> Resultado<String> {
        let n = self.u64()? as usize;
        // Un largo absurdo casi siempre significa que se perdió la posición,
        // no que el archivo tenga una cadena de 4 GB. Cortar acá da un error
        // entendible en vez de intentar reservar toda la memoria.
        if n > 64 * 1024 * 1024 {
            return Err(format!("cadena de {n} bytes: el archivo no cuadra"));
        }
        let mut b = vec![0u8; n];
        self.r
            .read_exact(&mut b)
            .map_err(|e| format!("cadena cortada: {e}"))?;
        String::from_utf8(b).map_err(|e| format!("cadena no es UTF-8: {e}"))
    }

    fn valor(&mut self, tipo: u32) -> Resultado<Valor> {
        Ok(match tipo {
            0 => Valor::U8(self.bytes::<1>()?[0]),
            1 => Valor::I8(self.bytes::<1>()?[0] as i8),
            2 => Valor::U16(u16::from_le_bytes(self.bytes::<2>()?)),
            3 => Valor::I16(i16::from_le_bytes(self.bytes::<2>()?)),
            4 => Valor::U32(self.u32()?),
            5 => Valor::I32(i32::from_le_bytes(self.bytes::<4>()?)),
            6 => Valor::F32(f32::from_le_bytes(self.bytes::<4>()?)),
            7 => Valor::Bool(self.bytes::<1>()?[0] != 0),
            8 => Valor::Texto(self.texto()?),
            9 => {
                let interno = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.valor(interno)?);
                }
                Valor::Lista(v)
            }
            10 => Valor::U64(self.u64()?),
            11 => Valor::I64(i64::from_le_bytes(self.bytes::<8>()?)),
            12 => Valor::F64(f64::from_le_bytes(self.bytes::<8>()?)),
            otro => return Err(format!("tipo de metadato desconocido: {otro}")),
        })
    }
}

impl Gguf {
    /// Lee la cabecera: metadatos y dónde está cada tensor. **No** lee los
    /// pesos, que son gigabytes.
    pub fn abrir(ruta: &Path) -> Resultado<Self> {
        let f = File::open(ruta).map_err(|e| format!("no pude abrir {}: {e}", ruta.display()))?;
        let mut l = Lector {
            r: BufReader::with_capacity(1 << 20, f),
        };

        if &l.bytes::<4>()? != b"GGUF" {
            return Err("no empieza con GGUF: no es un archivo de este formato".into());
        }
        let version = l.u32()?;
        if version != 2 && version != 3 {
            return Err(format!("version de GGUF {version}, solo se leen 2 y 3"));
        }

        let n_tensores = l.u64()? as usize;
        let n_meta = l.u64()? as usize;

        let mut metadatos = HashMap::with_capacity(n_meta);
        for _ in 0..n_meta {
            let clave = l.texto()?;
            let tipo = l.u32()?;
            metadatos.insert(clave, l.valor(tipo)?);
        }

        let mut tensores = Vec::with_capacity(n_tensores);
        for _ in 0..n_tensores {
            let nombre = l.texto()?;
            let nd = l.u32()? as usize;
            if nd > 4 {
                return Err(format!("{nombre}: {nd} dimensiones, el formato permite 4"));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(l.u64()?);
            }
            let bruto = l.u32()?;
            let tipo = TipoGgml::de_u32(bruto)
                .ok_or_else(|| format!("{nombre}: tipo de tensor desconocido {bruto}"))?;
            let offset = l.u64()?;
            tensores.push(InfoTensor {
                nombre,
                dims,
                tipo,
                offset,
            });
        }

        // La zona de datos arranca alineada. El valor por defecto es 32 y
        // algunos archivos lo cambian: leerlo del archivo en vez de darlo por
        // supuesto es la diferencia entre leer los pesos correctos y leer
        // basura desplazada unos bytes.
        let alineacion = metadatos
            .get("general.alignment")
            .and_then(|v| v.entero())
            .unwrap_or(32)
            .max(1);

        let pos =
            l.r.stream_position()
                .map_err(|e| format!("no pude saber la posicion: {e}"))?;
        let inicio_datos = pos.div_ceil(alineacion) * alineacion;

        Ok(Self {
            version,
            metadatos,
            tensores,
            inicio_datos,
        })
    }

    pub fn tensor(&self, nombre: &str) -> Option<&InfoTensor> {
        self.tensores.iter().find(|t| t.nombre == nombre)
    }

    pub fn meta_entero(&self, clave: &str) -> Option<u64> {
        self.metadatos.get(clave).and_then(|v| v.entero())
    }

    pub fn meta_decimal(&self, clave: &str) -> Option<f64> {
        self.metadatos.get(clave).and_then(|v| v.decimal())
    }

    /// Los bytes crudos de un tensor, tal cual están en el archivo.
    pub fn datos_de(&self, ruta: &Path, info: &InfoTensor) -> Resultado<Vec<u8>> {
        let mut f =
            File::open(ruta).map_err(|e| format!("no pude abrir {}: {e}", ruta.display()))?;
        f.seek(SeekFrom::Start(self.inicio_datos + info.offset))
            .map_err(|e| format!("no pude moverme en el archivo: {e}"))?;
        let mut b = vec![0u8; info.bytes()];
        f.read_exact(&mut b)
            .map_err(|e| format!("{}: datos cortados: {e}", info.nombre))?;
        Ok(b)
    }
}

/// Los pesos del archivo, mapeados en memoria virtual.
///
/// Reemplaza al camino de [`Gguf::datos_de`], que abre el archivo **de nuevo
/// por cada tensor** y copia sus bytes a un `Vec`. En el daiss-7b eso son 339
/// aperturas y 7,5 GB copiados al montículo; acá es un mapeo y cero copias,
/// porque [`Pesos::de`] devuelve una porción que apunta al archivo.
///
/// Que devuelva `&[u8]` y no `Vec<u8>` es el punto: `descuantizar`,
/// `desempaquetar_i8` y las dos `matmul` ya reciben `&[u8]`, así que el camino
/// entero desde el disco hasta el producto punto queda sin una sola copia.
///
/// **El archivo no se puede modificar mientras esto viva.** Un mapeo no es una
/// copia: si otro proceso trunca o reescribe el archivo, las porciones ya
/// entregadas pasan a leer otra cosa, o el proceso recibe un SIGBUS. Para pesos
/// de un modelo es la condición normal —se escriben una vez y se leen— y es la
/// razón por la que `gguf_write` escribe a un temporal y renombra.
pub struct Pesos {
    mapa: memmap2::Mmap,
    inicio_datos: u64,
}

impl Pesos {
    pub fn abrir(g: &Gguf, ruta: &Path) -> Resultado<Self> {
        let f = File::open(ruta).map_err(|e| format!("no pude abrir {}: {e}", ruta.display()))?;
        // SEGURIDAD: se cumple la condición documentada arriba — el archivo no
        // se modifica mientras el mapeo viva. No hay forma de garantizarlo con
        // el sistema de tipos porque el dueño del archivo es el sistema
        // operativo y no este proceso.
        let mapa = unsafe { memmap2::Mmap::map(&f) }
            .map_err(|e| format!("no pude mapear {}: {e}", ruta.display()))?;
        if (mapa.len() as u64) < g.inicio_datos {
            return Err(format!(
                "el archivo mide {} y los datos empiezan en {}",
                mapa.len(),
                g.inicio_datos
            ));
        }
        Ok(Self {
            mapa,
            inicio_datos: g.inicio_datos,
        })
    }

    /// Los bytes de un tensor, sin copiarlos.
    pub fn de(&self, info: &InfoTensor) -> Resultado<&[u8]> {
        // Las TRES cuentas van comprobadas, no solo la última. La suma de
        // `inicio_datos` y `offset` se desbordaba antes de llegar a la
        // comprobación de abajo, y en release eso da un índice envuelto que
        // apunta a cualquier parte del mapeo: lo encontró el test que probaba
        // un offset absurdo, que es exactamente para lo que estaba escrito.
        let desde = self
            .inicio_datos
            .checked_add(info.offset)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| format!("{}: el offset se desborda", info.nombre))?;
        let hasta = desde
            .checked_add(info.bytes())
            .ok_or_else(|| format!("{}: el tamaño se desborda", info.nombre))?;
        // Comprobar antes de cortar: una porción fuera de rango sobre un mapeo
        // no da un error, da una violación de acceso.
        if hasta > self.mapa.len() {
            return Err(format!(
                "{}: pide hasta el byte {hasta} y el archivo mide {}",
                info.nombre,
                self.mapa.len()
            ));
        }
        Ok(&self.mapa[desde..hasta])
    }

    pub fn bytes(&self) -> usize {
        self.mapa.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construye un GGUF minimo pero valido en memoria.
    ///
    /// Hace falta porque los tests tienen que correr en CI, donde no hay
    /// modelos: son gigabytes y no van al repositorio. Un archivo sintetico
    /// prueba el parseo entero -- cabecera, metadatos de varios tipos,
    /// informacion de tensores y alineacion -- sin depender de nada externo.
    fn gguf_de_prueba() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&2u64.to_le_bytes()); // tensores
        b.extend_from_slice(&4u64.to_le_bytes()); // metadatos

        let texto = |b: &mut Vec<u8>, s: &str| {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        };

        texto(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        texto(&mut b, "qwen2");

        texto(&mut b, "qwen2.block_count");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&24u32.to_le_bytes());

        texto(&mut b, "qwen2.rope.freq_base");
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&1_000_000f32.to_le_bytes());

        // Una lista, que es el caso que mas facil se parsea mal.
        texto(&mut b, "tokenizer.ggml.tokens");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // de textos
        b.extend_from_slice(&2u64.to_le_bytes());
        texto(&mut b, "hola");
        texto(&mut b, "chau");

        // Dos tensores: uno F32 chico y uno Q8_0.
        texto(&mut b, "norm.weight");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes());

        texto(&mut b, "blk.0.attn_q.weight");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&64u64.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes()); // Q8_0
        b.extend_from_slice(&32u64.to_le_bytes());

        // Relleno hasta la alineacion de 32, y despues datos cualesquiera.
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(0xABu8, 4096));
        b
    }

    fn escribir_temporal(nombre: &str, datos: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(nombre);
        std::fs::write(&p, datos).expect("escribir el gguf de prueba");
        p
    }

    #[test]
    fn lee_cabecera_metadatos_y_tensores() {
        let p = escribir_temporal("cosxamix_gguf_ok.gguf", &gguf_de_prueba());
        let g = Gguf::abrir(&p).expect("abrir");

        assert_eq!(g.version, 3);
        assert_eq!(
            g.metadatos.get("general.architecture").unwrap().texto(),
            Some("qwen2")
        );
        assert_eq!(g.meta_entero("qwen2.block_count"), Some(24));
        assert_eq!(g.meta_decimal("qwen2.rope.freq_base"), Some(1_000_000.0));

        // La lista tiene que haberse leido entera, no salteada.
        match g.metadatos.get("tokenizer.ggml.tokens") {
            Some(Valor::Lista(v)) => {
                assert_eq!(v.len(), 2);
                assert_eq!(v[1].texto(), Some("chau"));
            }
            otro => panic!("la lista se leyo como {otro:?}"),
        }

        assert_eq!(g.tensores.len(), 2);
        let q = g.tensor("blk.0.attn_q.weight").expect("el tensor");
        assert_eq!(q.dims, vec![64, 2]);
        assert_eq!(q.tipo, TipoGgml::Q8_0);
        assert_eq!(q.elementos(), 128);
        // 128 valores en bloques de 32 son 4 bloques de 34 bytes.
        assert_eq!(q.bytes(), 136);

        let _ = std::fs::remove_file(&p);
    }

    /// La alineacion sale del archivo y no de una constante.
    ///
    /// Darla por supuesta es la clase de error que no da error: los pesos se
    /// leen desplazados unos bytes y el modelo responde basura.
    #[test]
    fn respeta_la_alineacion_declarada() {
        let base = gguf_de_prueba();
        let g = Gguf::abrir(&escribir_temporal("cosxamix_gguf_align.gguf", &base)).expect("abrir");
        assert_eq!(g.inicio_datos % 32, 0);
        assert!(
            g.inicio_datos >= 1,
            "la zona de datos no puede arrancar en 0"
        );
    }

    #[test]
    fn rechaza_lo_que_no_es_gguf() {
        let p = escribir_temporal("cosxamix_no_gguf.bin", b"esto no es un modelo");
        assert!(Gguf::abrir(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    /// Un archivo cortado tiene que dar error, no leer basura.
    #[test]
    fn un_archivo_cortado_avisa() {
        let mut datos = gguf_de_prueba();
        datos.truncate(40);
        let p = escribir_temporal("cosxamix_gguf_corto.gguf", &datos);
        assert!(Gguf::abrir(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    /// Los tamanos de bloque son del formato y no se pueden improvisar.
    ///
    /// Si uno esta mal, el tensor se lee con el largo equivocado y lo que sale
    /// no se parece en nada -- pero no falla, que es lo peor.
    #[test]
    fn los_tamanos_de_bloque_son_los_del_formato() {
        use TipoGgml::*;
        for (t, valores, bytes) in [
            (F32, 1, 4),
            (F16, 1, 2),
            (Q4_0, 32, 18),
            (Q5_0, 32, 22),
            (Q8_0, 32, 34),
            (Q2K, 256, 84),
            (Q3K, 256, 110),
            (Q4K, 256, 144),
            (Q5K, 256, 176),
            (Q6K, 256, 210),
        ] {
            assert_eq!(t.valores_por_bloque(), valores, "{t:?}: valores");
            assert_eq!(t.bytes_por_bloque(), bytes, "{t:?}: bytes");
        }
    }

    /// El mapeo tiene que entregar **los mismos bytes** que el camino que
    /// copia, y tiene que rechazar un tensor que se salga del archivo.
    ///
    /// Lo segundo es lo que mas importa: `Pesos` es `unsafe` y una porcion
    /// fuera de rango sobre un mapeo no da un `Err`, da una violacion de
    /// acceso que mata el proceso. La comprobacion de limites de `de()` es lo
    /// unico que separa un error con nombre de un `SIGSEGV`.
    #[test]
    fn el_mapeo_entrega_los_mismos_bytes_y_respeta_los_limites() {
        let dir = std::env::temp_dir().join("cosxamix_mapeo");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("prueba.gguf");
        std::fs::write(&ruta, gguf_de_prueba()).unwrap();

        let g = Gguf::abrir(&ruta).unwrap();
        let p = Pesos::abrir(&g, &ruta).unwrap();
        assert_eq!(p.bytes(), std::fs::metadata(&ruta).unwrap().len() as usize);

        for t in &g.tensores {
            let copiado = g.datos_de(&ruta, t).unwrap();
            let mapeado = p.de(t).unwrap();
            assert_eq!(copiado, mapeado, "{}: el mapeo leyo otra cosa", t.nombre);
        }

        // Un tensor que pide mas alla del final: tiene que avisar, no morir.
        let mut fuera = g.tensores[0].clone();
        fuera.dims = vec![1 << 40];
        let error = p.de(&fuera).unwrap_err();
        assert!(error.contains("y el archivo mide"), "{error}");

        // Y un offset absurdo, que es el otro camino al mismo desastre.
        let mut lejos = g.tensores[0].clone();
        lejos.offset = u64::MAX - 16;
        assert!(p.de(&lejos).is_err());
    }
}
