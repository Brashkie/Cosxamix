//! Escribir GGUF: el camino de vuelta del lector de [`crate::gguf`].
//!
//! Hace falta para que "crear un modelo" signifique algo. Un modelo propio que
//! no se puede guardar en el formato que el resto del mundo lee no es un
//! modelo, es un experimento que muere al cerrar el proceso.
//!
//! **Escribe en flujo, un tensor a la vez.** No junta el modelo en memoria y
//! después lo volca: para un daiss grande eso pediría decenas de gigabytes de
//! RAM y la máquina tiene 7,3 libres. El escritor declara primero la forma de
//! cada tensor —nombre, dimensiones y tipo—, calcula con eso los offsets de la
//! cabecera, y después pide los bytes de cada uno por separado. El pico de
//! memoria es el del tensor más grande, no el del modelo.
//!
//! La verificación es la de siempre: lo que escribimos lo tiene que leer
//! nuestro lector **y** candle, que es una implementación independiente. Si
//! las dos lo leen igual, el archivo está bien.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::gguf::{TipoGgml, Valor};

type Resultado<T> = Result<T, String>;

/// La forma de un tensor, sin sus datos.
#[derive(Debug, Clone)]
pub struct Declaracion {
    pub nombre: String,
    /// En el orden del archivo: `[entradas, salidas]`.
    pub dims: Vec<u64>,
    pub tipo: TipoGgml,
}

impl Declaracion {
    pub fn nueva(nombre: impl Into<String>, dims: &[u64], tipo: TipoGgml) -> Self {
        Self {
            nombre: nombre.into(),
            dims: dims.to_vec(),
            tipo,
        }
    }

    pub fn elementos(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn bytes(&self) -> usize {
        self.tipo.bytes_de(self.elementos())
    }
}

/// Junta metadatos y declaraciones, y después escribe el archivo.
pub struct Escritor {
    alineacion: u64,
    metadatos: Vec<(String, Valor)>,
    tensores: Vec<Declaracion>,
}

impl Default for Escritor {
    fn default() -> Self {
        Self::nuevo()
    }
}

impl Escritor {
    pub fn nuevo() -> Self {
        Self {
            alineacion: 32,
            metadatos: Vec::new(),
            tensores: Vec::new(),
        }
    }

    /// Agrega un metadato. El orden en que se agregan es el del archivo.
    pub fn meta(&mut self, clave: impl Into<String>, valor: Valor) -> &mut Self {
        self.metadatos.push((clave.into(), valor));
        self
    }

    pub fn meta_u32(&mut self, clave: impl Into<String>, v: u32) -> &mut Self {
        self.meta(clave, Valor::U32(v))
    }

    pub fn meta_f32(&mut self, clave: impl Into<String>, v: f32) -> &mut Self {
        self.meta(clave, Valor::F32(v))
    }

    pub fn meta_texto(&mut self, clave: impl Into<String>, v: impl Into<String>) -> &mut Self {
        self.meta(clave, Valor::Texto(v.into()))
    }

    pub fn tensor(&mut self, d: Declaracion) -> &mut Self {
        self.tensores.push(d);
        self
    }

    pub fn declaraciones(&self) -> &[Declaracion] {
        &self.tensores
    }

    /// Lo que va a pesar la zona de datos, con el relleno de alineación.
    pub fn bytes_de_datos(&self) -> u64 {
        let mut total = 0u64;
        for d in &self.tensores {
            total = total.div_ceil(self.alineacion) * self.alineacion;
            total += d.bytes() as u64;
        }
        total
    }

    /// Escribe el archivo, pidiendo los bytes de cada tensor por separado.
    ///
    /// `fuente` recibe la declaración y el destino donde tiene que dejar
    /// exactamente [`Declaracion::bytes`] bytes. Se le llama una vez por
    /// tensor, en el orden en que se declararon, así que puede generar el
    /// tensor en ese momento y olvidarlo — que es lo que permite escribir un
    /// modelo más grande que la memoria.
    ///
    /// Devuelve el tamaño del archivo.
    pub fn escribir<F>(&self, ruta: &Path, mut fuente: F) -> Resultado<u64>
    where
        F: FnMut(&Declaracion, &mut dyn Write) -> Resultado<()>,
    {
        if self.tensores.is_empty() {
            return Err("un GGUF sin tensores no sirve para nada".into());
        }
        for d in &self.tensores {
            if d.dims.is_empty() || d.dims.len() > 4 {
                return Err(format!(
                    "{}: {} dimensiones, el formato permite entre 1 y 4",
                    d.nombre,
                    d.dims.len()
                ));
            }
            let por_bloque = d.tipo.valores_por_bloque();
            // La PRIMERA dimension es la que se recorre dentro de cada fila,
            // asi que es la que tiene que ser multiplo del bloque. Si no lo
            // es, el archivo se escribe igual y el lector devuelve basura
            // desplazada, que es el peor error posible: silencioso.
            if !(d.dims[0] as usize).is_multiple_of(por_bloque) {
                return Err(format!(
                    "{}: la primera dimension es {} y {:?} va en bloques de {por_bloque}",
                    d.nombre, d.dims[0], d.tipo
                ));
            }
        }

        // Se escribe a un temporal y se renombra al final. Un archivo cortado
        // a mitad de camino seria un GGUF con cabecera valida y datos
        // truncados, que falla al usarlo y no al escribirlo.
        let temporal = ruta.with_extension("parcial");
        let archivo = File::create(&temporal)
            .map_err(|e| format!("no pude crear {}: {e}", temporal.display()))?;
        let mut w = Contador {
            w: BufWriter::with_capacity(1 << 20, archivo),
            escrito: 0,
        };

        w.write_all(b"GGUF").map_err(err)?;
        w.write_all(&3u32.to_le_bytes()).map_err(err)?;
        w.write_all(&(self.tensores.len() as u64).to_le_bytes())
            .map_err(err)?;
        // La alineacion se escribe SIEMPRE, aunque sea la de por defecto: el
        // lector la busca en los metadatos, y dar por supuesto el 32 fue lo
        // que mas cerca estuvo de leer basura desplazada.
        let n_meta = self.metadatos.len() as u64 + 1;
        w.write_all(&n_meta.to_le_bytes()).map_err(err)?;

        escribir_texto(&mut w, "general.alignment")?;
        escribir_valor(&mut w, &Valor::U32(self.alineacion as u32))?;
        for (clave, valor) in &self.metadatos {
            escribir_texto(&mut w, clave)?;
            escribir_valor(&mut w, valor)?;
        }

        // Los offsets son relativos al inicio de la zona de datos, y cada uno
        // va alineado.
        let mut offset = 0u64;
        for d in &self.tensores {
            offset = offset.div_ceil(self.alineacion) * self.alineacion;
            escribir_texto(&mut w, &d.nombre)?;
            w.write_all(&(d.dims.len() as u32).to_le_bytes())
                .map_err(err)?;
            for v in &d.dims {
                w.write_all(&v.to_le_bytes()).map_err(err)?;
            }
            w.write_all(&(d.tipo as u32).to_le_bytes()).map_err(err)?;
            w.write_all(&offset.to_le_bytes()).map_err(err)?;
            offset += d.bytes() as u64;
        }

        // Relleno hasta el inicio de los datos.
        let inicio = w.escrito.div_ceil(self.alineacion) * self.alineacion;
        let relleno = inicio - w.escrito;
        rellenar(&mut w, relleno)?;

        for d in &self.tensores {
            let relativo = w.escrito - inicio;
            let quiere = relativo.div_ceil(self.alineacion) * self.alineacion;
            let relleno = quiere - relativo;
            rellenar(&mut w, relleno)?;

            let desde = w.escrito;
            fuente(d, &mut w)?;
            let escrito = w.escrito - desde;
            // Esta comprobacion convierte un bug de generacion en un error con
            // nombre. Sin ella el archivo queda con todos los tensores
            // siguientes corridos y el sintoma aparece mucho despues.
            if escrito != d.bytes() as u64 {
                return Err(format!(
                    "{}: la fuente escribio {escrito} bytes y el tensor son {}",
                    d.nombre,
                    d.bytes()
                ));
            }
        }

        let total = w.escrito;
        w.w.flush().map_err(err)?;
        drop(w);
        std::fs::rename(&temporal, ruta)
            .map_err(|e| format!("no pude renombrar a {}: {e}", ruta.display()))?;
        Ok(total)
    }
}

/// Un `Write` que cuenta lo que pasa por él.
///
/// Se cuenta en vez de preguntar la posición con `stream_position`, porque eso
/// vacía el buffer en cada llamada y acá se llama dos veces por tensor.
struct Contador<W> {
    w: W,
    escrito: u64,
}

impl<W: Write> Write for Contador<W> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let n = self.w.write(b)?;
        self.escrito += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

fn err(e: std::io::Error) -> String {
    format!("error al escribir: {e}")
}

fn rellenar(w: &mut impl Write, n: u64) -> Resultado<()> {
    const CERO: [u8; 64] = [0u8; 64];
    let mut falta = n;
    while falta > 0 {
        let trozo = falta.min(CERO.len() as u64) as usize;
        w.write_all(&CERO[..trozo]).map_err(err)?;
        falta -= trozo as u64;
    }
    Ok(())
}

fn escribir_texto(w: &mut impl Write, s: &str) -> Resultado<()> {
    w.write_all(&(s.len() as u64).to_le_bytes()).map_err(err)?;
    w.write_all(s.as_bytes()).map_err(err)
}

/// El codigo de tipo de cada variante, que son los numeros del formato.
fn codigo(v: &Valor) -> u32 {
    match v {
        Valor::U8(_) => 0,
        Valor::I8(_) => 1,
        Valor::U16(_) => 2,
        Valor::I16(_) => 3,
        Valor::U32(_) => 4,
        Valor::I32(_) => 5,
        Valor::F32(_) => 6,
        Valor::Bool(_) => 7,
        Valor::Texto(_) => 8,
        Valor::Lista(_) => 9,
        Valor::U64(_) => 10,
        Valor::I64(_) => 11,
        Valor::F64(_) => 12,
    }
}

fn escribir_valor(w: &mut impl Write, v: &Valor) -> Resultado<()> {
    w.write_all(&codigo(v).to_le_bytes()).map_err(err)?;
    escribir_cuerpo(w, v)
}

fn escribir_cuerpo(w: &mut impl Write, v: &Valor) -> Resultado<()> {
    match v {
        Valor::U8(x) => w.write_all(&[*x]).map_err(err),
        Valor::I8(x) => w.write_all(&[*x as u8]).map_err(err),
        Valor::U16(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::I16(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::U32(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::I32(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::F32(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::Bool(x) => w.write_all(&[u8::from(*x)]).map_err(err),
        Valor::Texto(x) => escribir_texto(w, x),
        Valor::U64(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::I64(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::F64(x) => w.write_all(&x.to_le_bytes()).map_err(err),
        Valor::Lista(xs) => {
            // Una lista vacia no dice de que es, y el formato pide el tipo
            // interno igual. Se escribe como lista de U32, que es lo que
            // hacen las demas implementaciones.
            let interno = xs.first().map(codigo).unwrap_or(4);
            w.write_all(&interno.to_le_bytes()).map_err(err)?;
            w.write_all(&(xs.len() as u64).to_le_bytes()).map_err(err)?;
            for x in xs {
                if codigo(x) != interno {
                    return Err("una lista de GGUF no puede mezclar tipos".into());
                }
                escribir_cuerpo(w, x)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;

    /// Escribir y volver a leer con NUESTRO lector tiene que devolver lo
    /// mismo: metadatos, formas, tipos y bytes.
    ///
    /// Es la prueba que vale, porque el lector ya está verificado contra
    /// candle sobre los cinco modelos reales. Si el escritor produce algo que
    /// ese lector entiende, produce GGUF de verdad.
    #[test]
    fn lo_escrito_se_vuelve_a_leer_igual() {
        let dir = std::env::temp_dir().join("cosxamix_escritor");
        // Se borra al EMPEZAR y no al terminar: si una asercion falla, el
        // archivo queda para mirarlo, y la proxima corrida no hereda basura.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("prueba.gguf");

        let mut e = Escritor::nuevo();
        e.meta_texto("general.architecture", "daiss")
            .meta_u32("daiss.block_count", 2)
            .meta_f32("daiss.attention.layer_norm_rms_epsilon", 1e-6)
            .meta(
                "tokenizer.ggml.tokens",
                Valor::Lista(vec![
                    Valor::Texto("<pad>".into()),
                    Valor::Texto("hola".into()),
                ]),
            )
            .tensor(Declaracion::nueva("uno.weight", &[64, 2], TipoGgml::F32))
            .tensor(Declaracion::nueva("dos.weight", &[32, 3], TipoGgml::Q8_0))
            .tensor(Declaracion::nueva("tres.bias", &[8], TipoGgml::F32));

        let esperado: Vec<Vec<u8>> = e
            .declaraciones()
            .iter()
            .map(|d| {
                let n = d.bytes();
                (0..n).map(|i| ((i * 31 + 7) % 251) as u8).collect()
            })
            .collect();

        let mut i = 0usize;
        let total = e
            .escribir(&ruta, |_, w| {
                w.write_all(&esperado[i]).map_err(|x| x.to_string())?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(i, 3, "la fuente tiene que llamarse una vez por tensor");
        assert_eq!(total, std::fs::metadata(&ruta).unwrap().len());
        assert!(
            !ruta.with_extension("parcial").exists(),
            "quedo el temporal"
        );

        let g = Gguf::abrir(&ruta).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.tensores.len(), 3);
        assert_eq!(
            g.metadatos
                .get("general.architecture")
                .and_then(|v| v.texto()),
            Some("daiss")
        );
        assert_eq!(g.meta_entero("daiss.block_count"), Some(2));
        // `1e-6` guardado en `f32` y ampliado a `f64` NO es `1e-6` de `f64`:
        // son 9,99999974737875e-7. Comparar contra la constante de `f64`
        // haria fallar a un escritor correcto.
        assert_eq!(
            g.meta_decimal("daiss.attention.layer_norm_rms_epsilon"),
            Some(1e-6f32 as f64)
        );
        match g.metadatos.get("tokenizer.ggml.tokens") {
            Some(Valor::Lista(xs)) => {
                assert_eq!(xs.len(), 2);
                assert_eq!(xs[1].texto(), Some("hola"));
            }
            otro => panic!("la lista volvio como {otro:?}"),
        }

        for (n, (d, bytes)) in e.declaraciones().iter().zip(&esperado).enumerate() {
            let t = g.tensor(&d.nombre).unwrap_or_else(|| panic!("falta {n}"));
            assert_eq!(t.dims, d.dims, "{}: dimensiones", d.nombre);
            assert_eq!(t.tipo, d.tipo, "{}: tipo", d.nombre);
            assert_eq!(&g.datos_de(&ruta, t).unwrap(), bytes, "{}: bytes", d.nombre);
        }
    }

    /// Una fuente que escribe de menos tiene que dar un error con nombre, no
    /// un archivo con todo lo siguiente corrido.
    #[test]
    fn una_fuente_que_miente_se_detecta() {
        let dir = std::env::temp_dir().join("cosxamix_escritor_corto");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut e = Escritor::nuevo();
        e.tensor(Declaracion::nueva("uno.weight", &[32, 1], TipoGgml::F32));
        let error = e
            .escribir(&dir.join("corto.gguf"), |_, w| {
                w.write_all(&[0u8; 10]).map_err(|x| x.to_string())
            })
            .unwrap_err();
        assert!(error.contains("escribio 10"), "{error}");
    }

    /// Una forma que no cuadra con el bloque del formato se rechaza ANTES de
    /// escribir nada, porque el sintoma seria silencioso.
    #[test]
    fn una_forma_que_no_cuadra_con_el_bloque_se_rechaza() {
        let mut e = Escritor::nuevo();
        e.tensor(Declaracion::nueva("mal.weight", &[100, 2], TipoGgml::Q8_0));
        let error = e
            .escribir(&std::env::temp_dir().join("nunca.gguf"), |_, _| Ok(()))
            .unwrap_err();
        assert!(error.contains("bloques de 32"), "{error}");
    }
}
