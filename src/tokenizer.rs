//! Tokenizador propio: BPE a nivel de bytes, con entrenamiento incluido.
//!
//! Es lo que bloquea los tamaños chicos de daiss, y no es un accesorio. Con el
//! vocabulario de Qwen2.5 —151.936 piezas— un modelo de 256 de ancho y 6 capas
//! es **97% tabla de búsqueda**: 77,8 M de sus 80 M de parámetros son
//! embeddings y proyección de salida, y el transformer es el resto. Un
//! vocabulario de 8.192 cambia esa cuenta por completo.
//!
//! ## A nivel de bytes, y por qué importa
//!
//! El vocabulario arranca con los **256 bytes**, así que cualquier texto se
//! puede representar siempre: no existe un token desconocido y no hace falta
//! un `<unk>`. Un tokenizador a nivel de caracteres tiene que decidir qué hace
//! con un emoji que no vio, o con un byte inválido en medio de un mensaje de
//! WhatsApp; este no tiene que decidir nada, porque todo byte ya es un token.
//!
//! Hay un test que lo afirma sobre secuencias de bytes al azar, incluidas las
//! que no son UTF-8 válido.
//!
//! ## Qué es BPE, en una frase
//!
//! Se cuentan los pares de símbolos adyacentes más frecuentes del corpus, se
//! funde el más común en un símbolo nuevo, y se repite hasta llegar al tamaño
//! de vocabulario pedido. Las fusiones quedan **ordenadas**, y codificar es
//! aplicarlas en ese mismo orden.

use std::collections::HashMap;
use std::path::Path;

type Resultado<T> = Result<T, String>;

/// Cuántos bytes hay, que es el tamaño mínimo del vocabulario.
pub const BASE: usize = 256;

/// Un BPE entrenado.
#[derive(Debug)]
pub struct Bpe {
    /// Los bytes de cada token. Los primeros 256 son un byte cada uno.
    vocab: Vec<Vec<u8>>,
    /// `(izquierda, derecha) -> (rango, token nuevo)`. El rango es el orden en
    /// que se aprendió la fusión, y codificar lo respeta.
    fusiones: HashMap<(u32, u32), (u32, u32)>,
}

impl Bpe {
    /// Un BPE sin fusiones: cada byte es un token.
    ///
    /// Sirve de referencia para medir: cualquier entrenamiento tiene que
    /// comprimir más que esto, y si no, algo está mal.
    pub fn solo_bytes() -> Self {
        Self {
            vocab: (0..BASE).map(|b| vec![b as u8]).collect(),
            fusiones: HashMap::new(),
        }
    }

    pub fn tamano(&self) -> usize {
        self.vocab.len()
    }

    pub fn fusiones(&self) -> usize {
        self.fusiones.len()
    }

    /// Los bytes de un token, si existe.
    pub fn pieza(&self, id: u32) -> Option<&[u8]> {
        self.vocab.get(id as usize).map(|v| v.as_slice())
    }

    /// Entrena sobre `corpus` hasta llegar a `objetivo` tokens.
    ///
    /// `objetivo` cuenta los 256 bytes, así que un objetivo de 8192 aprende
    /// 7936 fusiones. Un objetivo menor o igual a 256 devuelve
    /// [`Bpe::solo_bytes`] sin trabajar.
    pub fn entrenar(corpus: &str, objetivo: usize) -> Self {
        let mut bpe = Self::solo_bytes();
        if objetivo <= BASE {
            return bpe;
        }

        // Las palabras se cuentan una vez: el corpus tiene muchas repetidas y
        // contar pares sobre cada aparición sería hacer el mismo trabajo miles
        // de veces.
        let mut cuenta: HashMap<Vec<u8>, u64> = HashMap::new();
        for trozo in pretokenizar(corpus) {
            *cuenta.entry(trozo.as_bytes().to_vec()).or_insert(0) += 1;
        }

        // Cada palabra, como lista de tokens. Arranca en bytes.
        let mut palabras: Vec<(Vec<u32>, u64)> = cuenta
            .into_iter()
            .map(|(bytes, n)| (bytes.iter().map(|b| *b as u32).collect(), n))
            .collect();

        for rango in 0..(objetivo - BASE) as u32 {
            // Contar los pares adyacentes de todas las palabras, pesados por
            // cuántas veces aparece cada palabra.
            let mut pares: HashMap<(u32, u32), u64> = HashMap::new();
            for (simbolos, n) in &palabras {
                for par in simbolos.windows(2) {
                    *pares.entry((par[0], par[1])).or_insert(0) += n;
                }
            }

            // El más frecuente. El desempate es por el par más chico, para que
            // entrenar dos veces sobre el mismo corpus dé el mismo
            // tokenizador: `HashMap` no tiene orden y sin desempate el
            // resultado dependería de la semilla de la tabla.
            let Some((&mejor, &veces)) = pares
                .iter()
                .max_by_key(|(par, n)| (**n, std::cmp::Reverse(**par)))
            else {
                break;
            };
            // Un par que aparece una sola vez no vale un token: el
            // vocabulario se llenaría de rarezas del corpus.
            if veces < 2 {
                break;
            }

            let nuevo = bpe.vocab.len() as u32;
            let mut pieza = bpe.vocab[mejor.0 as usize].clone();
            pieza.extend_from_slice(&bpe.vocab[mejor.1 as usize]);
            bpe.vocab.push(pieza);
            bpe.fusiones.insert(mejor, (rango, nuevo));

            for (simbolos, _) in palabras.iter_mut() {
                fundir(simbolos, mejor, nuevo);
            }
        }

        bpe
    }

    /// Codifica un texto.
    pub fn codificar(&self, texto: &str) -> Vec<u32> {
        let mut salida = Vec::new();
        for trozo in pretokenizar(texto) {
            self.codificar_trozo(trozo.as_bytes(), &mut salida);
        }
        salida
    }

    /// Codifica bytes crudos, que pueden no ser UTF-8 válido.
    ///
    /// No pre-tokeniza, porque partir en palabras pide texto. Se usa para
    /// comprobar la propiedad de que nada es desconocido.
    pub fn codificar_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut salida = Vec::new();
        self.codificar_trozo(bytes, &mut salida);
        salida
    }

    /// Aplica las fusiones a una pieza, **siempre la de menor rango primero**.
    ///
    /// Ese orden no es un detalle: es lo que hace que codificar sea
    /// determinista y que coincida con lo que el entrenamiento aprendió. Si se
    /// fundiera de izquierda a derecha, el mismo texto daría tokens distintos
    /// según dónde empezara.
    fn codificar_trozo(&self, bytes: &[u8], salida: &mut Vec<u32>) {
        if bytes.is_empty() {
            return;
        }
        let mut simbolos: Vec<u32> = bytes.iter().map(|b| *b as u32).collect();

        loop {
            let mut elegido: Option<((u32, u32), u32, u32)> = None;
            for par in simbolos.windows(2) {
                if let Some(&(rango, nuevo)) = self.fusiones.get(&(par[0], par[1])) {
                    if elegido.is_none_or(|(_, r, _)| rango < r) {
                        elegido = Some(((par[0], par[1]), rango, nuevo));
                    }
                }
            }
            match elegido {
                Some((par, _, nuevo)) => fundir(&mut simbolos, par, nuevo),
                None => break,
            }
        }
        salida.extend_from_slice(&simbolos);
    }

    /// Los bytes exactos de una secuencia de tokens.
    pub fn decodificar_bytes(&self, ids: &[u32]) -> Resultado<Vec<u8>> {
        let mut out = Vec::new();
        for id in ids {
            let pieza = self.pieza(*id).ok_or_else(|| {
                format!(
                    "el token {id} no existe en un vocabulario de {}",
                    self.tamano()
                )
            })?;
            out.extend_from_slice(pieza);
        }
        Ok(out)
    }

    /// El texto de una secuencia de tokens.
    ///
    /// Si los tokens vinieron de [`Bpe::codificar`] el resultado es UTF-8
    /// válido y exacto. Una secuencia armada a mano puede cortar un carácter
    /// por la mitad, y ahí los bytes sueltos salen como el carácter de
    /// reemplazo en vez de hacer fallar la llamada: en generación, el modelo
    /// produce tokens de a uno y cortar un carácter a mitad es normal.
    pub fn decodificar(&self, ids: &[u32]) -> Resultado<String> {
        Ok(String::from_utf8_lossy(&self.decodificar_bytes(ids)?).into_owned())
    }

    /// Las fusiones en orden de rango, como pares `(izquierda, derecha)`.
    ///
    /// Es todo lo que hace falta para reconstruir el tokenizador: los 256
    /// bytes son implícitos y la pieza de cada token sale de su par. Lo usan
    /// el guardado en archivo y el incrustado en un GGUF de daiss.
    pub fn pares(&self) -> Vec<(u32, u32)> {
        let mut v: Vec<_> = self
            .fusiones
            .iter()
            .map(|((a, d), (r, _))| (*r, *a, *d))
            .collect();
        v.sort_unstable();
        v.into_iter().map(|(_, a, d)| (a, d)).collect()
    }

    /// Reconstruye un tokenizador desde la lista de pares de [`Bpe::pares`].
    pub fn de_pares(pares: &[(u32, u32)]) -> Resultado<Self> {
        let mut bpe = Self::solo_bytes();
        for (rango, (a, d)) in pares.iter().enumerate() {
            // Los dos lados tienen que existir ya: una fusión solo puede
            // apoyarse en tokens anteriores. Comprobarlo convierte una lista
            // corrupta en un error con nombre y no en un pánico por índice.
            if *a as usize >= bpe.vocab.len() || *d as usize >= bpe.vocab.len() {
                return Err(format!(
                    "la fusión {rango} usa los tokens {a} y {d}, y solo hay {}",
                    bpe.vocab.len()
                ));
            }
            let nuevo = bpe.vocab.len() as u32;
            let mut pieza = bpe.vocab[*a as usize].clone();
            pieza.extend_from_slice(&bpe.vocab[*d as usize]);
            bpe.vocab.push(pieza);
            bpe.fusiones.insert((*a, *d), (rango as u32, nuevo));
        }
        Ok(bpe)
    }

    /// Guarda el tokenizador.
    ///
    /// El formato es la lista de fusiones y nada más: los 256 bytes son
    /// implícitos y la pieza de cada token se reconstruye desde su par, con el
    /// rango igual a su posición. Son 8 bytes por fusión — un vocabulario de
    /// 8192 pesa 63 KB.
    pub fn guardar(&self, ruta: &Path) -> Resultado<()> {
        let mut b = Vec::with_capacity(12 + self.fusiones.len() * 8);
        b.extend_from_slice(b"COSXBPE1");
        b.extend_from_slice(&(self.fusiones.len() as u32).to_le_bytes());

        for (a, d) in self.pares() {
            b.extend_from_slice(&a.to_le_bytes());
            b.extend_from_slice(&d.to_le_bytes());
        }
        std::fs::write(ruta, b).map_err(|e| format!("no pude escribir {}: {e}", ruta.display()))
    }

    pub fn cargar(ruta: &Path) -> Resultado<Self> {
        let b = std::fs::read(ruta).map_err(|e| format!("no pude leer {}: {e}", ruta.display()))?;
        if b.len() < 12 || &b[..8] != b"COSXBPE1" {
            return Err("no es un tokenizador de Cosxamix".into());
        }
        let n = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
        if b.len() < 12 + n * 8 {
            return Err(format!("dice tener {n} fusiones y el archivo no alcanza"));
        }

        let pares: Vec<(u32, u32)> = (0..n)
            .map(|rango| {
                let o = 12 + rango * 8;
                (
                    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()),
                    u32::from_le_bytes(b[o + 4..o + 8].try_into().unwrap()),
                )
            })
            .collect();
        Self::de_pares(&pares)
    }
}

/// Reemplaza cada aparición de `par` por `nuevo`, en su sitio.
fn fundir(simbolos: &mut Vec<u32>, par: (u32, u32), nuevo: u32) {
    let mut i = 0;
    let mut escribir = 0;
    while i < simbolos.len() {
        if i + 1 < simbolos.len() && simbolos[i] == par.0 && simbolos[i + 1] == par.1 {
            simbolos[escribir] = nuevo;
            i += 2;
        } else {
            simbolos[escribir] = simbolos[i];
            i += 1;
        }
        escribir += 1;
    }
    simbolos.truncate(escribir);
}

/// Parte el texto en piezas antes de aplicar BPE.
///
/// Hace falta porque sin esto el BPE aprendería fusiones que cruzan palabras
/// —`de_la`, `que_el`— y gastaría el vocabulario en combinaciones que solo
/// sirven para ese corpus.
///
/// Las reglas, dichas enteras:
///
/// - un **espacio suelto pegado al principio** de una palabra o de un número
///   va con ella, que es lo que hace que `" casa"` y `"casa"` sean piezas
///   distintas y el modelo no tenga que aprender dónde van los espacios;
/// - una **corrida de letras** es una pieza;
/// - una **corrida de dígitos** es una pieza;
/// - **cualquier otra cosa** —puntuación, saltos de línea, emoji— va de a un
///   carácter.
///
/// Es más simple que la expresión regular de GPT-2, que además separa
/// contracciones del inglés (`'s`, `'re`) y agrupa los espacios en blanco. Lo
/// que se pierde con esta: en inglés, `don't` da tres piezas en vez de dos.
/// Para español, que es el idioma del corpus y del bot, no hay contracciones
/// con apóstrofo y la diferencia no aparece.
pub fn pretokenizar(texto: &str) -> Vec<&str> {
    let mut piezas = Vec::new();
    let bytes = texto.as_bytes();
    let mut iter = texto.char_indices().peekable();

    while let Some((i, c)) = iter.next() {
        let inicio = i;
        let mut clase = clase_de(c);

        // Un espacio suelto se pega a lo que viene, si lo que viene es una
        // palabra o un número.
        if c == ' ' {
            match iter.peek().map(|(_, c)| clase_de(*c)) {
                Some(Clase::Letra) | Some(Clase::Digito) => {
                    clase = clase_de(iter.peek().unwrap().1);
                    iter.next();
                }
                _ => {
                    piezas.push(&texto[inicio..inicio + c.len_utf8()]);
                    continue;
                }
            }
        } else if clase == Clase::Otro {
            piezas.push(&texto[inicio..inicio + c.len_utf8()]);
            continue;
        }

        // Consumir la corrida de la misma clase.
        let mut fin = inicio + espacio_mas(texto, inicio);
        while let Some(&(j, c2)) = iter.peek() {
            if clase_de(c2) != clase {
                break;
            }
            fin = j + c2.len_utf8();
            iter.next();
        }
        // `fin` puede quedar antes del final del primer carácter cuando la
        // corrida es de uno solo; se corrige con la longitud real.
        let fin = fin.max(inicio + espacio_mas(texto, inicio));
        piezas.push(std::str::from_utf8(&bytes[inicio..fin]).unwrap_or(""));
    }
    piezas
}

/// Longitud del primer carácter de `texto` desde `i`, más el espacio que lo
/// precede si lo hay.
fn espacio_mas(texto: &str, i: usize) -> usize {
    let mut iter = texto[i..].chars();
    let primero = iter.next().map(|c| c.len_utf8()).unwrap_or(0);
    if texto.as_bytes().get(i) == Some(&b' ') {
        primero + iter.next().map(|c| c.len_utf8()).unwrap_or(0)
    } else {
        primero
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Clase {
    Letra,
    Digito,
    Otro,
}

fn clase_de(c: char) -> Clase {
    if c.is_alphabetic() {
        Clase::Letra
    } else if c.is_ascii_digit() {
        Clase::Digito
    } else {
        Clase::Otro
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXTO: &str = "la multiplicacion cuantizada de Cosxamix mide antes de \
        decidir. La multiplicacion fusionada gana con una fila y pierde con 32, \
        y la de f32 gana el lote. Medir antes de tocar, 3 veces si hace falta.";

    /// Codificar y decodificar tiene que devolver **exactamente** el mismo
    /// texto, con fusiones y sin ellas.
    #[test]
    fn el_circulo_cierra_con_cualquier_texto() {
        let casos = [
            "",
            "a",
            " ",
            "hola",
            " hola mundo",
            "año tras año, ñandú",
            "emoji: 🙂🚀 y mas",
            "1234567890",
            "\n\n  tabulado\ty saltos\n",
            "mezcla 42 cosas, ¿sí? ¡claro!",
            TEXTO,
        ];
        for bpe in [Bpe::solo_bytes(), Bpe::entrenar(TEXTO, 400)] {
            for caso in casos {
                let ids = bpe.codificar(caso);
                let vuelta = bpe.decodificar(&ids).unwrap();
                assert_eq!(vuelta, caso, "con {} fusiones", bpe.fusiones());
            }
        }
    }

    /// **Nada es desconocido**: cualquier secuencia de bytes, incluso la que no
    /// es UTF-8 válido, se codifica y se recupera byte por byte.
    ///
    /// Es la propiedad por la que el tokenizador es a nivel de bytes y no de
    /// caracteres. Un mensaje de WhatsApp con un byte roto no puede tirar el
    /// bot.
    #[test]
    fn ninguna_secuencia_de_bytes_es_desconocida() {
        let bpe = Bpe::entrenar(TEXTO, 400);
        let mut estado = 0x1234_5678u64;
        for largo in [1usize, 2, 7, 33, 100] {
            let bytes: Vec<u8> = (0..largo)
                .map(|_| {
                    estado = estado.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (estado >> 33) as u8
                })
                .collect();
            let ids = bpe.codificar_bytes(&bytes);
            assert_eq!(bpe.decodificar_bytes(&ids).unwrap(), bytes);
        }
    }

    /// Entrenar tiene que comprimir, y más vocabulario no puede comprimir
    /// menos.
    ///
    /// No se exige que SIEMPRE comprima más, porque el entrenamiento **se
    /// detiene solo**: cuando no queda ningún par que aparezca dos veces, un
    /// token más no tiene nada que aprender. En un corpus de juguete eso pasa
    /// enseguida, y es lo correcto — llenar el vocabulario con rarezas que
    /// aparecen una vez sería gastarlo.
    #[test]
    fn mas_vocabulario_no_comprime_menos() {
        let corpus = TEXTO.repeat(5);
        let bytes = Bpe::solo_bytes().codificar(&corpus).len();

        let mut anterior = bytes;
        let mut mejoro = false;
        for objetivo in [300usize, 400, 600, 1000] {
            let bpe = Bpe::entrenar(&corpus, objetivo);
            let n = bpe.codificar(&corpus).len();
            assert!(
                n <= anterior,
                "con objetivo {objetivo} dio {n} tokens y con menos vocabulario {anterior}"
            );
            mejoro |= n < anterior;
            anterior = n;
        }
        assert!(mejoro, "mas vocabulario nunca comprimio mas");
        assert!(
            anterior < bytes / 3,
            "de {bytes} bytes a {anterior} tokens: comprime menos de 3x"
        );
    }

    /// El entrenamiento tiene que parar por su cuenta cuando ya no hay nada
    /// que fundir, en vez de inventar tokens de un solo uso.
    #[test]
    fn el_entrenamiento_se_detiene_solo() {
        let bpe = Bpe::entrenar(TEXTO, 100_000);
        assert!(
            bpe.fusiones() < 1_000,
            "aprendio {} fusiones de un texto de {} bytes",
            bpe.fusiones(),
            TEXTO.len()
        );
        // Y lo que aprendio sigue siendo correcto.
        assert_eq!(bpe.decodificar(&bpe.codificar(TEXTO)).unwrap(), TEXTO);
    }

    /// Entrenar dos veces sobre el mismo corpus tiene que dar el mismo
    /// tokenizador.
    ///
    /// Sin el desempate del entrenamiento esto fallaría: `HashMap` no tiene
    /// orden, así que el par elegido entre dos empatados dependería de la
    /// semilla de la tabla y el tokenizador cambiaría en cada corrida.
    #[test]
    fn entrenar_es_determinista() {
        let a = Bpe::entrenar(TEXTO, 400);
        let b = Bpe::entrenar(TEXTO, 400);
        assert_eq!(a.tamano(), b.tamano());
        assert_eq!(a.codificar(TEXTO), b.codificar(TEXTO));
    }

    /// Guardar y cargar tiene que dar un tokenizador que codifica igual, y un
    /// archivo corrupto tiene que avisar en vez de entrar en pánico.
    #[test]
    fn guardar_y_cargar_conserva_todo() {
        let dir = std::env::temp_dir().join("cosxamix_bpe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let ruta = dir.join("tok.bpe");

        let bpe = Bpe::entrenar(TEXTO, 500);
        bpe.guardar(&ruta).unwrap();
        let vuelta = Bpe::cargar(&ruta).unwrap();
        assert_eq!(vuelta.tamano(), bpe.tamano());
        assert_eq!(vuelta.codificar(TEXTO), bpe.codificar(TEXTO));
        for id in 0..bpe.tamano() as u32 {
            assert_eq!(vuelta.pieza(id), bpe.pieza(id), "token {id}");
        }

        // 8 bytes por fusión más la cabecera de 12.
        let pesa = std::fs::metadata(&ruta).unwrap().len();
        assert_eq!(pesa, 12 + bpe.fusiones() as u64 * 8);

        // Un archivo que dice tener más fusiones de las que trae.
        let mal = dir.join("mal.bpe");
        let mut b = std::fs::read(&ruta).unwrap();
        b[8..12].copy_from_slice(&999_999u32.to_le_bytes());
        std::fs::write(&mal, &b).unwrap();
        assert!(Bpe::cargar(&mal).unwrap_err().contains("no alcanza"));

        // Y uno que no es nuestro.
        let otro = dir.join("otro.bpe");
        std::fs::write(&otro, b"no soy un tokenizador").unwrap();
        assert!(Bpe::cargar(&otro).is_err());
    }

    /// La pre-tokenización tiene que respetar sus propias reglas, y sobre todo
    /// no perder ni duplicar un solo carácter.
    #[test]
    fn la_pretokenizacion_no_pierde_nada() {
        for caso in [
            "hola mundo",
            " con espacio al principio",
            "numeros 42 y 1000",
            "¿signos? ¡sí!",
            "año ñandú 🙂",
            "a  b",
            "",
        ] {
            let piezas = pretokenizar(caso);
            assert_eq!(
                piezas.concat(),
                caso,
                "las piezas {piezas:?} no reconstruyen el texto"
            );
        }

        // El espacio va pegado a la palabra que sigue.
        assert_eq!(pretokenizar("hola mundo"), vec!["hola", " mundo"]);
        // Dos espacios seguidos: el primero no tiene palabra detrás.
        assert_eq!(pretokenizar("a  b"), vec!["a", " ", " b"]);
        // Letras y digitos no se mezclan.
        assert_eq!(pretokenizar("abc123"), vec!["abc", "123"]);
    }
}
