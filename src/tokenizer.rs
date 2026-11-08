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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

use rayon::prelude::*;

use crate::kernels::classify::{primer_cero, Mascaras};

type Resultado<T> = Result<T, String>;

/// Cuántos bytes hay, que es el tamaño mínimo del vocabulario.
pub const BASE: usize = 256;

// ─── Las tablas ─────────────────────────────────────────────────────────────

/// Un hash de una multiplicación para las claves del entrenamiento. El de
/// `HashMap` por defecto es SipHash, que defiende de quien elige las claves a
/// propósito: acá las claves son del propio corpus, y entrenar pasa más tiempo
/// en esa defensa que contando.
#[derive(Default)]
struct HashRapido(u64);

const MEZCLA: u64 = 0x9E37_79B9_7F4A_7C15;

impl Hasher for HashRapido {
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for trozo in bytes.chunks(8) {
            let mut v = [0u8; 8];
            v[..trozo.len()].copy_from_slice(trozo);
            h = (h.rotate_left(5) ^ u64::from_le_bytes(v)).wrapping_mul(MEZCLA);
        }
        self.0 = h;
    }
    fn write_u32(&mut self, v: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(v)).wrapping_mul(MEZCLA);
    }
    fn finish(&self) -> u64 {
        // Los bits altos de una multiplicación son los buenos, y `HashMap` mira
        // los bajos.
        self.0 ^ (self.0 >> 32)
    }
}

type MapaRapido<K, V> = HashMap<K, V, BuildHasherDefault<HashRapido>>;

/// Las fusiones, `(izquierda, derecha) -> (rango, token nuevo)`, en una sola
/// tabla plana con direccionamiento abierto.
///
/// Es lo que más se consulta al codificar: una vez por cada par de símbolos,
/// y de nuevo tras cada fusión. Doce bytes por entrada, sin punteros que
/// seguir ni SipHash que calcular.
#[derive(Debug, Clone)]
struct TablaFusiones {
    claves: Vec<u64>,
    valores: Vec<(u32, u32)>,
    corrimiento: u32,
}

/// Una clave que ningún par puede dar: los ids no llegan a `u32::MAX`.
const VACIA: u64 = u64::MAX;

impl TablaFusiones {
    fn con_lugar_para(n: usize) -> Self {
        // Menos de la mitad llena: las sondas quedan en una o dos.
        let capacidad = (n * 2 + 2).next_power_of_two();
        Self {
            claves: vec![VACIA; capacidad],
            valores: vec![(0, 0); capacidad],
            corrimiento: 64 - capacidad.trailing_zeros(),
        }
    }

    #[inline]
    fn clave(a: u32, b: u32) -> u64 {
        (u64::from(a) << 32) | u64::from(b)
    }

    #[inline]
    fn lugar(&self, clave: u64) -> usize {
        (clave.wrapping_mul(MEZCLA) >> self.corrimiento) as usize
    }

    fn insertar(&mut self, a: u32, b: u32, rango: u32, nuevo: u32) {
        let clave = Self::clave(a, b);
        let mascara = self.claves.len() - 1;
        let mut i = self.lugar(clave);
        while self.claves[i] != VACIA && self.claves[i] != clave {
            i = (i + 1) & mascara;
        }
        self.claves[i] = clave;
        self.valores[i] = (rango, nuevo);
    }

    #[inline]
    fn buscar(&self, a: u32, b: u32) -> Option<(u32, u32)> {
        let clave = Self::clave(a, b);
        let mascara = self.claves.len() - 1;
        let mut i = self.lugar(clave);
        loop {
            // SAFETY: `i` sale de `lugar` o de enmascarar con `len - 1`, y la
            // capacidad es potencia de dos: siempre está dentro. Las dos
            // tablas miden lo mismo.
            let k = unsafe { *self.claves.get_unchecked(i) };
            if k == clave {
                return Some(unsafe { *self.valores.get_unchecked(i) });
            }
            if k == VACIA {
                return None;
            }
            i = (i + 1) & mascara;
        }
    }
}

/// Ver [`Bpe::partes_planas`].
pub type PartesPlanas<'a> = (&'a [u32; BASE], &'a [(u32, u32, u32)], &'a [u32], &'a [u8]);

/// Un BPE entrenado.
#[derive(Debug, Clone)]
pub struct Bpe {
    /// El token de cada byte. En el nuestro, el propio byte; en un vocabulario
    /// ajeno, el que diga el archivo.
    base: [u32; BASE],
    /// `(izquierda, derecha, token nuevo)`, por rango.
    pares: Vec<(u32, u32, u32)>,
    tabla: TablaFusiones,
    /// Los bytes de todos los tokens, uno detrás del otro: `inicios[id]` a
    /// `inicios[id + 1]`. Con 151.936 tokens, un `Vec<Vec<u8>>` son otros tantos
    /// bloques de memoria repartidos; decodificar copiaría desde todos lados.
    inicios: Vec<u32>,
    bytes: Vec<u8>,
}

/// Desde este largo, codificar recuerda las palabras ya vistas. Por debajo no
/// se repiten lo suficiente como para pagar la tabla.
const TEXTO_LARGO: usize = 8 << 10;

/// Desde este largo, codificar se reparte en los núcleos, de a trozos de esto.
const TROZO_PARALELO: usize = 1 << 20;

/// Una palabra más larga que esto se codifica con cola de prioridad en vez de
/// recorrer el vector: con un texto sin espacios —un base64, un enlace largo—
/// recorrerlo en cada fusión sería cuadrático.
const PALABRA_CORTA: usize = 48;

impl Bpe {
    /// Un BPE sin fusiones: cada byte es un token.
    ///
    /// Sirve de referencia para medir: cualquier entrenamiento tiene que
    /// comprimir más que esto, y si no, algo está mal.
    pub fn solo_bytes() -> Self {
        Self::de_pares(&[]).expect("sin fusiones no hay nada que pueda fallar")
    }

    /// Un BPE con vocabulario ajeno: el token de cada byte, las fusiones por
    /// rango y los bytes de cada token, en cualquier numeración.
    ///
    /// Es lo que hace falta para leer un `tokenizer.json`, donde los primeros
    /// 256 tokens no son los bytes en orden (ver `tokenizer_json.rs`).
    pub fn de_partes(
        base: [u32; BASE],
        pares: Vec<(u32, u32, u32)>,
        piezas: &[Vec<u8>],
    ) -> Resultado<Self> {
        let mut tabla = TablaFusiones::con_lugar_para(pares.len());
        for (rango, (a, d, nuevo)) in pares.iter().enumerate() {
            let n = piezas.len() as u32;
            if *a >= n || *d >= n || *nuevo >= n {
                return Err(format!(
                    "la fusión {rango} usa los tokens {a}, {d} y {nuevo}, y solo hay {n}"
                ));
            }
            tabla.insertar(*a, *d, rango as u32, *nuevo);
        }
        let total: usize = piezas.iter().map(Vec::len).sum();
        let mut bytes = Vec::with_capacity(total);
        let mut inicios = Vec::with_capacity(piezas.len() + 1);
        for p in piezas {
            inicios.push(bytes.len() as u32);
            bytes.extend_from_slice(p);
        }
        inicios.push(bytes.len() as u32);
        Ok(Self {
            base,
            pares,
            tabla,
            inicios,
            bytes,
        })
    }

    /// Las tablas tal cual, para guardarlas compiladas (`tokenizer_json.rs`):
    /// el token de cada byte, las fusiones, y dónde empieza cada pieza.
    pub fn partes_planas(&self) -> PartesPlanas<'_> {
        (&self.base, &self.pares, &self.inicios, &self.bytes)
    }

    /// Lo inverso de [`Bpe::partes_planas`]. Comprueba que todo apunte adentro:
    /// los datos vienen de un archivo.
    pub fn de_partes_planas(
        base: [u32; BASE],
        pares: Vec<(u32, u32, u32)>,
        inicios: Vec<u32>,
        bytes: Vec<u8>,
    ) -> Resultado<Self> {
        let n = inicios.len().saturating_sub(1) as u32;
        if n == 0 || inicios[0] != 0 || inicios[n as usize] as usize != bytes.len() {
            return Err("las piezas no cierran".into());
        }
        if inicios.windows(2).any(|w| w[0] > w[1]) {
            return Err("las piezas no van en orden".into());
        }
        if base.iter().any(|b| *b >= n) {
            return Err("un byte apunta fuera del vocabulario".into());
        }
        let mut tabla = TablaFusiones::con_lugar_para(pares.len());
        for (rango, (a, d, nuevo)) in pares.iter().enumerate() {
            if *a >= n || *d >= n || *nuevo >= n {
                return Err(format!("la fusión {rango} apunta fuera del vocabulario"));
            }
            tabla.insertar(*a, *d, rango as u32, *nuevo);
        }
        Ok(Self {
            base,
            pares,
            tabla,
            inicios,
            bytes,
        })
    }

    pub fn tamano(&self) -> usize {
        self.inicios.len() - 1
    }

    pub fn fusiones(&self) -> usize {
        self.pares.len()
    }

    /// Los bytes de un token, si existe.
    #[inline]
    pub fn pieza(&self, id: u32) -> Option<&[u8]> {
        let i = id as usize;
        let fin = *self.inicios.get(i + 1)? as usize;
        Some(&self.bytes[self.inicios[i] as usize..fin])
    }

    /// Entrena sobre `corpus` hasta llegar a `objetivo` tokens.
    ///
    /// `objetivo` cuenta los 256 bytes, así que un objetivo de 8192 aprende
    /// 7936 fusiones. Un objetivo menor o igual a 256 devuelve
    /// [`Bpe::solo_bytes`] sin trabajar.
    ///
    /// Es incremental: fundir un par solo toca las palabras que lo tienen, y
    /// el conteo de todo lo demás no cambia. Hasta la 0.7.0 cada fusión volvía
    /// a contar todos los pares de todas las palabras, y eso era lo que hacía
    /// imposible entrenar con el corpus de gigabytes que un vocabulario grande
    /// pide. Da **las mismas fusiones, en el mismo orden**: hay un test que lo
    /// compara contra aquel algoritmo.
    pub fn entrenar(corpus: &str, objetivo: usize) -> Self {
        if objetivo <= BASE {
            return Self::solo_bytes();
        }

        let mut palabras = contar_palabras(corpus);

        // Cuántas veces aparece cada par, y en qué palabras. La lista de cada
        // par puede quedar con palabras que ya no lo tienen: se comprueba al
        // usarla, que es más barato que mantenerla exacta.
        let mut cuenta: MapaRapido<(u32, u32), u64> = MapaRapido::default();
        let mut donde: MapaRapido<(u32, u32), Vec<u32>> = MapaRapido::default();
        for (i, (simbolos, n)) in palabras.iter().enumerate() {
            for par in simbolos.windows(2) {
                let par = (par[0], par[1]);
                *cuenta.entry(par).or_insert(0) += n;
                let lista = donde.entry(par).or_default();
                if lista.last() != Some(&(i as u32)) {
                    lista.push(i as u32);
                }
            }
        }

        // La cola guarda (cuenta, par más chico primero). Una entrada queda
        // vieja cuando la cuenta del par cambia, y se descarta al sacarla: lo
        // que vale es la cuenta de la tabla. Es el mismo desempate de siempre,
        // por el par más chico, que es lo que hace determinista el resultado.
        let mut cola: BinaryHeap<(u64, Reverse<(u32, u32)>)> =
            cuenta.iter().map(|(p, n)| (*n, Reverse(*p))).collect();

        let mut pares: Vec<(u32, u32, u32)> = Vec::with_capacity(objetivo - BASE);
        let mut tocados: Vec<(u32, u32)> = Vec::new();
        let mut siguiente = BASE as u32;

        while pares.len() < objetivo - BASE {
            let Some((veces, Reverse(mejor))) = cola.pop() else {
                break;
            };
            if cuenta.get(&mejor).copied().unwrap_or(0) != veces {
                continue;
            }
            // Un par que aparece una sola vez no vale un token: el
            // vocabulario se llenaría de rarezas del corpus. Y como la cola
            // saca de mayor a menor, tampoco vale ninguno de los que quedan.
            if veces < 2 {
                break;
            }

            let nuevo = siguiente;
            siguiente += 1;
            pares.push((mejor.0, mejor.1, nuevo));

            tocados.clear();
            let lista = donde.remove(&mejor).unwrap_or_default();
            for i in lista {
                let (simbolos, n) = &mut palabras[i as usize];
                let n = *n;
                if !simbolos
                    .windows(2)
                    .any(|w| w[0] == mejor.0 && w[1] == mejor.1)
                {
                    continue;
                }
                // Lo que la palabra aportaba sale entero y entra lo nuevo: con
                // palabras de diez símbolos es tan barato como calcular la
                // diferencia, y no hay casos especiales con `aaa`, donde los
                // pares se solapan.
                for par in simbolos.windows(2) {
                    let par = (par[0], par[1]);
                    let c = cuenta.get_mut(&par).expect("el par se contó al empezar");
                    *c -= n;
                    tocados.push(par);
                }
                fundir(simbolos, mejor, nuevo);
                for par in simbolos.windows(2) {
                    let par = (par[0], par[1]);
                    *cuenta.entry(par).or_insert(0) += n;
                    let l = donde.entry(par).or_default();
                    if l.last() != Some(&i) {
                        l.push(i);
                    }
                    tocados.push(par);
                }
            }
            tocados.sort_unstable();
            tocados.dedup();
            for par in &tocados {
                match cuenta.get(par).copied() {
                    Some(0) | None => {
                        cuenta.remove(par);
                    }
                    Some(c) => cola.push((c, Reverse(*par))),
                }
            }
        }

        let solo: Vec<(u32, u32)> = pares.iter().map(|(a, d, _)| (*a, *d)).collect();
        Self::de_pares(&solo).expect("las fusiones salen de tokens anteriores")
    }

    /// Codifica un texto.
    ///
    /// Un prompt pasa derecho. Un texto largo recuerda las palabras que ya vio,
    /// y uno de varios megabytes —un corpus— además se reparte en todos los
    /// núcleos, cortando en saltos de línea, que siempre son una pieza suelta:
    /// el resultado es el mismo, token por token.
    pub fn codificar(&self, texto: &str) -> Vec<u32> {
        if texto.len() < TEXTO_LARGO {
            // Una aproximación por lo alto: unos 3 bytes por token.
            let mut salida = Vec::with_capacity(texto.len() / 2 + 8);
            recorrer_piezas(texto, |trozo| {
                self.codificar_trozo(trozo.as_bytes(), &mut salida)
            });
            return salida;
        }
        if texto.len() < TROZO_PARALELO {
            return self.codificar_recordando(texto);
        }
        let cortes = cortes_en_lineas(texto, TROZO_PARALELO);
        let partes: Vec<Vec<u32>> = cortes
            .par_windows(2)
            .map(|c| self.codificar_recordando(&texto[c[0]..c[1]]))
            .collect();
        let mut salida = Vec::with_capacity(partes.iter().map(Vec::len).sum());
        for p in partes {
            salida.extend_from_slice(&p);
        }
        salida
    }

    /// Codifica recordando qué dio cada palabra: en un texto largo, una
    /// palabra ya vista es copiar sus tokens. Lo recordado apunta a la propia
    /// salida, así que no cuesta memoria aparte.
    fn codificar_recordando(&self, texto: &str) -> Vec<u32> {
        let mut salida: Vec<u32> = Vec::with_capacity(texto.len() / 2 + 8);
        let mut vistas: MapaRapido<&[u8], (u32, u32)> = MapaRapido::default();
        recorrer_piezas(texto, |trozo| {
            let b = trozo.as_bytes();
            if b.len() == 1 {
                salida.push(self.base[b[0] as usize]);
                return;
            }
            if let Some((i, n)) = vistas.get(b) {
                let i = *i as usize;
                salida.extend_from_within(i..i + *n as usize);
                return;
            }
            let i = salida.len();
            self.codificar_trozo(b, &mut salida);
            vistas.insert(b, (i as u32, (salida.len() - i) as u32));
        });
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
    pub fn codificar_trozo(&self, bytes: &[u8], salida: &mut Vec<u32>) {
        match bytes.len() {
            0 => {}
            1 => salida.push(self.base[bytes[0] as usize]),
            n if n <= PALABRA_CORTA => self.codificar_corta(bytes, salida),
            _ => self.codificar_larga(bytes, salida),
        }
    }

    /// Una palabra normal: todo en la pila y sin reservar nada. Cada par
    /// recuerda su rango, así que tras una fusión solo se consultan los dos
    /// pares vecinos y no toda la palabra otra vez.
    #[inline]
    fn codificar_corta(&self, bytes: &[u8], salida: &mut Vec<u32>) {
        const NADA: u32 = u32::MAX;
        let mut simbolos = [0u32; PALABRA_CORTA];
        // `rangos[i]` y `nuevos[i]` son los del par (i, i + 1).
        let mut rangos = [NADA; PALABRA_CORTA];
        let mut nuevos = [0u32; PALABRA_CORTA];
        let mut n = bytes.len();
        for (s, b) in simbolos.iter_mut().zip(bytes) {
            *s = self.base[*b as usize];
        }
        for i in 0..n - 1 {
            if let Some((r, t)) = self.tabla.buscar(simbolos[i], simbolos[i + 1]) {
                rangos[i] = r;
                nuevos[i] = t;
            }
        }
        while n > 1 {
            // El de menor rango; ante un empate, el de más a la izquierda.
            let (mut i, mut menor) = (0, NADA);
            for (k, r) in rangos[..n - 1].iter().enumerate() {
                if *r < menor {
                    menor = *r;
                    i = k;
                }
            }
            if menor == NADA {
                break;
            }
            simbolos[i] = nuevos[i];
            simbolos.copy_within(i + 2..n, i + 1);
            // Los pares de más a la derecha se corren uno; si la fusión fue la
            // del último par, no queda ninguno.
            if i + 2 < n - 1 {
                rangos.copy_within(i + 2..n - 1, i + 1);
                nuevos.copy_within(i + 2..n - 1, i + 1);
            }
            n -= 1;
            if i + 1 < n {
                let (r, t) = self
                    .tabla
                    .buscar(simbolos[i], simbolos[i + 1])
                    .unwrap_or((NADA, 0));
                rangos[i] = r;
                nuevos[i] = t;
            }
            if i > 0 {
                let (r, t) = self
                    .tabla
                    .buscar(simbolos[i - 1], simbolos[i])
                    .unwrap_or((NADA, 0));
                rangos[i - 1] = r;
                nuevos[i - 1] = t;
            }
        }
        salida.extend_from_slice(&simbolos[..n]);
    }

    /// Una palabra larga: lista enlazada y cola de prioridad, `O(n log n)`.
    fn codificar_larga(&self, bytes: &[u8], salida: &mut Vec<u32>) {
        const FIN: u32 = u32::MAX;
        let n = bytes.len();
        let mut simbolos: Vec<u32> = bytes.iter().map(|b| self.base[*b as usize]).collect();
        let mut sig: Vec<u32> = (1..=n as u32).collect();
        sig[n - 1] = FIN;
        let mut ant: Vec<u32> = (0..n as u32).map(|i| i.wrapping_sub(1)).collect();
        ant[0] = FIN;
        let mut cola: BinaryHeap<Reverse<(u32, u32, u32, u32)>> = BinaryHeap::new();
        // Cada entrada lleva los dos símbolos que vio: si alguno cambió, quedó
        // vieja. La posición desempata a favor del de más a la izquierda.
        let poner = |cola: &mut BinaryHeap<Reverse<(u32, u32, u32, u32)>>,
                     simbolos: &[u32],
                     i: u32,
                     d: u32| {
            let (a, b) = (simbolos[i as usize], simbolos[d as usize]);
            if let Some((r, _)) = self.tabla.buscar(a, b) {
                cola.push(Reverse((r, i, a, b)));
            }
        };
        for i in 0..n as u32 - 1 {
            poner(&mut cola, &simbolos, i, i + 1);
        }
        while let Some(Reverse((_, i, a, b))) = cola.pop() {
            let d = sig[i as usize];
            if simbolos[i as usize] != a || d == FIN || simbolos[d as usize] != b {
                continue;
            }
            let (_, nuevo) = self.tabla.buscar(a, b).expect("estaba al ponerla");
            simbolos[i as usize] = nuevo;
            // `d` deja de ser parte de la lista; `FIN` como símbolo hace que
            // una entrada vieja suya no pase la comprobación de arriba.
            simbolos[d as usize] = FIN;
            let dd = sig[d as usize];
            sig[i as usize] = dd;
            if dd != FIN {
                ant[dd as usize] = i;
                poner(&mut cola, &simbolos, i, dd);
            }
            let aa = ant[i as usize];
            if aa != FIN {
                poner(&mut cola, &simbolos, aa, i);
            }
        }
        let mut i = 0u32;
        while i != FIN {
            salida.push(simbolos[i as usize]);
            i = sig[i as usize];
        }
    }

    /// Los bytes exactos de una secuencia de tokens.
    pub fn decodificar_bytes(&self, ids: &[u32]) -> Resultado<Vec<u8>> {
        let mut out = Vec::with_capacity(ids.len() * 4);
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
        self.pares.iter().map(|(a, d, _)| (*a, *d)).collect()
    }

    /// Reconstruye un tokenizador desde la lista de pares de [`Bpe::pares`].
    pub fn de_pares(pares: &[(u32, u32)]) -> Resultado<Self> {
        let total = BASE + pares.len();
        let mut inicios = Vec::with_capacity(total + 1);
        let mut bytes: Vec<u8> = (0..=255u8).collect();
        inicios.extend(0..=BASE as u32);
        let mut tabla = TablaFusiones::con_lugar_para(pares.len());
        let mut trios = Vec::with_capacity(pares.len());
        for (rango, (a, d)) in pares.iter().enumerate() {
            // Los dos lados tienen que existir ya: una fusión solo puede
            // apoyarse en tokens anteriores. Comprobarlo convierte una lista
            // corrupta en un error con nombre y no en un pánico por índice.
            let hay = inicios.len() - 1;
            if *a as usize >= hay || *d as usize >= hay {
                return Err(format!(
                    "la fusión {rango} usa los tokens {a} y {d}, y solo hay {hay}"
                ));
            }
            let nuevo = hay as u32;
            for lado in [*a as usize, *d as usize] {
                let (i, f) = (inicios[lado] as usize, inicios[lado + 1] as usize);
                bytes.extend_from_within(i..f);
            }
            inicios.push(bytes.len() as u32);
            tabla.insertar(*a, *d, rango as u32, nuevo);
            trios.push((*a, *d, nuevo));
        }
        let mut base = [0u32; BASE];
        for (i, b) in base.iter_mut().enumerate() {
            *b = i as u32;
        }
        Ok(Self {
            base,
            pares: trios,
            tabla,
            inicios,
            bytes,
        })
    }

    /// Guarda el tokenizador.
    ///
    /// El formato es la lista de fusiones y nada más: los 256 bytes son
    /// implícitos y la pieza de cada token se reconstruye desde su par, con el
    /// rango igual a su posición. Son 8 bytes por fusión — un vocabulario de
    /// 8192 pesa 63 KB.
    pub fn guardar(&self, ruta: &Path) -> Resultado<()> {
        let mut b = Vec::with_capacity(12 + self.fusiones() * 8);
        b.extend_from_slice(b"COSXBPE1");
        b.extend_from_slice(&(self.fusiones() as u32).to_le_bytes());

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

/// Las palabras distintas del corpus con cuántas veces aparece cada una, como
/// listas de bytes, y en orden de bytes: el orden no cambia el resultado, pero
/// así no depende de cómo cayeron las tablas.
///
/// En paralelo, partiendo en saltos de línea: un salto de línea siempre es una
/// pieza suelta, así que cortar ahí no cambia ninguna palabra.
fn contar_palabras(corpus: &str) -> Vec<(Vec<u32>, u64)> {
    let cuenta = cortes_en_lineas(corpus, TROZO_PARALELO)
        .par_windows(2)
        .map(|c| {
            let mut m: MapaRapido<&[u8], u64> = MapaRapido::default();
            recorrer_piezas(&corpus[c[0]..c[1]], |p| {
                *m.entry(p.as_bytes()).or_insert(0) += 1
            });
            m
        })
        .reduce(MapaRapido::default, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        });
    let mut palabras: Vec<(&[u8], u64)> = cuenta.into_iter().collect();
    palabras.par_sort_unstable();
    palabras
        .into_iter()
        .map(|(p, n)| (p.iter().map(|b| u32::from(*b)).collect(), n))
        .collect()
}

/// Dónde cortar un texto en trozos de al menos `trozo` bytes, siempre después
/// de un salto de línea: un salto de línea es una pieza suelta, así que cortar
/// ahí no cambia ninguna palabra.
fn cortes_en_lineas(texto: &str, trozo: usize) -> Vec<usize> {
    let mut cortes = vec![0];
    while *cortes.last().unwrap() + trozo < texto.len() {
        let desde = cortes.last().unwrap() + trozo;
        match texto.as_bytes()[desde..].iter().position(|b| *b == b'\n') {
            Some(p) => cortes.push(desde + p + 1),
            None => break,
        }
    }
    if *cortes.last().unwrap() != texto.len() {
        cortes.push(texto.len());
    }
    cortes
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
    recorrer_piezas(texto, |p| piezas.push(p));
    piezas
}

/// Por debajo de esto no se calculan máscaras: pedir memoria para ellas cuesta
/// más que mirar un puñado de bytes.
const MINIMO_PARA_MASCARAS: usize = 128;

/// [`pretokenizar`] sin armar la lista: codificar no la necesita.
#[inline]
pub fn recorrer_piezas<'a>(texto: &'a str, f: impl FnMut(&'a str)) {
    if texto.len() >= MINIMO_PARA_MASCARAS {
        recorrer_con_mascaras(texto, &Mascaras::de(texto.as_bytes()), f);
    } else {
        recorrer_de_a_uno(texto, f);
    }
}

/// Lo mismo con las máscaras del kernel: el fin de una corrida ASCII sale de un
/// `tzcnt`, y solo los bytes altos —las letras con acento— se miran uno a uno.
fn recorrer_con_mascaras<'a>(texto: &'a str, m: &Mascaras, mut f: impl FnMut(&'a str)) {
    let bytes = texto.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let inicio = i;
        let (mut clase, largo) = clase_en(texto, i);
        i += largo;
        if bytes[inicio] == b' ' && i < n {
            let (c, l) = clase_en(texto, i);
            if c != Clase::Otro {
                clase = c;
                i += l;
            }
        }
        match clase {
            Clase::Otro => {}
            Clase::Digito => i = primer_cero(&m.digitos, i),
            Clase::Letra => loop {
                i = primer_cero(&m.letras, i);
                if i >= n || bytes[i] < 0x80 {
                    break;
                }
                let c = texto[i..].chars().next().expect("i está en un límite");
                if !c.is_alphabetic() {
                    break;
                }
                i += c.len_utf8();
            },
        }
        f(&texto[inicio..i]);
    }
}

/// Lo mismo un carácter a la vez, para los textos chicos.
fn recorrer_de_a_uno<'a>(texto: &'a str, mut f: impl FnMut(&'a str)) {
    let bytes = texto.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        let inicio = i;
        let (mut clase, largo) = clase_en(texto, i);
        i += largo;

        if bytes[inicio] == b' ' {
            // Un espacio suelto se pega a lo que viene, si lo que viene es una
            // palabra o un número.
            if i < n {
                let (c, l) = clase_en(texto, i);
                if c != Clase::Otro {
                    clase = c;
                    i += l;
                }
            }
            if clase == Clase::Otro {
                f(&texto[inicio..i]);
                continue;
            }
        } else if clase == Clase::Otro {
            f(&texto[inicio..i]);
            continue;
        }

        // Consumir la corrida de la misma clase.
        while i < n {
            let (c, l) = clase_en(texto, i);
            if c != clase {
                break;
            }
            i += l;
        }
        f(&texto[inicio..i]);
    }
}

/// La clase del carácter que empieza en `i` y cuánto mide.
#[inline]
fn clase_en(texto: &str, i: usize) -> (Clase, usize) {
    let b = texto.as_bytes()[i];
    if b < 0x80 {
        let clase = if b.is_ascii_alphabetic() {
            Clase::Letra
        } else if b.is_ascii_digit() {
            Clase::Digito
        } else {
            Clase::Otro
        };
        return (clase, 1);
    }
    let c = texto[i..].chars().next().expect("i está en un límite");
    (clase_de(c), c.len_utf8())
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

    // ── Los algoritmos de antes, como referencia ────────────────────────────
    // Hasta la 0.7.0 esto era el código. Es lento y obvio, y por eso sirve para
    // exigirle al de ahora **exactamente** el mismo resultado.

    fn entrenar_referencia(corpus: &str, objetivo: usize) -> Vec<(u32, u32)> {
        let mut cuenta: HashMap<Vec<u8>, u64> = HashMap::new();
        for trozo in pretokenizar_referencia(corpus) {
            *cuenta.entry(trozo.as_bytes().to_vec()).or_insert(0) += 1;
        }
        let mut palabras: Vec<(Vec<u32>, u64)> = cuenta
            .into_iter()
            .map(|(bytes, n)| (bytes.iter().map(|b| *b as u32).collect(), n))
            .collect();
        let mut salida = Vec::new();
        for siguiente in BASE as u32..objetivo.max(BASE) as u32 {
            let mut pares: HashMap<(u32, u32), u64> = HashMap::new();
            for (simbolos, n) in &palabras {
                for par in simbolos.windows(2) {
                    *pares.entry((par[0], par[1])).or_insert(0) += n;
                }
            }
            let Some((&mejor, &veces)) = pares.iter().max_by_key(|(par, n)| (**n, Reverse(**par)))
            else {
                break;
            };
            if veces < 2 {
                break;
            }
            salida.push(mejor);
            for (simbolos, _) in palabras.iter_mut() {
                fundir(simbolos, mejor, siguiente);
            }
        }
        salida
    }

    fn codificar_referencia(bpe: &Bpe, bytes: &[u8]) -> Vec<u32> {
        let mut simbolos: Vec<u32> = bytes.iter().map(|b| *b as u32).collect();
        loop {
            let mut elegido: Option<((u32, u32), u32, u32)> = None;
            for par in simbolos.windows(2) {
                if let Some((rango, nuevo)) = bpe.tabla.buscar(par[0], par[1]) {
                    if elegido.is_none_or(|(_, r, _)| rango < r) {
                        elegido = Some(((par[0], par[1]), rango, nuevo));
                    }
                }
            }
            match elegido {
                Some((par, _, nuevo)) => fundir(&mut simbolos, par, nuevo),
                None => return simbolos,
            }
        }
    }

    fn pretokenizar_referencia(texto: &str) -> Vec<&str> {
        fn espacio_mas(texto: &str, i: usize) -> usize {
            let mut iter = texto[i..].chars();
            let primero = iter.next().map(|c| c.len_utf8()).unwrap_or(0);
            if texto.as_bytes().get(i) == Some(&b' ') {
                primero + iter.next().map(|c| c.len_utf8()).unwrap_or(0)
            } else {
                primero
            }
        }
        let mut piezas = Vec::new();
        let bytes = texto.as_bytes();
        let mut iter = texto.char_indices().peekable();
        while let Some((i, c)) = iter.next() {
            let inicio = i;
            let mut clase = clase_de(c);
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
            let mut fin = inicio + espacio_mas(texto, inicio);
            while let Some(&(j, c2)) = iter.peek() {
                if clase_de(c2) != clase {
                    break;
                }
                fin = j + c2.len_utf8();
                iter.next();
            }
            let fin = fin.max(inicio + espacio_mas(texto, inicio));
            piezas.push(std::str::from_utf8(&bytes[inicio..fin]).unwrap_or(""));
        }
        piezas
    }

    /// Un texto largo y variado, reproducible: palabras con acentos, números,
    /// emoji, espacios raros y palabras repetidas para que haya qué fundir.
    fn corpus_de_prueba(largo: usize) -> String {
        const PIEZAS: &[&str] = &[
            "la",
            "multiplicación",
            "cuantizada",
            "de",
            "Cosxamix",
            "ñandú",
            "año",
            "que",
            "el",
            "gana",
            "pierde",
            "matriz",
            "aaaa",
            "abab",
            "🙂",
            "🚀",
            "42",
            "2026",
            "ñ",
            "¿sí?",
            "¡claro!",
            "x",
            "\n",
            "\n\n",
            "\t",
            "  ",
            ".",
            ",",
            "日本語",
            "Ω",
            "٣",
        ];
        let mut azar = 0x0123_4567_89AB_CDEF_u64;
        let mut t = String::new();
        while t.len() < largo {
            azar ^= azar << 13;
            azar ^= azar >> 7;
            azar ^= azar << 17;
            t.push_str(PIEZAS[(azar % PIEZAS.len() as u64) as usize]);
            if azar & 0x300 != 0 {
                t.push(' ');
            }
        }
        t
    }

    #[test]
    fn entrenar_da_las_mismas_fusiones_que_el_algoritmo_de_antes() {
        for (corpus, objetivo) in [
            (TEXTO.to_string(), 400),
            (corpus_de_prueba(40_000), 900),
            ("aaaa aaaa aaa aa a abab abab baba".to_string(), 300),
            ("sin nada que repetir".to_string(), 1000),
            (String::new(), 500),
        ] {
            let nuevo = Bpe::entrenar(&corpus, objetivo).pares();
            let viejo = entrenar_referencia(&corpus, objetivo);
            assert_eq!(nuevo, viejo, "corpus de {} bytes", corpus.len());
        }
    }

    #[test]
    fn codificar_da_lo_mismo_que_el_algoritmo_de_antes() {
        let corpus = corpus_de_prueba(40_000);
        let bpe = Bpe::entrenar(&corpus, 900);
        // Las piezas de verdad, y pegadas en una sola para pasar por el camino
        // de las palabras largas.
        for p in pretokenizar(&corpus) {
            assert_eq!(
                bpe.codificar_bytes(p.as_bytes()),
                codificar_referencia(&bpe, p.as_bytes())
            );
        }
        for largo in [
            PALABRA_CORTA - 1,
            PALABRA_CORTA,
            PALABRA_CORTA + 1,
            500,
            5000,
        ] {
            let b = &corpus.as_bytes()[..largo];
            assert_eq!(
                bpe.codificar_bytes(b),
                codificar_referencia(&bpe, b),
                "largo {largo}"
            );
        }
        for largo in [PALABRA_CORTA, PALABRA_CORTA + 1, 4000] {
            let b = vec![b'a'; largo];
            assert_eq!(
                bpe.codificar_bytes(&b),
                codificar_referencia(&bpe, &b),
                "aaa… {largo}"
            );
        }
    }

    /// Recordar palabras y repartir en núcleos no cambia ni un token.
    #[test]
    fn un_texto_largo_da_lo_mismo_que_de_a_una_palabra() {
        let bpe = Bpe::entrenar(&corpus_de_prueba(40_000), 900);
        for largo in [
            TEXTO_LARGO - 1,
            TEXTO_LARGO,
            100_000,
            TROZO_PARALELO,
            3 << 20,
        ] {
            let t = corpus_de_prueba(largo);
            let mut directo = Vec::new();
            recorrer_piezas(&t, |p| bpe.codificar_trozo(p.as_bytes(), &mut directo));
            assert_eq!(bpe.codificar(&t), directo, "largo {largo}");
        }
    }

    #[test]
    fn pretokenizar_da_lo_mismo_que_el_algoritmo_de_antes() {
        let largo = corpus_de_prueba(40_000);
        for t in [
            "",
            " ",
            "  ",
            " a",
            "a ",
            " 1",
            " é",
            "é",
            "1a",
            "a1",
            " .",
            TEXTO,
            largo.as_str(),
        ] {
            assert_eq!(pretokenizar(t), pretokenizar_referencia(t), "{t:?}");
            // Los dos caminos y todas las copias del kernel, sin importar el largo.
            let mut uno = Vec::new();
            recorrer_de_a_uno(t, |p| uno.push(p));
            assert_eq!(uno, pretokenizar_referencia(t), "de a uno, {t:?}");
            for copia in crate::kernels::classify::disponibles() {
                let mut con = Vec::new();
                recorrer_con_mascaras(t, &Mascaras::con(copia, t.as_bytes()), |p| con.push(p));
                assert_eq!(con, pretokenizar_referencia(t), "{copia:?}, {t:?}");
            }
        }
    }

    /// Una palabra enorme sin espacios no puede tardar cuadrático: un mensaje
    /// con un base64 de 1 MB pasaría de milisegundos a minutos.
    ///
    /// Se mide cómo crece y no cuántos segundos tarda: con 4 veces más texto,
    /// lineal tarda unas 4 veces más y cuadrático 16. Un límite en segundos
    /// fallaba en CI, que compila sin optimizar: 5 a 7,7 s contra 0,1 en release.
    #[test]
    fn una_palabra_enorme_no_es_cuadratica() {
        let corpus = corpus_de_prueba(40_000);
        let bpe = Bpe::entrenar(&corpus, 900);
        let medir = |n: usize| {
            // El mejor de dos, para que una pausa del sistema no decida.
            (0..2)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let ids = bpe.codificar_bytes(&vec![b'a'; n]);
                    let s = t.elapsed().as_secs_f64();
                    assert_eq!(bpe.decodificar_bytes(&ids).unwrap().len(), n);
                    s
                })
                .fold(f64::MAX, f64::min)
        };
        let (chico, grande) = (medir(100_000), medir(400_000));
        let veces = grande / chico.max(1e-6);
        assert!(
            veces < 8.0,
            "4 veces el texto tardó {veces:.1} veces: {chico:.3} s y {grande:.3} s"
        );
    }
}
