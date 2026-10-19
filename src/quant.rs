//! Descuantización propia de los formatos de GGML.
//!
//! ## Dos decisiones de diseño, y las dos son de memoria
//!
//! **1. Trabaja sobre `&[u8]`, no sobre structs.** Convertir un buffer de
//! archivo en `&[BlockQ3K]` necesita que los bytes estén alineados, y un
//! `Vec<u8>` leído de disco no lo garantiza. Candle hace esa conversión y ahí
//! vive el `as_t_slice` que devuelve un slice hacia un `Cow` ya liberado —el
//! bug que en este proyecto corrompía la primera fila de cada lote—. Leyendo
//! los campos con `from_le_bytes` el problema no existe.
//!
//! **2. Escribe en un buffer que da quien llama.** No asigna nada. Eso no es
//! cosmética: descuantizar la tabla de embeddings entera costaba 519 MB,
//! más que el modelo completo, cuando un forward solo necesita las filas de
//! los tokens que entran. Bajó de 938 a 506 MB residentes. Una API que
//! devuelve `Vec<f32>` invita a ese error; una que pide el destino, no.
//!
//! ## Cómo se sabe que está bien
//!
//! Contra candle, exigiendo **igualdad exacta** y no una tolerancia: no hay
//! nada que redondear distinto, cada valor es una cuenta determinista de
//! enteros y una multiplicación. Un solo bit de diferencia significa que algún
//! desplazamiento está mal.

use crate::gguf::TipoGgml;

/// Cuántos valores entran en un superbloque de los k-quants.
const QK_K: usize = 256;

/// Convierte un `f16` a `f32`.
///
/// Escrito acá y no tomado de un crate porque es quince líneas y porque el
/// objetivo es que el motor no dependa de nadie para leer sus propios pesos.
/// Contempla los subnormales: un `f16` subnormal vale `mantisa · 2⁻²⁴`, y
/// tratarlos como cero daría pesos chicos exactamente en cero, que es un error
/// que no rompe nada y empeora el modelo en silencio.
#[inline]
pub fn f16_a_f32(bits: u16) -> f32 {
    let signo = ((bits >> 15) as u32) << 31;
    let exponente = ((bits >> 10) & 0x1F) as u32;
    let mantisa = (bits & 0x3FF) as u32;

    match exponente {
        0 if mantisa == 0 => f32::from_bits(signo),
        0 => {
            // Subnormal: se normaliza corriendo la mantisa hasta que aparezca
            // el bit implícito.
            let mut e = 1i32;
            let mut m = mantisa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits(signo | (((e - 15 + 127) as u32) << 23) | (m << 13))
        }
        0x1F => f32::from_bits(signo | 0x7F80_0000 | (mantisa << 13)),
        _ => f32::from_bits(signo | ((exponente + 127 - 15) << 23) | (mantisa << 13)),
    }
}

#[inline]
pub(crate) fn leer_f16(b: &[u8], i: usize) -> f32 {
    f16_a_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

#[inline]
fn leer_u32(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

/// Las escalas de Q4_K y Q5_K vienen empaquetadas a 6 bits en 12 bytes.
///
/// Los primeros cuatro pares salen directos; los otros cuatro arman cada valor
/// con cuatro bits de un byte y dos del byte de seis posiciones antes. Es la
/// parte del formato donde es más fácil equivocarse y no notarlo: un error acá
/// da pesos plausibles pero mal escalados.
#[inline]
/// Las dieciséis escalas de 6 bits de un bloque Q3_K.
///
/// Vienen repartidas en 12 bytes: los cuatro bits bajos en los ocho primeros y
/// los dos altos en los cuatro últimos. Se rearman en cuatro palabras de 32 y
/// salen como enteros con signo a los que hay que restarles 32.
///
/// Estaba escrito dos veces, idéntico, y ahora lo usan tres sitios: los dos
/// descuantizadores y el kernel fusionado.
pub(crate) fn escalas_q3k(b: &[u8], o: usize) -> [i8; 16] {
    const MASCARA1: u32 = 0x0303_0303;
    const MASCARA2: u32 = 0x0F0F_0F0F;

    let mut aux = [0u32; 4];
    for (j, a) in aux.iter_mut().take(3).enumerate() {
        *a = leer_u32(b, o + 96 + j * 4);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & MASCARA2) | (((tmp >> 4) & MASCARA1) << 4);
    aux[3] = ((aux[1] >> 4) & MASCARA2) | (((tmp >> 6) & MASCARA1) << 4);
    aux[0] = (aux[0] & MASCARA2) | ((tmp & MASCARA1) << 4);
    aux[1] = (aux[1] & MASCARA2) | (((tmp >> 2) & MASCARA1) << 4);

    let mut esc = [0i8; 16];
    for (j, e) in esc.iter_mut().enumerate() {
        *e = (aux[j / 4] >> (8 * (j % 4))) as u8 as i8;
    }
    esc
}

pub(crate) fn escala_y_minimo(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Descuantiza `bloques` bytes crudos hacia `salida`.
///
/// `salida.len()` decide cuántos valores se producen y tiene que ser múltiplo
/// del tamaño de bloque del tipo. Devuelve error en vez de entrar en pánico
/// porque los datos vienen de un archivo, y un archivo puede estar mal.
pub fn descuantizar(tipo: TipoGgml, bloques: &[u8], salida: &mut [f32]) -> Result<(), String> {
    let por_bloque = tipo.valores_por_bloque();
    if !salida.len().is_multiple_of(por_bloque) {
        return Err(format!(
            "{tipo:?}: pedí {} valores y el bloque es de {por_bloque}",
            salida.len()
        ));
    }
    let n_bloques = salida.len() / por_bloque;
    let necesarios = n_bloques * tipo.bytes_por_bloque();
    if bloques.len() < necesarios {
        return Err(format!(
            "{tipo:?}: hacen falta {necesarios} bytes y hay {}",
            bloques.len()
        ));
    }

    match tipo {
        TipoGgml::F32 => {
            for (i, s) in salida.iter_mut().enumerate() {
                *s = f32::from_bits(leer_u32(bloques, i * 4));
            }
        }
        TipoGgml::F16 => {
            for (i, s) in salida.iter_mut().enumerate() {
                *s = leer_f16(bloques, i * 2);
            }
        }
        TipoGgml::Q8_0 => q8_0(bloques, salida, n_bloques),
        TipoGgml::Q5_0 => q5_0(bloques, salida, n_bloques),
        TipoGgml::Q2K => q2_k(bloques, salida, n_bloques),
        TipoGgml::Q3K => q3_k(bloques, salida, n_bloques),
        TipoGgml::Q4K => q4_k(bloques, salida, n_bloques),
        TipoGgml::Q5K => q5_k(bloques, salida, n_bloques),
        TipoGgml::Q6K => q6_k(bloques, salida, n_bloques),
        otro => return Err(format!("{otro:?}: todavía no está implementado")),
    }
    Ok(())
}

/// Descuantiza solo las filas pedidas de una matriz guardada por filas.
///
/// Es la operación que de verdad hace falta en el camino caliente: la tabla de
/// embeddings del 3B son 151936 filas y un forward usa las de los tokens que
/// entran — 3,5 KB por token contra 519 MB por la tabla entera.
///
/// Cada fila ocupa un número entero de bloques, así que se puede recortar sin
/// desarmar nada: 896 elementos en bloques de 32 son 28 bloques exactos.
pub fn descuantizar_filas(
    tipo: TipoGgml,
    datos: &[u8],
    ancho: usize,
    filas: &[usize],
    salida: &mut [f32],
) -> Result<(), String> {
    let por_bloque = tipo.valores_por_bloque();
    if !ancho.is_multiple_of(por_bloque) {
        return Err(format!(
            "una fila de {ancho} no es múltiplo del bloque de {por_bloque}: este tipo no puede guardarla"
        ));
    }
    if salida.len() != filas.len() * ancho {
        return Err(format!(
            "salida de {} para {} filas de {ancho}",
            salida.len(),
            filas.len()
        ));
    }
    let bytes_fila = ancho / por_bloque * tipo.bytes_por_bloque();

    for (n, &fila) in filas.iter().enumerate() {
        let desde = fila * bytes_fila;
        let hasta = desde + bytes_fila;
        if hasta > datos.len() {
            return Err(format!("la fila {fila} se sale de los datos"));
        }
        descuantizar(
            tipo,
            &datos[desde..hasta],
            &mut salida[n * ancho..(n + 1) * ancho],
        )?;
    }
    Ok(())
}

/// A que unidad cambia la escala en cada formato.
///
/// Quien multiplica necesita saberlo **antes** de cuantizar la entrada, porque
/// los dos lados del producto tienen que usar el mismo tamano de bloque.
pub fn sub_bloque(tipo: TipoGgml) -> Option<usize> {
    Some(match tipo {
        TipoGgml::Q8_0 | TipoGgml::Q5_0 | TipoGgml::Q4K | TipoGgml::Q5K => 32,
        TipoGgml::Q2K | TipoGgml::Q3K | TipoGgml::Q6K => 16,
        _ => return None,
    })
}

/// Desempaqueta a `i8` con una escala y un desplazamiento por sub-bloque,
/// **sin pasar por `f32`**.
///
/// Los pesos ya vienen como enteros chicos con una escala: convertirlos a
/// `f32` para volver a convertirlos a `i8` es un viaje de ida y vuelta que no
/// aporta nada y cuesta. Medido, ese rodeo era la mayor parte del tiempo de la
/// multiplicacion en enteros — unos 7 ms de 14.
///
/// La forma a la que se desempaqueta es `valor = escala * q + desplazamiento`,
/// que cubre a todos los formatos de GGML: los que no tienen minimo —Q8_0,
/// Q5_0, Q3_K, Q6_K— dejan el desplazamiento en cero, y los que si —Q2_K,
/// Q4_K, Q5_K— lo usan.
///
/// El desplazamiento no estorba al producto punto: `suma x·(s·q + o)` es
/// `s·suma(x·q) + o·suma(x)`, y la suma de la entrada se calcula una vez.
///
/// Devuelve el tamano de sub-bloque que uso, que depende del formato.
pub fn desempaquetar_i8(
    tipo: TipoGgml,
    bloques: &[u8],
    q: &mut [i8],
    escalas: &mut [f32],
    desplazamientos: &mut [f32],
) -> Result<usize, String> {
    let por_bloque = tipo.valores_por_bloque();
    if !q.len().is_multiple_of(por_bloque) {
        return Err(format!(
            "{tipo:?}: pedi {} valores y el bloque es de {por_bloque}",
            q.len()
        ));
    }
    let n = q.len() / por_bloque;
    if bloques.len() < n * tipo.bytes_por_bloque() {
        return Err(format!("{tipo:?}: faltan bytes"));
    }

    // El sub-bloque es la unidad a la que cambia la escala en cada formato.
    let sub =
        sub_bloque(tipo).ok_or_else(|| format!("{tipo:?}: sin desempaquetado directo a i8"))?;
    let n_sub = q.len() / sub;
    if escalas.len() != n_sub || desplazamientos.len() != n_sub {
        return Err(format!(
            "{tipo:?}: hacen falta {n_sub} escalas y llegaron {}",
            escalas.len()
        ));
    }
    desplazamientos.fill(0.0);

    match tipo {
        TipoGgml::Q8_0 => {
            for i in 0..n {
                let o = i * 34;
                escalas[i] = leer_f16(bloques, o);
                let destino = &mut q[i * 32..i * 32 + 32];
                for (d, b) in destino.iter_mut().zip(&bloques[o + 2..o + 34]) {
                    *d = *b as i8;
                }
            }
        }
        TipoGgml::Q4K => {
            for i in 0..n {
                let o = i * 144;
                let d = leer_f16(bloques, o);
                let minimo = leer_f16(bloques, o + 2);
                let esc = &bloques[o + 4..o + 16];
                let qs = &bloques[o + 16..o + 144];

                // Ocho sub-bloques de 32. Los nibbles bajos de los 32 bytes
                // son el sub-bloque par y los altos el impar, que es como los
                // guarda el formato.
                for par in 0..4 {
                    let base = par * 32;
                    let (sc, m) = escala_y_minimo(par * 2, esc);
                    let (sc2, m2) = escala_y_minimo(par * 2 + 1, esc);
                    let s0 = i * 8 + par * 2;
                    escalas[s0] = d * sc as f32;
                    desplazamientos[s0] = -(minimo * m as f32);
                    escalas[s0 + 1] = d * sc2 as f32;
                    desplazamientos[s0 + 1] = -(minimo * m2 as f32);

                    let desde = i * 256 + par * 64;
                    let (bajos, altos) = q[desde..desde + 64].split_at_mut(32);
                    for ((b, lo), hi) in qs[base..base + 32].iter().zip(bajos).zip(altos) {
                        *lo = (b & 0xF) as i8;
                        *hi = (b >> 4) as i8;
                    }
                }
            }
        }
        TipoGgml::Q5_0 => {
            for i in 0..n {
                let o = i * 22;
                escalas[i] = leer_f16(bloques, o);
                let qh = leer_u32(bloques, o + 2);
                let qs = &bloques[o + 6..o + 22];
                let (bajos, altos) = q[i * 32..i * 32 + 32].split_at_mut(16);
                for j in 0..16 {
                    let alto_0 = (((qh >> j) << 4) & 0x10) as u8;
                    let alto_1 = ((qh >> (j + 12)) & 0x10) as u8;
                    bajos[j] = ((qs[j] & 0x0F) | alto_0) as i8 - 16;
                    altos[j] = ((qs[j] >> 4) | alto_1) as i8 - 16;
                }
            }
        }
        TipoGgml::Q5K => {
            for i in 0..n {
                let o = i * 176;
                let d = leer_f16(bloques, o);
                let minimo = leer_f16(bloques, o + 2);
                let esc = &bloques[o + 4..o + 16];
                let qh = &bloques[o + 16..o + 48];
                let ql = &bloques[o + 48..o + 176];

                let mut is = 0usize;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                let mut k = i * QK_K;
                for j in (0..QK_K).step_by(64) {
                    let qv = &ql[j / 2..j / 2 + 32];
                    let (sc, m) = escala_y_minimo(is, esc);
                    escalas[i * 8 + is] = d * sc as f32;
                    desplazamientos[i * 8 + is] = -(minimo * m as f32);
                    let (sc2, m2) = escala_y_minimo(is + 1, esc);
                    escalas[i * 8 + is + 1] = d * sc2 as f32;
                    desplazamientos[i * 8 + is + 1] = -(minimo * m2 as f32);

                    let (bajos, altos) = q[k..k + 64].split_at_mut(32);
                    for ((v, h), d) in qv.iter().zip(qh).zip(bajos) {
                        let extra = if h & u1 != 0 { 16 } else { 0 };
                        *d = ((v & 0xF) + extra) as i8;
                    }
                    for ((v, h), d) in qv.iter().zip(qh).zip(altos) {
                        let extra = if h & u2 != 0 { 16 } else { 0 };
                        *d = ((v >> 4) + extra) as i8;
                    }
                    k += 64;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        TipoGgml::Q2K => {
            for i in 0..n {
                let o = i * 84;
                let esc = &bloques[o..o + 16];
                let qs = &bloques[o + 16..o + 80];
                let d = leer_f16(bloques, o + 80);
                let minimo = leer_f16(bloques, o + 82);

                let mut is = 0usize;
                let mut k = i * QK_K;
                for mitad in 0..2 {
                    let qs = &qs[mitad * 32..mitad * 32 + 32];
                    let mut desplazamiento = 0;
                    for _ in 0..4 {
                        // Los dos sub-bloques de 16 que vienen seguidos se
                        // recorren en UN bucle de 32. Difieren solo en la
                        // escala; el origen y el destino son contiguos entre
                        // ellos, asi que partirlos en dos dejaba el bucle a
                        // medio registro de AVX2 y no habia ningun motivo.
                        for t in 0..2 {
                            let sc = esc[is + t];
                            escalas[i * 16 + is + t] = d * (sc & 0xF) as f32;
                            desplazamientos[i * 16 + is + t] = -(minimo * (sc >> 4) as f32);
                        }
                        is += 2;
                        for (d, v) in q[k..k + 32].iter_mut().zip(qs.iter()) {
                            *d = ((v >> desplazamiento) & 3) as i8;
                        }
                        k += 32;
                        desplazamiento += 2;
                    }
                }
            }
        }
        TipoGgml::Q3K => {
            for i in 0..n {
                let o = i * 110;
                let hmask = &bloques[o..o + 32];
                let qs = &bloques[o + 32..o + 96];
                let d_todo = leer_f16(bloques, o + 108);
                let esc = escalas_q3k(bloques, o);

                let mut m: u8 = 1;
                let mut is = 0usize;
                let mut k = i * QK_K;
                for mitad in 0..2 {
                    let qs = &qs[mitad * 32..mitad * 32 + 32];
                    let mut desplazamiento = 0;
                    for _ in 0..4 {
                        // Igual que en Q2_K: los dos sub-bloques van seguidos
                        // y comparten el corrimiento y la mascara, asi que son
                        // un solo bucle de 32.
                        for t in 0..2 {
                            escalas[i * 16 + is + t] = d_todo * (esc[is + t] as f32 - 32.0);
                        }
                        is += 2;
                        for ((d, v), h) in q[k..k + 32].iter_mut().zip(&qs[..32]).zip(&hmask[..32])
                        {
                            let alto = if h & m == 0 { 4i8 } else { 0 };
                            *d = ((v >> desplazamiento) & 3) as i8 - alto;
                        }
                        k += 32;
                        desplazamiento += 2;
                        m <<= 1;
                    }
                }
            }
        }
        TipoGgml::Q6K => {
            for i in 0..n {
                let o = i * 210;
                let ql = &bloques[o..o + 128];
                let qh = &bloques[o + 128..o + 192];
                let esc = &bloques[o + 192..o + 208];
                let d = leer_f16(bloques, o + 208);

                for mitad in 0..2 {
                    let base = i * QK_K + mitad * 128;
                    let ql = &ql[64 * mitad..];
                    let qh = &qh[32 * mitad..];
                    let sc = &esc[8 * mitad..];

                    // Cada 16 posiciones dentro de la mitad cambian de escala,
                    // y el indice es sencillamente `posicion / 16`.
                    for p in 0..8 {
                        escalas[i * 16 + mitad * 8 + p] = d * sc[p] as i8 as f32;
                    }

                    let ventana = &mut q[base..base + 128];
                    let (primera, resto) = ventana.split_at_mut(32);
                    let (segunda, resto) = resto.split_at_mut(32);
                    let (tercera, cuarta) = resto.split_at_mut(32);
                    let (ql_bajo, ql_alto) = (&ql[..32], &ql[32..64]);
                    for (l, h) in qh[..32].iter().enumerate() {
                        primera[l] = ((ql_bajo[l] & 0xF) | ((h & 3) << 4)) as i8 - 32;
                        segunda[l] = ((ql_alto[l] & 0xF) | (((h >> 2) & 3) << 4)) as i8 - 32;
                        tercera[l] = ((ql_bajo[l] >> 4) | (((h >> 4) & 3) << 4)) as i8 - 32;
                        cuarta[l] = ((ql_alto[l] >> 4) | (((h >> 6) & 3) << 4)) as i8 - 32;
                    }
                }
            }
        }
        otro => return Err(format!("{otro:?}: sin desempaquetado directo a i8 todavia")),
    }
    Ok(sub)
}

// ─── Los formatos, uno por uno ───────────────────────────────────────────────

fn q8_0(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 34;
        let d = leer_f16(b, o);
        for (s, v) in y[i * 32..i * 32 + 32].iter_mut().zip(&b[o + 2..o + 34]) {
            *s = *v as i8 as f32 * d;
        }
    }
}

fn q5_0(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 22;
        let d = leer_f16(b, o);
        let qh = leer_u32(b, o + 2);
        let qs = &b[o + 6..o + 22];
        let (bajos, altos) = y[i * 32..i * 32 + 32].split_at_mut(16);
        for j in 0..16 {
            // El quinto bit de cada valor vive aparte, en `qh`. El de la
            // mitad baja sale del bit j y el de la alta del bit j+16, que es
            // de donde viene el desplazamiento de 12 (16 - 4).
            let alto_0 = (((qh >> j) << 4) & 0x10) as u8;
            let alto_1 = ((qh >> (j + 12)) & 0x10) as u8;
            bajos[j] = (((qs[j] & 0x0F) | alto_0) as i32 - 16) as f32 * d;
            altos[j] = (((qs[j] >> 4) | alto_1) as i32 - 16) as f32 * d;
        }
    }
}

fn q2_k(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 84;
        let escalas = &b[o..o + 16];
        let qs = &b[o + 16..o + 80];
        let d = leer_f16(b, o + 80);
        let minimo = leer_f16(b, o + 82);

        let mut is = 0;
        let mut k = i * QK_K;
        for mitad in 0..2 {
            let qs = &qs[mitad * 32..mitad * 32 + 32];
            let mut desplazamiento = 0;
            for _ in 0..4 {
                for tramo in 0..2 {
                    let sc = escalas[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = minimo * (sc >> 4) as f32;
                    let desde = k + tramo * 16;
                    for (s, q) in y[desde..desde + 16]
                        .iter_mut()
                        .zip(&qs[tramo * 16..tramo * 16 + 16])
                    {
                        *s = dl * ((q >> desplazamiento) & 3) as f32 - ml;
                    }
                }
                k += 32;
                desplazamiento += 2;
            }
        }
    }
}

fn q3_k(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 110;
        let hmask = &b[o..o + 32];
        let qs = &b[o + 32..o + 96];
        let d_todo = leer_f16(b, o + 108);
        let escalas = escalas_q3k(b, o);

        let mut m: u8 = 1;
        let mut is = 0;
        let mut k = i * QK_K;
        for mitad in 0..2 {
            let qs = &qs[mitad * 32..mitad * 32 + 32];
            let mut desplazamiento = 0;
            for _ in 0..4 {
                // Los dos sub-bloques de 16 van seguidos en el origen y en el
                // destino, y comparten corrimiento y máscara: se recorren con
                // una ventana de 32 partida en dos, cada mitad con su escala.
                // Partirlos en dos bucles dejaba el índice de `y` calculado en
                // tiempo de ejecución, con su comprobación de límites por
                // valor, y eso le prohíbe vectorizar: medido, 6,5 veces más
                // lento.
                let (primera, segunda) = y[k..k + 32].split_at_mut(16);
                for (tramo, destino) in [primera, segunda].into_iter().enumerate() {
                    let dl = d_todo * (escalas[is] as f32 - 32.0);
                    is += 1;
                    let desde = tramo * 16;
                    for ((s, q), h) in destino
                        .iter_mut()
                        .zip(&qs[desde..desde + 16])
                        .zip(&hmask[desde..desde + 16])
                    {
                        // El tercer bit está en `hmask`, y cuando falta hay
                        // que restar 4: el valor es de 2 bits con signo
                        // corrido, no de 3 bits sin signo.
                        let alto = if h & m == 0 { 4i8 } else { 0 };
                        *s = dl * (((q >> desplazamiento) & 3) as i8 - alto) as f32;
                    }
                }
                k += 32;
                desplazamiento += 2;
                m <<= 1;
            }
        }
    }
}

fn q4_k(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 144;
        let d = leer_f16(b, o);
        let minimo = leer_f16(b, o + 2);
        let escalas = &b[o + 4..o + 16];
        let qs = &b[o + 16..o + 144];

        let mut is = 0;
        let mut k = i * QK_K;
        for j in (0..QK_K).step_by(64) {
            let q = &qs[j / 2..j / 2 + 32];
            let (sc, m) = escala_y_minimo(is, escalas);
            let (d1, m1) = (d * sc as f32, minimo * m as f32);
            let (sc, m) = escala_y_minimo(is + 1, escalas);
            let (d2, m2) = (d * sc as f32, minimo * m as f32);
            let (bajos, altos) = y[k..k + 64].split_at_mut(32);
            for (s, q) in bajos.iter_mut().zip(q) {
                *s = d1 * (q & 0xF) as f32 - m1;
            }
            for (s, q) in altos.iter_mut().zip(q) {
                *s = d2 * (q >> 4) as f32 - m2;
            }
            k += 64;
            is += 2;
        }
    }
}

fn q5_k(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 176;
        let d = leer_f16(b, o);
        let minimo = leer_f16(b, o + 2);
        let escalas = &b[o + 4..o + 16];
        let qh = &b[o + 16..o + 48];
        let ql = &b[o + 48..o + 176];

        let mut is = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        let mut k = i * QK_K;
        for j in (0..QK_K).step_by(64) {
            let q = &ql[j / 2..j / 2 + 32];
            let (sc, m) = escala_y_minimo(is, escalas);
            let (d1, m1) = (d * sc as f32, minimo * m as f32);
            let (sc, m) = escala_y_minimo(is + 1, escalas);
            let (d2, m2) = (d * sc as f32, minimo * m as f32);
            let (bajos, altos) = y[k..k + 64].split_at_mut(32);
            for ((q, h), s) in q.iter().zip(qh).zip(bajos) {
                let extra = if h & u1 != 0 { 16.0 } else { 0.0 };
                *s = d1 * ((q & 0xF) as f32 + extra) - m1;
            }
            for ((q, h), s) in q.iter().zip(qh).zip(altos) {
                let extra = if h & u2 != 0 { 16.0 } else { 0.0 };
                *s = d2 * ((q >> 4) as f32 + extra) - m2;
            }
            k += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn q6_k(b: &[u8], y: &mut [f32], n: usize) {
    for i in 0..n {
        let o = i * 210;
        let ql = &b[o..o + 128];
        let qh = &b[o + 128..o + 192];
        let escalas = &b[o + 192..o + 208];
        let d = leer_f16(b, o + 208);

        for mitad in 0..2 {
            let base = i * QK_K + mitad * 128;
            let ql = &ql[64 * mitad..];
            let qh = &qh[32 * mitad..];
            let sc = &escalas[8 * mitad..];
            let ventana = &mut y[base..base + 128];
            let (primera, resto) = ventana.split_at_mut(32);
            let (segunda, resto) = resto.split_at_mut(32);
            let (tercera, cuarta) = resto.split_at_mut(32);
            for l in 0..32 {
                let is = l / 16;
                // Cuatro valores por byte bajo: dos nibbles, y los dos bits
                // altos de cada uno repartidos en `qh`.
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                primera[l] = d * sc[is] as i8 as f32 * q1 as f32;
                segunda[l] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                tercera[l] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                cuarta[l] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
}

// ─── El camino de vuelta: de `f32` al formato ────────────────────────────────

/// `f32` a `f16`, con subnormales y redondeo al par más cercano.
///
/// Es el inverso de [`f16_a_f32`] y hace falta para **escribir** modelos, no
/// solo para leerlos: las escalas de cada bloque de los formatos cuantizados
/// se guardan en `f16`, así que sin esto no se puede producir un GGUF.
///
/// El redondeo es al par y no truncando. Truncar sesga cada escala hacia cero,
/// y una escala sesgada desvía los 32 o 256 valores que cuelgan de ella.
pub fn f32_a_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let signo = ((bits >> 16) & 0x8000) as u16;
    let magnitud = bits & 0x7FFF_FFFF;

    // NaN e infinito: el exponente se satura y la mantisa se conserva para no
    // convertir un NaN en un infinito, que son cosas distintas.
    if magnitud >= 0x7F80_0000 {
        let mantisa = (magnitud & 0x007F_FFFF) != 0;
        return signo | 0x7C00 | if mantisa { 0x0200 } else { 0 };
    }

    let exponente = (magnitud >> 23) as i32 - 127;

    if exponente > 15 {
        // Más grande de lo que `f16` representa: satura a infinito. Devolver
        // el máximo finito mentiría sobre el orden de magnitud.
        return signo | 0x7C00;
    }

    if exponente < -14 {
        // Subnormal de `f16`, o cero. Se corre la mantisa con el bit implícito
        // puesto y se redondea sobre el resultado corrido.
        let corrimiento = (-14 - exponente) as u32;
        if corrimiento > 24 {
            return signo;
        }
        let mantisa = (magnitud & 0x007F_FFFF) | 0x0080_0000;
        let valor = mantisa >> (corrimiento + 13);
        let resto = mantisa & ((1 << (corrimiento + 13)) - 1);
        let mitad = 1 << (corrimiento + 12);
        let redondeo = u32::from(resto > mitad || (resto == mitad && valor & 1 == 1));
        return signo | (valor + redondeo) as u16;
    }

    let mantisa = magnitud & 0x007F_FFFF;
    let valor = ((exponente + 15) as u32) << 10 | (mantisa >> 13);
    let resto = mantisa & 0x1FFF;
    let redondeo = u32::from(resto > 0x1000 || (resto == 0x1000 && valor & 1 == 1));
    signo | (valor + redondeo) as u16
}

/// Pasa `f32` al formato pedido y lo agrega a `salida`.
///
/// Es el inverso de [`descuantizar`] y cubre los tres formatos con los que
/// tiene sentido **escribir** un modelo nuevo:
///
/// - `F32`, que no pierde nada y pesa cuatro bytes por valor;
/// - `F16`, mitad de tamaño y el error de redondeo de `f16`;
/// - `Q8_0`, un byte por valor más una escala `f16` cada 32, o sea 1,0625
///   bytes por valor — nueve veces menos que `F32` por diez.
///
/// Los k-quants no están, y a propósito: su cuantizador busca la escala y el
/// mínimo que minimizan el error de cada sub-bloque con una búsqueda
/// iterativa, no es una formula cerrada, y escribirlo sin tener contra qué
/// compararlo valor por valor sería justo lo que este proyecto no hace. Un
/// modelo propio se escribe en `Q8_0` y se recuantiza después con una
/// herramienta que ya esté verificada.
pub fn cuantizar_bloques(tipo: TipoGgml, x: &[f32], salida: &mut Vec<u8>) -> Result<(), String> {
    let por_bloque = tipo.valores_por_bloque();
    if !x.len().is_multiple_of(por_bloque) {
        return Err(format!(
            "{tipo:?}: {} valores no es multiplo de {por_bloque}",
            x.len()
        ));
    }
    salida.reserve(tipo.bytes_de(x.len()));

    match tipo {
        TipoGgml::F32 => {
            for v in x {
                salida.extend_from_slice(&v.to_le_bytes());
            }
        }
        TipoGgml::F16 => {
            for v in x {
                salida.extend_from_slice(&f32_a_f16(*v).to_le_bytes());
            }
        }
        TipoGgml::Q8_0 => {
            for trozo in x.chunks(32) {
                // La escala es `max|x| / 127`, igual que en los cuantizadores
                // de GGML, y por eso el resultado entra en [-127, 127].
                let amax = trozo.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                salida.extend_from_slice(&f32_a_f16(d).to_le_bytes());
                // La escala se guarda en `f16`, así que al leerla de vuelta no
                // es `d` sino su redondeo. Cuantizar con la que SE VA A LEER y
                // no con la original quita ese error de un paso entero.
                let leida = f16_a_f32(f32_a_f16(d));
                let inv = if leida > 0.0 { 1.0 / leida } else { 0.0 };
                for v in trozo {
                    salida.push(((v * inv).round() as i32).clamp(-127, 127) as i8 as u8);
                }
            }
        }
        otro => return Err(format!("{otro:?}: sin cuantizador para escribir")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Los casos de `f16` que más fácil se escriben mal.
    #[test]
    fn f16_cubre_los_casos_raros() {
        assert_eq!(f16_a_f32(0x0000), 0.0);
        assert_eq!(f16_a_f32(0x8000), -0.0);
        assert_eq!(f16_a_f32(0x3C00), 1.0);
        assert_eq!(f16_a_f32(0xBC00), -1.0);
        assert_eq!(f16_a_f32(0x4000), 2.0);
        assert_eq!(f16_a_f32(0x7BFF), 65504.0); // el mayor f16 normal
        assert!(f16_a_f32(0x7C00).is_infinite());
        assert!(f16_a_f32(0x7E00).is_nan());
        // El subnormal más chico: 2⁻²⁴. Tratarlo como cero no rompe nada y
        // empeora el modelo en silencio, que es peor.
        assert_eq!(f16_a_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_a_f32(0x03FF), 1023.0 * 2f32.powi(-24));
    }

    /// Un bloque Q8_0 armado a mano, donde la cuenta se puede hacer de cabeza.
    #[test]
    fn q8_0_a_mano() {
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
        b[2] = 5;
        b[3] = (-3i8) as u8;
        let mut y = vec![0f32; 32];
        descuantizar(TipoGgml::Q8_0, &b, &mut y).unwrap();
        assert_eq!(y[0], 5.0);
        assert_eq!(y[1], -3.0);
    }

    #[test]
    fn avisa_si_el_destino_no_es_multiplo_del_bloque() {
        let b = vec![0u8; 34];
        let mut y = vec![0f32; 17];
        assert!(descuantizar(TipoGgml::Q8_0, &b, &mut y).is_err());
    }

    #[test]
    fn avisa_si_faltan_bytes() {
        let b = vec![0u8; 10];
        let mut y = vec![0f32; 32];
        assert!(descuantizar(TipoGgml::Q8_0, &b, &mut y).is_err());
    }

    /// Sacar filas sueltas da lo mismo que descuantizar todo y recortar.
    ///
    /// Es la propiedad que hace que el ahorro de memoria sea gratis: si no se
    /// cumpliera, la optimizacion de los embeddings estaria cambiando el
    /// resultado.
    #[test]
    fn por_filas_coincide_con_entero() {
        let ancho = 64usize;
        let n_filas = 5usize;
        let bytes = ancho / 32 * 34 * n_filas;
        let datos: Vec<u8> = (0..bytes).map(|i| (i * 37 % 251) as u8).collect();

        let mut todo = vec![0f32; ancho * n_filas];
        descuantizar(TipoGgml::Q8_0, &datos, &mut todo).unwrap();

        let pedidas = [3usize, 0, 4];
        let mut parte = vec![0f32; ancho * pedidas.len()];
        descuantizar_filas(TipoGgml::Q8_0, &datos, ancho, &pedidas, &mut parte).unwrap();

        for (n, &f) in pedidas.iter().enumerate() {
            assert_eq!(
                &parte[n * ancho..(n + 1) * ancho],
                &todo[f * ancho..(f + 1) * ancho],
                "la fila {f} no coincide"
            );
        }
    }

    /// Una fila que no cabe en bloques enteros no se puede recortar, y hay que
    /// decirlo en vez de leer bytes de la fila de al lado.
    /// Reconstruir desde el desempaquetado a `i8` tiene que dar **exactamente**
    /// lo mismo que descuantizar a `f32`.
    ///
    /// Es exacto y no aproximado porque no hay recuantizacion de por medio: el
    /// `i8` guarda el mismo entero que el formato traia, y la escala es la
    /// misma. Si difiriera, el desempaquetado estaria leyendo otros bits.
    #[test]
    fn el_desempaquetado_reconstruye_la_descuantizacion() {
        for (tipo, bytes_bloque, valores, sub_esperado) in [
            (TipoGgml::Q8_0, 34usize, 32usize, 32usize),
            (TipoGgml::Q5_0, 22, 32, 32),
            (TipoGgml::Q2K, 84, 256, 16),
            (TipoGgml::Q3K, 110, 256, 16),
            (TipoGgml::Q4K, 144, 256, 32),
            (TipoGgml::Q5K, 176, 256, 32),
            (TipoGgml::Q6K, 210, 256, 16),
        ] {
            let n = 3usize;
            let datos: Vec<u8> = (0..n * bytes_bloque)
                .map(|i| ((i * 97 + 13) % 251) as u8)
                .collect();

            let mut esperado = vec![0f32; n * valores];
            descuantizar(tipo, &datos, &mut esperado).unwrap();

            let mut q = vec![0i8; n * valores];
            let mut esc = vec![0f32; n * valores / sub_esperado];
            let mut des = vec![0f32; n * valores / sub_esperado];
            let sub = desempaquetar_i8(tipo, &datos, &mut q, &mut esc, &mut des).unwrap();
            assert_eq!(sub, sub_esperado, "{tipo:?}: sub-bloque");

            for b in 0..q.len() / sub {
                for j in 0..sub {
                    let idx = b * sub + j;
                    let rec = esc[b] * q[idx] as f32 + des[b];
                    assert_eq!(
                        rec, esperado[idx],
                        "{tipo:?} valor {idx}: reconstruido {rec} contra {}",
                        esperado[idx]
                    );
                }
            }
        }
    }

    /// `f32` a `f16` y de vuelta tiene que devolver el mismo numero cuando el
    /// numero **es** representable en `f16`.
    ///
    /// Se prueban los casos que rompen una conversion mal escrita: cero y cero
    /// negativo, el subnormal mas chico, el normal mas chico, el finito mas
    /// grande, y potencias de dos a los dos lados del limite de los
    /// subnormales.
    #[test]
    fn f16_ida_y_vuelta_en_los_valores_representables() {
        let exactos = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0, // el finito mas grande de f16
            -65504.0,
            6.103_515_6e-5, // el normal mas chico
            5.960_464_5e-8, // el subnormal mas chico
            1.0 / 1024.0,
        ];
        for v in exactos {
            let vuelta = f16_a_f32(f32_a_f16(v));
            assert_eq!(
                vuelta.to_bits(),
                v.to_bits(),
                "{v:e} volvio como {vuelta:e}"
            );
        }

        // Fuera de rango satura a infinito y no al maximo finito, que diria
        // una magnitud equivocada.
        assert!(f16_a_f32(f32_a_f16(1e30)).is_infinite());
        assert!(f16_a_f32(f32_a_f16(-1e30)).is_infinite());
        assert!(f16_a_f32(f32_a_f16(f32::NAN)).is_nan());

        // Y el redondeo es al par mas cercano, no truncando: a mitad de
        // camino entre dos f16 consecutivos tiene que caer en el par.
        let medio = (f16_a_f32(0x3C00) + f16_a_f32(0x3C01)) / 2.0;
        assert_eq!(f32_a_f16(medio), 0x3C00, "1.0 es el par de los dos");
    }

    /// Cuantizar y descuantizar de vuelta tiene que dar lo mismo en los
    /// formatos sin perdida, y un error acotado en Q8_0.
    ///
    /// La cota no es un numero elegido a gusto: Q8_0 guarda un entero de 8
    /// bits con signo por valor y una escala por bloque de 32, asi que el
    /// error de cada valor no puede pasar de media escala, y la escala es
    /// `max|x| / 127`. O sea `max|x| / 254` por valor, mas lo que agregue el
    /// redondeo de la escala a `f16`.
    #[test]
    fn cuantizar_y_volver_cierra_el_circulo() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.11).sin() * 3.0).collect();
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));

        for (tipo, cota) in [
            (TipoGgml::F32, 0.0f32),
            (TipoGgml::F16, amax * 1e-3),
            (TipoGgml::Q8_0, amax / 254.0 * 1.01),
        ] {
            let mut bytes = Vec::new();
            cuantizar_bloques(tipo, &x, &mut bytes).unwrap();
            assert_eq!(
                bytes.len(),
                tipo.bytes_de(x.len()),
                "{tipo:?}: tamano escrito"
            );

            let mut vuelta = vec![0f32; x.len()];
            descuantizar(tipo, &bytes, &mut vuelta).unwrap();

            let peor = x
                .iter()
                .zip(&vuelta)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                peor <= cota,
                "{tipo:?}: el peor error fue {peor} y la cota es {cota}"
            );
        }
    }

    /// Un bloque entero en cero no puede dar NaN.
    ///
    /// La escala sale de dividir por el maximo absoluto, y en un modelo
    /// cuantizado los bloques en cero existen de verdad.
    #[test]
    fn un_bloque_en_cero_no_da_nan() {
        let x = vec![0f32; 64];
        let mut bytes = Vec::new();
        cuantizar_bloques(TipoGgml::Q8_0, &x, &mut bytes).unwrap();
        let mut vuelta = vec![9f32; 64];
        descuantizar(TipoGgml::Q8_0, &bytes, &mut vuelta).unwrap();
        assert!(vuelta.iter().all(|v| *v == 0.0), "{vuelta:?}");
    }

    #[test]
    fn una_fila_que_no_es_multiplo_del_bloque_avisa() {
        let datos = vec![0u8; 1024];
        let mut y = vec![0f32; 100];
        assert!(descuantizar_filas(TipoGgml::Q8_0, &datos, 100, &[0], &mut y).is_err());
    }
}
