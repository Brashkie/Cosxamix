//! Escribir k-quants: Q2_K, Q3_K, Q4_K y Q5_K.
//!
//! Hasta la 0.7.0 un modelo propio solo se podía guardar en `Q8_0` o más
//! grande, y eso es lo que más lo frena: el techo de generación es
//! `tokens/s ≈ ancho de banda / tamaño del modelo`, así que un archivo el doble
//! de grande rinde la mitad. Un daiss de 3B en `Q8_0` son 3,4 GB, y en `Q3_K`,
//! 1,5 GB. Y un modelo grande, directamente, no entra en memoria si no.
//!
//! El algoritmo es el de los cuantizadores de GGML —la misma búsqueda de la
//! escala y el mínimo de cada sub-bloque—, y lo que se compara es **el error
//! contra los pesos**: `examples/measure_kquants.rs` cuantiza con este código
//! los pesos de Qwen2.5 3B y mide contra los archivos oficiales del mismo
//! modelo (DECISIONES). Que el formato está bien lo dice el lector
//! ([`crate::quant::descuantizar`], verificado valor por valor contra candle).

use crate::quant::{f16_a_f32, f32_a_f16};

/// Por debajo de esto un sub-bloque es todo ceros.
const EPSILON: f32 = 1e-15;

#[inline]
fn entero(x: f32) -> i32 {
    x.round() as i32
}

/// Cuantiza al entero de `[-nmax, nmax - 1]` con la escala que minimiza el
/// error ponderado por `x²`. Deja en `l` los enteros corridos a `[0, 2 nmax)` y
/// devuelve la escala.
fn cuantos_q3(x: &[f32; 16], nmax: i32, l: &mut [i32; 16]) -> f32 {
    let (mut mayor, mut amax) = (0f32, 0f32);
    for v in x {
        if v.abs() > amax {
            amax = v.abs();
            mayor = *v;
        }
    }
    if amax < EPSILON {
        l.fill(0);
        return 0.0;
    }
    let iescala = -(nmax as f32) / mayor;
    let (mut sumlx, mut suml2) = (0f32, 0f32);
    for (v, li) in x.iter().zip(l.iter_mut()) {
        *li = entero(iescala * v).clamp(-nmax, nmax - 1);
        let w = v * v;
        sumlx += w * v * *li as f32;
        suml2 += w * (*li * *li) as f32;
    }
    // Afinar: cada valor prueba el entero que le conviene con la escala que
    // sale de los otros quince, y se queda si baja el error.
    for _ in 0..5 {
        let mut cambios = 0;
        for (v, li) in x.iter().zip(l.iter_mut()) {
            let w = v * v;
            let actual = *li as f32;
            let mut slx = sumlx - w * v * actual;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * actual * actual;
                let nuevo = entero(v * sl2 / slx).clamp(-nmax, nmax - 1);
                if nuevo != *li {
                    slx += w * v * nuevo as f32;
                    sl2 += w * (nuevo * nuevo) as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        *li = nuevo;
                        sumlx = slx;
                        suml2 = sl2;
                        cambios += 1;
                    }
                }
            }
        }
        if cambios == 0 {
            break;
        }
    }
    for li in l.iter_mut() {
        *li += nmax;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Cómo busca [`cuantos_qkx`].
struct Busqueda {
    nmax: i32,
    desde: f32,
    paso: f32,
    pasos: i32,
    /// Medir el error absoluto en vez del cuadrático.
    absoluto: bool,
}

/// Cuantiza a `[0, nmax]` con escala y mínimo: `x ≈ escala * l - minimo`. Prueba
/// varias escalas alrededor de la obvia y resuelve el mínimo y la escala por
/// cuadrados mínimos para cada una. Devuelve `(escala, minimo)`.
fn cuantos_qkx(x: &[f32], w: &[f32], l: &mut [i32], aux: &mut [i32], b: &Busqueda) -> (f32, f32) {
    let (mut min, mut max) = (x[0], x[0]);
    let (mut suma_w, mut suma_x) = (w[0], w[0] * x[0]);
    for (v, p) in x.iter().zip(w).skip(1) {
        min = min.min(*v);
        max = max.max(*v);
        suma_w += p;
        suma_x += p * v;
    }
    // El mínimo no puede ser positivo: el formato lo guarda como algo que se
    // resta, sin signo.
    if min > 0.0 {
        min = 0.0;
    }
    if min == max {
        l.fill(0);
        return (0.0, -min);
    }
    let error = |d: f32| if b.absoluto { d.abs() } else { d * d };
    let mut iescala = b.nmax as f32 / (max - min);
    let mut escala = 1.0 / iescala;
    let mut mejor = 0f32;
    for ((v, p), li) in x.iter().zip(w).zip(l.iter_mut()) {
        *li = entero(iescala * (v - min)).clamp(0, b.nmax);
        mejor += p * error(escala * *li as f32 + min - v);
    }
    for k in 0..=b.pasos {
        iescala = (b.desde + b.paso * k as f32 + b.nmax as f32) / (max - min);
        let (mut suma_l, mut suma_l2, mut suma_xl) = (0f32, 0f32, 0f32);
        for ((v, p), a) in x.iter().zip(w).zip(aux.iter_mut()) {
            *a = entero(iescala * (v - min)).clamp(0, b.nmax);
            let lf = *a as f32;
            let wl = p * lf;
            suma_l += wl;
            suma_l2 += wl * lf;
            suma_xl += wl * v;
        }
        let d = suma_w * suma_l2 - suma_l * suma_l;
        if d > 0.0 {
            let mut esta_escala = (suma_w * suma_xl - suma_x * suma_l) / d;
            let mut este_min = (suma_l2 * suma_x - suma_l * suma_xl) / d;
            if este_min > 0.0 {
                este_min = 0.0;
                esta_escala = suma_xl / suma_l2;
            }
            let mut e = 0f32;
            for ((v, p), a) in x.iter().zip(w).zip(aux.iter()) {
                e += p * error(esta_escala * *a as f32 + este_min - v);
            }
            if e < mejor {
                l.copy_from_slice(aux);
                mejor = e;
                escala = esta_escala;
                min = este_min;
            }
        }
    }
    (escala, -min)
}

/// Agrega un bloque Q3_K de 256 valores: 110 bytes.
pub fn bloque_q3k(x: &[f32], salida: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), 256);
    let mut l = [[0i32; 16]; 16];
    let mut escalas = [0f32; 16];
    let (mut mayor, mut amax) = (0f32, 0f32);
    for j in 0..16 {
        let sub: &[f32; 16] = x[j * 16..j * 16 + 16].try_into().expect("mide 16");
        escalas[j] = cuantos_q3(sub, 4, &mut l[j]);
        if escalas[j].abs() > amax {
            amax = escalas[j].abs();
            mayor = escalas[j];
        }
    }

    let mut hmask = [0u8; 32];
    let mut qs = [0u8; 64];
    let mut empaquetadas = [0u8; 12];
    let d_guardada;
    if amax < EPSILON {
        // Un bloque en cero: los enteros en el centro —4 es el 0, con el bit
        // alto puesto— y escala cero.
        d_guardada = 0u16;
        for fila in l.iter_mut() {
            fila.fill(4);
        }
    } else {
        // La escala de cada sub-bloque en 6 bits con signo, en -32..=31.
        let iescala = -32.0 / mayor;
        d_guardada = f32_a_f16(1.0 / iescala);
        let d = f16_a_f32(d_guardada);
        for j in 0..16 {
            let e = entero(iescala * escalas[j]).clamp(-32, 31);
            let sc = d * e as f32;
            if sc != 0.0 {
                let inv = 1.0 / sc;
                for (li, v) in l[j].iter_mut().zip(&x[j * 16..j * 16 + 16]) {
                    *li = entero(v * inv).clamp(-4, 3) + 4;
                }
            } else {
                l[j].fill(4);
            }
            let u = (e + 32) as u8;
            // Los 4 bits bajos en los primeros 8 bytes, de a dos por byte, y
            // los 2 altos en los últimos 4 (ver `quant::escalas_q3k`).
            if j < 8 {
                empaquetadas[j] |= u & 0xF;
            } else {
                empaquetadas[j - 8] |= (u & 0xF) << 4;
            }
            empaquetadas[8 + j % 4] |= (u >> 4) << (2 * (j / 4));
        }
    }

    // Los valores: 2 bits en `qs` y el tercero en `hmask`.
    for (idx, valor) in (0..256).map(|i| (i, l[i / 16][i % 16] as u8)) {
        let mitad = idx / 128;
        let ronda = (idx % 128) / 32;
        let lugar = idx % 32;
        qs[mitad * 32 + lugar] |= (valor & 3) << (ronda * 2);
        if valor & 4 != 0 {
            hmask[lugar] |= 1 << (mitad * 4 + ronda);
        }
    }

    salida.extend_from_slice(&hmask);
    salida.extend_from_slice(&qs);
    salida.extend_from_slice(&empaquetadas);
    salida.extend_from_slice(&d_guardada.to_le_bytes());
}

/// Agrega un bloque Q2_K de 256 valores: 84 bytes.
pub fn bloque_q2k(x: &[f32], salida: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), 256);
    let mut l = [[0i32; 16]; 16];
    let mut aux = [0i32; 16];
    let mut w = [0f32; 16];
    let (mut escalas, mut minimos) = ([0f32; 16], [0f32; 16]);
    let (mut max_esc, mut max_min) = (0f32, 0f32);
    let busqueda = Busqueda {
        nmax: 3,
        desde: -0.5,
        paso: 0.1,
        pasos: 15,
        absoluto: true,
    };
    for j in 0..16 {
        let sub = &x[j * 16..j * 16 + 16];
        for (p, v) in w.iter_mut().zip(sub) {
            *p = v.abs();
        }
        let (e, m) = cuantos_qkx(sub, &w, &mut l[j], &mut aux, &busqueda);
        escalas[j] = e;
        minimos[j] = m;
        max_esc = max_esc.max(e);
        max_min = max_min.max(m);
    }

    // Escala y mínimo de cada sub-bloque, en 4 bits cada uno.
    let inv_esc = if max_esc > 0.0 { 15.0 / max_esc } else { 0.0 };
    let inv_min = if max_min > 0.0 { 15.0 / max_min } else { 0.0 };
    let d_guardada = f32_a_f16(if max_esc > 0.0 { 1.0 / inv_esc } else { 0.0 });
    let m_guardado = f32_a_f16(if max_min > 0.0 { 1.0 / inv_min } else { 0.0 });
    let (d, dm) = (f16_a_f32(d_guardada), f16_a_f32(m_guardado));

    let mut paquete = [0u8; 16];
    for j in 0..16 {
        let le = entero(inv_esc * escalas[j]).clamp(0, 15);
        let lm = entero(inv_min * minimos[j]).clamp(0, 15);
        paquete[j] = (le | (lm << 4)) as u8;
        let (dl, ml) = (d * le as f32, dm * lm as f32);
        for (li, v) in l[j].iter_mut().zip(&x[j * 16..j * 16 + 16]) {
            *li = if dl > 0.0 {
                entero((v + ml) / dl).clamp(0, 3)
            } else {
                0
            };
        }
    }

    let mut qs = [0u8; 64];
    for idx in 0..256 {
        let mitad = idx / 128;
        let ronda = (idx % 128) / 32;
        let valor = l[idx / 16][idx % 16] as u8;
        qs[mitad * 32 + idx % 32] |= valor << (ronda * 2);
    }

    salida.extend_from_slice(&paquete);
    salida.extend_from_slice(&qs);
    salida.extend_from_slice(&d_guardada.to_le_bytes());
    salida.extend_from_slice(&m_guardado.to_le_bytes());
}

/// Lo que comparten Q4_K y Q5_K: ocho sub-bloques de 32, con escala y mínimo en
/// 6 bits. Devuelve las dos escalas del bloque, el paquete y los enteros.
fn sub_bloques_de_32(x: &[f32], b: &Busqueda) -> ([u8; 2], [u8; 2], [u8; 12], [[i32; 32]; 8]) {
    let mut l = [[0i32; 32]; 8];
    let mut aux = [0i32; 32];
    let mut w = [0f32; 32];
    let (mut escalas, mut minimos) = ([0f32; 8], [0f32; 8]);
    let (mut max_esc, mut max_min) = (0f32, 0f32);
    let media = (x.iter().map(|v| v * v).sum::<f32>() / 256.0).sqrt();
    for j in 0..8 {
        let sub = &x[j * 32..j * 32 + 32];
        for (p, v) in w.iter_mut().zip(sub) {
            *p = media + v.abs();
        }
        let (e, m) = cuantos_qkx(sub, &w, &mut l[j], &mut aux, b);
        escalas[j] = e;
        minimos[j] = m;
        max_esc = max_esc.max(e);
        max_min = max_min.max(m);
    }

    let inv_esc = if max_esc > 0.0 { 63.0 / max_esc } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let d_guardada = f32_a_f16(if max_esc > 0.0 { 1.0 / inv_esc } else { 0.0 });
    let m_guardado = f32_a_f16(if max_min > 0.0 { 1.0 / inv_min } else { 0.0 });
    let (d, dm) = (f16_a_f32(d_guardada), f16_a_f32(m_guardado));

    let mut paquete = [0u8; 12];
    for j in 0..8 {
        let le = entero(inv_esc * escalas[j]).clamp(0, 63) as u8;
        let lm = entero(inv_min * minimos[j]).clamp(0, 63) as u8;
        // Ver `quant::escala_y_minimo`.
        if j < 4 {
            paquete[j] |= le;
            paquete[j + 4] |= lm;
        } else {
            paquete[j + 4] = (le & 0xF) | ((lm & 0xF) << 4);
            paquete[j - 4] |= (le >> 4) << 6;
            paquete[j] |= (lm >> 4) << 6;
        }
        let (dl, ml) = (d * le as f32, dm * lm as f32);
        for (li, v) in l[j].iter_mut().zip(&x[j * 32..j * 32 + 32]) {
            *li = if dl > 0.0 {
                entero((v + ml) / dl).clamp(0, b.nmax)
            } else {
                0
            };
        }
    }
    (
        d_guardada.to_le_bytes(),
        m_guardado.to_le_bytes(),
        paquete,
        l,
    )
}

/// Agrega un bloque Q4_K de 256 valores: 144 bytes.
pub fn bloque_q4k(x: &[f32], salida: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), 256);
    let busqueda = Busqueda {
        nmax: 15,
        desde: -1.0,
        paso: 0.1,
        pasos: 20,
        absoluto: false,
    };
    let (d, m, paquete, l) = sub_bloques_de_32(x, &busqueda);
    salida.extend_from_slice(&d);
    salida.extend_from_slice(&m);
    salida.extend_from_slice(&paquete);
    for [bajo, alto] in l.as_chunks::<2>().0 {
        for (b, a) in bajo.iter().zip(alto) {
            salida.push((b | (a << 4)) as u8);
        }
    }
}

/// Agrega un bloque Q5_K de 256 valores: 176 bytes. Como Q4_K, con el quinto bit
/// de cada valor aparte, en `qh`.
pub fn bloque_q5k(x: &[f32], salida: &mut Vec<u8>) {
    debug_assert_eq!(x.len(), 256);
    let busqueda = Busqueda {
        nmax: 31,
        desde: -0.5,
        paso: 0.1,
        pasos: 15,
        absoluto: false,
    };
    let (d, m, paquete, l) = sub_bloques_de_32(x, &busqueda);
    let mut qh = [0u8; 32];
    for (p, [bajo, alto]) in l.as_chunks::<2>().0.iter().enumerate() {
        for ((h, b), a) in qh.iter_mut().zip(bajo).zip(alto) {
            *h |= ((b >> 4) as u8) << (2 * p);
            *h |= ((a >> 4) as u8) << (2 * p + 1);
        }
    }
    salida.extend_from_slice(&d);
    salida.extend_from_slice(&m);
    salida.extend_from_slice(&paquete);
    salida.extend_from_slice(&qh);
    for [bajo, alto] in l.as_chunks::<2>().0 {
        for (b, a) in bajo.iter().zip(alto) {
            salida.push(((b & 0xF) | ((a & 0xF) << 4)) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::gguf::TipoGgml;
    use crate::quant::{cuantizar_bloques, descuantizar};

    /// Normales reproducibles, como las que inicializan un modelo.
    fn normales(n: usize, sigma: f32) -> Vec<f32> {
        let mut azar = crate::daiss::Azar::con_semilla(7);
        (0..n).map(|_| azar.normales(sigma).0).collect()
    }

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32).sqrt()
    }

    fn ida_y_vuelta(tipo: TipoGgml, x: &[f32]) -> Vec<f32> {
        let mut b = Vec::new();
        cuantizar_bloques(tipo, x, &mut b).unwrap();
        assert_eq!(b.len(), tipo.bytes_de(x.len()), "{tipo:?}");
        let mut y = vec![0f32; x.len()];
        descuantizar(tipo, &b, &mut y).unwrap();
        y
    }

    /// El error relativo que da cada formato con pesos normales: más bits,
    /// menos error, y todos dentro de lo que su cantidad de bits permite.
    #[test]
    fn el_error_es_el_que_corresponde_a_cada_formato() {
        let x = normales(256 * 64, 0.02);
        let e = |t| rmse(&x, &ida_y_vuelta(t, &x)) / 0.02;
        let (q2, q3, q4) = (e(TipoGgml::Q2K), e(TipoGgml::Q3K), e(TipoGgml::Q4K));
        assert!(q4 < q3 && q3 < q2, "{q4} {q3} {q2}");
        assert!(e(TipoGgml::Q5K) < q4 / 1.6, "Q5_K: {}", e(TipoGgml::Q5K));
        assert!(q4 < 0.09, "Q4_K: {q4}");
        assert!(q3 < 0.2, "Q3_K: {q3}");
        assert!(q2 < 0.42, "Q2_K: {q2}");
    }

    #[test]
    fn lo_que_no_tiene_informacion_vuelve_igual() {
        for tipo in [TipoGgml::Q2K, TipoGgml::Q3K, TipoGgml::Q4K, TipoGgml::Q5K] {
            let ceros = vec![0f32; 256];
            assert_eq!(ida_y_vuelta(tipo, &ceros), ceros, "{tipo:?}");
            // Un sub-bloque en cero en medio de valores: no se contagia.
            let mut x = normales(256, 0.5);
            x[32..64].fill(0.0);
            let y = ida_y_vuelta(tipo, &x);
            assert!(rmse(&x, &y) < 0.3, "{tipo:?}");
            assert!(y.iter().all(|v| v.is_finite()), "{tipo:?}");
        }
    }

    /// Valores muy desparejos dentro del bloque —un sub-bloque grande y otro
    /// chico— y todos positivos o todos negativos.
    #[test]
    fn soporta_escalas_desparejas_y_un_solo_signo() {
        for tipo in [TipoGgml::Q2K, TipoGgml::Q3K, TipoGgml::Q4K, TipoGgml::Q5K] {
            let mut x = normales(256, 0.001);
            for v in &mut x[..16] {
                *v *= 1000.0;
            }
            let y = ida_y_vuelta(tipo, &x);
            assert!(rmse(&x[..16], &y[..16]) < 0.5, "{tipo:?}");
            assert!(y.iter().all(|v| v.is_finite()), "{tipo:?}");

            let pos: Vec<f32> = normales(256, 0.1).iter().map(|v| v.abs()).collect();
            let neg: Vec<f32> = pos.iter().map(|v| -v).collect();
            assert!(
                rmse(&pos, &ida_y_vuelta(tipo, &pos)) < 0.05,
                "{tipo:?} positivos"
            );
            assert!(
                rmse(&neg, &ida_y_vuelta(tipo, &neg)) < 0.05,
                "{tipo:?} negativos"
            );
        }
    }

    /// Cada byte del empaquetado llega a su lugar: la escala de cada
    /// sub-bloque tiene que afectar a su sub-bloque y a ningún otro.
    #[test]
    fn cada_sub_bloque_conserva_su_escala() {
        for (tipo, sub) in [
            (TipoGgml::Q2K, 16),
            (TipoGgml::Q3K, 16),
            (TipoGgml::Q4K, 32),
            (TipoGgml::Q5K, 32),
        ] {
            let n = 256 / sub;
            let mut x = vec![0f32; 256];
            for j in 0..n {
                let amplitud = 0.1 * (1.0 + j as f32 / n as f32);
                for (i, v) in x[j * sub..(j + 1) * sub].iter_mut().enumerate() {
                    *v = if i % 2 == 0 { amplitud } else { -amplitud };
                }
            }
            let y = ida_y_vuelta(tipo, &x);
            for j in 0..n {
                let (a, b) = (&x[j * sub..(j + 1) * sub], &y[j * sub..(j + 1) * sub]);
                assert!(rmse(a, b) < 0.06, "{tipo:?} sub-bloque {j}: {}", rmse(a, b));
            }
        }
    }
}
