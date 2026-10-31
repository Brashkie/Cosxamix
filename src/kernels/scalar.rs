//! La implementación de referencia: Rust portable, sin intrínsecos.
//!
//! Esta es la que define qué resultado es el correcto. Corre en cualquier
//! arquitectura y **nunca se borra**, por dos motivos:
//!
//! * Es el respaldo cuando la CPU no tiene nada mejor — un ARM viejo, un x86
//!   sin AVX2, lo que sea. Un binario que solo trae kernels vectorizados no
//!   arranca ahí.
//! * Es contra la que se comparan todas las demás, en corrección y en
//!   velocidad. Sin ella no hay forma de saber si un kernel en otro lenguaje
//!   aporta algo o solo parece que sí.
//!
//! No se intenta que sea rápida. Se intenta que sea obviamente correcta.

/// Producto punto, elemento a elemento.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // `zip` acota los dos a la vez, así que el compilador se ahorra comprobar
    // el índice en cada vuelta. Se deja que autovectorice si quiere: el punto
    // de esta versión es ser legible, no impedirle al compilador que ayude.
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalización RMS, la que usan los transformers modernos en cada capa.
///
/// `salida[i] = entrada[i] / sqrt(media(entrada²) + eps) * peso[i]`
///
/// A diferencia de LayerNorm no resta la media: solo escala por la raíz
/// cuadrática media. Eso la hace más barata y es el motivo de que se haya
/// impuesto.
///
/// `eps` evita dividir por cero cuando el vector es todo ceros, que pasa de
/// verdad en las primeras iteraciones de un entrenamiento y en vectores de
/// relleno.
pub fn rmsnorm(salida: &mut [f32], entrada: &[f32], peso: &[f32], eps: f32) {
    debug_assert_eq!(salida.len(), entrada.len());
    debug_assert_eq!(entrada.len(), peso.len());

    if entrada.is_empty() {
        return;
    }

    // Se acumula en f64 a propósito. Con vectores de miles de elementos, la
    // suma de cuadrados en f32 pierde precisión de forma visible, y esto es
    // la REFERENCIA: su trabajo es ser correcta, no rápida. Los backends
    // vectorizados acumulan en f32 y por eso se comparan con tolerancia.
    let suma: f64 = entrada.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let escala = 1.0 / ((suma / entrada.len() as f64) + eps as f64).sqrt();

    for ((s, &e), &p) in salida.iter_mut().zip(entrada).zip(peso) {
        *s = (e as f64 * escala) as f32 * p;
    }
}

/// SiLU de referencia: `x * sigmoid(x)`, o sea `x / (1 + exp(-x))`.
///
/// Es la activacion del MLP de Qwen2, y corre sobre `intermedio` elementos
/// —11008 en el 3B— por capa y por token.
///
/// Sin trucos a proposito: esta implementacion define que resultado es el
/// correcto, y para eso vale mas ser evidente que ser rapida.
pub fn silu(salida: &mut [f32], entrada: &[f32]) {
    for (s, &x) in salida.iter_mut().zip(entrada) {
        *s = x / (1.0 + (-x).exp());
    }
}

/// Producto punto de dos vectores de `i8`, acumulando en `i32`.
///
/// Es la operacion que de verdad hace una multiplicacion cuantizada: los pesos
/// y la activacion se guardan como enteros chicos y una escala aparte, y el
/// producto se hace en enteros. Un `i32` no desborda ni cerca -- el peor caso
/// con 2048 elementos de 127 es 33 millones contra los 2147 millones que
/// aguanta.
///
/// Define que resultado es el correcto. Al ser enteros, "correcto" aca es
/// exacto: las otras implementaciones tienen que dar el MISMO numero, no uno
/// parecido.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| *x as i32 * *y as i32).sum()
}

/// Producto punto de dos vectores cuantizados por bloques.
///
/// Cada bloque de 32 trae su propia escala en los dos lados, asi que el
/// resultado es `suma_b escala_a[b] * escala_b[b] * dot_i8(bloque_b)`.
///
/// Existe como una sola operacion y no como `dot_i8` llamado en un bucle por
/// una razon medida: `dot_i8` termina con una suma horizontal, que es barata
/// sobre un vector largo y carisima repetida una vez por bloque. Componer el
/// kernel equivocado costo que la version en enteros saliera **2x mas lenta**
/// que la de f32, pese a que el kernel entero es 4x mas rapido.
pub fn dot_i8_escalado(a: &[i8], b: &[i8], esc_a: &[f32], esc_b: &[f32], bloque: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut total = 0f32;
    for (i, (ea, eb)) in esc_a.iter().zip(esc_b).enumerate() {
        let r = i * bloque;
        let hasta = (r + bloque).min(a.len());
        let parcial: i32 = a[r..hasta]
            .iter()
            .zip(&b[r..hasta])
            .map(|(x, y)| *x as i32 * *y as i32)
            .sum();
        total += ea * eb * parcial as f32;
    }
    total
}

/// Producto punto de una fila en **Q4_K** contra una entrada ya cuantizada a
/// `i8`, **sin materializar los pesos**.
///
/// Es la referencia del kernel fusionado. El camino de hoy desempaqueta la
/// fila entera a un buffer y después multiplica; medido, ese desempaquetado es
/// el 94% del trabajo de un hilo. Fusionar quita el buffer: cada sub-bloque se
/// desempaqueta y se consume en el acto.
///
/// La aritmética que hay que reproducir exactamente es la de
/// `ops::matmul_i8_directo`:
///
/// ```text
///     total = Σ_b  xe[b] · ( d·sc_b · Σ_i xq_i·q_i  +  (-dmin·m_b) · xs[b] )
/// ```
///
/// donde `xs[b]` es la suma de los `xq` del sub-bloque `b`. El desplazamiento
/// sale afuera del producto punto porque `Σ x·(s·q + o)` es
/// `s·Σ(x·q) + o·Σx`, y la suma de la entrada se calcula una vez por fila.
///
/// `bloques` son los 144 bytes por cada 256 valores; `xq`, `xe` y `xs` cubren
/// la fila entera, con un sub-bloque de 32.
pub fn dot_q4k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());

    let mut total = 0f32;
    for bl in 0..xq.len() / 256 {
        let o = bl * 144;
        let d = crate::quant::leer_f16(bloques, o);
        let dmin = crate::quant::leer_f16(bloques, o + 2);
        let esc = &bloques[o + 4..o + 16];
        let qs = &bloques[o + 16..o + 144];

        for par in 0..4 {
            let base = par * 32;
            for mitad in 0..2 {
                let j = par * 2 + mitad;
                let (sc, mn) = crate::quant::escala_y_minimo(j, esc);
                let sub = bl * 8 + j;
                let x = &xq[bl * 256 + par * 64 + mitad * 32..][..32];

                let mut entero = 0i32;
                for (i, v) in x.iter().enumerate() {
                    // Los nibbles bajos son el sub-bloque par y los altos el
                    // impar, que es como lo guarda el formato.
                    let q = if mitad == 0 {
                        qs[base + i] & 0x0F
                    } else {
                        qs[base + i] >> 4
                    };
                    entero += *v as i32 * q as i32;
                }

                total += xe[sub] * (d * sc as f32 * entero as f32 - dmin * mn as f32 * xs[sub]);
            }
        }
    }
    total
}

/// Producto punto de una fila en **Q3_K** contra una entrada en `i8`, sin
/// materializar los pesos. Referencia del kernel fusionado.
///
/// Q3_K necesita un rodeo que Q4_K no: sus valores van de **−4 a 3**, con
/// signo, y `vpmaddubsw` quiere un operando sin signo. El rodeo es sumarles 4
/// —con lo que quedan en 0 a 7— y descontarlo después, porque
///
/// ```text
///     Σ x·q  =  Σ x·(q + 4)  −  4·Σx
/// ```
///
/// y `Σx` por sub-bloque es exactamente `xs`, que el multiplicador ya calcula
/// una vez por fila para el término del desplazamiento. O sea que la
/// corrección sale gratis.
///
/// El sub-bloque de Q3_K es de **16** y no de 32, así que hay dos escalas por
/// cada 32 valores.
pub fn dot_q3k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 16);
    debug_assert_eq!(xs.len(), xe.len());

    let mut total = 0f32;
    for bl in 0..xq.len() / 256 {
        let o = bl * 110;
        let hmask = &bloques[o..o + 32];
        let qs = &bloques[o + 32..o + 96];
        let d = crate::quant::leer_f16(bloques, o + 108);
        let esc = crate::quant::escalas_q3k(bloques, o);

        // Ocho grupos de 32 valores. En cada uno cambian el corrimiento de los
        // dos bits bajos y el bit de `hmask` que aporta el tercero.
        for g in 0..8 {
            let qsg = &qs[(g / 4) * 32..][..32];
            let corrimiento = 2 * (g % 4);
            let m = 1u8 << g;
            let x = &xq[bl * 256 + g * 32..][..32];

            for medio in 0..2 {
                let sub = bl * 16 + g * 2 + medio;
                let escala = d * (esc[g * 2 + medio] as f32 - 32.0);
                let mut entero = 0i32;
                for l in 0..16 {
                    let idx = medio * 16 + l;
                    // El tercer bit vive en `hmask`, y cuando esta puesto el
                    // valor sube 4. Sumarselo es lo que lo deja sin signo.
                    let extra = if hmask[idx] & m != 0 { 4i32 } else { 0 };
                    let q = ((qsg[idx] >> corrimiento) & 3) as i32 + extra;
                    entero += x[idx] as i32 * q;
                }
                // Y aca se descuenta el 4 que se le sumo a cada valor.
                total += xe[sub] * escala * (entero as f32 - 4.0 * xs[sub]);
            }
        }
    }
    total
}

/// Producto de una fila Q6_K contra una entrada cuantizada en sub-bloques de
/// 16, sin desempaquetar a ningún buffer. Referencia de
/// [`super::x86::dot_q6k_i8_avx2`].
///
/// Un bloque de 256 son 128 bytes de nibbles bajos (`ql`), 64 de pares de
/// bits altos (`qh`), 16 escalas `i8` y una `d`. Cada mitad de 128 valores
/// son cuatro grupos de 32 seguidos; el grupo `g` de la mitad toma:
///
/// | `g` | 4 bits bajos | 2 bits altos |
/// |---|---|---|
/// | 0 | nibble bajo de `ql[0..32]` | bits 0-1 de `qh` |
/// | 1 | nibble bajo de `ql[32..64]` | bits 2-3 |
/// | 2 | nibble alto de `ql[0..32]` | bits 4-5 |
/// | 3 | nibble alto de `ql[32..64]` | bits 6-7 |
///
/// El valor va de 0 a 63 y se le resta 32, que acá se descuenta con la suma
/// de la entrada, igual que el 4 de Q3_K.
pub fn dot_q6k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 16);
    debug_assert_eq!(xs.len(), xe.len());

    let mut total = 0f32;
    for bl in 0..xq.len() / 256 {
        let o = bl * 210;
        let esc = &bloques[o + 192..o + 208];
        let d = crate::quant::leer_f16(bloques, o + 208);
        for g in 0..8 {
            let (mitad, gi) = (g / 4, g % 4);
            let ql = &bloques[o + 64 * mitad + 32 * (gi % 2)..][..32];
            let qh = &bloques[o + 128 + 32 * mitad..][..32];
            let x = &xq[bl * 256 + g * 32..][..32];
            for medio in 0..2 {
                let sub = bl * 16 + g * 2 + medio;
                let escala = d * esc[g * 2 + medio] as i8 as f32;
                let mut entero = 0i32;
                for l in 0..16 {
                    let i = medio * 16 + l;
                    let nib = if gi < 2 { ql[i] & 0x0F } else { ql[i] >> 4 };
                    let q = (nib | (((qh[i] >> (2 * gi)) & 3) << 4)) as i32;
                    entero += x[i] as i32 * q;
                }
                total += xe[sub] * escala * (entero as f32 - 32.0 * xs[sub]);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caso_a_mano() {
        // Calculado aparte: 1*4 + 2*5 + 3*6 = 32.
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn vacio() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    fn los_signos_no_se_pierden() {
        assert_eq!(dot(&[1.0, -2.0], &[3.0, 4.0]), 3.0 - 8.0);
    }
    #[test]
    fn rmsnorm_con_pesos_en_uno_normaliza() {
        // Con peso 1, la salida tiene que tener RMS ~1.
        let e = [1.0f32, 2.0, 3.0, 4.0];
        let p = [1.0f32; 4];
        let mut s = [0.0f32; 4];
        rmsnorm(&mut s, &e, &p, 1e-6);
        let rms = (s.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms={rms}");
    }

    #[test]
    fn rmsnorm_aplica_el_peso() {
        let e = [1.0f32, 1.0, 1.0, 1.0];
        let p = [2.0f32, 3.0, 4.0, 5.0];
        let mut s = [0.0f32; 4];
        rmsnorm(&mut s, &e, &p, 1e-6);
        // Con entrada constante, la normalizacion da 1 y queda el peso tal cual.
        for (a, b) in s.iter().zip(&p) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    fn rmsnorm_todo_ceros_no_divide_por_cero() {
        // Pasa de verdad con vectores de relleno. Sin eps seria NaN.
        let e = [0.0f32; 8];
        let p = [1.0f32; 8];
        let mut s = [9.9f32; 8];
        rmsnorm(&mut s, &e, &p, 1e-6);
        assert!(s.iter().all(|x| x.is_finite()), "{s:?}");
    }
}
