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
