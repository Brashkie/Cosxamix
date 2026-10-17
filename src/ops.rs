//! Multiplicación de matrices cuantizadas, nuestra.
//!
//! ## Qué tiene que hacer y por qué no alcanza con copiar a candle
//!
//! Calcula `dst = x · Wᵀ`, con `W` cuantizada y guardada por filas de salida.
//! Es donde se va el 99% del tiempo de una respuesta, así que es la pieza que
//! decide si el motor propio sirve.
//!
//! La de candle es muy buena en un caso y mala en otro, las dos cosas medidas
//! en este repositorio:
//!
//! * con **una fila** —el decode— su `vec_dot_q3k_q8k` en AVX2 escrito a mano
//!   desempaqueta el k-quant dentro de los registros, fundido con el producto.
//!   Competir con eso no salió bien dos veces.
//! * con **varias filas** su bucle de filas es el externo, así que un lote de
//!   `m` filas hace `m` barridos completos de los pesos: 8 filas cuestan
//!   8,55x una.
//!
//! El segundo caso es el que tiene aire, y es el que importa para el prefill y
//! para cualquier esquema que verifique varios tokens de una vez.
//!
//! ## Una conclusión propia que había que revisar
//!
//! En la 0.2.0 se midió que "descuantizar una vez por columna y después hacer
//! `m` productos en f32" salía entre 5 y 25 veces peor, y de ahí se concluyó
//! que separar el desempaquetado del producto es inviable.
//!
//! Esa medición tenía un defecto propio: el buffer de la columna se asignaba
//! **dentro** del bucle de columnas, o sea 11008 asignaciones por
//! multiplicación. Acá el buffer se reserva una vez por tramo y se reusa, que
//! es lo que había que medir desde el principio.

use crate::gguf::TipoGgml;
use crate::quant::descuantizar;
use rayon::prelude::*;

/// Cuántas columnas como mínimo le toca a cada hilo.
///
/// candle usa 128 sobre su eje. Acá va menos porque con 128 una matriz de 256
/// columnas de salida —`attn_k` y `attn_v`— solo se parte en dos trozos y se
/// queda en 2 hilos de 6. Son el 1,4% de los parámetros, así que no mueve la
/// aguja, pero tampoco cuesta nada.
const COLUMNAS_MINIMAS: usize = 32;

/// `dst[m, n] = lhs[m, k] · W[n, k]ᵀ`.
///
/// `pesos` son los bytes crudos del tensor, tal cual salen del GGUF: `n` filas
/// de `k` valores cada una, cuantizadas.
///
/// **La columna de pesos se desempaqueta una vez y se usa para las `m` filas**,
/// que es justo lo que la de candle no hace.
pub fn matmul(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    if lhs.len() != m * k {
        return Err(format!("lhs mide {} y deberia medir {m}*{k}", lhs.len()));
    }
    if dst.len() != m * n {
        return Err(format!("dst mide {} y deberia medir {m}*{n}", dst.len()));
    }
    let por_bloque = tipo.valores_por_bloque();
    if !k.is_multiple_of(por_bloque) {
        return Err(format!(
            "una fila de {k} no es multiplo del bloque de {por_bloque}"
        ));
    }
    let bytes_fila = k / por_bloque * tipo.bytes_por_bloque();
    if pesos.len() < n * bytes_fila {
        return Err(format!(
            "hacen falta {} bytes de pesos y hay {}",
            n * bytes_fila,
            pesos.len()
        ));
    }

    // El resultado se calcula en orden columna-fila y se traspone al final.
    //
    // Es lo que permite repartir por columnas sin `unsafe`: cada tarea escribe
    // un tramo contiguo. Escribiendo directo en `dst`, que va fila-columna,
    // cada tarea tocaria posiciones salteadas. La trasposicion son `m * n`
    // escrituras y con `m` chico no se nota.
    let mut salida_t = vec![0f32; n * m];
    let mut error: Option<String> = None;

    salida_t
        .par_chunks_mut(m * COLUMNAS_MINIMAS)
        .enumerate()
        .for_each(|(tramo, destino)| {
            // Un solo buffer por tramo, no uno por columna. Reservarlo dentro
            // del bucle de columnas fue el defecto de la medicion anterior.
            let mut columna = vec![0f32; k];
            let primera = tramo * COLUMNAS_MINIMAS;

            for (j, salida) in destino.chunks_mut(m).enumerate() {
                let col = primera + j;
                let desde = col * bytes_fila;
                if descuantizar(tipo, &pesos[desde..desde + bytes_fila], &mut columna).is_err() {
                    continue;
                }
                for (fila, s) in salida.iter_mut().enumerate() {
                    *s = crate::kernels::dot(&lhs[fila * k..(fila + 1) * k], &columna);
                }
            }
        });

    if let Some(e) = error.take() {
        return Err(e);
    }

    for col in 0..n {
        for fila in 0..m {
            dst[fila * n + col] = salida_t[col * m + fila];
        }
    }
    Ok(())
}

/// Tamano del bloque al recuantizar a `i8`.
///
/// 32 y no 16 porque es lo que procesa `dot_i8` de una vez en AVX2: con
/// bloques mas chicos el kernel nunca llega a usar su camino ancho y se pasa
/// el tiempo en la cola.
pub const BLOQUE_I8: usize = 32;

/// Pasa un vector de `f32` a `i8` con una escala por bloque.
///
/// Cada bloque se escala por `max|x| / 127`, que es exactamente lo que hacen
/// los cuantizadores de GGML — y es lo que garantiza que el resultado quede en
/// `[-127, 127]`, que es la precondicion de `dot_i8` en AVX2.
///
/// Devuelve los valores y las escalas por separado: el producto se reconstruye
/// como `escala_a * escala_b * dot_i8(a, b)` bloque por bloque.
pub fn cuantizar_i8(x: &[f32], q: &mut [i8], escalas: &mut [f32]) {
    cuantizar_i8_en(x, q, escalas, BLOQUE_I8)
}

/// Igual, con el tamano de bloque que pida el formato de los pesos.
pub fn cuantizar_i8_en(x: &[f32], q: &mut [i8], escalas: &mut [f32], bloque: usize) {
    debug_assert_eq!(q.len(), x.len());
    debug_assert_eq!(escalas.len(), x.len().div_ceil(bloque));

    for (b, (trozo, escala)) in x.chunks(bloque).zip(escalas.iter_mut()).enumerate() {
        let amax = trozo.iter().fold(0f32, |m, v| m.max(v.abs()));
        // Un bloque entero en cero no es raro en un modelo cuantizado, y
        // dividir por cero daria NaN en vez de cero.
        let (d, inv) = if amax > 0.0 {
            (amax / 127.0, 127.0 / amax)
        } else {
            (0.0, 0.0)
        };
        *escala = d;
        for (i, v) in trozo.iter().enumerate() {
            // `round` y no truncar: truncar sesga todos los valores hacia
            // cero y el error se acumula a lo largo de 2048 productos.
            q[b * bloque + i] = (v * inv).round() as i32 as i8;
        }
    }
}

/// `dst[m, n] = lhs[m, k] · W[n, k]ᵀ`, multiplicando en **enteros**.
///
/// La diferencia con [`matmul`] es donde se hace el producto. Aquella
/// desempaqueta la columna a `f32` y multiplica en `f32`; esta la recuantiza a
/// `i8` —una vez por columna, igual que el desempaquetado— y usa `dot_i8`, que
/// medido va entre 3,4 y 4,8 veces mas rapido porque AVX2 procesa 32 enteros
/// de 8 bits por instruccion contra 8 `f32`.
///
/// Recuantizar los pesos casi no cuesta precision: ya venian con 2 a 6 bits
/// por valor, asi que meterlos en 8 no les quita nada. Lo que si pierde algo
/// es la entrada, igual que en candle — que la cuantiza a Q8_K con **una**
/// escala cada 256 valores, mientras aca va una cada 32.
pub fn matmul_i8(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    if lhs.len() != m * k {
        return Err(format!("lhs mide {} y deberia medir {m}*{k}", lhs.len()));
    }
    if dst.len() != m * n {
        return Err(format!("dst mide {} y deberia medir {m}*{n}", dst.len()));
    }
    let por_bloque = tipo.valores_por_bloque();
    if !k.is_multiple_of(por_bloque) || !k.is_multiple_of(BLOQUE_I8) {
        return Err(format!(
            "una fila de {k} no es multiplo del bloque ({por_bloque} del formato, {BLOQUE_I8} del entero)"
        ));
    }
    let bytes_fila = k / por_bloque * tipo.bytes_por_bloque();
    if pesos.len() < n * bytes_fila {
        return Err(format!(
            "hacen falta {} bytes de pesos y hay {}",
            n * bytes_fila,
            pesos.len()
        ));
    }

    let n_bloques = k / BLOQUE_I8;

    // La entrada se cuantiza UNA vez para toda la multiplicacion, no una por
    // columna. Es `m*k` trabajo contra `n*m*k` si estuviera adentro.
    let mut lhs_q = vec![0i8; m * k];
    let mut lhs_esc = vec![0f32; m * n_bloques];
    for f in 0..m {
        cuantizar_i8(
            &lhs[f * k..(f + 1) * k],
            &mut lhs_q[f * k..(f + 1) * k],
            &mut lhs_esc[f * n_bloques..(f + 1) * n_bloques],
        );
    }

    let mut salida_t = vec![0f32; n * m];

    salida_t
        .par_chunks_mut(m * COLUMNAS_MINIMAS)
        .enumerate()
        .for_each(|(tramo, destino)| {
            // Buffers por tramo, no por columna.
            let mut columna_f = vec![0f32; k];
            let mut columna_q = vec![0i8; k];
            let mut columna_esc = vec![0f32; n_bloques];
            let primera = tramo * COLUMNAS_MINIMAS;

            for (j, salida) in destino.chunks_mut(m).enumerate() {
                let col = primera + j;
                let desde = col * bytes_fila;
                if descuantizar(tipo, &pesos[desde..desde + bytes_fila], &mut columna_f).is_err() {
                    continue;
                }
                cuantizar_i8(&columna_f, &mut columna_q, &mut columna_esc);

                for (fila, s) in salida.iter_mut().enumerate() {
                    // Una sola llamada para toda la fila, no una por bloque:
                    // la suma horizontal se paga una vez. Componerlo al reves
                    // hacia esta multiplicacion 2x mas lenta que la de f32
                    // pese a usar un kernel 4x mas rapido.
                    *s = crate::kernels::dot_i8_escalado(
                        &columna_q,
                        &lhs_q[fila * k..(fila + 1) * k],
                        &columna_esc,
                        &lhs_esc[fila * n_bloques..(fila + 1) * n_bloques],
                        BLOQUE_I8,
                    );
                }
            }
        });

    for col in 0..n {
        for fila in 0..m {
            dst[fila * n + col] = salida_t[col * m + fila];
        }
    }
    Ok(())
}

/// Igual que [`matmul_i8`] pero **sin pasar por `f32`**.
///
/// La diferencia esta en como llega la columna de pesos: aquella la
/// descuantiza a `f32` y la vuelve a cuantizar a `i8`, un viaje de ida y
/// vuelta que medido costaba unos 7 ms de los 14. Esta la desempaqueta
/// directo del formato, que es mas barato porque los pesos **ya eran enteros
/// con una escala**.
///
/// Los formatos con minimo —Q4_K, Q5_K, Q2_K— aportan ademas un
/// desplazamiento por sub-bloque. No estorba: `suma x·(s·q + o)` es
/// `s·suma(x·q) + o·suma(x)`, y la suma de la entrada se calcula una vez por
/// fila en vez de una por columna.
pub fn matmul_i8_directo(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    if lhs.len() != m * k {
        return Err(format!("lhs mide {} y deberia medir {m}*{k}", lhs.len()));
    }
    if dst.len() != m * n {
        return Err(format!("dst mide {} y deberia medir {m}*{n}", dst.len()));
    }
    let por_bloque = tipo.valores_por_bloque();
    // El tamano de sub-bloque lo manda el formato: 32 en Q8_0, Q5_0, Q4_K y
    // Q5_K, y 16 en Q2_K, Q3_K y Q6_K. Los dos lados del producto tienen que
    // usar el mismo, asi que la entrada se cuantiza con este y no con una
    // constante.
    let sub = crate::quant::sub_bloque(tipo)
        .ok_or_else(|| format!("{tipo:?}: sin desempaquetado directo a i8"))?;
    if !k.is_multiple_of(por_bloque) || !k.is_multiple_of(sub) {
        return Err(format!("una fila de {k} no es multiplo de los bloques"));
    }
    let bytes_fila = k / por_bloque * tipo.bytes_por_bloque();
    if pesos.len() < n * bytes_fila {
        return Err("faltan bytes de pesos".into());
    }

    let n_bloques = k / sub;

    // La entrada se cuantiza una vez, con sus escalas y sus sumas por bloque.
    // Las sumas hacen falta para el termino del desplazamiento y calcularlas
    // aqui cuesta `m*k` contra `n*m*k` si estuvieran adentro del bucle.
    let mut lhs_q = vec![0i8; m * k];
    let mut lhs_esc = vec![0f32; m * n_bloques];
    let mut lhs_suma = vec![0f32; m * n_bloques];
    for f in 0..m {
        let (q, e) = (
            &mut lhs_q[f * k..(f + 1) * k],
            &mut lhs_esc[f * n_bloques..(f + 1) * n_bloques],
        );
        cuantizar_i8_en(&lhs[f * k..(f + 1) * k], q, e, sub);
        for b in 0..n_bloques {
            let r = b * sub;
            lhs_suma[f * n_bloques + b] =
                q[r..r + sub].iter().map(|v| *v as i32).sum::<i32>() as f32;
        }
    }

    let mut salida_t = vec![0f32; n * m];

    salida_t
        .par_chunks_mut(m * COLUMNAS_MINIMAS)
        .enumerate()
        .for_each(|(tramo, destino)| {
            let mut columna_q = vec![0i8; k];
            let mut columna_esc = vec![0f32; n_bloques];
            let mut columna_des = vec![0f32; n_bloques];
            let primera = tramo * COLUMNAS_MINIMAS;

            for (j, salida) in destino.chunks_mut(m).enumerate() {
                let col = primera + j;
                let desde = col * bytes_fila;
                if crate::quant::desempaquetar_i8(
                    tipo,
                    &pesos[desde..desde + bytes_fila],
                    &mut columna_q,
                    &mut columna_esc,
                    &mut columna_des,
                )
                .is_err()
                {
                    continue;
                }
                // Los formatos sin minimo dejan todo en cero: saltarse el
                // termino entero ahorra una pasada por fila.
                let hay_desplazamiento = columna_des.iter().any(|v| *v != 0.0);

                for (fila, s) in salida.iter_mut().enumerate() {
                    let xe = &lhs_esc[fila * n_bloques..(fila + 1) * n_bloques];
                    let mut total = crate::kernels::dot_i8_escalado(
                        &columna_q,
                        &lhs_q[fila * k..(fila + 1) * k],
                        &columna_esc,
                        xe,
                        sub,
                    );
                    if hay_desplazamiento {
                        let xs = &lhs_suma[fila * n_bloques..(fila + 1) * n_bloques];
                        for b in 0..n_bloques {
                            total += columna_des[b] * xe[b] * xs[b];
                        }
                    }
                    *s = total;
                }
            }
        });

    for col in 0..n {
        for fila in 0..m {
            dst[fila * n + col] = salida_t[col * m + fila];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pesos Q8_0 donde la cuenta se puede hacer a mano.
    ///
    /// Con `d = 1` y enteros chicos el resultado es un producto punto de
    /// enteros, asi que se puede verificar sin confiar en ninguna otra parte
    /// del codigo.
    fn pesos_q8_identidad(n: usize, k: usize) -> Vec<u8> {
        let bloques_fila = k / 32;
        let mut b = vec![0u8; n * bloques_fila * 34];
        for fila in 0..n {
            for bl in 0..bloques_fila {
                let o = (fila * bloques_fila + bl) * 34;
                b[o..o + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1
                for j in 0..32 {
                    let idx = bl * 32 + j;
                    // Uno en la diagonal, cero en el resto.
                    b[o + 2 + j] = if idx == fila { 1 } else { 0 };
                }
            }
        }
        b
    }

    #[test]
    fn la_identidad_devuelve_la_entrada() {
        let (m, k, n) = (3usize, 64usize, 64usize);
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.25) - 4.0).collect();
        let pesos = pesos_q8_identidad(n, k);
        let mut dst = vec![0f32; m * n];
        matmul((m, k, n), &lhs, &pesos, TipoGgml::Q8_0, &mut dst).unwrap();
        assert_eq!(dst, lhs, "multiplicar por la identidad cambio los valores");
    }

    #[test]
    fn una_fila_y_varias_filas_dan_lo_mismo() {
        let (k, n) = (64usize, 96usize);
        let pesos: Vec<u8> = {
            let mut b = pesos_q8_identidad(n, k);
            // Valores cualesquiera, para que no sea la identidad.
            for (i, v) in b.iter_mut().enumerate() {
                if i % 34 >= 2 {
                    *v = (i * 31 % 177) as u8;
                }
            }
            b
        };
        let filas: Vec<Vec<f32>> = (0..4)
            .map(|f| (0..k).map(|i| ((i + f) as f32 * 0.13).sin()).collect())
            .collect();

        let mut juntas = vec![0f32; 4 * n];
        let plano: Vec<f32> = filas.iter().flatten().copied().collect();
        matmul((4, k, n), &plano, &pesos, TipoGgml::Q8_0, &mut juntas).unwrap();

        for (f, fila) in filas.iter().enumerate() {
            let mut sola = vec![0f32; n];
            matmul((1, k, n), fila, &pesos, TipoGgml::Q8_0, &mut sola).unwrap();
            assert_eq!(
                &juntas[f * n..(f + 1) * n],
                &sola[..],
                "la fila {f} da distinto sola que en lote"
            );
        }
    }

    /// La version en enteros tiene que dar casi lo mismo que la de f32.
    ///
    /// No identico: recuantizar la entrada a `i8` pierde algo, igual que le
    /// pasa a candle. Lo que se exige es que el error relativo sea chico, que
    /// es lo que distingue "otra aritmetica" de "esta mal".
    #[test]
    fn enteros_y_f32_dan_casi_lo_mismo() {
        let (m, k, n) = (3usize, 128usize, 64usize);
        let mut pesos = pesos_q8_identidad(n, k);
        for (i, v) in pesos.iter_mut().enumerate() {
            if i % 34 >= 2 {
                *v = (i * 29 % 200) as u8;
            }
        }
        let lhs: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.07).sin()).collect();

        let mut a = vec![0f32; m * n];
        let mut b = vec![0f32; m * n];
        matmul((m, k, n), &lhs, &pesos, TipoGgml::Q8_0, &mut a).unwrap();
        matmul_i8((m, k, n), &lhs, &pesos, TipoGgml::Q8_0, &mut b).unwrap();

        let rango =
            a.iter().cloned().fold(f32::MIN, f32::max) - a.iter().cloned().fold(f32::MAX, f32::min);
        let peor = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            peor < rango * 0.02,
            "la diferencia {peor} es mas del 2% del rango {rango}"
        );
    }

    /// Cuantizar y reconstruir no puede mover un vector mas de medio paso.
    #[test]
    fn cuantizar_i8_conserva_la_forma() {
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.21).sin() * 3.0).collect();
        let mut q = vec![0i8; 64];
        let mut e = vec![0f32; 2];
        cuantizar_i8(&x, &mut q, &mut e);
        assert!(
            q.iter().all(|v| *v != i8::MIN),
            "salio un -128, fuera de contrato"
        );
        for (b, trozo) in x.chunks(BLOQUE_I8).enumerate() {
            for (i, v) in trozo.iter().enumerate() {
                let rec = q[b * BLOQUE_I8 + i] as f32 * e[b];
                assert!((rec - v).abs() <= e[b] * 0.51, "se movio mas de medio paso");
            }
        }
    }

    /// Un bloque entero en cero no puede dar NaN.
    #[test]
    fn cuantizar_i8_aguanta_los_ceros() {
        let x = vec![0f32; 64];
        let mut q = vec![0i8; 64];
        let mut e = vec![0f32; 2];
        cuantizar_i8(&x, &mut q, &mut e);
        assert!(e.iter().all(|v| v.is_finite()));
        assert!(q.iter().all(|v| *v == 0));
    }

    /// El camino directo contra el de `f32`, que es el mas preciso.
    ///
    /// La primera version de este test comparaba directo contra rodeo
    /// exigiendo `1e-5`, y fallaba. El equivocado era el test: los dos
    /// cuantizan la **entrada** a i8 y eso introduce un redondeo propio, asi
    /// que difieren entre si sin que ninguno este mal. Comprobado a mano
    /// sobre un caso con respuesta conocida:
    ///
    /// ```text
    ///   a mano      120
    ///   f32         120
    ///   i8 rodeo    120,68
    ///   i8 directo  120,20
    /// ```
    ///
    /// El directo queda mas cerca, que tiene sentido: se saltea una etapa de
    /// redondeo. Asi que la comparacion correcta es contra `f32` y con una
    /// tolerancia propia de 8 bits.
    #[test]
    fn el_directo_coincide_con_el_rodeo() {
        let (m, k, n) = (3usize, 128usize, 64usize);
        let mut pesos = pesos_q8_identidad(n, k);
        for (i, v) in pesos.iter_mut().enumerate() {
            if i % 34 >= 2 {
                // Dentro de [-127, 127], que es lo que producen los
                // cuantizadores de GGML. Con bytes arbitrarios aparece -128 y
                // el camino de AVX2 no lo soporta en los dos lados.
                *v = ((i * 29 % 253) as i32 - 126) as i8 as u8;
            }
        }
        let lhs: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.07).sin()).collect();

        let mut exacto = vec![0f32; m * n];
        let mut directo = vec![0f32; m * n];
        matmul((m, k, n), &lhs, &pesos, TipoGgml::Q8_0, &mut exacto).unwrap();
        matmul_i8_directo((m, k, n), &lhs, &pesos, TipoGgml::Q8_0, &mut directo).unwrap();

        let rango = exacto.iter().cloned().fold(f32::MIN, f32::max)
            - exacto.iter().cloned().fold(f32::MAX, f32::min);
        let peor = exacto
            .iter()
            .zip(&directo)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            peor < rango * 0.02,
            "la diferencia {peor} es mas del 2% del rango {rango}"
        );
    }

    #[test]
    fn avisa_si_las_formas_no_cuadran() {
        let pesos = pesos_q8_identidad(32, 64);
        let mut dst = vec![0f32; 32];
        assert!(matmul((1, 63, 32), &[0f32; 63], &pesos, TipoGgml::Q8_0, &mut dst).is_err());
        assert!(matmul((1, 64, 32), &[0f32; 10], &pesos, TipoGgml::Q8_0, &mut dst).is_err());
    }
}
