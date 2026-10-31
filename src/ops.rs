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

/// `dst[m, n] = lhs[m, k] · W[n, k]ᵀ` con el **kernel fusionado**, sin
/// desempaquetar los pesos a ningún buffer.
///
/// La diferencia con [`matmul_i8_directo`] es que ahí cada columna se
/// desempaqueta a un buffer de `k` valores y después se multiplica, y medido
/// ese desempaquetado era el 94% del trabajo de un hilo — 2,4 ms contra 1,0
/// del kernel. Acá no hay buffer: [`crate::kernels::dot_q4k_i8`] lee los
/// nibbles a registros vectoriales y acumula el producto al vuelo.
///
/// Hay un segundo efecto que importa más de lo que parece: una fila en Q4_K de
/// 2048 valores son **1152 bytes**, así que cabe de sobra en L1 y las `m`
/// filas de salida la reusan desde ahí. El camino con buffer tocaba `k` bytes
/// de `i8` más sus escalas por cada columna.
///
/// **Q4_K y Q3_K**; cualquier otro formato cae a [`matmul_i8_directo`], que
/// los cubre todos. Q4_K sale directo porque sus nibbles ya son enteros sin
/// signo, y Q3_K necesita sumarle 4 a cada valor y descontarlo con la suma de
/// la entrada — ver [`crate::kernels::dot_q3k_i8`]. Los otros cinco restan
/// cosas distintas y cada uno pide su propio rodeo.
pub fn matmul_i8_fusionado(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    // Los dos formatos con kernel fusionado, con su sub-bloque y sus bytes
    // por cada 256 valores. Cualquier otro cae al camino general, que los
    // cubre todos.
    let (sub, bytes_bloque) = match tipo {
        TipoGgml::Q4K => (32usize, 144usize),
        TipoGgml::Q3K => (16, 110),
        TipoGgml::Q6K => (16, 210),
        _ => return matmul_i8_directo((m, k, n), lhs, pesos, tipo, dst),
    };
    if !k.is_multiple_of(256) {
        return matmul_i8_directo((m, k, n), lhs, pesos, tipo, dst);
    }
    if lhs.len() != m * k {
        return Err(format!("lhs mide {} y deberia medir {m}*{k}", lhs.len()));
    }
    if dst.len() != m * n {
        return Err(format!("dst mide {} y deberia medir {m}*{n}", dst.len()));
    }
    let bytes_fila = k / 256 * bytes_bloque;
    if pesos.len() < n * bytes_fila {
        return Err("faltan bytes de pesos".into());
    }

    let n_bloques = k / sub;

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
            let primera = tramo * COLUMNAS_MINIMAS;
            for (j, salida) in destino.chunks_mut(m).enumerate() {
                let desde = (primera + j) * bytes_fila;
                let fila = &pesos[desde..desde + bytes_fila];
                for (f, s) in salida.iter_mut().enumerate() {
                    let xq = &lhs_q[f * k..(f + 1) * k];
                    let xe = &lhs_esc[f * n_bloques..(f + 1) * n_bloques];
                    let xs = &lhs_suma[f * n_bloques..(f + 1) * n_bloques];
                    *s = match tipo {
                        TipoGgml::Q4K => crate::kernels::dot_q4k_i8(fila, xq, xe, xs),
                        TipoGgml::Q6K => crate::kernels::dot_q6k_i8(fila, xq, xe, xs),
                        _ => crate::kernels::dot_q3k_i8(fila, xq, xe, xs),
                    };
                }
            }
        });

    for col in 0..n {
        for f in 0..m {
            dst[f * n + col] = salida_t[col * m + f];
        }
    }
    Ok(())
}

/// `dst[f, col] = salida_t[col, f]`, en paralelo y de a 16 filas.
///
/// Las multiplicaciones por tramos de columnas dejan la salida transpuesta, y
/// darla vuelta en un hilo costaba hasta 11 ms en `ffn_gate` con 333 filas —un
/// 20% de la multiplicación— con los demás núcleos esperando. De a 16 filas,
/// cada columna aporta una línea de caché entera (16 `f32` seguidos) en vez
/// de un valor suelto.
fn transponer_en_paralelo(salida_t: &[f32], m: usize, n: usize, dst: &mut [f32]) {
    const FILAS: usize = 16;
    dst.par_chunks_mut(FILAS * n)
        .enumerate()
        .for_each(|(g, bloque)| {
            let f0 = g * FILAS;
            let filas = bloque.len() / n;
            for col in 0..n {
                let origen = &salida_t[col * m + f0..col * m + f0 + filas];
                for (r, v) in origen.iter().enumerate() {
                    bloque[r * n + col] = *v;
                }
            }
        });
}

/// Las escalas por sub-bloque, expandidas a una por carril de 32 bits.
///
/// Un trozo de 32 bytes de `i8` sale de `vpmaddubsw` + `vpmaddwd` como ocho
/// `i32`, y el carril `l` suma los bytes `4l..4l+3`. Así que su escala es la
/// del sub-bloque donde caen esos bytes: con sub-bloques de 16, los cuatro
/// primeros carriles llevan una y los otros cuatro la siguiente; con 32, los
/// ocho la misma. Dejarlas así una vez por fila y por columna es lo que le
/// ahorra al kernel armar el vector de escalas en cada trozo.
///
/// Sin divisiones: la primera versión calculaba el sub-bloque de cada carril
/// con una, y con 18 filas la expansión pesaba más que lo que ahorraba.
fn expandir_escalas(esc: &[f32], sub: usize, destino: &mut [f32]) {
    if sub == 16 {
        for (t, d) in destino.chunks_exact_mut(8).enumerate() {
            d[..4].fill(esc[2 * t]);
            d[4..].fill(esc[2 * t + 1]);
        }
    } else {
        let por_sub = sub / 32;
        for (t, d) in destino.chunks_exact_mut(8).enumerate() {
            d.fill(esc[t / por_sub]);
        }
    }
}

/// `dst[m, n] = lhs[m, k] · W[n, k]ᵀ` **por bloques**: el camino del prefill.
///
/// Hace las mismas cuentas que [`matmul_i8_directo`] —las mismas columnas
/// desempaquetadas a `i8`, la misma entrada cuantizada, las mismas escalas
/// por sub-bloque— en otro orden. Allá, cada producto punto era una llamada:
/// por cada columna se recorría la entrada entera, fila por fila, y cada byte
/// traído de la caché se usaba una sola vez. Con 333 filas de 2048 valores son
/// 680 KB por columna, que no entran en L2: la multiplicación esperaba a la
/// caché, no a las cuentas.
///
/// Acá:
///
/// - cada hilo desempaqueta su tramo de columnas una vez;
/// - la entrada se recorre en franjas de filas que **sí** entran en L2 junto
///   con el tramo, y la franja se reusa para todas las columnas del tramo;
/// - adentro, una tesela de 4 filas × 2 columnas en registros
///   ([`crate::kernels::x86::tesela_i8_avx2`]): cada trozo de columna cargado
///   se usa 4 veces y cada trozo de fila 2;
/// - cuantizar la entrada y transponer la salida van en paralelo. En un hilo
///   eran entre el 24% y el 64% de cada multiplicación —medido por etapas—,
///   y eran lo que de verdad la frenaba: con ellas en paralelo la mejora
///   pasó de 1,65x a 2x.
///
/// Se probaron y se sacaron dos cosas más, con `examples/measure_precision`
/// para el error: **escalas enteras** en los k-quants —la entrada con una
/// escala cada 256 valores, como el Q8_K de candle, y el producto de un
/// bloque en enteros— ganaba un 15% más pero con 3 a 7 veces más error en
/// los logits (KL 0,0034 a 0,0071 contra 0,0011); y **tajadas a lo largo de
/// `k`** para que todo entrara en L2, que no le ganó a no partir.
///
/// Solo con AVX2; sin él, o con `k` que no sea múltiplo de 32, cae a
/// [`matmul_i8_directo`].
pub fn matmul_i8_bloques(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    #[cfg(target_arch = "x86_64")]
    if crate::kernels::usar_avx2() && k.is_multiple_of(32) {
        return matmul_i8_bloques_avx2((m, k, n), lhs, pesos, tipo, dst);
    }
    matmul_i8_directo((m, k, n), lhs, pesos, tipo, dst)
}

/// Filas y columnas de la tesela. 4 × 2 son 8 acumuladores más 2 columnas
/// con su valor absoluto y su escala: 14 de los 16 registros.
#[cfg(target_arch = "x86_64")]
const TESELA_FILAS: usize = 4;
#[cfg(target_arch = "x86_64")]
const TESELA_COLS: usize = 2;

/// Bytes de `i8` de entrada por franja: lo que se deja en L2 mientras pasan
/// las columnas del tramo.
#[cfg(target_arch = "x86_64")]
const FRANJA_BYTES: usize = 128 * 1024;

#[cfg(target_arch = "x86_64")]
fn matmul_i8_bloques_avx2(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    use crate::kernels::x86::{tesela_i8_avx2, tesela_u8_avx2};

    if lhs.len() != m * k {
        return Err(format!("lhs mide {} y deberia medir {m}*{k}", lhs.len()));
    }
    if dst.len() != m * n {
        return Err(format!("dst mide {} y deberia medir {m}*{n}", dst.len()));
    }
    let por_bloque = tipo.valores_por_bloque();
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
    let trozos = k / 32;
    let ancho_esc = trozos * 8;

    // La entrada, igual que en `matmul_i8_directo`, más sus escalas
    // expandidas y, para el término del desplazamiento, escala por suma.
    let mut lhs_q = vec![0i8; m * k];
    let mut lhs_esc = vec![0f32; m * n_bloques];
    let mut lhs_e8 = vec![0f32; m * ancho_esc];
    let mut lhs_es = vec![0f32; m * n_bloques];
    // En paralelo, fila por fila: en un hilo eran 3 a 14 ms por llamada,
    // hasta el 20% de la multiplicación, con los demás núcleos esperando.
    lhs_q
        .par_chunks_mut(k)
        .zip(lhs_esc.par_chunks_mut(n_bloques))
        .zip(lhs_e8.par_chunks_mut(ancho_esc))
        .zip(lhs_es.par_chunks_mut(n_bloques))
        .zip(lhs.par_chunks(k))
        .for_each(|((((q, e), e8), es), x)| {
            cuantizar_i8_en(x, q, e, sub);
            expandir_escalas(e, sub, e8);
            for b in 0..n_bloques {
                let r = b * sub;
                let suma = q[r..r + sub].iter().map(|v| *v as i32).sum::<i32>() as f32;
                es[b] = e[b] * suma;
            }
        });

    let franja = (FRANJA_BYTES / k).max(TESELA_FILAS) / TESELA_FILAS * TESELA_FILAS;
    let mut salida_t = vec![0f32; n * m];

    salida_t
        .par_chunks_mut(m * COLUMNAS_MINIMAS)
        .enumerate()
        .for_each(|(tramo, destino)| {
            let primera = tramo * COLUMNAS_MINIMAS;
            let ncols = destino.len() / m;

            // El tramo entero desempaquetado, una vez.
            let mut cols_q = vec![0i8; ncols * k];
            let mut cols_e8 = vec![0f32; ncols * ancho_esc];
            let mut cols_des = vec![0f32; ncols * n_bloques];
            let mut esc = vec![0f32; n_bloques];
            let mut hay_des = vec![false; ncols];
            for j in 0..ncols {
                let desde = (primera + j) * bytes_fila;
                let des = &mut cols_des[j * n_bloques..(j + 1) * n_bloques];
                if crate::quant::desempaquetar_i8(
                    tipo,
                    &pesos[desde..desde + bytes_fila],
                    &mut cols_q[j * k..(j + 1) * k],
                    &mut esc,
                    des,
                )
                .is_err()
                {
                    continue;
                }
                expandir_escalas(&esc, sub, &mut cols_e8[j * ancho_esc..(j + 1) * ancho_esc]);
            }

            // Pesos sin signo, si el tramo lo permite: se corren todos por
            // `-mínimo` y el kernel se ahorra el rodeo del signo. El
            // corrimiento se descuenta en el término del desplazamiento:
            // `s·Σ x·(u - c) = s·Σ x·u - c·s·Σ x`, y `Σ x` ya está por fila.
            // Todos los k-quants entran —Q3_K va de -4 a 3, Q6_K de -32 a 31—;
            // Q8_0 puede ir de -127 a 127 y sigue por el camino con signo.
            let (min, max) = cols_q
                .iter()
                .fold((i8::MAX, i8::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            let corrimiento = -(min as i32);
            let sin_signo = (max as i32) + corrimiento <= 129;
            if sin_signo && corrimiento != 0 {
                for v in cols_q.iter_mut() {
                    *v = ((*v as i32) + corrimiento) as u8 as i8;
                }
            }
            for j in 0..ncols {
                let des = &mut cols_des[j * n_bloques..(j + 1) * n_bloques];
                if sin_signo && corrimiento != 0 {
                    let e8 = &cols_e8[j * ancho_esc..(j + 1) * ancho_esc];
                    for (b, d) in des.iter_mut().enumerate() {
                        // La escala del sub-bloque `b`, leída de la expandida.
                        *d -= corrimiento as f32 * e8[b * sub / 4];
                    }
                }
                hay_des[j] = des.iter().any(|v| *v != 0.0);
            }

            let fila = |f: usize| (lhs_q[f * k..].as_ptr(), lhs_e8[f * ancho_esc..].as_ptr());
            let col = |j: usize| (cols_q[j * k..].as_ptr(), cols_e8[j * ancho_esc..].as_ptr());

            // Partir también a lo largo de `k`, en tajadas que entraran en L2
            // con la franja, se probó con 512, 1024 y 2048 valores: ninguna le
            // ganó a no partir (1,67x, 1,93x y 2,11x contra 2,05x), y se sacó.
            let mut f0 = 0;
            while f0 < m {
                let f1 = (f0 + franja).min(m);
                let mut j = 0;
                while j < ncols {
                    let nc = (ncols - j).min(TESELA_COLS);
                    let mut f = f0;
                    while f < f1 {
                        let nf = (f1 - f).min(TESELA_FILAS);
                        // La tesela entera o, en los bordes, de a un producto.
                        if nf == TESELA_FILAS && nc == TESELA_COLS {
                            let (x0, e0) = fila(f);
                            let (x1, e1) = fila(f + 1);
                            let (x2, e2) = fila(f + 2);
                            let (x3, e3) = fila(f + 3);
                            let (c0, d0) = col(j);
                            let (c1, d1) = col(j + 1);
                            let (xs, es, cs, ds) =
                                ([x0, x1, x2, x3], [e0, e1, e2, e3], [c0, c1], [d0, d1]);
                            // SAFETY: AVX2 comprobado en `matmul_i8_bloques`;
                            // cada puntero cubre `k` valores y sus
                            // escalas, la entrada está en -127..=127, y con
                            // `sin_signo` los pesos en 0..=129.
                            let r = unsafe {
                                if sin_signo {
                                    tesela_u8_avx2::<4, 2>(&xs, &es, &cs, &ds, trozos)
                                } else {
                                    tesela_i8_avx2::<4, 2>(&xs, &es, &cs, &ds, trozos)
                                }
                            };
                            for (df, fila_r) in r.iter().enumerate() {
                                for (dc, v) in fila_r.iter().enumerate() {
                                    destino[(j + dc) * m + f + df] = *v;
                                }
                            }
                        } else {
                            for df in 0..nf {
                                for dc in 0..nc {
                                    let (x, e) = fila(f + df);
                                    let (c, d) = col(j + dc);
                                    // SAFETY: lo mismo que arriba.
                                    let r = unsafe {
                                        if sin_signo {
                                            tesela_u8_avx2::<1, 1>(&[x], &[e], &[c], &[d], trozos)
                                        } else {
                                            tesela_i8_avx2::<1, 1>(&[x], &[e], &[c], &[d], trozos)
                                        }
                                    };
                                    destino[(j + dc) * m + f + df] = r[0][0];
                                }
                            }
                        }
                        f += nf;
                    }
                    j += nc;
                }
                f0 = f1;
            }

            // El término del desplazamiento, en los formatos que lo tienen.
            for j in 0..ncols {
                if !hay_des[j] {
                    continue;
                }
                let des = &cols_des[j * n_bloques..(j + 1) * n_bloques];
                for f in 0..m {
                    destino[j * m + f] +=
                        crate::kernels::dot(des, &lhs_es[f * n_bloques..(f + 1) * n_bloques]);
                }
            }
        });

    transponer_en_paralelo(&salida_t, m, n, dst);
    Ok(())
}

/// Si `tipo` tiene kernel fusionado —sin buffer— para filas de `k` valores.
///
/// Q6_K solo con AVX2: su kernel fusionado no tiene versión vectorizada en
/// otras arquitecturas, y el escalar perdería contra el camino con buffer.
/// `COSXAMIX_SIN_Q6K_FUSIONADO` lo devuelve al camino con buffer, para
/// comparar.
fn hay_fusionado(tipo: TipoGgml, k: usize) -> bool {
    static Q6K_FUSIONADO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let q6k = *Q6K_FUSIONADO.get_or_init(|| {
        crate::kernels::usar_avx2() && std::env::var_os("COSXAMIX_SIN_Q6K_FUSIONADO").is_none()
    });
    (matches!(tipo, TipoGgml::Q4K | TipoGgml::Q3K) || (tipo == TipoGgml::Q6K && q6k))
        && k.is_multiple_of(256)
}

/// La entrada de una fila, cuantizada en sub-bloques de `sub`, con su escala
/// y su suma por sub-bloque.
struct EntradaI8 {
    q: Vec<i8>,
    escalas: Vec<f32>,
    sumas: Vec<f32>,
}

impl EntradaI8 {
    fn nueva(x: &[f32], sub: usize) -> Self {
        let mut q = vec![0i8; x.len()];
        let mut escalas = vec![0f32; x.len() / sub];
        cuantizar_i8_en(x, &mut q, &mut escalas, sub);
        let sumas = q
            .chunks(sub)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        Self { q, escalas, sumas }
    }
}

/// Varias proyecciones de **un mismo vector** en una sola pasada paralela:
/// `destinos[i] = x · pesos[i]ᵀ`. Es el decode de q, k y v, o de gate y up.
///
/// Hace exactamente lo que harían las llamadas a [`matmul_auto`] por separado
/// con una fila —los mismos kernels, la misma entrada cuantizada— y da el
/// mismo resultado bit a bit. Lo que cambia es que hay **una** región
/// paralela en vez de tres: cada llamada cuesta unos 10 µs fijos de repartir
/// y juntar los hilos, y `attn_k` y `attn_v` del 3B son tan chicas (0,2 MB)
/// que eso era casi todo su costo. Medido dentro del forward, q, k y v iban
/// a 12 GB/s contra 21 de las ffn.
///
/// `pesos[i]` es `(bytes, tipo)` de una matriz de `destinos[i].len()` filas
/// de `x.len()` valores. Los formatos sin desempaquetado a `i8` (F32, F16)
/// van por [`matmul_auto`], uno por uno.
pub fn matmul_varias_una_fila(
    x: &[f32],
    pesos: &[(&[u8], TipoGgml)],
    destinos: &mut [&mut [f32]],
) -> Result<(), String> {
    if pesos.len() != destinos.len() {
        return Err("hace falta un destino por matriz".into());
    }
    let k = x.len();
    let mut filas = Vec::with_capacity(pesos.len());
    for ((bytes, tipo), d) in pesos.iter().zip(destinos.iter()) {
        let Some(sub) = crate::quant::sub_bloque(*tipo) else {
            filas.push(None);
            continue;
        };
        let por_bloque = tipo.valores_por_bloque();
        if !k.is_multiple_of(por_bloque) || !k.is_multiple_of(sub) {
            return Err(format!("una fila de {k} no es multiplo de los bloques"));
        }
        let bytes_fila = k / por_bloque * tipo.bytes_por_bloque();
        if bytes.len() < d.len() * bytes_fila {
            return Err("faltan bytes de pesos".into());
        }
        filas.push(Some((sub, bytes_fila)));
    }
    // Los que no tienen camino en enteros, aparte.
    for (i, f) in filas.iter().enumerate() {
        if f.is_none() {
            let (bytes, tipo) = pesos[i];
            let n = destinos[i].len();
            matmul_auto((1, k, n), x, bytes, tipo, destinos[i])?;
        }
    }

    // La entrada, una vez por tamaño de sub-bloque que haga falta.
    let necesita = |s: usize| filas.iter().flatten().any(|(sub, _)| *sub == s);
    let e16 = necesita(16).then(|| EntradaI8::nueva(x, 16));
    let e32 = necesita(32).then(|| EntradaI8::nueva(x, 32));

    // Las tareas: tramos de columnas de todas las matrices, mezclados en una
    // sola lista para que los hilos se repartan todo de una vez.
    let mut tareas: Vec<(usize, usize, &mut [f32])> = Vec::new();
    for (i, d) in destinos.iter_mut().enumerate() {
        if filas[i].is_none() {
            continue;
        }
        for (t, trozo) in d.chunks_mut(COLUMNAS_MINIMAS).enumerate() {
            tareas.push((i, t * COLUMNAS_MINIMAS, trozo));
        }
    }

    tareas.into_par_iter().for_each(|(i, primera, salida)| {
        let (bytes, tipo) = pesos[i];
        let (sub, bytes_fila) = filas[i].expect("filtrado arriba");
        let e = if sub == 16 {
            e16.as_ref()
        } else {
            e32.as_ref()
        }
        .expect("cuantizada arriba");
        let fila = |j: usize| &bytes[(primera + j) * bytes_fila..(primera + j + 1) * bytes_fila];

        if hay_fusionado(tipo, k) {
            for (j, s) in salida.iter_mut().enumerate() {
                *s = match tipo {
                    TipoGgml::Q4K => {
                        crate::kernels::dot_q4k_i8(fila(j), &e.q, &e.escalas, &e.sumas)
                    }
                    TipoGgml::Q6K => {
                        crate::kernels::dot_q6k_i8(fila(j), &e.q, &e.escalas, &e.sumas)
                    }
                    _ => crate::kernels::dot_q3k_i8(fila(j), &e.q, &e.escalas, &e.sumas),
                };
            }
            return;
        }

        // El camino con buffer, el mismo de `matmul_i8_directo` con una fila.
        let n_bloques = k / sub;
        let mut columna_q = vec![0i8; k];
        let mut columna_esc = vec![0f32; n_bloques];
        let mut columna_des = vec![0f32; n_bloques];
        for (j, s) in salida.iter_mut().enumerate() {
            if crate::quant::desempaquetar_i8(
                tipo,
                fila(j),
                &mut columna_q,
                &mut columna_esc,
                &mut columna_des,
            )
            .is_err()
            {
                continue;
            }
            let mut total =
                crate::kernels::dot_i8_escalado(&columna_q, &e.q, &columna_esc, &e.escalas, sub);
            if columna_des.iter().any(|v| *v != 0.0) {
                for ((d, xe), xs) in columna_des.iter().zip(&e.escalas).zip(&e.sumas) {
                    total += d * xe * xs;
                }
            }
            *s = total;
        }
    });
    Ok(())
}

/// Desde cuántas filas conviene [`matmul_i8_bloques`]: el cruce medido está
/// entre 8 (0,79x) y 16 (1,07x), y desde 24 ya gana 1,34x.
pub const BLOQUES_DESDE: usize = 24;

/// Multiplica eligiendo el camino, que es lo que el motor debería hacer en vez
/// de obligar a quien llama a saberse la tabla.
///
/// Hay tres caminos y **ninguno gana siempre**. Medido sobre el `ffn_gate` del
/// 3B, 2048 → 11008, dos pasadas por formato y todo en el mismo proceso:
///
/// | filas | Q4_K | Q3_K |
/// |---:|---|---|
/// | 1 a 4 | **fusionado** | **fusionado** |
/// | 8 y más | **fusionado** | con buffer |
///
/// **Esta tabla se rehízo entera.** La anterior mandaba el lote de Q4_K al
/// camino de `f32` y el de Q3_K al buffer desde dos filas, y era correcta —
/// para el kernel fusionado de entonces. Vectorizar el cálculo de las escalas
/// lo hizo casi el doble de rápido y movió los dos cruces. Es el riesgo de
/// escribir umbrales medidos: hay que volver a medirlos cuando cambia lo que
/// miden.
///
/// Por qué Q3_K todavía cruza y Q4_K no: **el fusionado desempaqueta una vez
/// por fila de salida** y el camino con buffer una vez por columna, así que
/// al crecer el lote paga `m` desempaquetados contra uno. El de Q3_K cuesta
/// bastante más —máscara de bits para el tercer bit, y 16 escalas de 6 bits
/// por bloque contra 8— y por eso le alcanza el buffer antes.
///
/// Desde [`BLOQUES_DESDE`] filas va todo por [`matmul_i8_bloques`], que con
/// lotes grandes gana en todos los formatos. Medido sobre todas las matrices
/// de dos capas del 3B Q3_K_M (`examples/measure_blocked_matmul`):
///
/// | filas | antes | por bloques | |
/// |---:|---:|---:|---:|
/// | 8 | 19,1 ms | 24,2 ms | 0,79x |
/// | 16 | 32,5 ms | 30,2 ms | 1,07x |
/// | 24 | 47,6 ms | 35,4 ms | 1,34x |
/// | 32 | 61,4 ms | 41,6 ms | 1,48x |
/// | 128 | 249 ms | 127 ms | 1,95x |
/// | 333 | 643 ms | 323 ms | 1,99x |
///
/// Los umbrales salen de **un tensor y esta máquina**. Son un valor por
/// defecto medido, no una ley: en otro hardware el cruce se mueve, y la forma
/// de saberlo es `cargo run --release --features candle --example
/// measure_own_matmul`.
pub fn matmul_auto(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    pesos: &[u8],
    tipo: TipoGgml,
    dst: &mut [f32],
) -> Result<(), String> {
    // Los formatos SIN cuantizar —F32 y F16— no tienen nada que desempaquetar
    // y su unico camino posible es el de `f32`. Mandarlos al de enteros, que
    // es lo que pasaba, falla con "sin desempaquetado directo a i8": lo
    // encontro el experimento del modelo convertido a F32, que es justo el
    // caso que nadie habia probado porque todos los modelos vienen
    // cuantizados.
    if crate::quant::sub_bloque(tipo).is_none() {
        return matmul((m, k, n), lhs, pesos, tipo, dst);
    }

    // `COSXAMIX_SIN_BLOQUES` lo apaga, para comparar con y sin él.
    static CON_BLOQUES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let con_bloques =
        *CON_BLOQUES.get_or_init(|| std::env::var_os("COSXAMIX_SIN_BLOQUES").is_none());
    if con_bloques && m >= BLOQUES_DESDE {
        return matmul_i8_bloques((m, k, n), lhs, pesos, tipo, dst);
    }

    let fusionable = hay_fusionado(tipo, k);

    // En Q4_K el fusionado gana en todo el rango medido, de 1 a 32 filas. En
    // Q3_K gana hasta 4 y pierde desde 8.
    let tope_fusion = if tipo == TipoGgml::Q4K { usize::MAX } else { 4 };

    if fusionable && m <= tope_fusion {
        return matmul_i8_fusionado((m, k, n), lhs, pesos, tipo, dst);
    }
    // Lo demas por el camino con buffer, que cubre los siete formatos y nunca
    // es el peor de los tres.
    matmul_i8_directo((m, k, n), lhs, pesos, tipo, dst)
}

// ─── RoPE y softmax: lo que falta de la fase 2 ───────────────────────────────

/// Las tablas de cosenos y senos de RoPE, de `contexto × (dim_cabeza / 2)`.
///
/// Se precalculan porque dependen solo de la posición y de la dimensión, no de
/// los datos: calcularlas en cada token sería pagar `contexto` senos y cosenos
/// por paso cuando alcanza una tabla.
///
/// La frecuencia del par `j` es `1 / base^(2j / dim_cabeza)`, y el ángulo de
/// la posición `p` es `p` por esa frecuencia. `base` sale del archivo del
/// modelo —`qwen2.rope.freq_base`—, y en Qwen2.5 vale 1.000.000 para los
/// tamaños grandes y 10.000 para el 0,5B: usar el equivocado da un modelo que
/// responde mal sin fallar, porque las posiciones quedan mal escaladas.
pub fn frecuencias_rope(dim_cabeza: usize, base: f32, contexto: usize) -> (Vec<f32>, Vec<f32>) {
    let mitad = dim_cabeza / 2;
    let mut cos = vec![0f32; contexto * mitad];
    let mut sin = vec![0f32; contexto * mitad];

    for j in 0..mitad {
        let frecuencia = 1.0 / base.powf(2.0 * j as f32 / dim_cabeza as f32);
        for p in 0..contexto {
            // El ángulo se calcula en `f64` y se guarda en `f32`. Con
            // `contexto` de 32.768 y frecuencia 1, el ángulo llega a 32.768
            // radianes, y ahí un `f32` tiene unos 0,002 de resolución: el
            // seno saldría con un error visible. En `f64` el error es
            // despreciable y la tabla queda igual que la de candle, que
            // también calcula en coma flotante de 32 bits pero sobre un
            // producto exacto.
            let angulo = (p as f64) * (frecuencia as f64);
            cos[p * mitad + j] = angulo.cos() as f32;
            sin[p * mitad + j] = angulo.sin() as f32;
        }
    }
    (cos, sin)
}

/// Aplica RoPE en su sitio sobre `x`, de `tokens × (cabezas · dim_cabeza)`.
///
/// **La variante es la de mitades partidas**, no la intercalada, porque es la
/// que usa Qwen2 — `candle_nn::rotary_emb::rope`, no `rope_i`. Para cada
/// cabeza, el valor `i` se rota contra el `i + dim_cabeza/2`:
///
/// ```text
///     salida[i]        = x[i]·cos − x[i + d/2]·sin
///     salida[i + d/2]  = x[i]·sin + x[i + d/2]·cos
/// ```
///
/// Elegir la variante equivocada **no da ningún error**: el modelo carga,
/// genera, y dice cosas sin sentido. Es el tipo de bug que solo aparece
/// comparando contra una implementación que ya funciona, y por eso hay un test
/// que lo hace.
///
/// `posicion` es la posición absoluta del primer token, que en decode es el
/// largo del contexto ya procesado. Pasar cero ahí con un caché caliente
/// rotaría cada token como si fuera el primero.
pub fn rope(
    x: &mut [f32],
    (tokens, cabezas, dim_cabeza): (usize, usize, usize),
    cos: &[f32],
    sin: &[f32],
    posicion: usize,
) -> Result<(), String> {
    if !dim_cabeza.is_multiple_of(2) {
        return Err(format!("dim_cabeza {dim_cabeza} tiene que ser par"));
    }
    let ancho = cabezas * dim_cabeza;
    if x.len() != tokens * ancho {
        return Err(format!(
            "x mide {} y deberia medir {tokens}*{cabezas}*{dim_cabeza}",
            x.len()
        ));
    }
    let mitad = dim_cabeza / 2;
    if cos.len() != sin.len() || !cos.len().is_multiple_of(mitad) {
        return Err("las tablas de cos y sin no cuadran con dim_cabeza".into());
    }
    let contexto = cos.len() / mitad;
    if posicion + tokens > contexto {
        return Err(format!(
            "se piden las posiciones {posicion}..{} y la tabla llega a {contexto}",
            posicion + tokens
        ));
    }

    for (t, fila) in x.chunks_mut(ancho).enumerate() {
        let desde = (posicion + t) * mitad;
        let c = &cos[desde..desde + mitad];
        let s = &sin[desde..desde + mitad];
        for cabeza in fila.chunks_mut(dim_cabeza) {
            let (bajos, altos) = cabeza.split_at_mut(mitad);
            for (((a, b), c), s) in bajos.iter_mut().zip(altos).zip(c).zip(s) {
                let (x0, x1) = (*a, *b);
                *a = x0 * c - x1 * s;
                *b = x0 * s + x1 * c;
            }
        }
    }
    Ok(())
}

/// Softmax en su sitio, por filas de `ancho`.
///
/// Resta el máximo antes de exponenciar. No es un refinamiento: los logits de
/// un modelo de 151.936 palabras llegan tranquilamente a 30, y `exp(30)` ya es
/// 10^13 — sin restar el máximo, una fila con un logit de 90 desborda a
/// infinito y la fila entera sale `NaN`.
///
/// Una fila entera en `-inf` —que es lo que deja una máscara causal en la
/// primera posición de algunas implementaciones— daría `0/0`. Acá devuelve una
/// distribución uniforme en vez de `NaN`, que es lo que no rompe el muestreo
/// de más arriba.
pub fn softmax(x: &mut [f32], ancho: usize) -> Result<(), String> {
    if ancho == 0 || !x.len().is_multiple_of(ancho) {
        return Err(format!("{} valores no es multiplo de {ancho}", x.len()));
    }

    for fila in x.chunks_mut(ancho) {
        let maximo = fila.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        if !maximo.is_finite() {
            // Todo en -inf, o algun NaN: no hay nada que normalizar.
            let p = 1.0 / ancho as f32;
            fila.fill(p);
            continue;
        }
        let mut total = 0f32;
        for v in fila.iter_mut() {
            *v = (*v - maximo).exp();
            total += *v;
        }
        let inv = 1.0 / total;
        for v in fila.iter_mut() {
            *v *= inv;
        }
    }
    Ok(())
}

/// Atención con máscara causal y cabezas agrupadas (GQA).
///
/// `q` son `tokens × (cabezas · dim_cabeza)`; `k` y `v` son
/// `total × (cabezas_kv · dim_cabeza)` e incluyen **todo el caché**, no solo
/// los tokens nuevos. `total − tokens` es el prefijo ya procesado, y de ahí
/// sale la posición absoluta de cada token nuevo.
///
/// Tres cosas que esta función tiene que acertar y que se equivocan en
/// silencio:
///
/// - **la máscara causal con caché caliente.** El token `t` de esta tanda está
///   en la posición absoluta `total − tokens + t`, y puede mirar las claves
///   `0..=` esa posición. Usar `t` en vez de la posición absoluta deja al
///   primer token nuevo viendo solo una clave cuando debería ver todo el
///   prefijo: el modelo responde como si hubiera perdido la conversación, sin
///   error;
/// - **el reparto de GQA.** La cabeza `h` de la consulta usa la cabeza de
///   clave y valor `h / (cabezas / cabezas_kv)`. En el 3B son 16 cabezas
///   contra 2, o sea 8 consultas por grupo; repartir con `h % cabezas_kv`
///   compila igual y mezcla las cabezas;
/// - **la escala.** Los productos se dividen por `sqrt(dim_cabeza)` antes del
///   softmax. Sin eso, con 128 de dimensión los logits salen ~11 veces más
///   grandes y el softmax se vuelve casi un máximo: la atención deja de
///   promediar y copia un solo valor.
///
/// La salida es `tokens × (cabezas · dim_cabeza)`, la misma forma que `q`.
pub fn atencion(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    (tokens, total, cabezas, cabezas_kv, dim_cabeza): (usize, usize, usize, usize, usize),
    salida: &mut [f32],
) -> Result<(), String> {
    if cabezas_kv == 0 || !cabezas.is_multiple_of(cabezas_kv) {
        return Err(format!(
            "{cabezas} cabezas no se reparten en {cabezas_kv} grupos"
        ));
    }
    if tokens > total {
        return Err(format!("{tokens} tokens nuevos y solo {total} en total"));
    }
    let ancho_q = cabezas * dim_cabeza;
    let ancho_kv = cabezas_kv * dim_cabeza;
    for (quien, tiene, debe) in [
        ("q", q.len(), tokens * ancho_q),
        ("k", k.len(), total * ancho_kv),
        ("v", v.len(), total * ancho_kv),
        ("salida", salida.len(), tokens * ancho_q),
    ] {
        if tiene != debe {
            return Err(format!("{quien} mide {tiene} y deberia medir {debe}"));
        }
    }

    let por_grupo = cabezas / cabezas_kv;
    let escala = 1.0 / (dim_cabeza as f32).sqrt();
    let desplazamiento = total - tokens;

    // Se reparte por token: cada uno escribe su propia fila de la salida y
    // lee todo el caché, asi que no hay nada compartido que mutar.
    salida
        .par_chunks_mut(ancho_q)
        .enumerate()
        .for_each(|(t, fila)| {
            // Hasta donde puede mirar este token, por la máscara causal.
            let visibles = desplazamiento + t + 1;
            let mut puntos = vec![0f32; visibles];

            for h in 0..cabezas {
                let qh = &q[t * ancho_q + h * dim_cabeza..][..dim_cabeza];
                let grupo = h / por_grupo;
                let base_kv = grupo * dim_cabeza;

                for (p, punto) in puntos.iter_mut().enumerate() {
                    let kh = &k[p * ancho_kv + base_kv..][..dim_cabeza];
                    *punto = crate::kernels::dot(qh, kh) * escala;
                }
                // La máscara causal ya está aplicada por construcción: solo se
                // calcularon los puntos visibles, asi que no hace falta
                // rellenar con -inf y dejar que el softmax lo anule.
                let _ = softmax(&mut puntos, visibles);

                let destino = &mut fila[h * dim_cabeza..][..dim_cabeza];
                destino.fill(0.0);
                for (p, peso) in puntos.iter().enumerate() {
                    let vh = &v[p * ancho_kv + base_kv..][..dim_cabeza];
                    for (d, x) in destino.iter_mut().zip(vh) {
                        *d += peso * x;
                    }
                }
            }
        });
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

    /// Por bloques y con buffer hacen las mismas cuentas en otro orden: tienen
    /// que coincidir salvo el redondeo de sumar en otro orden, en los seis
    /// formatos con desempaquetado a `i8`.
    ///
    /// Las formas son a propósito incómodas: 37 filas no son múltiplo de la
    /// tesela de 4, 70 columnas dejan un último tramo de 6 —par— y 71 uno de 7
    /// —impar—, que es donde viven los bordes. Q8_0 va por el camino con
    /// signo (sus pesos van de -127 a 127) y los k-quants por el sin signo.
    #[test]
    fn por_bloques_coincide_con_el_de_buffer() {
        let k = 512usize;
        for (tipo, bytes_bloque, valores_bloque) in [
            (TipoGgml::Q3K, 110usize, 256usize),
            (TipoGgml::Q4K, 144, 256),
            (TipoGgml::Q5K, 176, 256),
            (TipoGgml::Q6K, 210, 256),
            (TipoGgml::Q2K, 84, 256),
            (TipoGgml::Q8_0, 34, 32),
        ] {
            for n in [70usize, 71] {
                let pesos: Vec<u8> = if tipo == TipoGgml::Q8_0 {
                    let mut b = pesos_q8_identidad(n, k);
                    for (i, v) in b.iter_mut().enumerate() {
                        if i % 34 >= 2 {
                            *v = ((i * 29 % 253) as i32 - 126) as i8 as u8;
                        }
                    }
                    b
                } else {
                    (0..n * k / valores_bloque * bytes_bloque)
                        .map(|i| ((i * 97 + 29) % 251) as u8)
                        .collect()
                };
                for m in [4usize, 37] {
                    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.021).cos()).collect();
                    let mut con_buffer = vec![0f32; m * n];
                    let mut bloques = vec![0f32; m * n];
                    matmul_i8_directo((m, k, n), &x, &pesos, tipo, &mut con_buffer).unwrap();
                    matmul_i8_bloques((m, k, n), &x, &pesos, tipo, &mut bloques).unwrap();
                    let escala = con_buffer.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let peor = con_buffer
                        .iter()
                        .zip(&bloques)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        peor <= escala * 1e-4,
                        "{tipo:?} {m}x{n}: diferencia {peor} sobre {escala}"
                    );
                }
            }
        }
    }

    /// El fusionado de Q6_K —el de la matriz de salida en el decode— contra
    /// el camino con buffer, que desempaqueta con `quant::desempaquetar_i8`:
    /// si el kernel leyera mal el orden de los bits, no coincidirían. Se
    /// prueban el de AVX2 (por `matmul_i8_fusionado`) y el escalar.
    #[test]
    fn el_fusionado_de_q6k_coincide_con_el_de_buffer() {
        let (k, n) = (512usize, 40usize);
        let mut pesos: Vec<u8> = (0..n * k / 256 * 210)
            .map(|i| ((i * 97 + 29) % 251) as u8)
            .collect();
        // Una `d` de verdad en cada bloque: con bytes al azar puede salir un
        // f16 infinito.
        for b in pesos.chunks_exact_mut(210) {
            b[208..210].copy_from_slice(&0x2C00u16.to_le_bytes());
        }
        let m = 1;
        let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.021).cos()).collect();
        let mut con_buffer = vec![0f32; m * n];
        let mut fusionado = vec![0f32; m * n];
        matmul_i8_directo((m, k, n), &x, &pesos, TipoGgml::Q6K, &mut con_buffer).unwrap();
        matmul_i8_fusionado((m, k, n), &x, &pesos, TipoGgml::Q6K, &mut fusionado).unwrap();
        let escala = con_buffer.iter().fold(0f32, |a, v| a.max(v.abs()));
        for (a, b) in con_buffer.iter().zip(&fusionado) {
            assert!((a - b).abs() <= escala * 1e-5, "{a} contra {b}");
        }

        // El escalar, fila por fila, con la misma entrada cuantizada.
        let mut xq = vec![0i8; k];
        let mut xe = vec![0f32; k / 16];
        cuantizar_i8_en(&x, &mut xq, &mut xe, 16);
        let xs: Vec<f32> = xq
            .chunks(16)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        let bytes_fila = k / 256 * 210;
        for (j, esperado) in con_buffer.iter().enumerate() {
            let fila = &pesos[j * bytes_fila..(j + 1) * bytes_fila];
            let v = crate::kernels::scalar::dot_q6k_i8(fila, &xq, &xe, &xs);
            assert!(
                (v - esperado).abs() <= escala * 1e-5,
                "escalar: {v} contra {esperado}"
            );
        }
    }

    /// Varias proyecciones en una pasada tienen que dar **lo mismo bit a bit**
    /// que una llamada a `matmul_auto` por matriz: son los mismos kernels con
    /// la misma entrada. Se mezclan formatos con kernel fusionado (Q3_K,
    /// Q4_K, Q6_K) y con buffer (Q5_K, Q8_0), y tamaños que no son múltiplo
    /// del tramo de columnas.
    #[test]
    fn varias_en_una_pasada_dan_lo_mismo_que_de_a_una() {
        let k = 512usize;
        let armar = |tipo: TipoGgml, n: usize| -> Vec<u8> {
            if tipo == TipoGgml::Q8_0 {
                let mut b = pesos_q8_identidad(n, k);
                for (i, v) in b.iter_mut().enumerate() {
                    if i % 34 >= 2 {
                        *v = ((i * 29 % 253) as i32 - 126) as i8 as u8;
                    }
                }
                return b;
            }
            let bytes = tipo.bytes_por_bloque();
            let mut b: Vec<u8> = (0..n * k / 256 * bytes)
                .map(|i| ((i * 97 + 29) % 251) as u8)
                .collect();
            // `d` —y en Q4_K y Q5_K también `dmin`— finitas en cada bloque: con
            // bytes al azar puede salir un f16 infinito y el resultado, NaN.
            let campos: &[usize] = match tipo {
                TipoGgml::Q3K => &[108],
                TipoGgml::Q6K => &[208],
                _ => &[0, 2],
            };
            for bl in b.chunks_exact_mut(bytes) {
                for &d in campos {
                    bl[d..d + 2].copy_from_slice(&0x2C00u16.to_le_bytes());
                }
            }
            b
        };
        let matrices = [
            (TipoGgml::Q3K, 70usize),
            (TipoGgml::Q4K, 33),
            (TipoGgml::Q5K, 40),
            (TipoGgml::Q6K, 64),
            (TipoGgml::Q8_0, 5),
        ];
        let pesos: Vec<Vec<u8>> = matrices.iter().map(|(t, n)| armar(*t, *n)).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.021).cos()).collect();

        let mut juntas: Vec<Vec<f32>> = matrices.iter().map(|(_, n)| vec![0f32; *n]).collect();
        let refs: Vec<(&[u8], TipoGgml)> = pesos
            .iter()
            .zip(&matrices)
            .map(|(p, (t, _))| (p.as_slice(), *t))
            .collect();
        let mut destinos: Vec<&mut [f32]> = juntas.iter_mut().map(|v| v.as_mut_slice()).collect();
        matmul_varias_una_fila(&x, &refs, &mut destinos).unwrap();

        for (i, (tipo, n)) in matrices.iter().enumerate() {
            let mut sola = vec![0f32; *n];
            matmul_auto((1, k, *n), &x, &pesos[i], *tipo, &mut sola).unwrap();
            assert_eq!(sola, juntas[i], "{tipo:?}: no coincide bit a bit");
        }
    }

    #[test]
    fn las_escalas_se_expanden_por_carril() {
        // Sub-bloques de 16: cuatro carriles para cada uno.
        let mut d = vec![0f32; 16];
        expandir_escalas(&[1.0, 2.0, 3.0, 4.0], 16, &mut d);
        assert_eq!(
            d,
            [1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4.]
        );
        // De 32: los ocho iguales.
        let mut d = vec![0f32; 16];
        expandir_escalas(&[5.0, 6.0], 32, &mut d);
        assert_eq!(
            d,
            [5., 5., 5., 5., 5., 5., 5., 5., 6., 6., 6., 6., 6., 6., 6., 6.]
        );
    }

    #[test]
    fn avisa_si_las_formas_no_cuadran() {
        let pesos = pesos_q8_identidad(32, 64);
        let mut dst = vec![0f32; 32];
        assert!(matmul((1, 63, 32), &[0f32; 63], &pesos, TipoGgml::Q8_0, &mut dst).is_err());
        assert!(matmul((1, 64, 32), &[0f32; 10], &pesos, TipoGgml::Q8_0, &mut dst).is_err());
    }

    /// `matmul_auto` tiene que dar lo mismo sin importar qué camino elija.
    ///
    /// Se compara contra el de `f32`, que es la referencia, con la tolerancia
    /// que corresponde: los caminos de enteros cuantizan la entrada y el de
    /// `f32` no, así que no pueden coincidir al bit. Se prueban los dos
    /// formatos fusionables y los dos lados de cada umbral, que es donde un
    /// error de despacho se esconde.
    #[test]
    fn el_despacho_automatico_da_lo_mismo_por_cualquier_camino() {
        use crate::gguf::TipoGgml;

        let k = 256usize;
        let n = 64usize;
        // F32 y F16 entran a proposito: no tienen desempaquetado a `i8`, y
        // sin ellos el despacho los mandaba al camino de enteros y fallaba.
        for (tipo, bytes_bloque) in [
            (TipoGgml::Q4K, 144usize),
            (TipoGgml::Q3K, 110),
            (TipoGgml::F32, 1024),
            (TipoGgml::F16, 512),
        ] {
            // Para los cuantizados cualquier patron de bytes es valido. Para
            // F32 y F16 NO: bytes al azar dan infinitos y NaN, y entonces el
            // rango de la referencia sale `-inf` y la tolerancia no significa
            // nada. Asi que esos se arman desde numeros de verdad.
            let pesos: Vec<u8> = if crate::quant::sub_bloque(tipo).is_none() {
                let valores: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.019).sin()).collect();
                let mut b = Vec::new();
                crate::quant::cuantizar_bloques(tipo, &valores, &mut b).unwrap();
                b
            } else {
                (0..n * bytes_bloque)
                    .map(|i| ((i * 97 + 29) % 251) as u8)
                    .collect()
            };

            for m in [1usize, 2, 4, 8, BLOQUES_DESDE - 1, BLOQUES_DESDE, 37] {
                let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.021).cos()).collect();

                let mut referencia = vec![0f32; m * n];
                matmul((m, k, n), &x, &pesos, tipo, &mut referencia).unwrap();
                let rango = referencia.iter().cloned().fold(f32::MIN, f32::max)
                    - referencia.iter().cloned().fold(f32::MAX, f32::min);

                let mut auto = vec![0f32; m * n];
                matmul_auto((m, k, n), &x, &pesos, tipo, &mut auto).unwrap();

                let peor = referencia
                    .iter()
                    .zip(&auto)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    peor < rango * 0.03,
                    "{tipo:?} con {m} filas: el peor error fue {peor} sobre un rango de {rango}"
                );
            }
        }
    }

    /// Softmax tiene que sumar 1, ser invariante a sumarle una constante, y no
    /// dar `NaN` en los dos casos que lo producirian.
    #[test]
    fn softmax_suma_uno_y_no_desborda() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut x, 4).unwrap();
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > x[2] && x[2] > x[1], "no conserva el orden: {x:?}");

        // Invariante a sumar una constante, que es la propiedad que hace que
        // restar el maximo sea legitimo.
        let mut y = vec![101.0f32, 102.0, 103.0, 104.0];
        softmax(&mut y, 4).unwrap();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-6, "{x:?} contra {y:?}");
        }

        // Logits grandes: sin restar el maximo esto seria infinito sobre
        // infinito.
        let mut z = vec![90.0f32, 0.0, -90.0];
        softmax(&mut z, 3).unwrap();
        assert!(z.iter().all(|v| v.is_finite()), "{z:?}");
        assert!((z[0] - 1.0).abs() < 1e-6, "{z:?}");

        // Una fila entera enmascarada: uniforme y no NaN.
        let mut m = vec![f32::NEG_INFINITY; 4];
        softmax(&mut m, 4).unwrap();
        assert!(m.iter().all(|v| (*v - 0.25).abs() < 1e-6), "{m:?}");

        // Varias filas a la vez, cada una por su cuenta.
        let mut dos = vec![0.0f32, 10.0, 10.0, 0.0];
        softmax(&mut dos, 2).unwrap();
        assert!((dos[0] + dos[1] - 1.0).abs() < 1e-6);
        assert!((dos[2] + dos[3] - 1.0).abs() < 1e-6);
        assert!(dos[1] > dos[0] && dos[2] > dos[3]);

        assert!(softmax(&mut [0.0; 3], 2).is_err());
    }

    /// RoPE es una rotacion, asi que tiene que **conservar la norma** de cada
    /// par, y en la posicion cero tiene que no hacer nada.
    ///
    /// Las dos propiedades se pueden comprobar sin ninguna otra
    /// implementacion, y juntas atrapan casi cualquier error de signo o de
    /// indice.
    #[test]
    fn rope_rota_sin_cambiar_la_norma() {
        let (cabezas, dim) = (2usize, 8usize);
        let tokens = 3usize;
        let (cos, sin) = frecuencias_rope(dim, 10_000.0, 16);

        let original: Vec<f32> = (0..tokens * cabezas * dim)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();

        // Posicion cero: cos = 1 y sin = 0, asi que es la identidad.
        let mut x = original[..cabezas * dim].to_vec();
        rope(&mut x, (1, cabezas, dim), &cos, &sin, 0).unwrap();
        for (a, b) in x.iter().zip(&original) {
            assert!((a - b).abs() < 1e-6, "la posicion cero no es la identidad");
        }

        // Y en cualquier posicion, la norma de cada par se conserva.
        let mut x = original.clone();
        rope(&mut x, (tokens, cabezas, dim), &cos, &sin, 5).unwrap();
        let mitad = dim / 2;
        for t in 0..tokens {
            for h in 0..cabezas {
                let base = t * cabezas * dim + h * dim;
                for i in 0..mitad {
                    let antes = original[base + i].powi(2) + original[base + mitad + i].powi(2);
                    let despues = x[base + i].powi(2) + x[base + mitad + i].powi(2);
                    assert!(
                        (antes - despues).abs() < 1e-5,
                        "el par {i} de la cabeza {h} cambio de norma: {antes} a {despues}"
                    );
                }
            }
        }
        // Y algo tiene que haber cambiado, o el test no probaria nada.
        assert!(x.iter().zip(&original).any(|(a, b)| (a - b).abs() > 1e-3));

        // Pedir mas posiciones de las que tiene la tabla avisa.
        assert!(rope(&mut x, (tokens, cabezas, dim), &cos, &sin, 14).is_err());
        assert!(rope(&mut [0.0; 6], (1, 1, 3), &cos, &sin, 0).is_err());
    }

    /// RoPE y softmax tienen que coincidir con candle **valor por valor**.
    ///
    /// Es la prueba que de verdad importa en RoPE: hay dos convenciones —la de
    /// mitades partidas y la intercalada— y elegir la equivocada no da ningun
    /// error. El modelo carga, genera y dice cosas sin sentido.
    #[cfg(feature = "candle")]
    #[test]
    fn rope_y_softmax_coinciden_con_candle() {
        use candle_core::{DType, Device, Tensor};

        let dev = Device::Cpu;
        let (cabezas, dim, tokens, contexto) = (4usize, 64usize, 5usize, 32usize);
        let base = 1_000_000f32;
        let posicion = 7usize;

        // Las tablas, calculadas como las calcula el modelo.
        let (cos, sin) = frecuencias_rope(dim, base, contexto);
        let mitad = dim / 2;
        let theta: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();
        let theta_t = Tensor::new(theta.as_slice(), &dev).unwrap();
        let idx = Tensor::arange(0, contexto as u32, &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((contexto, 1))
            .unwrap()
            .matmul(&theta_t.reshape((1, mitad)).unwrap())
            .unwrap();
        let cos_c = idx.cos().unwrap();
        let sin_c = idx.sin().unwrap();

        // Las tablas propias tienen que coincidir con las suyas.
        let suyos = cos_c.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let peor = cos
            .iter()
            .zip(&suyos)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(peor < 1e-5, "las tablas de coseno difieren en {peor}");

        // Y la rotacion.
        let datos: Vec<f32> = (0..tokens * cabezas * dim)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();

        // candle quiere (lote, cabezas, tokens, dim); lo nuestro es
        // tokens x (cabezas*dim), asi que hay que transponer para comparar.
        let mut suyo = vec![0f32; datos.len()];
        for t in 0..tokens {
            for h in 0..cabezas {
                for d in 0..dim {
                    suyo[h * tokens * dim + t * dim + d] = datos[t * cabezas * dim + h * dim + d];
                }
            }
        }
        let x_c = Tensor::from_vec(suyo, (1, cabezas, tokens, dim), &dev).unwrap();
        let cos_n = cos_c.narrow(0, posicion, tokens).unwrap();
        let sin_n = sin_c.narrow(0, posicion, tokens).unwrap();
        let salida_c = candle_nn::rotary_emb::rope(&x_c.contiguous().unwrap(), &cos_n, &sin_n)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut nuestro = datos.clone();
        rope(&mut nuestro, (tokens, cabezas, dim), &cos, &sin, posicion).unwrap();

        let mut peor = 0f32;
        for t in 0..tokens {
            for h in 0..cabezas {
                for d in 0..dim {
                    let a = nuestro[t * cabezas * dim + h * dim + d];
                    let b = salida_c[h * tokens * dim + t * dim + d];
                    peor = peor.max((a - b).abs());
                }
            }
        }
        assert!(
            peor < 1e-5,
            "RoPE difiere de candle en {peor}: puede ser la convencion intercalada"
        );

        // Softmax, sobre valores con rango amplio.
        let logits: Vec<f32> = (0..3 * 128)
            .map(|i| (i as f32 * 0.17).sin() * 20.0)
            .collect();
        let suyo = candle_nn::ops::softmax_last_dim(
            &Tensor::from_vec(logits.clone(), (3, 128), &dev).unwrap(),
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        let mut nuestro = logits;
        softmax(&mut nuestro, 128).unwrap();
        let peor = nuestro
            .iter()
            .zip(&suyo)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(peor < 1e-6, "softmax difiere de candle en {peor}");
    }

    /// La atencion tiene que cumplir tres propiedades que no dependen de
    /// ninguna otra implementacion, y que juntas atrapan los errores de
    /// mascara, de reparto de GQA y de escala.
    #[test]
    fn la_atencion_promedia_y_respeta_la_causalidad() {
        let (cabezas, cabezas_kv, dim) = (4usize, 2usize, 8usize);
        let ancho_q = cabezas * dim;
        let ancho_kv = cabezas_kv * dim;

        // Con un solo token y una sola clave, la salida ES el valor: el
        // softmax de un elemento da 1.
        let q: Vec<f32> = (0..ancho_q).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..ancho_kv).map(|i| (i as f32 * 0.7).cos()).collect();
        let v: Vec<f32> = (0..ancho_kv).map(|i| i as f32).collect();
        let mut salida = vec![0f32; ancho_q];
        atencion(&q, &k, &v, (1, 1, cabezas, cabezas_kv, dim), &mut salida).unwrap();
        for h in 0..cabezas {
            let grupo = h / (cabezas / cabezas_kv);
            for d in 0..dim {
                assert!(
                    (salida[h * dim + d] - v[grupo * dim + d]).abs() < 1e-5,
                    "cabeza {h}, dim {d}: {} contra {}",
                    salida[h * dim + d],
                    v[grupo * dim + d]
                );
            }
        }

        // La salida es una combinacion convexa de los valores, asi que cada
        // componente queda entre el minimo y el maximo de esa columna. Si la
        // escala o el softmax estuvieran mal, esto seguiria valiendo; si la
        // suma de pesos no fuera 1, no.
        let total = 6usize;
        let tokens = 3usize;
        let k: Vec<f32> = (0..total * ancho_kv)
            .map(|i| (i as f32 * 0.21).sin())
            .collect();
        let v: Vec<f32> = (0..total * ancho_kv)
            .map(|i| (i as f32 * 0.13).cos())
            .collect();
        let q: Vec<f32> = (0..tokens * ancho_q)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let mut salida = vec![0f32; tokens * ancho_q];
        atencion(
            &q,
            &k,
            &v,
            (tokens, total, cabezas, cabezas_kv, dim),
            &mut salida,
        )
        .unwrap();
        for h in 0..cabezas {
            let grupo = h / (cabezas / cabezas_kv);
            for d in 0..dim {
                let columna: Vec<f32> = (0..total)
                    .map(|p| v[p * ancho_kv + grupo * dim + d])
                    .collect();
                let (lo, hi) = (
                    columna.iter().cloned().fold(f32::MAX, f32::min),
                    columna.iter().cloned().fold(f32::MIN, f32::max),
                );
                for t in 0..tokens {
                    let x = salida[t * ancho_q + h * dim + d];
                    assert!(
                        x >= lo - 1e-5 && x <= hi + 1e-5,
                        "la salida {x} se sale de [{lo}, {hi}]: no es una combinacion convexa"
                    );
                }
            }
        }

        // Causalidad: cambiar un valor FUTURO no puede mover la salida de un
        // token anterior. Es la prueba que detecta la mascara mal puesta.
        let mut v2 = v.clone();
        for d in 0..ancho_kv {
            v2[(total - 1) * ancho_kv + d] += 100.0;
        }
        let mut salida2 = vec![0f32; tokens * ancho_q];
        atencion(
            &q,
            &k,
            &v2,
            (tokens, total, cabezas, cabezas_kv, dim),
            &mut salida2,
        )
        .unwrap();
        // El ultimo token nuevo esta en la posicion `total-1`, asi que SI ve
        // el cambio; los anteriores, no.
        for t in 0..tokens - 1 {
            for i in 0..ancho_q {
                assert!(
                    (salida[t * ancho_q + i] - salida2[t * ancho_q + i]).abs() < 1e-5,
                    "el token {t} vio un valor futuro: la mascara causal esta mal"
                );
            }
        }
        assert!(
            (0..ancho_q).any(|i| (salida[(tokens - 1) * ancho_q + i]
                - salida2[(tokens - 1) * ancho_q + i])
                .abs()
                > 1e-3),
            "el ultimo token no vio su propia clave"
        );

        assert!(atencion(&q, &k, &v, (tokens, total, 4, 3, dim), &mut salida).is_err());
        assert!(atencion(&q, &k, &v, (7, 6, cabezas, cabezas_kv, dim), &mut salida).is_err());
    }

    /// Y la atencion tiene que coincidir con candle, armada con sus tensores.
    ///
    /// Es la prueba que vale: las propiedades de arriba no distinguen un
    /// reparto de GQA con `%` de uno con `/`, porque las dos reparten "algo" a
    /// cada cabeza. Esta si.
    #[cfg(feature = "candle")]
    #[test]
    fn la_atencion_coincide_con_candle() {
        use candle_core::{Device, Tensor};

        let dev = Device::Cpu;
        let (cabezas, cabezas_kv, dim) = (16usize, 2usize, 64usize);
        let (tokens, total) = (4usize, 9usize);
        let ancho_q = cabezas * dim;
        let ancho_kv = cabezas_kv * dim;
        let por_grupo = cabezas / cabezas_kv;

        let q: Vec<f32> = (0..tokens * ancho_q)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let k: Vec<f32> = (0..total * ancho_kv)
            .map(|i| (i as f32 * 0.017).cos())
            .collect();
        let v: Vec<f32> = (0..total * ancho_kv)
            .map(|i| (i as f32 * 0.023).sin())
            .collect();

        let mut nuestro = vec![0f32; tokens * ancho_q];
        atencion(
            &q,
            &k,
            &v,
            (tokens, total, cabezas, cabezas_kv, dim),
            &mut nuestro,
        )
        .unwrap();

        // Lo mismo con candle: (cabezas, tokens, dim) y (cabezas, total, dim),
        // con las cabezas de kv repetidas para cubrir el grupo.
        let mut qc = vec![0f32; tokens * ancho_q];
        for t in 0..tokens {
            for h in 0..cabezas {
                for d in 0..dim {
                    qc[h * tokens * dim + t * dim + d] = q[t * ancho_q + h * dim + d];
                }
            }
        }
        let mut kc = vec![0f32; cabezas * total * dim];
        let mut vc = vec![0f32; cabezas * total * dim];
        for h in 0..cabezas {
            let grupo = h / por_grupo;
            for p in 0..total {
                for d in 0..dim {
                    kc[h * total * dim + p * dim + d] = k[p * ancho_kv + grupo * dim + d];
                    vc[h * total * dim + p * dim + d] = v[p * ancho_kv + grupo * dim + d];
                }
            }
        }
        let qt = Tensor::from_vec(qc, (cabezas, tokens, dim), &dev).unwrap();
        let kt = Tensor::from_vec(kc, (cabezas, total, dim), &dev).unwrap();
        let vt = Tensor::from_vec(vc, (cabezas, total, dim), &dev).unwrap();

        let escala = 1.0 / (dim as f64).sqrt();
        let puntos = (qt.matmul(&kt.transpose(1, 2).unwrap()).unwrap() * escala).unwrap();

        // La mascara causal con cache caliente: el token t esta en la posicion
        // total-tokens+t.
        let desplazamiento = total - tokens;
        let mascara: Vec<f32> = (0..tokens)
            .flat_map(|t| {
                (0..total).map(move |p| {
                    if p <= desplazamiento + t {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mascara = Tensor::from_vec(mascara, (tokens, total), &dev)
            .unwrap()
            .broadcast_as((cabezas, tokens, total))
            .unwrap();
        let puntos = puntos.broadcast_add(&mascara).unwrap();
        let pesos = candle_nn::ops::softmax_last_dim(&puntos).unwrap();
        let suyo = pesos
            .matmul(&vt)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut peor = 0f32;
        for t in 0..tokens {
            for h in 0..cabezas {
                for d in 0..dim {
                    let a = nuestro[t * ancho_q + h * dim + d];
                    let b = suyo[h * tokens * dim + t * dim + d];
                    peor = peor.max((a - b).abs());
                }
            }
        }
        assert!(peor < 1e-5, "la atencion difiere de candle en {peor}");
    }
}
