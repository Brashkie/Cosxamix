//! El kernel fusionado de Q4_K con el bucle en ensamblador.
//!
//! El cuerpo está en `kernels/q4k_avx2_win64.s` y `kernels/q4k_avx2_sysv.s`,
//! uno por convención de llamada; el de System V sirve para ELF y Mach-O a la
//! vez. Cada plataforma pide solo el suyo. Acá está todo lo demás: el puente,
//! el envoltorio seguro, los tests contra la referencia y contra los
//! intrínsecos, y el bench.
//!
//! Va detrás de la feature `asm_q4k`, apagada por defecto:
//!
//! ```text
//! cargo test  --features asm_q4k
//! cargo bench --features asm_q4k --bench kernels -- dot_q4k_i8
//! ```
//!
//! ## Dos contratos, y por qué el segundo
//!
//! **Por bloque**, el primero:
//!
//! ```c
//! void cosxamix_q4k_pares(const uint8_t *qs,    // 32 bytes por par
//!                         const int8_t  *xq,    // 64 valores por par
//!                         const float   *esc,   // 2 escalas por par
//!                         size_t         pares,
//!                         float         *acc);  // 8 f32, lee y escribe
//! ```
//!
//! Se llama una vez por bloque, con `pares = 4`. Es correcto y mide 6,4x sobre
//! el escalar, pero 1,16x detrás de los intrínsecos, y la causa está contada:
//! de las 82 instrucciones por bloque, **14 son de armar y desarmar la
//! llamada** —leer `acc`, reconstruir las dos constantes, guardar `acc`,
//! volver—. Los intrínsecos hacen eso una vez por fila; este, ocho veces.
//!
//! **Por fila**, el que cierra esa distancia:
//!
//! ```c
//! void cosxamix_q4k_fila(const uint8_t *bloques,  // 144 bytes por bloque
//!                        const int8_t  *xq,       // 256 valores por bloque
//!                        const float   *esc,      // 8 escalas por bloque
//!                        size_t         n_bloques,
//!                        float         *acc);     // 8 f32, lee y escribe
//! ```
//!
//! Los **mismos cinco argumentos en las mismas posiciones**, así que la
//! entrada de la ABI no cambia. Lo que cambia es qué recorre:
//!
//! - `bloques` apunta a los bloques **enteros**, con su encabezado. Dentro de
//!   cada uno, `qs` empieza 16 bytes después del comienzo, y del bloque
//!   siguiente lo separan 144. Es el único de los tres punteros que **no** es
//!   contiguo a lo largo de la fila;
//! - `xq` y `esc` **sí** son contiguos de punta a punta: 256 valores y 8
//!   escalas por bloque, uno detrás del otro. Lo que avanzan por par es lo
//!   mismo que en el contrato por bloque;
//! - el acumulador y las dos constantes se arman **una vez** y viven en
//!   registros toda la fila. Eso es exactamente lo que el contrato por bloque
//!   pagaba ocho veces.
//!
//! Rust calcula antes las escalas de toda la fila —de a 64 bloques, que entra
//! en 2 KB de pila— con el mismo código que los intrínsecos, y llama una vez
//! por tanda. Una fila del 3B son 8 bloques en la atención y 43 en el FFN: una
//! sola llamada en los dos casos.
//!
//! **La especificación ejecutable es [`super::scalar::dot_q4k_i8`]**, en Rust
//! y ya verificada contra candle sobre los modelos reales. Los tests comparan
//! contra ella.
//!
//! ## El tercero: el ensamblador hace todo
//!
//! Medido aislado, el bucle por fila es 14% más rápido que el por bloque y
//! cuesta menos que el kernel entero de intrínsecos. De punta a punta perdía
//! por el envoltorio: Rust calculaba las escalas en una pasada aparte y se
//! las entregaba por un buffer. Este contrato deja de entregárselas:
//!
//! ```c
//! void cosxamix_q4k_completo(const uint8_t *bloques,  // 144 bytes por bloque
//!                            const int8_t  *xq,       // 256 valores por bloque
//!                            const float   *xe,       // 8 por bloque
//!                            const float   *xs,       // 8 por bloque
//!                            size_t         n_bloques,
//!                            float         *acc);     // 16 f32, lee y escribe
//! ```
//!
//! **Seis argumentos**, uno más que los otros dos: en Windows x64 el quinto
//! llega en `[rsp+40]` y el sexto en `[rsp+48]`; en System V, `r8` y `r9`.
//!
//! Lo que tiene que calcular, por bloque, es exactamente lo que hacen hoy
//! [`super::x86::escalas_q4k`] y el bucle de [`super::x86::dot_q4k_i8_avx2`],
//! que son la especificación legible:
//!
//! | bytes del bloque | qué es |
//! |---|---|
//! | 0–1 | `d`, en `f16` |
//! | 2–3 | `dmin`, en `f16` |
//! | 4–15 | las 8 escalas y los 8 mínimos, de 6 bits, empaquetados |
//! | 16–143 | `qs`: 4 pares de 32 bytes, nibbles bajos el sub-bloque par y altos el impar |
//!
//! - **`d` y `dmin`** son `f16`. `vcvtph2ps` los pasa a `f32` con una
//!   instrucción, y está en todo procesador con AVX2 —F16C es anterior—. El
//!   Rust de hoy los convierte por software, así que acá el ensamblador tiene
//!   algo que el otro camino no usa;
//! - **las escalas `sc[j]` y los mínimos `mn[j]`**, `j` de 0 a 7, salen de los
//!   12 bytes con el desempaquetado sin ramas de `escalas_q4k`: tres palabras de
//!   32 bits y unas máscaras. Está comentado línea por línea ahí;
//! - **la escala del sub-bloque `j`** es `d · sc[j] · xe[j]`, y es la que
//!   multiplica sus productos en el bucle, igual que ahora;
//! - **el desplazamiento** acumula `dmin · mn[j] · xe[j] · xs[j]` **en
//!   positivo**, carril `j` del vector, en `acc[8..16]`. Rust lo resta al
//!   final.
//!
//! `acc` son **16** `f32`: los 8 primeros son el acumulador de los productos,
//! como en los otros contratos, y los 8 siguientes el del desplazamiento. Los
//! dos entran y salen, así que se puede llamar por tandas sin perder nada.
//! Rust hace solo la reducción final: `Σ acc[0..8] − Σ acc[8..16]`, con el
//! mismo código que los intrínsecos.
//!
//! La ganancia esperable está acotada por lo medido: el bucle aislado cuesta
//! 61 ns y el kernel entero de intrínsecos, con las escalas, 92. Lo que se
//! juega es cuánto del trabajo de las escalas consigue superponerse con el
//! bucle cuando están intercalados en la misma pasada.
//!
//! ## Cómo se detecta cuál hay
//!
//! `build.rs` busca en el `.s` del target la **etiqueta** de cada función
//! —`cosxamix_q4k_pares:`, `cosxamix_q4k_fila:` y `cosxamix_q4k_completo:`,
//! con los dos puntos— y habilita las que encuentre. Con la etiqueta y no con el nombre a secas,
//! porque el nombre aparece en los comentarios de la cabecera y eso daba por
//! presente una función que no estaba. Puede haber cualquier combinación, y
//! el bench compara las que haya.
//!
//! ## Las tres cosas que dependen de la plataforma
//!
//! 1. **dónde llegan los argumentos.** Windows x64 usa `rcx rdx r8 r9` y el
//!    quinto en `[rsp+40]` —8 de la dirección de retorno más 32 de sombra—.
//!    System V usa `rdi rsi rdx rcx r8`;
//! 2. **qué registros hay que preservar.** En Windows x64 los `xmm6` a
//!    `xmm15` son no volátiles: si se usan, hay que salvarlos y restaurarlos.
//!    Con el contrato por fila eso pasa una vez por fila y no por bloque, así
//!    que deja de ser caro si hace falta un registro más;
//! 3. **cómo se declara el símbolo en cada formato de objeto.** COFF pide
//!    `.def`/`.scl`/`.endef`, ELF pide `.type`, y Mach-O no acepta `.type` y
//!    decora los símbolos con un guion bajo. El `sysv.s` define el símbolo con
//!    y sin guion bajo y sin la directiva de ELF.
//!
//! **La sintaxis es AT&T**, la que `rustc` emite por defecto, declarada con
//! `options(att_syntax)`. `dot_avx2_*.s` está en Intel; conviven porque cada
//! `global_asm!` declara la suya.

#![cfg(all(
    feature = "asm_q4k",
    any(hay_q4k_asm, hay_q4k_fila, hay_q4k_completo),
    target_arch = "x86_64"
))]

use std::arch::global_asm;
use std::arch::x86_64::_mm256_loadu_ps;
#[cfg(any(hay_q4k_asm, hay_q4k_fila))]
use std::arch::x86_64::_mm256_setzero_ps;

#[cfg(target_os = "windows")]
global_asm!(
    include_str!("../../kernels/q4k_avx2_win64.s"),
    options(att_syntax)
);

#[cfg(not(target_os = "windows"))]
global_asm!(
    include_str!("../../kernels/q4k_avx2_sysv.s"),
    options(att_syntax)
);

unsafe extern "C" {
    /// Contrato por bloque: `pares * 64` valores, `acc` entra y sale.
    #[cfg(hay_q4k_asm)]
    fn cosxamix_q4k_pares(
        qs: *const u8,
        xq: *const i8,
        esc: *const f32,
        pares: usize,
        acc: *mut f32,
    );

    /// Contrato por fila: `n_bloques` bloques enteros, `acc` entra y sale.
    #[cfg(hay_q4k_fila)]
    fn cosxamix_q4k_fila(
        bloques: *const u8,
        xq: *const i8,
        esc: *const f32,
        n_bloques: usize,
        acc: *mut f32,
    );

    /// Contrato completo: el ensamblador calcula las escalas y el
    /// desplazamiento. `acc` son 16 `f32`, entran y salen.
    #[cfg(hay_q4k_completo)]
    fn cosxamix_q4k_completo(
        bloques: *const u8,
        xq: *const i8,
        xe: *const f32,
        xs: *const f32,
        n_bloques: usize,
        acc: *mut f32,
    );
}

/// El fusionado entero en ensamblador: escalas, desplazamiento y bucle, en una
/// sola pasada y una sola llamada por fila.
///
/// Rust no hace nada más que llamarlo y reducir los dos acumuladores, con el
/// mismo código que los intrínsecos.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. Quien llame tiene que haberlo comprobado.
#[cfg(hay_q4k_completo)]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4k_i8_asm_completo(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());
    debug_assert!(bloques.len() >= xq.len() / 256 * 144);

    let mut acc = [0f32; 16];
    unsafe {
        cosxamix_q4k_completo(
            bloques.as_ptr(),
            xq.as_ptr(),
            xe.as_ptr(),
            xs.as_ptr(),
            xq.len() / 256,
            acc.as_mut_ptr(),
        );
        super::x86::reducir_f32(_mm256_loadu_ps(acc.as_ptr()))
            - super::x86::reducir_f32(_mm256_loadu_ps(acc.as_ptr().add(8)))
    }
}

/// El fusionado con el bucle en ensamblador, **una llamada por bloque**.
///
/// Las escalas y la reducción final las hace Rust con el mismo código que los
/// intrínsecos —[`super::x86::escalas_q4k`] y [`super::x86::reducir_f32`]—,
/// así que lo único que difiere entre los dos es el bucle.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. Quien llame tiene que haberlo comprobado.
#[cfg(hay_q4k_asm)]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4k_i8_asm(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());

    let mut acc = [0f32; 8];
    let mut desp = _mm256_setzero_ps();

    for bl in 0..xq.len() / 256 {
        let o = bl * 144;
        let escalas = unsafe {
            super::x86::escalas_q4k(
                &bloques[o..o + 144],
                &xe[bl * 8..bl * 8 + 8],
                &xs[bl * 8..bl * 8 + 8],
                &mut desp,
            )
        };
        unsafe {
            cosxamix_q4k_pares(
                bloques.as_ptr().add(o + 16),
                xq.as_ptr().add(bl * 256),
                escalas.as_ptr(),
                4,
                acc.as_mut_ptr(),
            );
        }
    }

    unsafe {
        super::x86::reducir_f32(_mm256_loadu_ps(acc.as_ptr())) - super::x86::reducir_f32(desp)
    }
}

/// Cuántos bloques se mandan por llamada en el contrato por fila.
///
/// Sus escalas tienen que estar calculadas antes, y 64 bloques son 512 `f32`,
/// 2 KB de pila. Una fila del 3B son 8 o 43 bloques, una llamada; la del FFN
/// del 7B, 74, dos.
#[cfg(hay_q4k_fila)]
const TANDA: usize = 64;

/// El fusionado con el bucle en ensamblador, **una llamada por fila**.
///
/// Calcula primero las escalas de toda la tanda y después llama una vez, así
/// que el ensamblador arma el acumulador y las constantes una sola vez y no
/// una por bloque, que es lo que costaba el 17% del contrato por bloque.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. Quien llame tiene que haberlo comprobado.
#[cfg(hay_q4k_fila)]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4k_i8_asm_fila(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));
    debug_assert_eq!(xe.len(), xq.len() / 32);
    debug_assert_eq!(xs.len(), xe.len());

    let n = xq.len() / 256;
    let mut acc = [0f32; 8];
    let mut desp = _mm256_setzero_ps();
    // SIN poner en cero. `[0f32; TANDA * 8]` escribia 2 KB en cada llamada
    // aunque una fila de 8 bloques usa 256 bytes: medido, era la mayor parte
    // de lo que el contrato por fila perdia contra el por bloque. Cada
    // posicion que el ensamblador lee se escribe antes, asi que nunca se lee
    // memoria sin inicializar.
    let mut escalas = [const { std::mem::MaybeUninit::<f32>::uninit() }; TANDA * 8];

    let mut desde = 0usize;
    while desde < n {
        let cuantos = (n - desde).min(TANDA);
        for i in 0..cuantos {
            let bl = desde + i;
            let o = bl * 144;
            let e = unsafe {
                super::x86::escalas_q4k(
                    &bloques[o..o + 144],
                    &xe[bl * 8..bl * 8 + 8],
                    &xs[bl * 8..bl * 8 + 8],
                    &mut desp,
                )
            };
            // Copia sin comprobacion de limites: `i < cuantos <= TANDA`, asi
            // que `i * 8 + 8 <= TANDA * 8` siempre.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    e.as_ptr(),
                    escalas.as_mut_ptr().add(i * 8) as *mut f32,
                    8,
                );
            }
        }
        unsafe {
            cosxamix_q4k_fila(
                bloques.as_ptr().add(desde * 144),
                xq.as_ptr().add(desde * 256),
                escalas.as_ptr() as *const f32,
                cuantos,
                acc.as_mut_ptr(),
            );
        }
        desde += cuantos;
    }

    unsafe {
        super::x86::reducir_f32(_mm256_loadu_ps(acc.as_ptr())) - super::x86::reducir_f32(desp)
    }
}

/// SOLO el bucle por bloque, con las escalas ya calculadas.
///
/// Mide el ensamblador **aislado** del calculo de escalas, que es lo que permite
/// saber si una diferencia de punta a punta viene del bucle o de lo que lo
/// rodea. Asi se supo que el bucle por fila es 14% mas rapido que este, aunque
/// el kernel completo por fila salga mas lento.
///
/// # Safety
///
/// AVX2 y FMA; `esc` con 8 escalas por bloque.
#[cfg(hay_q4k_asm)]
pub unsafe fn bucle_por_bloque(bloques: &[u8], xq: &[i8], esc: &[f32], acc: &mut [f32; 8]) {
    for bl in 0..xq.len() / 256 {
        unsafe {
            cosxamix_q4k_pares(
                bloques.as_ptr().add(bl * 144 + 16),
                xq.as_ptr().add(bl * 256),
                esc.as_ptr().add(bl * 8),
                4,
                acc.as_mut_ptr(),
            );
        }
    }
}

/// SOLO el bucle por fila, con las escalas ya calculadas.
///
/// # Safety
///
/// AVX2 y FMA; `esc` con 8 escalas por bloque.
#[cfg(hay_q4k_fila)]
pub unsafe fn bucle_por_fila(bloques: &[u8], xq: &[i8], esc: &[f32], acc: &mut [f32; 8]) {
    unsafe {
        cosxamix_q4k_fila(
            bloques.as_ptr(),
            xq.as_ptr(),
            esc.as_ptr(),
            xq.len() / 256,
            acc.as_mut_ptr(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Kernel = unsafe fn(&[u8], &[i8], &[f32], &[f32]) -> f32;

    /// Los kernels de ensamblador que haya en el `.s`, con su nombre.
    fn kernels() -> Vec<(&'static str, Kernel)> {
        // El modulo solo se compila si hay al menos uno, asi que nunca queda
        // vacio.
        [
            #[cfg(hay_q4k_asm)]
            ("por bloque", dot_q4k_i8_asm as Kernel),
            #[cfg(hay_q4k_fila)]
            ("por fila", dot_q4k_i8_asm_fila as Kernel),
            #[cfg(hay_q4k_completo)]
            ("completo", dot_q4k_i8_asm_completo as Kernel),
        ]
        .into_iter()
        .collect()
    }

    /// F16C tambien: el contrato completo usa `vcvtph2ps`. Todo procesador con
    /// AVX2 lo tiene, pero un test no deberia suponerlo.
    fn hay_avx2() -> bool {
        std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
    }

    /// Bloques Q4_K con bytes al azar pero **encabezados validos**.
    ///
    /// Los cuatro primeros bytes de cada bloque son `d` y `dmin` en `f16`, y
    /// un byte alto con el exponente lleno los vuelve infinito o NaN. Con 17
    /// bloques al azar paso, y como NaN no es igual a NaN el test fallaba
    /// aunque el kernel y la referencia dijeran lo mismo.
    fn bloques_validos(n_bloques: usize, semilla: usize) -> Vec<u8> {
        let mut b: Vec<u8> = (0..n_bloques * 144)
            .map(|i| ((i * 131 + semilla) % 251) as u8)
            .collect();
        for bl in 0..n_bloques {
            let o = bl * 144;
            let d = crate::quant::f32_a_f16(0.01 + (bl % 50) as f32 * 0.003);
            let dmin = crate::quant::f32_a_f16(0.005 + (bl % 50) as f32 * 0.001);
            b[o..o + 2].copy_from_slice(&d.to_le_bytes());
            b[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
        }
        b
    }

    /// Una fila de `n_bloques` con la entrada cuantizada como lo hace el
    /// multiplicador.
    fn fila(n_bloques: usize, semilla: usize) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<f32>) {
        let n = n_bloques * 256;
        let pesos = bloques_validos(n_bloques, semilla);
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
        let mut xq = vec![0i8; n];
        let mut xe = vec![0f32; n / 32];
        crate::ops::cuantizar_i8(&x, &mut xq, &mut xe);
        let xs: Vec<f32> = xq
            .chunks(32)
            .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
            .collect();
        (pesos, xq, xe, xs)
    }

    /// Cada kernel que haya tiene que dar lo mismo que la referencia escalar,
    /// que es la especificación.
    ///
    /// La tolerancia es relativa y chica: suman en distinto orden, así que la
    /// igualdad de bits sería pedir de más, pero un error de índice o de signo
    /// se sale de esto por varios órdenes de magnitud.
    #[test]
    fn coincide_con_la_referencia() {
        if !hay_avx2() {
            return;
        }
        let (pesos, xq, xe, xs) = fila(8, 17);
        let esperado = crate::kernels::scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs);
        for (nombre, k) in kernels() {
            let nuestro = unsafe { k(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
                "{nombre}: dio {nuestro} y la referencia {esperado}"
            );
        }
    }

    /// Y lo mismo que los intrínsecos, que es contra quien compite.
    #[test]
    fn coincide_con_los_intrinsecos() {
        if !hay_avx2() {
            return;
        }
        let (pesos, xq, xe, xs) = fila(8, 17);
        let suyo = unsafe { crate::kernels::x86::dot_q4k_i8_avx2(&pesos, &xq, &xe, &xs) };
        for (nombre, k) in kernels() {
            let nuestro = unsafe { k(&pesos, &xq, &xe, &xs) };
            assert!(
                (nuestro - suyo).abs() / suyo.abs().max(1.0) < 1e-5,
                "{nombre}: dio {nuestro} y los intrinsecos {suyo}"
            );
        }
    }

    /// Filas de un bloque, de pocos, y de más de una tanda.
    ///
    /// Uno: el bucle puede dar una sola vuelta, el error clásico. Diecisiete:
    /// el bucle de bloques del contrato por fila tiene que avanzar 144 en los
    /// pesos y 256 en la entrada, y un paso mal contado aparece acá y no con
    /// uno. Ochenta: más de las 64 de una tanda, así que se parte en dos
    /// llamadas y el acumulador tiene que pasar de una a la otra.
    #[test]
    fn aguanta_uno_pocos_y_mas_de_una_tanda() {
        if !hay_avx2() {
            return;
        }
        for n_bloques in [1usize, 2, 17, 80] {
            let (pesos, xq, xe, xs) = fila(n_bloques, 41);
            let esperado = crate::kernels::scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs);
            assert!(
                esperado.is_finite(),
                "con {n_bloques} bloques la referencia dio {esperado}: datos mal armados"
            );
            for (nombre, k) in kernels() {
                let nuestro = unsafe { k(&pesos, &xq, &xe, &xs) };
                assert!(
                    (nuestro - esperado).abs() / esperado.abs().max(1.0) < 1e-4,
                    "{nombre} con {n_bloques} bloques: dio {nuestro} y la referencia {esperado}"
                );
            }
        }
    }
}
