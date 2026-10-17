//! La frontera por donde entra el código de otros lenguajes.
//!
//! Cosxamix no se escribe solo en Rust: hay sitio para Zig, C, ensamblador y lo
//! que haga falta. Pero el riesgo de eso no son los lenguajes, es tener **N
//! sitios distintos** donde cada uno se mete. Entonces hay una sola frontera,
//! y es esta.
//!
//! ## Las tres reglas
//!
//! 1. **Todo kernel tiene una implementación de referencia en Rust portable.**
//!    Es la que define qué resultado es el correcto y la que corre cuando la
//!    CPU no tiene nada mejor. Nunca se borra.
//!
//! 2. **Toda implementación se compara contra la referencia**, en los tests.
//!    Un kernel que va más rápido y da otro número no es más rápido: está mal.
//!
//! 3. **Toda implementación tiene que justificarse midiendo**: ganarle a la
//!    referencia en velocidad, **o** habilitar una plataforma donde no había
//!    aceleración. Si no hace ninguna de las dos, se borra — no se mantiene
//!    un backend porque "ya lo escribimos" ni porque el lenguaje sea
//!    interesante. El kernel de Zig está acá por lo segundo: pierde un 8%
//!    en x86-64 y es el único que corre en ARM y en Android.
//!
//! ## Cómo se elige
//!
//! En tiempo de ejecución, mirando lo que la CPU de verdad soporta — no en
//! tiempo de compilación. Un binario compilado en una máquina con AVX-512 y
//! ejecutado en una que no lo tiene se cae con instrucción ilegal; detectar al
//! arrancar es lo que permite publicar **un solo binario** que anda en todas y
//! usa lo mejor de cada una. Eso es "más soporte", no menos.
//!
//! `COSXAMIX_KERNEL` fuerza uno concreto. Existe para poder medir uno contra otro
//! en la misma máquina, que es justo lo que pide la regla 3.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod x86;

/// El mismo `dot`, con el bucle caliente escrito a mano en ensamblador.
///
/// Existe para contestar midiendo una pregunta que el proyecto venia
/// razonando: si los intrinsecos dejan algo sobre la mesa. No esta enchufado
/// al despachador hasta que gane, que es lo que pide la regla 3.
#[cfg(target_arch = "x86_64")]
pub mod x86_asm;

#[cfg(feature = "zig")]
pub mod zig;

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Rust portable. La referencia: siempre disponible, siempre correcta.
    Escalar,
    /// Rust con intrínsecos AVX2 + FMA.
    Avx2,
    /// Zig, enlazado por FFI con ABI de C.
    Zig,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Escalar => "escalar (Rust portable)",
            Self::Avx2 => "avx2 (Rust + intrínsecos)",
            Self::Zig => "zig",
        })
    }
}

/// Los backends que esta compilación trae Y esta CPU puede ejecutar.
pub fn disponibles() -> Vec<Backend> {
    // El `mut` hace falta en x86-64 y con la feature `zig`, y en ningun otro
    // caso: en un ARM sin `zig` no se empuja nada y el compilador avisa. No se
    // puede poner el `mut` tras un `cfg` porque la mutabilidad no es un
    // atributo, asi que se silencia aqui, acotado a esta funcion.
    #[allow(unused_mut)]
    let mut v = vec![Backend::Escalar];

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        v.push(Backend::Avx2);
    }

    #[cfg(feature = "zig")]
    v.push(Backend::Zig);

    v
}

/// El que se usa. Se resuelve una vez y se cachea: preguntarle a la CPU en
/// cada llamada a un kernel sería pagar la detección dentro del bucle caliente.
pub fn elegido() -> Backend {
    use std::sync::OnceLock;
    static ELEGIDO: OnceLock<Backend> = OnceLock::new();

    *ELEGIDO.get_or_init(|| {
        let disp = disponibles();

        if let Ok(forzado) = std::env::var("COSXAMIX_KERNEL") {
            let pedido = match forzado.to_lowercase().as_str() {
                "escalar" => Some(Backend::Escalar),
                "avx2" => Some(Backend::Avx2),
                "zig" => Some(Backend::Zig),
                _ => None,
            };
            match pedido {
                // Se avisa y se sigue en vez de caer en silencio al que toque:
                // quien forzó un backend está midiendo, y medir el equivocado
                // sin enterarse es peor que no medir.
                Some(b) if disp.contains(&b) => return b,
                Some(b) => tracing::warn!("COSXAMIX_KERNEL={forzado}: {b} no está disponible acá"),
                None => tracing::warn!("COSXAMIX_KERNEL={forzado} no es un backend conocido"),
            }
        }

        // El último de la lista es el más específico.
        *disp.last().unwrap()
    })
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

/// Producto punto de dos vectores.
///
/// Es el primer kernel porque es el que más aparece: una multiplicación de
/// matrices son productos punto, y la atención también. Si la frontera no
/// sirve para este, no sirve para ninguno.
///
/// # Pánico
///
/// Si los dos vectores no miden lo mismo. Es un error de programación del
/// llamador, no una condición de ejecución: devolver un `Result` obligaría a
/// comprobarlo en el bucle caliente para algo que nunca debería pasar.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot: longitudes distintas ({} y {})",
        a.len(),
        b.len()
    );

    match elegido() {
        Backend::Escalar => scalar::dot(a, b),

        #[cfg(target_arch = "x86_64")]
        // Seguro: `elegido()` solo devuelve Avx2 si la CPU reporta avx2 y fma.
        Backend::Avx2 => unsafe { x86::dot_avx2(a, b) },
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx2 => scalar::dot(a, b),

        #[cfg(feature = "zig")]
        Backend::Zig => zig::dot(a, b),
        #[cfg(not(feature = "zig"))]
        Backend::Zig => scalar::dot(a, b),
    }
}

/// Normalización RMS: `salida[i] = entrada[i] / sqrt(media(entrada²) + eps) * peso[i]`.
///
/// Es el segundo kernel, y entró para comprobar que esta frontera generaliza.
/// Su forma es distinta a la de `dot` —dos pasadas, una raíz, y escribe en un
/// buffer en vez de devolver un escalar— y aun así no hizo falta cambiar nada
/// del mecanismo de selección. Eso es lo que se quería saber.
///
/// # Pánico
///
/// Si los tres vectores no miden lo mismo.
/// SiLU: `x * sigmoid(x)`, la activacion del MLP.
///
/// **No hay camino AVX2 aca, y es el dato interesante del kernel.** Los otros
/// dos de la frontera tienen tres implementaciones; este tiene dos, porque
/// vectorizar una exponencial en Rust con intrinsecos obliga a escribir a mano
/// una aproximacion polinomica con reduccion de rango, mientras que el `@exp`
/// de Zig acepta un vector y baja al intrinseco de LLVM.
///
/// Asi que en x86-64 esto elige **Zig si esta compilado, y la referencia si
/// no**, que es al reves de lo que pasa con `dot` y `rmsnorm`. Es la primera
/// vez que la ventaja de Zig en este proyecto no es "corre en ARM" sino "el
/// lenguaje trae la herramienta".
/// Producto punto de `i8`, acumulando en `i32`.
///
/// Es la operacion de una multiplicacion cuantizada de verdad: pesos y
/// activacion como enteros chicos, y la escala aparte. AVX2 hace 32 por
/// instruccion contra los 8 del camino en f32, y ahi esta la diferencia entre
/// nuestra multiplicacion y la de candle.
///
/// No hay camino Zig todavia: la instruccion que hace falta, `vpmaddubsw`, no
/// tiene equivalente portable en `@Vector`, asi que en Zig habria que escribir
/// el mismo rodeo con intrinsecos y deja de ser una fuente para todas las
/// plataformas. Queda pendiente mirarlo para NEON, donde `sdot` hace esto en
/// una sola instruccion y no hace falta ningun rodeo.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: los vectores miden distinto");

    #[cfg(target_arch = "x86_64")]
    {
        if matches!(elegido(), Backend::Avx2 | Backend::Zig) {
            return unsafe { x86::dot_i8_avx2(a, b) };
        }
    }
    scalar::dot_i8(a, b)
}

/// Producto punto cuantizado por bloques, con una escala por bloque en cada
/// lado.
///
/// Es la operacion que de verdad necesita una multiplicacion cuantizada, y es
/// una sola operacion y no `dot_i8` en un bucle por un motivo medido: la suma
/// horizontal del final es barata una vez y carisima una por bloque.
pub fn dot_i8_escalado(a: &[i8], b: &[i8], esc_a: &[f32], esc_b: &[f32], bloque: usize) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "dot_i8_escalado: los vectores miden distinto"
    );

    // AVX2 primero en x86-64, y no es la eleccion que parecia. El de Zig es
    // mas correcto —acepta todo el rango de i8— y mas portable, pero el banco
    // dijo otra cosa: 85 ns contra 259 en n=2048, o sea **3x**.
    //
    // El motivo es estructural y no se arregla afinando: ensanchar i8 a i32
    // para multiplicar da 8 carriles por vector, mientras el rodeo del signo
    // de `vpmaddubsw` trabaja con 32. El rodeo existe justamente para eso.
    //
    // Asi que Zig se queda por el segundo brazo de la regla 3 —habilita ARM,
    // donde el camino de Rust usa `core::arch::x86_64` y no existe— y no por
    // velocidad.
    #[cfg(target_arch = "x86_64")]
    {
        if (bloque == 16 || bloque.is_multiple_of(32))
            && a.len().is_multiple_of(bloque.max(32))
            && matches!(elegido(), Backend::Avx2 | Backend::Zig)
        {
            return unsafe { x86::dot_i8_escalado_avx2(a, b, esc_a, esc_b, bloque) };
        }
    }

    #[cfg(all(feature = "zig", not(target_arch = "x86_64")))]
    {
        if bloque.is_multiple_of(8) && a.len().is_multiple_of(bloque) && elegido() == Backend::Zig {
            return zig::dot_i8_escalado(a, b, esc_a, esc_b, bloque);
        }
    }

    scalar::dot_i8_escalado(a, b, esc_a, esc_b, bloque)
}

pub fn silu(salida: &mut [f32], entrada: &[f32]) {
    assert_eq!(
        salida.len(),
        entrada.len(),
        "silu: salida y entrada miden distinto"
    );

    #[cfg(feature = "zig")]
    {
        if elegido() == Backend::Zig {
            return zig::silu(salida, entrada);
        }
    }
    scalar::silu(salida, entrada)
}

pub fn rmsnorm(salida: &mut [f32], entrada: &[f32], peso: &[f32], eps: f32) {
    assert_eq!(
        salida.len(),
        entrada.len(),
        "rmsnorm: salida y entrada miden distinto"
    );
    assert_eq!(
        entrada.len(),
        peso.len(),
        "rmsnorm: entrada y peso miden distinto"
    );

    match elegido() {
        Backend::Escalar => scalar::rmsnorm(salida, entrada, peso, eps),

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => unsafe { x86::rmsnorm_avx2(salida, entrada, peso, eps) },
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx2 => scalar::rmsnorm(salida, entrada, peso, eps),

        #[cfg(feature = "zig")]
        Backend::Zig => zig::rmsnorm(salida, entrada, peso, eps),
        #[cfg(not(feature = "zig"))]
        Backend::Zig => scalar::rmsnorm(salida, entrada, peso, eps),
    }
}

#[cfg(test)]
mod tests {

    /// Sub-bloques de 16, que es lo que usan Q2_K, Q3_K y Q6_K.
    #[test]
    fn dot_i8_escalado_con_bloques_de_16() {
        for k in [32usize, 64, 2048] {
            let nb = k / 16;
            let a: Vec<i8> = (0..k)
                .map(|i| ((i * 37) % 251) as i32 as i8)
                .map(|v| v.max(-127))
                .collect();
            let b: Vec<i8> = (0..k)
                .map(|i| ((i * 53) % 251) as i32 as i8)
                .map(|v| v.max(-127))
                .collect();
            let ea: Vec<f32> = (0..nb).map(|i| 0.01 + i as f32 * 0.003).collect();
            let eb: Vec<f32> = (0..nb).map(|i| 0.02 - i as f32 * 0.0007).collect();
            let esperado = scalar::dot_i8_escalado(&a, &b, &ea, &eb, 16);
            let obtenido = dot_i8_escalado(&a, &b, &ea, &eb, 16);
            let tol = esperado.abs() * 1e-5 + 1e-4;
            assert!(
                (esperado - obtenido).abs() <= tol,
                "k={k}: {esperado} contra {obtenido}"
            );
        }
    }

    /// El de Zig contra la referencia, y con -128 adentro.
    ///
    /// Ese valor esta FUERA del contrato del camino de AVX2 —el rodeo del
    /// signo lo devuelve con el signo cambiado— y DENTRO del de Zig, que
    /// ensancha a i32. Es la diferencia concreta entre los dos, probada y no
    /// afirmada.
    #[cfg(feature = "zig")]
    #[test]
    fn zig_aguanta_todo_el_rango_de_i8() {
        for k in [32usize, 64, 2048] {
            let nb = k / 32;
            // -128 puesto a proposito, no esperando que caiga por
            // aritmetica modular: es el valor que se quiere probar.
            let mut a: Vec<i8> = (0..k).map(|i| ((i * 37) % 251) as i32 as i8).collect();
            let mut b: Vec<i8> = (0..k).map(|i| ((i * 53) % 251) as i32 as i8).collect();
            a[0] = i8::MIN;
            a[k / 2] = i8::MIN;
            b[1] = i8::MIN;
            b[k - 1] = i8::MIN;
            let ea: Vec<f32> = (0..nb).map(|i| 0.01 + i as f32 * 0.003).collect();
            let eb: Vec<f32> = (0..nb).map(|i| 0.02 - i as f32 * 0.001).collect();

            let esperado = scalar::dot_i8_escalado(&a, &b, &ea, &eb, 32);
            let obtenido = zig::dot_i8_escalado(&a, &b, &ea, &eb, 32);
            let tol = esperado.abs() * 1e-5 + 1e-4;
            assert!(
                (esperado - obtenido).abs() <= tol,
                "k={k}: referencia {esperado} contra zig {obtenido}"
            );
        }
    }

    /// El camino de AVX2 contra la referencia, con bloques y escalas.
    #[test]
    fn dot_i8_escalado_coincide_con_la_referencia() {
        for k in [32usize, 64, 256, 2048] {
            let nb = k / 32;
            let a: Vec<i8> = (0..k)
                .map(|i| ((i * 37) % 251) as i32 as i8)
                .map(|v| v.max(-127))
                .collect();
            let b: Vec<i8> = (0..k)
                .map(|i| ((i * 53) % 251) as i32 as i8)
                .map(|v| v.max(-127))
                .collect();
            let ea: Vec<f32> = (0..nb).map(|i| 0.01 + i as f32 * 0.003).collect();
            let eb: Vec<f32> = (0..nb).map(|i| 0.02 - i as f32 * 0.001).collect();

            let esperado = scalar::dot_i8_escalado(&a, &b, &ea, &eb, 32);
            let obtenido = dot_i8_escalado(&a, &b, &ea, &eb, 32);
            let tol = esperado.abs() * 1e-5 + 1e-4;
            assert!(
                (esperado - obtenido).abs() <= tol,
                "k={k}: referencia {esperado} contra avx2 {obtenido}"
            );
        }
    }

    /// El `dot_i8` de AVX2 contra la referencia, con igualdad EXACTA.
    ///
    /// Son enteros: no hay nada que redondear distinto, asi que una diferencia
    /// de uno significa que el rodeo del signo esta mal o que la cola se
    /// quedo afuera. Se prueban los extremos de i8 a proposito, que es donde
    /// el truco de `vpmaddubsw` podria saturar: -128 no tiene opuesto en i8.
    #[test]
    fn dot_i8_coincide_con_la_referencia() {
        for n in [0usize, 1, 7, 31, 32, 33, 64, 100, 2048, 2049] {
            // Dentro de [-127, 127], que es lo que producen los
            // cuantizadores de GGML y lo unico que el kernel promete.
            let a: Vec<i8> = (0..n)
                .map(|i| ((i * 37) % 255) as i32 as i8)
                .map(|v| v.max(-127))
                .collect();
            let b: Vec<i8> = (0..n)
                .map(|i| (((i * 53) % 255) as i32 - 127) as i8)
                .map(|v| v.max(-127))
                .collect();
            assert_eq!(scalar::dot_i8(&a, &b), dot_i8(&a, &b), "n={n}");
        }
    }

    /// El peor caso dentro del contrato: todo en el extremo del rango.
    ///
    /// -128 no esta porque **no es un valor valido** para este kernel: en i8
    /// no tiene opuesto, y el rodeo del signo de AVX2 devuelve el producto con
    /// el signo cambiado. Los cuantizadores de GGML calculan la escala como
    /// `amax / 127`, asi que producen [-127, 127] y nunca -128. Lo encontro
    /// este test cuando probaba -128, y la respuesta correcta no fue arreglar
    /// el kernel sino escribir la precondicion.
    #[test]
    fn dot_i8_en_los_extremos() {
        let n = 2048;
        for (x, y) in [(127i8, 127i8), (-127, 127), (-127, -127), (127, -127)] {
            let a = vec![x; n];
            let b = vec![y; n];
            assert_eq!(
                scalar::dot_i8(&a, &b),
                dot_i8(&a, &b),
                "con {x} y {y} el resultado difiere"
            );
        }
    }

    /// La SiLU de Zig contra la referencia.
    ///
    /// Con tolerancia y no igualdad exacta, a proposito: `@exp` de Zig baja al
    /// intrinseco vectorial de LLVM y la referencia usa el `exp` escalar de
    /// la libreria matematica. No son la misma funcion, son la misma
    /// operacion por dos caminos, asi que los ultimos bits difieren.
    ///
    /// La tolerancia es relativa porque SiLU crece sin limite: un error
    /// absoluto de 1e-5 no significa lo mismo en x=0,1 que en x=80.
    #[cfg(feature = "zig")]
    #[test]
    fn silu_de_zig_coincide_con_la_referencia() {
        for n in [0usize, 1, 7, 8, 31, 1024, 4097] {
            // Rango ancho a proposito: la exponencial se desborda rapido y es
            // donde una aproximacion mala se delata.
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 40.0).collect();
            let mut esperado = vec![0f32; n];
            let mut obtenido = vec![0f32; n];
            scalar::silu(&mut esperado, &x);
            zig::silu(&mut obtenido, &x);
            for (k, (e, o)) in esperado.iter().zip(&obtenido).enumerate() {
                let tol = 1e-5 * e.abs().max(1.0);
                assert!(
                    (e - o).abs() <= tol,
                    "n={n} i={k} x={}: referencia {e} contra zig {o}",
                    x[k]
                );
            }
        }
    }
    use super::*;

    /// Datos con decimales y signos mezclados: con enteros chicos, una suma mal
    /// ordenada o una conversión perdida daría igual y el test no vería nada.
    fn datos(n: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let b = (0..n)
            .map(|i| (i as f32 * 0.11).cos() * 2.5 - 1.0)
            .collect();
        (a, b)
    }

    /// La regla 2, en código: todo backend tiene que coincidir con la referencia.
    #[test]
    fn todos_los_backends_coinciden_con_la_referencia() {
        // Se prueban longitudes alrededor de los múltiplos de 8, que es el
        // ancho de un registro AVX2: ahí es donde un kernel vectorizado se
        // olvida de la cola y deja elementos sin sumar.
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1024, 4096, 4097] {
            let (a, b) = datos(n);
            let esperado = scalar::dot(&a, &b);

            for backend in disponibles() {
                let obtenido = match backend {
                    Backend::Escalar => scalar::dot(&a, &b),
                    #[cfg(target_arch = "x86_64")]
                    Backend::Avx2 => unsafe { x86::dot_avx2(&a, &b) },
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::Avx2 => continue,
                    #[cfg(feature = "zig")]
                    Backend::Zig => zig::dot(&a, &b),
                    #[cfg(not(feature = "zig"))]
                    Backend::Zig => continue,
                };

                // Tolerancia relativa: sumar en otro orden cambia el último
                // bit, y exigir igualdad exacta rechazaría kernels correctos.
                let tol = 1e-4 * esperado.abs().max(1.0);
                assert!(
                    (obtenido - esperado).abs() <= tol,
                    "{backend} con n={n}: esperaba {esperado}, obtuve {obtenido}",
                );
            }
        }
    }

    #[test]
    fn la_referencia_siempre_esta_disponible() {
        assert!(disponibles().contains(&Backend::Escalar));
    }

    #[test]
    fn el_vector_vacio_da_cero() {
        assert_eq!(dot(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "longitudes distintas")]
    fn longitudes_distintas_avisan() {
        dot(&[1.0, 2.0], &[1.0]);
    }
    /// La regla 2 para el segundo kernel.
    #[test]
    fn rmsnorm_todos_los_backends_coinciden() {
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 100, 1024, 4096, 4097] {
            let (entrada, peso) = datos(n);
            let mut esperado = vec![0.0f32; n];
            scalar::rmsnorm(&mut esperado, &entrada, &peso, 1e-5);

            for backend in disponibles() {
                let mut obtenido = vec![0.0f32; n];
                match backend {
                    Backend::Escalar => scalar::rmsnorm(&mut obtenido, &entrada, &peso, 1e-5),
                    #[cfg(target_arch = "x86_64")]
                    Backend::Avx2 => unsafe {
                        x86::rmsnorm_avx2(&mut obtenido, &entrada, &peso, 1e-5)
                    },
                    #[cfg(not(target_arch = "x86_64"))]
                    Backend::Avx2 => continue,
                    #[cfg(feature = "zig")]
                    Backend::Zig => zig::rmsnorm(&mut obtenido, &entrada, &peso, 1e-5),
                    #[cfg(not(feature = "zig"))]
                    Backend::Zig => continue,
                }

                for (i, (o, e)) in obtenido.iter().zip(&esperado).enumerate() {
                    let tol = 1e-3 * e.abs().max(1.0);
                    assert!(
                        (o - e).abs() <= tol,
                        "{backend} con n={n}, indice {i}: esperaba {e}, obtuve {o}",
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "miden distinto")]
    fn rmsnorm_longitudes_distintas_avisan() {
        let mut s = [0.0f32; 2];
        rmsnorm(&mut s, &[1.0, 2.0], &[1.0], 1e-5);
    }
}
