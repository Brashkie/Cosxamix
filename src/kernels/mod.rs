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

/// El hueco para el kernel fusionado de Q4_K escrito a mano, con el andamio
/// —puente, envoltorio, tests y bench— ya puesto.
///
/// Detras de la feature `asm_q4k`, apagada por defecto: el cuerpo del kernel
/// lo escribe una persona en `kernels/q4k_avx2_*.s` y hasta que existan esos
/// archivos este modulo no se compila. El contrato esta documentado adentro, y
/// la especificacion de lo que tiene que calcular es `scalar::dot_q4k_i8`.
#[cfg(all(
    feature = "asm_q4k",
    any(hay_q4k_asm, hay_q4k_fila, hay_q4k_completo),
    target_arch = "x86_64"
))]
pub mod q4k_asm;

/// El mismo contrato completo de Q4_K en ensamblador para ARM64, con `sdot`,
/// que es la unica forma de usar esa instruccion desde este proyecto: Rust
/// estable no la expone y Zig no la genera. Fuera del despachador hasta que se
/// mida en ARM de verdad. Ver la cabecera del modulo.
#[cfg(all(feature = "asm_q4k", hay_q4k_neon, target_arch = "aarch64"))]
pub mod q4k_neon;

#[cfg(feature = "zig")]
pub mod zig;

/// El fusionado de Q3_K en ensamblador. Ver la cabecera del módulo.
#[cfg(all(feature = "asm_q3k", target_arch = "x86_64"))]
pub mod q3k_asm;

/// Kernels en C, compilados con `zig cc`. Ver la cabecera del módulo.
#[cfg(all(feature = "c", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod c;

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

    // F16C tambien: el kernel fusionado de Q4_K convierte los f16 de cada
    // bloque con `vcvtph2ps`, que es la mitad de su ventaja. Todo procesador
    // con AVX2 tiene F16C —salio antes, en Ivy Bridge y Piledriver—, asi que en
    // la practica no excluye a nadie; se pregunta igual porque una maquina
    // virtual puede esconder una extension y dejar la otra.
    #[cfg(target_arch = "x86_64")]
    if cpu_avx2() {
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

/// Si la CPU tiene AVX2, FMA y F16C. Se pregunta una vez.
///
/// `COSXAMIX_SIN_AVX2` hace como si no los tuviera, para probar en una
/// máquina moderna el camino de las que no los tienen. Solo tiene sentido en
/// la compilación sin `target-cpu=x86-64-v3`: en la de siempre el compilador
/// ya usó AVX2 en todo el binario.
pub fn cpu_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static AVX2: OnceLock<bool> = OnceLock::new();
        *AVX2.get_or_init(|| {
            std::env::var_os("COSXAMIX_SIN_AVX2").is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Si los kernels de Rust con AVX2 se pueden usar.
///
/// No alcanza con mirar el backend. Con `zig`, el elegido es Zig también en
/// una CPU sin AVX2 —en la compilación sin `target-cpu`, sus kernels tienen
/// una copia baseline—, y los kernels que Zig no tiene o pierde (`dot_i8`, el
/// Q4_K de AVX2) se tomaban de Rust con AVX2 dando por hecho que "Zig" quería
/// decir "x86-64 con AVX2": en una CPU sin AVX2, una instrucción ilegal.
pub fn usar_avx2() -> bool {
    match elegido() {
        Backend::Escalar => false,
        Backend::Avx2 => true,
        Backend::Zig => cpu_avx2(),
    }
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
        if usar_avx2() {
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
            && usar_avx2()
        {
            return unsafe { x86::dot_i8_escalado_avx2(a, b, esc_a, esc_b, bloque) };
        }
    }

    // En ARM es el único camino vectorizado; en x86-64 se llega acá sin AVX2
    // (la copia baseline de Zig, en SSE2) o con un bloque que el de AVX2 no
    // acepta.
    #[cfg(feature = "zig")]
    {
        if bloque.is_multiple_of(8) && a.len().is_multiple_of(bloque) && elegido() == Backend::Zig {
            return zig::dot_i8_escalado(a, b, esc_a, esc_b, bloque);
        }
    }

    scalar::dot_i8_escalado(a, b, esc_a, esc_b, bloque)
}

/// Producto punto de una fila en **Q4_K** contra una entrada en `i8`, sin
/// desempaquetar los pesos a ningun buffer.
///
/// Es el kernel fusionado. El camino de [`dot_i8_escalado`] necesita que
/// alguien haya desempaquetado la fila antes, y medido eso era el 94% del
/// trabajo de un hilo; aca cada sub-bloque se desempaqueta en registros y se
/// consume en el acto.
///
/// Solo Q4_K por ahora, y a proposito: es el unico formato cuyos valores son
/// enteros **sin signo** sin corrimiento, que es justo lo que `vpmaddubsw`
/// pide. Los demas necesitan restar antes —32 en Q6_K, 16 en Q5_0— y eso rompe
/// el rango sin signo.
pub fn dot_q4k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(
        xq.len().is_multiple_of(256),
        "dot_q4k_i8: la fila tiene que ser multiplo de 256"
    );
    assert_eq!(xe.len(), xq.len() / 32, "dot_q4k_i8: faltan escalas");
    assert_eq!(xs.len(), xe.len(), "dot_q4k_i8: faltan sumas");
    assert!(
        bloques.len() >= xq.len() / 256 * 144,
        "dot_q4k_i8: faltan bytes de pesos"
    );

    // El ensamblador del contrato COMPLETO va primero cuando esta: le gana a
    // los intrinsecos en tres pasadas de tres, entre 1,04x y 1,08x, con los
    // dos usando `vcvtph2ps`. Es el primer ensamblador del proyecto en el
    // camino caliente, y entro por la regla 3 como cualquier otro kernel:
    // midiendo. Los contratos por bloque y por fila no estan aca porque los
    // dos pierden contra los intrinsecos —ver DECISIONES §5—.
    #[cfg(all(feature = "asm_q4k", hay_q4k_completo, target_arch = "x86_64"))]
    {
        if usar_avx2() {
            return unsafe { q4k_asm::dot_q4k_i8_asm_completo(bloques, xq, xe, xs) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if usar_avx2() {
            return unsafe { x86::dot_q4k_i8_avx2(bloques, xq, xe, xs) };
        }
    }

    // En ARM64 con `dotprod`, el de C con `sdot` va primero: medido en el
    // runner ARM de CI (Neoverse N2), 162 ns contra 382 del de Zig, 2,4x
    // (DECISIONES §22). En un ARM sin la extension —A53, A72— sigue el de Zig.
    #[cfg(all(feature = "c", target_arch = "aarch64"))]
    {
        if usar_sdot() {
            return unsafe { c::dot_q4k_i8_neon(bloques, xq, xe, xs) };
        }
    }

    // Fuera de x86-64 el de Zig es el UNICO camino vectorizado sin `sdot`: el
    // de arriba usa `core::arch::x86_64` y ni siquiera compila. Sin esto, ARM
    // y Termux corren el bucle mas caliente del motor byte a byte.
    #[cfg(feature = "zig")]
    {
        if elegido() == Backend::Zig {
            return zig::dot_q4k_i8(bloques, xq, xe, xs);
        }
    }

    scalar::dot_q4k_i8(bloques, xq, xe, xs)
}

/// Si se usan los kernels de C con `sdot`: la CPU tiene `dotprod` y no se
/// forzó la referencia con `COSXAMIX_KERNEL=escalar`, que tiene que seguir
/// midiendo la referencia y nada más.
#[cfg(all(feature = "c", target_arch = "aarch64"))]
fn usar_sdot() -> bool {
    elegido() != Backend::Escalar && c::hay_dotprod()
}

/// Igual que [`dot_q4k_i8`] pero para **Q3_K**, que es el formato dominante
/// del modelo por defecto.
///
/// Necesita un rodeo que Q4_K no: sus valores llevan signo, −4 a 3, asi que se
/// les suma 4 y se descuenta con la suma de la entrada. El detalle, en
/// [`scalar::dot_q3k_i8`].
pub fn dot_q3k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(
        xq.len().is_multiple_of(256),
        "dot_q3k_i8: la fila tiene que ser multiplo de 256"
    );
    assert_eq!(xe.len(), xq.len() / 16, "dot_q3k_i8: faltan escalas");
    assert_eq!(xs.len(), xe.len(), "dot_q3k_i8: faltan sumas");
    assert!(
        bloques.len() >= xq.len() / 256 * 110,
        "dot_q3k_i8: faltan bytes de pesos"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if usar_avx2() {
            return unsafe { x86::dot_q3k_i8_avx2(bloques, xq, xe, xs) };
        }
    }

    // En ARM64 con `dotprod`, el de C con `sdot`: 367 ns contra 1998 de la
    // referencia en el runner ARM de CI, 5,4x. Es el primer camino
    // vectorizado de Q3_K en ARM, que es el formato del modelo por defecto.
    #[cfg(all(feature = "c", target_arch = "aarch64"))]
    {
        if usar_sdot() {
            return unsafe { c::dot_q3k_i8_neon(bloques, xq, xe, xs) };
        }
    }

    scalar::dot_q3k_i8(bloques, xq, xe, xs)
}

/// Producto de una fila Q6_K contra la entrada cuantizada en sub-bloques de
/// 16, fundido con el desempaquetado. Ver [`x86::dot_q6k_i8_avx2`].
///
/// Solo hay versión vectorizada en x86-64 con AVX2; en lo demás va la
/// escalar, y por eso `ops::matmul_auto` lo elige solo con AVX2.
pub fn dot_q6k_i8(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    assert!(
        xq.len().is_multiple_of(256),
        "dot_q6k_i8: la fila tiene que ser multiplo de 256"
    );
    assert_eq!(xe.len(), xq.len() / 16, "dot_q6k_i8: faltan escalas");
    assert_eq!(xs.len(), xe.len(), "dot_q6k_i8: faltan sumas");
    assert!(
        bloques.len() >= xq.len() / 256 * 210,
        "dot_q6k_i8: faltan bytes de pesos"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if usar_avx2() {
            return unsafe { x86::dot_q6k_i8_avx2(bloques, xq, xe, xs) };
        }
    }

    scalar::dot_q6k_i8(bloques, xq, xe, xs)
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

    /// El kernel fusionado tiene que dar lo mismo que desempaquetar y despues
    /// multiplicar, que es el camino ya verificado contra candle.
    ///
    /// No se exige igualdad de bits: los dos suman en otro orden y en `f32` el
    /// orden importa. Se exige que la diferencia sea despreciable frente al
    /// tamano del resultado, que es lo unico que se puede exigir.
    #[test]
    fn el_fusionado_coincide_con_desempaquetar_y_multiplicar() {
        use crate::gguf::TipoGgml;

        for (tipo, bytes_bloque) in [(TipoGgml::Q4K, 144usize), (TipoGgml::Q3K, 110)] {
            let n_bloques = 8usize; // 2048 valores, el ancho del 3B
            let k = n_bloques * 256;
            let sub = crate::quant::sub_bloque(tipo).unwrap();
            let pesos: Vec<u8> = (0..n_bloques * bytes_bloque)
                .map(|i| ((i * 131 + 17) % 251) as u8)
                .collect();

            let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.013).sin()).collect();
            let mut xq = vec![0i8; k];
            let mut xe = vec![0f32; k / sub];
            crate::ops::cuantizar_i8_en(&x, &mut xq, &mut xe, sub);
            let xs: Vec<f32> = xq
                .chunks(sub)
                .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
                .collect();

            // El camino de referencia: desempaquetar y multiplicar.
            let mut wq = vec![0i8; k];
            let mut we = vec![0f32; k / sub];
            let mut wd = vec![0f32; k / sub];
            assert_eq!(
                crate::quant::desempaquetar_i8(tipo, &pesos, &mut wq, &mut we, &mut wd).unwrap(),
                sub
            );
            let mut esperado = dot_i8_escalado(&wq, &xq, &we, &xe, sub);
            for b in 0..k / sub {
                esperado += wd[b] * xe[b] * xs[b];
            }

            let (escalar, despachado) = if tipo == TipoGgml::Q4K {
                (
                    scalar::dot_q4k_i8(&pesos, &xq, &xe, &xs),
                    dot_q4k_i8(&pesos, &xq, &xe, &xs),
                )
            } else {
                (
                    scalar::dot_q3k_i8(&pesos, &xq, &xe, &xs),
                    dot_q3k_i8(&pesos, &xq, &xe, &xs),
                )
            };

            #[cfg(target_arch = "x86_64")]
            if tipo != TipoGgml::Q4K && cpu_avx2() {
                #[cfg(feature = "c")]
                {
                    let en_c = unsafe { c::dot_q3k_i8(&pesos, &xq, &xe, &xs) };
                    assert!(
                        (en_c - escalar).abs() / esperado.abs().max(1.0) < 1e-5,
                        "q3k en C dio {en_c} y la referencia {escalar}"
                    );
                }
            }

            // El de Zig se comprueba aparte, porque en x86-64 el despachador
            // elige AVX2 y nunca lo llamaria: sin esto, el unico camino que
            // tiene ARM se quedaria sin test en la maquina donde se desarrolla.
            #[cfg(feature = "zig")]
            if tipo == TipoGgml::Q4K {
                let suyo = zig::dot_q4k_i8(&pesos, &xq, &xe, &xs);
                assert!(
                    (suyo - escalar).abs() / esperado.abs().max(1.0) < 1e-4,
                    "zig dio {suyo} y la referencia {escalar}"
                );
            }

            let escala = esperado.abs().max(1.0);
            for (quien, v) in [("escalar", escalar), ("despachado", despachado)] {
                assert!(
                    (v - esperado).abs() / escala < 1e-4,
                    "{tipo:?} {quien} dio {v} y el camino verificado {esperado}"
                );
            }

            // Y el escalar y el vectorizado tienen que coincidir entre si, que
            // es lo que detecta un error en los intrinsecos y no en la formula.
            assert!(
                (escalar - despachado).abs() / escala < 1e-5,
                "{tipo:?}: escalar {escalar} contra vectorizado {despachado}"
            );
        }
    }

    /// El desempaquetado sin ramas de las escalas de Q4_K tiene que dar
    /// **exactamente** lo mismo que `escala_y_minimo`, que es la versión
    /// legible y la que está verificada contra candle.
    ///
    /// Exacto y no aproximado: las escalas se calculan con las mismas
    /// multiplicaciones en el mismo orden, así que un solo bit de diferencia
    /// significa una máscara o un corrimiento mal puesto. Se prueban miles de
    /// bloques porque cada uno de los 96 bits de las escalas tiene que caer en
    /// su sitio.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn las_escalas_sin_ramas_coinciden_con_las_legibles() {
        if !usar_avx2() {
            return;
        }
        let mut estado = 0x9E37_79B9_7F4A_7C15u64;
        let mut azar = || {
            estado ^= estado << 13;
            estado ^= estado >> 7;
            estado ^= estado << 17;
            estado
        };
        for _ in 0..5000 {
            let mut bloque = [0u8; 144];
            for b in bloque.iter_mut() {
                *b = azar() as u8;
            }
            // `d` y `dmin` validos: con bytes al azar podrian ser infinito.
            let d = 0.001 + (azar() % 1000) as f32 * 1e-5;
            let dmin = 0.0005 + (azar() % 1000) as f32 * 1e-5;
            bloque[0..2].copy_from_slice(&crate::quant::f32_a_f16(d).to_le_bytes());
            bloque[2..4].copy_from_slice(&crate::quant::f32_a_f16(dmin).to_le_bytes());
            let xe: Vec<f32> = (0..8).map(|_| (azar() % 1000) as f32 * 1e-3).collect();
            let xs: Vec<f32> = (0..8).map(|_| (azar() % 4000) as f32 - 2000.0).collect();

            let mut desp = unsafe { std::arch::x86_64::_mm256_setzero_ps() };
            let nuevas = unsafe { x86::escalas_q4k(&bloque, &xe, &xs, &mut desp) };

            let d = crate::quant::leer_f16(&bloque, 0);
            for j in 0..8 {
                let (sc, _) = crate::quant::escala_y_minimo(j, &bloque[4..16]);
                let esperada = d * sc as f32 * xe[j];
                assert_eq!(
                    nuevas[j].to_bits(),
                    esperada.to_bits(),
                    "escala {j}: sin ramas {} y legible {esperada}",
                    nuevas[j]
                );
            }

            // Y el desplazamiento, con tolerancia: ahi hay un FMA, que
            // redondea una vez donde la cuenta escalar redondea dos.
            let dmin = crate::quant::leer_f16(&bloque, 2);
            let mut esperado = 0f32;
            for j in 0..8 {
                let (_, mn) = crate::quant::escala_y_minimo(j, &bloque[4..16]);
                esperado += dmin * mn as f32 * xe[j] * xs[j];
            }
            let mut v = [0f32; 8];
            unsafe { std::arch::x86_64::_mm256_storeu_ps(v.as_mut_ptr(), desp) };
            let obtenido: f32 = v.iter().sum();
            assert!(
                (obtenido - esperado).abs() <= esperado.abs() * 1e-5 + 1e-6,
                "desplazamiento: sin ramas {obtenido} y legible {esperado}"
            );
        }
    }
}
