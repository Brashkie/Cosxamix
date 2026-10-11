//! Producto punto en Zig, para comparar contra la referencia en Rust.
//!
//! Esto existe para contestar una pregunta con un número: ¿un kernel en Zig le
//! gana a uno en Rust? Si no le gana, este archivo se borra — es la regla 3 de
//! `src/kernels/mod.rs`.
//!
//! Lo que Zig aporta acá y se puede ver en el código:
//!
//!   * `@Vector(N, f32)` es SIMD portable **en el lenguaje y estable**. En Rust
//!     estable hoy habría que bajar a los intrínsecos de cada arquitectura
//!     (ver `src/kernels/x86.rs`), porque `std::simd` sigue en nightly.
//!   * `comptime` permitiría generar el mismo kernel para varios anchos de
//!     registro desde una sola fuente, sin macros ni copias.
//!
//! Se compila con `zig build-lib` desde `build.rs`, solo con la feature `zig`.

const std = @import("std");

/// Ancho del vector: 8 f32 = 256 bits = un registro AVX2.
const ANCHO = 8;
const V = @Vector(ANCHO, f32);

/// Producto punto. ABI de C para que Rust lo llame por FFI.
///
/// Recibe punteros y longitud en vez de un slice porque el otro lado es Rust y
/// los slices no cruzan la frontera: un `*const f32` y un `usize` sí son lo
/// mismo en los dos lenguajes.
export fn cosxamix_dot_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    // Cuatro acumuladores por el mismo motivo que la versión de Rust: la
    // latencia de la multiplicación-suma es de varios ciclos y con una sola
    // cadena cada iteración espera a la anterior.
    var s0: V = @splat(0);
    var s1: V = @splat(0);
    var s2: V = @splat(0);
    var s3: V = @splat(0);

    var i: usize = 0;

    while (i + ANCHO * 4 <= n) : (i += ANCHO * 4) {
        s0 += cargar(a, i) * cargar(b, i);
        s1 += cargar(a, i + ANCHO) * cargar(b, i + ANCHO);
        s2 += cargar(a, i + ANCHO * 2) * cargar(b, i + ANCHO * 2);
        s3 += cargar(a, i + ANCHO * 3) * cargar(b, i + ANCHO * 3);
    }

    while (i + ANCHO <= n) : (i += ANCHO) {
        s0 += cargar(a, i) * cargar(b, i);
    }

    var total = @reduce(.Add, (s0 + s1) + (s2 + s3));

    // La cola. Olvidarla es el error clásico: con n múltiplo de 8 todo pasa y
    // con 4097 elementos se pierde uno.
    while (i < n) : (i += 1) {
        total += a[i] * b[i];
    }

    return total;
}

/// Lee ANCHO elementos desde `p[i]` como un vector.
///
/// La alineación declarada es la de un `f32` (4 bytes), **no la del vector**
/// (32). Es lo único que el otro lado garantiza: un slice de Rust está alineado
/// para su elemento y nada más. Pedir `@alignCast` acá sería afirmar algo que
/// no es cierto, y en una CPU que exige alineación —o con la comprobación de
/// seguridad activada— se cae. Una carga desalineada en AVX2 no cuesta nada
/// medible frente a eso.
inline fn cargar(p: [*]const f32, i: usize) V {
    return @as(*align(@alignOf(f32)) const V, @ptrCast(p + i)).*;
}

/// Normalización RMS. Misma fuente para x86-64, ARM64 y Android.
///
/// Es el segundo kernel y sirve para comprobar que la frontera generaliza: la
/// forma es distinta a la de `dot` —dos pasadas, una raíz, y escribe en un
/// buffer de salida en vez de devolver un escalar— y aun así cruza la FFI con
/// punteros y longitud, sin nada nuevo.
export fn cosxamix_rmsnorm_f32(
    salida: [*]f32,
    entrada: [*]const f32,
    peso: [*]const f32,
    n: usize,
    eps: f32,
) void {
    if (n == 0) return;

    // Primera pasada: suma de cuadrados.
    var s0: V = @splat(0);
    var s1: V = @splat(0);

    var i: usize = 0;
    while (i + ANCHO * 2 <= n) : (i += ANCHO * 2) {
        const a = cargar(entrada, i);
        const b = cargar(entrada, i + ANCHO);
        s0 += a * a;
        s1 += b * b;
    }
    while (i + ANCHO <= n) : (i += ANCHO) {
        const a = cargar(entrada, i);
        s0 += a * a;
    }

    var suma = @reduce(.Add, s0 + s1);
    while (i < n) : (i += 1) {
        suma += entrada[i] * entrada[i];
    }

    // La raíz y la división, una sola vez: son dos operaciones contra las 2n
    // de las pasadas.
    const escala = 1.0 / @sqrt(suma / @as(f32, @floatFromInt(n)) + eps);
    const vescala: V = @splat(escala);

    // Segunda pasada: escalar y aplicar el peso.
    i = 0;
    while (i + ANCHO <= n) : (i += ANCHO) {
        const r = cargar(entrada, i) * vescala * cargar(peso, i);
        @as(*align(@alignOf(f32)) V, @ptrCast(salida + i)).* = r;
    }
    while (i < n) : (i += 1) {
        salida[i] = entrada[i] * escala * peso[i];
    }
}

/// No hace nada. Existe para medir el coste de cruzar la FFI.
///
/// La guía de teoría afirmaba que invocar un kernel externo cuesta "menos de
/// 1,5 ns por llamada". Ese número no salía de ninguna medición, y el propio
/// documento dice que en Cosxamix nada se decide sin medir. Esto permite
/// comprobarlo: el banco llama a esto y a una función de Rust igual de vacía,
/// y la diferencia es el coste real de la frontera.
export fn cosxamix_noop(x: f32) f32 {
    return x;
}

/// SiLU (o Swish): `x * sigmoid(x)`, que es `x / (1 + exp(-x))`.
///
/// Es la activacion del MLP de Qwen2, y el MLP es el 79% de los pesos del
/// modelo, asi que esto corre sobre 11008 elementos por capa y por token.
///
/// ## Por que este kernel esta en Zig y no en Rust con intrinsecos
///
/// Aca Zig aporta algo que no es cobertura de plataforma: **`@exp` funciona
/// sobre un vector**. `@exp(@Vector(8, f32))` baja al intrinseco vectorial de
/// LLVM y se vectoriza solo.
///
/// Rust no tiene eso. `f32::exp` es escalar y no hay forma de aplicarlo a un
/// `__m256` sin escribir a mano una aproximacion polinomica con su reduccion
/// de rango — unas cuarenta lineas delicadas donde un coeficiente mal puesto
/// no rompe nada, solo da respuestas un poco peores.
///
/// O sea que para esta operacion Zig da gratis lo que en Rust hay que ganarse.
/// Es la primera vez en este proyecto que la ventaja de Zig no es "corre en
/// ARM" sino "el lenguaje trae la herramienta".
export fn cosxamix_silu_f32(salida: [*]f32, entrada: [*]const f32, n: usize) void {
    const uno: V = @splat(1.0);
    var i: usize = 0;

    while (i + ANCHO <= n) : (i += ANCHO) {
        const x = cargar(entrada, i);
        // `@exp` sobre el vector entero: ocho exponenciales de una vez.
        const y = x / (uno + @exp(-x));
        const p: *align(@alignOf(f32)) V = @ptrCast(salida + i);
        p.* = y;
    }

    // La cola, de a uno. Olvidarse de esto es el error clasico de un kernel
    // vectorizado: con n multiplo de 8 todos los tests pasan.
    while (i < n) : (i += 1) {
        const x = entrada[i];
        salida[i] = x / (1.0 + @exp(-x));
    }
}
