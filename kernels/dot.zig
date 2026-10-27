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
//! No exporta nada por sí mismo: los nombres que ve Rust los pone
//! `exports.zig`, una vez sin sufijo (`lib.zig`) y en x86-64 otra con `_v3`
//! (`lib_v3.zig`).

const std = @import("std");

/// Ancho del vector: 8 f32 = 256 bits = un registro AVX2.
const ANCHO = 8;
const V = @Vector(ANCHO, f32);

/// Producto punto. ABI de C para que Rust lo llame por FFI.
///
/// Recibe punteros y longitud en vez de un slice porque el otro lado es Rust y
/// los slices no cruzan la frontera: un `*const f32` y un `usize` sí son lo
/// mismo en los dos lenguajes.
pub fn cosxamix_dot_f32(a: [*]const f32, b: [*]const f32, n: usize) callconv(.C) f32 {
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
pub fn cosxamix_rmsnorm_f32(
    salida: [*]f32,
    entrada: [*]const f32,
    peso: [*]const f32,
    n: usize,
    eps: f32,
) callconv(.C) void {
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
pub fn cosxamix_noop(x: f32) callconv(.C) f32 {
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
pub fn cosxamix_silu_f32(salida: [*]f32, entrada: [*]const f32, n: usize) callconv(.C) void {
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

/// Producto punto cuantizado por bloques, con una escala por bloque de cada
/// lado: `suma_b esc_a[b] * esc_b[b] * dot(bloque_b)`.
///
/// ## Por que esto en Zig y no solo en AVX2
///
/// En el camino de Rust este kernel usa `vpmaddubsw`, que multiplica un
/// operando SIN signo por uno con signo, asi que hace falta el rodeo de mover
/// el signo con `vpsignb`. Ese rodeo trae una precondicion incomoda: **con
/// -128 da el resultado con el signo cambiado**, porque en i8 ese valor no
/// tiene opuesto.
///
/// Aca no hace falta ningun rodeo. `@Vector` permite **ensanchar** i8 a i32 y
/// multiplicar directo, asi que el kernel es correcto en todo el rango de i8
/// sin precondiciones. Y como es portable, es ademas el unico camino
/// vectorizado de esta operacion en ARM, donde el de Rust ni existe.
///
/// ## La estructura, que es lo que importa
///
/// El acumulador de i32 se vacia **una vez por bloque** hacia un acumulador de
/// f32 vectorial, y la suma horizontal va al final de todo. Reducir por bloque
/// es lo que hizo que la primera version de esto saliera mas lenta que el
/// camino de f32 pese a usar instrucciones mas rapidas.
pub fn cosxamix_dot_i8_escalado_f32(
    a: [*]const i8,
    b: [*]const i8,
    esc_a: [*]const f32,
    esc_b: [*]const f32,
    n: usize,
    bloque: usize,
) callconv(.C) f32 {
    const VI8 = @Vector(8, i8);
    const VI32 = @Vector(8, i32);
    const VF = @Vector(8, f32);

    var acumulador: VF = @splat(0.0);
    const n_bloques = n / bloque;

    var bl: usize = 0;
    while (bl < n_bloques) : (bl += 1) {
        var parcial: VI32 = @splat(0);
        var i: usize = bl * bloque;
        const fin = i + bloque;

        while (i + 8 <= fin) : (i += 8) {
            const pa: *align(@alignOf(i8)) const VI8 = @ptrCast(a + i);
            const pb: *align(@alignOf(i8)) const VI8 = @ptrCast(b + i);
            // Ensanchar y multiplicar: sin rodeos de signo, correcto en todo
            // el rango de i8 incluido -128.
            parcial += @as(VI32, pa.*) * @as(VI32, pb.*);
        }

        const escala: VF = @splat(esc_a[bl] * esc_b[bl]);
        acumulador += @as(VF, @floatFromInt(parcial)) * escala;
    }

    return @reduce(.Add, acumulador);
}

// ─── El kernel fusionado, que en x86 lo hace AVX2 y aca nadie ───────────────

/// Lee un `f16` little-endian de un flujo de bytes.
///
/// Zig tiene `f16` nativo, asi que esto es un `bitCast` y una ampliacion: no
/// hace falta la aritmetica de exponente y mantisa que pide hacerlo a mano.
fn leerF16(b: [*]const u8, i: usize) f32 {
    const bits: u16 = @as(u16, b[i]) | (@as(u16, b[i + 1]) << 8);
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

/// La escala y el minimo de 6 bits del sub-bloque `j` de un bloque Q4_K.
///
/// Los doce bytes guardan ocho pares de 6 bits de una forma que no es
/// regular: los primeros cuatro pares van enteros en los ocho primeros bytes,
/// y los otros cuatro reparten sus dos bits altos en los cuatro ultimos.
fn escalaYMinimo(j: usize, q: [*]const u8) struct { u8, u8 } {
    if (j < 4) {
        return .{ q[j] & 63, q[j + 4] & 63 };
    }
    return .{
        (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
        (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
    };
}

/// Producto punto de una fila en **Q4_K** contra una entrada en `i8`, sin
/// desempaquetar los pesos a ningun buffer.
///
/// ## Por que existe teniendo la version de AVX2
///
/// Porque la de AVX2 **no existe fuera de x86-64**. En ARM y en Termux, que
/// son objetivo declarado del proyecto, el camino de Rust usa
/// `core::arch::x86_64` y no compila, asi que el despachador cae a la
/// referencia escalar: byte a byte, en el bucle mas caliente del motor. Esta
/// funcion es la que evita eso, y entra por el **segundo brazo de la regla 3**
/// —habilita una plataforma— y no por ganarle a nadie en velocidad.
///
/// ## Lo que puede y lo que no
///
/// No reproduce el truco de `vpmaddubsw`, que suma pares de productos de 8
/// bits dentro de un registro de 16 y es lo que le da a AVX2 sus 32 carriles.
/// Aca se ensancha a `i32` y se multiplica, que son 8 carriles con NEON de 128
/// bits — cuatro veces menos ancho.
///
/// A cambio es **mas simple y mas portable**: `@Vector` deja que LLVM elija la
/// instruccion de cada arquitectura, y en ARMv8.2 con `dotprod` puede salir
/// `sdot`, que hace exactamente el producto punto de cuatro `i8` en un `i32`.
/// Si el objetivo no lo tiene, sale la version con `smull` y sigue andando.
///
/// `n` son los valores de la fila, multiplo de 256. `xe` y `xs` tienen un
/// valor por cada 32.
pub fn cosxamix_dot_q4k_i8_f32(
    bloques: [*]const u8,
    xq: [*]const i8,
    xe: [*]const f32,
    xs: [*]const f32,
    n: usize,
) callconv(.C) f32 {
    const VU8 = @Vector(ANCHO, u8);
    const VI8 = @Vector(ANCHO, i8);
    const VI32 = @Vector(ANCHO, i32);

    const mascara: VU8 = @splat(0x0F);
    const cuatro: @Vector(ANCHO, u3) = @splat(4);

    var total: f32 = 0.0;
    const n_bloques = n / 256;

    var bl: usize = 0;
    while (bl < n_bloques) : (bl += 1) {
        const o = bl * 144;
        const d = leerF16(bloques, o);
        const dmin = leerF16(bloques, o + 2);
        const esc = bloques + o + 4;
        const qs = bloques + o + 16;

        var par: usize = 0;
        while (par < 4) : (par += 1) {
            const base = par * 32;
            var mitad: usize = 0;
            while (mitad < 2) : (mitad += 1) {
                const j = par * 2 + mitad;
                const sc_m = escalaYMinimo(j, esc);
                const sub = bl * 8 + j;
                const desde = bl * 256 + par * 64 + mitad * 32;

                // El acumulador entero se vacia una vez por sub-bloque, que
                // es donde cambia la escala. Reducir mas seguido fue el error
                // que hizo lenta a la primera version de `dot_i8`.
                var parcial: VI32 = @splat(0);
                var i: usize = 0;
                while (i < 32) : (i += ANCHO) {
                    const pq: *align(1) const VU8 = @ptrCast(qs + base + i);
                    // Los nibbles bajos son el sub-bloque par y los altos el
                    // impar, que es como lo guarda el formato.
                    const nib: VU8 = if (mitad == 0) pq.* & mascara else pq.* >> cuatro;
                    const px: *align(1) const VI8 = @ptrCast(xq + desde + i);
                    parcial += @as(VI32, nib) * @as(VI32, px.*);
                }

                const entero: f32 = @floatFromInt(@reduce(.Add, parcial));
                const escala = d * @as(f32, @floatFromInt(sc_m[0]));
                const minimo = dmin * @as(f32, @floatFromInt(sc_m[1]));
                total += xe[sub] * (escala * entero - minimo * xs[sub]);
            }
        }
    }

    return total;
}
