//! Compila los kernels que no están en Rust.
//!
//! Solo corre con la feature `zig`. Sin ella no hace nada, y Cosxamix se compila
//! sin tener Zig instalado — que es la condición para que "varios lenguajes"
//! no se convierta en "hay que instalar varios compiladores para construir
//! esto".

fn main() {
    println!("cargo:rerun-if-changed=kernels/");

    #[cfg(feature = "zig")]
    compilar_zig();

    comprobar_asm_q4k();
}

/// Decide si el kernel de Q4_K escrito a mano se puede compilar.
///
/// El cuerpo de ese kernel lo escribe una persona en dos `.s` que todavia no
/// existen, y `include_str!` falla si el archivo falta. Dejar que falle
/// romperia `--all-features`, que es el comando con el que se verifica el
/// proyecto entero: un hueco a medio hacer no puede tirar abajo la
/// verificacion de todo lo demas.
///
/// Asi que el modulo se compila solo si los dos archivos estan, y si la
/// feature esta encendida sin ellos se avisa en vez de fallar.
fn comprobar_asm_q4k() {
    // Hace falta declararlos o `unexpected_cfgs` los marca: son cfg propios y
    // el compilador no puede saber que existen.
    println!("cargo:rustc-check-cfg=cfg(hay_q4k_asm)");
    println!("cargo:rustc-check-cfg=cfg(hay_q4k_fila)");
    println!("cargo:rustc-check-cfg=cfg(hay_q4k_completo)");
    println!("cargo:rustc-check-cfg=cfg(hay_q4k_neon)");
    for f in [
        "kernels/q4k_avx2_win64.s",
        "kernels/q4k_avx2_sysv.s",
        "kernels/q4k_neon.s",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }

    // En ARM64 hay un solo archivo para los tres sistemas, porque Linux,
    // Android y macOS usan la misma convencion, AAPCS64. Y solo el contrato
    // completo: los otros dos se escribieron en x86 para medir, y perdieron.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64") {
        let texto = std::fs::read_to_string("kernels/q4k_neon.s").unwrap_or_default();
        if define_etiqueta(&texto, "cosxamix_q4k_completo") {
            println!("cargo:rustc-cfg=hay_q4k_neon");
        } else if std::env::var_os("CARGO_FEATURE_ASM_Q4K").is_some() {
            println!(
                "cargo:warning=la feature asm_q4k esta encendida pero kernels/q4k_neon.s no define cosxamix_q4k_completo: el kernel NEON no se compila. El contrato esta en src/kernels/q4k_neon.rs."
            );
        }
        return;
    }

    // Un archivo por convencion de llamada, y se mira SOLO el del target que
    // se compila. `CARGO_CFG_TARGET_OS` es el sistema del target y no el del
    // host, que es lo que importa cuando se cruza.
    let archivo = match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "kernels/q4k_avx2_win64.s",
        // Linux y macOS comparten el de System V, como `dot_avx2_sysv.s`.
        _ => "kernels/q4k_avx2_sysv.s",
    };
    let texto = std::fs::read_to_string(archivo).unwrap_or_default();

    let pares = define_etiqueta(&texto, "cosxamix_q4k_pares");
    let fila = define_etiqueta(&texto, "cosxamix_q4k_fila");
    let completo = define_etiqueta(&texto, "cosxamix_q4k_completo");

    if pares {
        println!("cargo:rustc-cfg=hay_q4k_asm");
    }
    if fila {
        println!("cargo:rustc-cfg=hay_q4k_fila");
    }
    if completo {
        println!("cargo:rustc-cfg=hay_q4k_completo");
    }
    if !pares && !fila && !completo && std::env::var_os("CARGO_FEATURE_ASM_Q4K").is_some() {
        println!(
            "cargo:warning=la feature asm_q4k esta encendida pero {archivo} no define ni cosxamix_q4k_pares ni cosxamix_q4k_fila: el kernel en ensamblador no se compila. El contrato esta en src/kernels/q4k_asm.rs."
        );
    }
}

/// Si el ensamblador define la funcion, mirando la ETIQUETA con los dos puntos.
///
/// No el nombre a secas: el nombre aparece en los comentarios de la cabecera,
/// y buscarlo asi daba por presente una funcion que no estaba. En Mach-O la
/// etiqueta lleva un guion bajo delante. Se quitan los comentarios `//`, que
/// el ensamblador de LLVM acepta tanto en x86 con AT&T como en ARM64, ELF y
/// Mach-O: por eso los `.s` comentan siempre asi.
fn define_etiqueta(texto: &str, nombre: &str) -> bool {
    texto.lines().any(|l| {
        let l = l.split("//").next().unwrap_or("").trim();
        l == format!("{nombre}:") || l == format!("_{nombre}:")
    })
}

/// Traduce el target de Rust al de Zig, con el `-mcpu` que corresponde.
///
/// **`-mcpu` no es opcional si se va a comparar contra la versión de Rust.**
/// Sin esto, Zig compila para el x86-64 básico de 2003: su `@Vector(8, f32)`
/// se parte en operaciones de 128 bits o directamente en escalares, y el
/// kernel pierde contra el de Rust por motivos que no tienen nada que ver con
/// el lenguaje. Medido: así perdía entre un 3% y un 26%.
///
/// `x86_64_v3` es exactamente AVX2 + FMA, que es lo que habilita el camino de
/// Rust con `#[target_feature]`. Comparar contra otra cosa sería comparar dos
/// targets, no dos lenguajes. No se usa `native`: ataría el binario a la CPU
/// que lo compiló, y la idea es publicar uno que ande en todas y elija en
/// tiempo de ejecución.
///
/// En aarch64 va `baseline` **a propósito**: NEON es obligatorio en ARMv8-A,
/// así que la línea base ya trae los 128 bits de SIMD. No hay un `v3` que
/// pedir, y nombrar una microarquitectura concreta (`apple_m1`, `cortex_a76`)
/// ataría el binario a un teléfono.
#[cfg(feature = "zig")]
fn zig_target(rust_target: &str) -> (&'static str, &'static str) {
    match rust_target {
        "x86_64-unknown-linux-gnu" => ("x86_64-linux-gnu", "x86_64_v3"),
        "aarch64-unknown-linux-gnu" => ("aarch64-linux-gnu", "baseline"),
        "aarch64-linux-android" => ("aarch64-linux-android", "baseline"),
        "x86_64-pc-windows-msvc" => ("x86_64-windows-msvc", "x86_64_v3"),
        "aarch64-apple-darwin" => ("aarch64-macos", "baseline"),
        "x86_64-apple-darwin" => ("x86_64-macos", "x86_64_v3"),
        otro => panic!(
            "no sé traducir el target {otro} al de Zig.\n\
             Agregalo en `zig_target()` de build.rs, o compilá sin la feature:\n\
             \n    cargo build --release --features candle\n\n\
             Sin `zig` el motor usa el camino de Rust: AVX2 en x86-64 y la\n\
             referencia escalar en todo lo demás."
        ),
    }
}

#[cfg(feature = "zig")]
fn compilar_zig() {
    use std::path::PathBuf;
    use std::process::Command;

    let salida = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));

    // `TARGET` y `HOST` los pone cargo para los build scripts. Hay que usar
    // estos y NO `cfg!(target_os = ...)`: un build script se compila para el
    // host, así que sus `cfg!` describen la máquina que construye y no la que
    // va a ejecutar. Dando por bueno `cfg!`, cruzar de Linux a Windows nombra
    // la librería `libcosxamix_kernels.a` mientras Zig escribe un `.lib`, y el
    // linker no encuentra nada por un motivo que no se parece a la causa.
    let target = std::env::var("TARGET").expect("TARGET");
    let host = std::env::var("HOST").expect("HOST");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");

    let (zig_tgt, mcpu) = zig_target(&target);

    // `build-lib -static` deja una librería estática que el linker de Rust
    // mete en el binario final. Nada de .dll ni .so al lado: Cosxamix sigue
    // siendo un archivo.
    //
    // `-O ReleaseFast` y no `ReleaseSafe`: un kernel numérico no quiere
    // comprobaciones de desbordamiento en el bucle caliente. Lo que protege
    // de que esté mal no es el modo de compilación, son los tests que lo
    // comparan contra la referencia en Rust.
    let destino = salida.join(nombre_lib("cosxamix_kernels", &target_os));

    let mut cmd = Command::new("zig");
    cmd.args(["build-lib", "kernels/dot.zig", "-static", "-OReleaseFast"]);
    cmd.arg(format!("-mcpu={mcpu}"));

    // Compilando para la propia máquina no se pasa `-target`, que es el camino
    // de todos los días y el que está probado. Al cruzar sí, y ahí Zig hace el
    // trabajo que si no pediría un toolchain entero por plataforma: `dot.zig`
    // solo importa `std` y usa `@Vector`, sin una sola llamada a libc, así que
    // no necesita cabeceras ni SDK del otro lado.
    //
    // Y acá `-target` va en DOS argumentos, al revés que `-femit-bin=`, que va
    // en uno. Zig no es consistente entre sus propios flags: `-mcpu=` acepta
    // el igual y `-target=` no, y el error que da —"unrecognized parameter"—
    // no insinúa que el problema sea la forma de pasarlo.
    if target != host {
        cmd.arg("-target").arg(zig_tgt);
    }

    // `-femit-bin=RUTA` va como UN argumento, no como dos. Pasarlos separados
    // hace que zig escriba su salida con el nombre por defecto y que el linker
    // después no encuentre nada, con un error que no menciona el verdadero
    // motivo.
    cmd.arg(format!("-femit-bin={}", destino.display()));

    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!(
            "zig build-lib falló con código {s:?} para {target}.\n\
             Si no tenés Zig instalado, compilá sin la feature: cargo build --release"
        ),
        Err(e) => panic!(
            "no pude ejecutar `zig`: {e}\n\
             Instalalo, o compilá sin la feature: cargo build --release"
        ),
    }

    println!("cargo:rustc-link-search=native={}", salida.display());
    println!("cargo:rustc-link-lib=static=cosxamix_kernels");
}

/// Cada plataforma nombra sus librerías estáticas a su manera, y `zig
/// build-lib` respeta esa convención. El linker de Rust pide el nombre
/// desnudo, pero el archivo en disco tiene que llamarse como toca.
///
/// Toma el sistema del **target**, no del host, por lo dicho arriba.
#[cfg(feature = "zig")]
fn nombre_lib(base: &str, target_os: &str) -> String {
    if target_os == "windows" {
        format!("{base}.lib")
    } else {
        format!("lib{base}.a")
    }
}
