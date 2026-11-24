//! Compila los kernels que no están en Rust.
//!
//! Solo corre con la feature `zig`. Sin ella no hace nada, y Cosxamix se compila
//! sin tener Zig instalado — que es la condición para que "varios lenguajes"
//! no se convierta en "hay que instalar varios compiladores para construir
//! esto".

fn main() {
    println!("cargo:rerun-if-changed=kernels/");
    // La copia de los kernels de Zig para x86-64-v3 (ver compilar_zig).
    println!("cargo:rustc-check-cfg=cfg(cosxamix_v3)");

    #[cfg(feature = "zig")]
    compilar_zig();

    #[cfg(feature = "c")]
    compilar_c();

    #[cfg(feature = "fortran")]
    compilar_fortran();

    #[cfg(feature = "ada")]
    compilar_ada();

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
/// En x86-64 Zig sigue a lo que se le pidió a Rust (ver compilar_zig): si
/// Rust se compila para `x86-64-v3` —lo de siempre, por `.cargo/config.toml`—,
/// Zig también, porque el binario igual no arranca sin AVX2. Si Rust se
/// compila para el x86-64 básico —la compilación aparte para las CPU de antes
/// de 2015—, Zig va en `baseline` y lleva una segunda copia para `x86_64_v3`
/// que se elige al arrancar. No se usa `native`: ataría el binario a la CPU
/// que lo compiló.
///
/// En aarch64 va `baseline` **a propósito**: NEON es obligatorio en ARMv8-A,
/// así que la línea base ya trae los 128 bits de SIMD. No hay un `v3` que
/// pedir, y nombrar una microarquitectura concreta (`apple_m1`, `cortex_a76`)
/// ataría el binario a un teléfono.
#[cfg(any(feature = "zig", feature = "c"))]
fn zig_target(rust_target: &str) -> (&'static str, &'static str) {
    match rust_target {
        "x86_64-unknown-linux-gnu" => ("x86_64-linux-gnu", "baseline"),
        "aarch64-unknown-linux-gnu" => ("aarch64-linux-gnu", "baseline"),
        "aarch64-linux-android" => ("aarch64-linux-android", "baseline"),
        "x86_64-pc-windows-msvc" => ("x86_64-windows-msvc", "baseline"),
        "aarch64-apple-darwin" => ("aarch64-macos", "baseline"),
        "x86_64-apple-darwin" => ("x86_64-macos", "baseline"),
        otro => panic!(
            "no sé traducir el target {otro} al de Zig.\n\
             Agregalo en `zig_target()` de build.rs, o compilá sin la feature:\n\
             \n    cargo build --release\n\n\
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
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");

    let (zig_tgt, mcpu_base) = zig_target(&target);

    // Lo que Rust ya da por hecho en este binario. `CARGO_CFG_TARGET_FEATURE`
    // trae lo que sale de `target-cpu`, del target y no del host.
    let rasgos = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let rust_v3 = ["avx2", "fma"]
        .iter()
        .all(|f| rasgos.split(',').any(|r| r == *f));
    let x86 = target_arch == "x86_64";

    // Con Rust en x86-64-v3, una sola copia de Zig y también en v3: el
    // binario igual no arranca sin AVX2, así que una copia baseline sería peso
    // muerto. Hasta la 0.4.0 Zig iba SIEMPRE en v3, y la compilación aparte
    // para las CPU sin AVX2 —sin el `target-cpu` del config— moría con
    // "instrucción ilegal" en el primer kernel de Zig.
    let mcpu = if x86 && rust_v3 {
        "x86_64_v3"
    } else {
        mcpu_base
    };

    // Con Rust en el x86-64 básico, los mismos kernels otra vez para x86-64-v3
    // (AVX2 + FMA), con sufijo `_v3` (`kernels/lib_v3.zig`): un objeto aparte
    // que entra en la misma librería, para que ese binario también rinda en
    // una CPU moderna. `COSXAMIX_SIN_V3` lo saltea, para comprobar que la
    // copia baseline sola compila y pasa los tests.
    println!("cargo:rerun-if-env-changed=COSXAMIX_SIN_V3");
    let objeto_v3 = (x86 && !rust_v3 && std::env::var_os("COSXAMIX_SIN_V3").is_none()).then(|| {
        let obj = salida.join(if target_os == "windows" {
            "cosxamix_v3.obj"
        } else {
            "cosxamix_v3.o"
        });
        let mut c = Command::new("zig");
        c.args([
            "build-obj",
            "kernels/lib_v3.zig",
            "-OReleaseFast",
            "-fPIC",
            "-mcpu=x86_64_v3",
        ]);
        if target != host {
            c.arg("-target").arg(zig_tgt);
        }
        c.arg(format!("-femit-bin={}", obj.display()));
        match c.status() {
            Ok(st) if st.success() => obj,
            Ok(st) => panic!("zig build-obj (x86-64-v3) falló con código {st:?} para {target}"),
            Err(e) => panic!(
                "no pude ejecutar `zig`: {e}\n\
                 Instalalo, o compilá sin la feature: cargo build --release"
            ),
        }
    });

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
    cmd.args([
        "build-lib",
        "kernels/lib.zig",
        "-static",
        "-OReleaseFast",
        "-fPIC",
    ]);
    cmd.arg(format!("-mcpu={mcpu}"));
    if let Some(obj) = &objeto_v3 {
        cmd.arg(obj);
    }

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
    if objeto_v3.is_some() {
        println!("cargo:rustc-cfg=cosxamix_v3");
    }
}

/// Compila los kernels de C con `zig cc`, en una librería aparte.
///
/// Sin libc (`-ffreestanding`): no usan nada de ella, y así cruzar de un
/// sistema a otro no pide el SDK del otro lado. En baseline: cada kernel pide
/// sus extensiones con `__attribute__((target))` —AVX2 en x86-64, `dotprod` en
/// ARM64—, así que el objeto es seguro en cualquier CPU de su arquitectura y el
/// que lo llama decide si la CPU sirve.
///
/// `-ffp-contract=off` porque clang fusiona `a*b + c` en una FMA por su
/// cuenta, y eso redondea distinto que Rust, que nunca fusiona. Los kernels
/// piden sus FMA explícitas con intrínsecos.
///
/// Un archivo por arquitectura: `q3k_avx2.c` en x86-64 y `neon_dotprod.c` en
/// ARM64. En otra no se compila nada.
#[cfg(feature = "c")]
fn compilar_c() {
    use std::path::PathBuf;
    use std::process::Command;

    let fuentes: &[&str] = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("x86_64") => &[
            "kernels/q3k_avx2.c",
            "kernels/classify_avx2.c",
            "kernels/q2k_avx2.c",
            "kernels/q5k_avx2.c",
            "kernels/tile_groups_avx2.c",
        ],
        Ok("aarch64") => &["kernels/neon_dotprod.c", "kernels/q2k_neon.c"],
        _ => return,
    };
    let salida = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let target = std::env::var("TARGET").expect("TARGET");
    let host = std::env::var("HOST").expect("HOST");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    let destino = salida.join(nombre_lib("cosxamix_c", &target_os));

    let mut cmd = Command::new("zig");
    cmd.args([
        "build-lib",
        "-static",
        "-OReleaseFast",
        "-fPIC",
        "-mcpu=baseline",
    ]);
    if target != host {
        cmd.arg("-target").arg(zig_target(&target).0);
    }
    // Las banderas de C van entre `-cflags` y `--`, y valen para los .c que
    // siguen: puestas después del archivo, Zig las ignora sin avisar.
    cmd.args([
        "-cflags",
        "-std=c11",
        "-ffreestanding",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        "--",
    ]);
    cmd.args(fuentes);
    cmd.arg(format!("-femit-bin={}", destino.display()));
    match cmd.status() {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("zig build-lib (C) falló con código {s:?} para {target}"),
        Err(e) => panic!(
            "no pude ejecutar `zig`: {e}\n\
             La feature `c` compila con `zig cc`: instalá Zig, o compilá sin ella."
        ),
    }
    println!("cargo:rustc-link-search=native={}", salida.display());
    println!("cargo:rustc-link-lib=static=cosxamix_c");
}

/// Compila los kernels de Fortran con gfortran, en una librería aparte.
///
/// Sin el runtime de Fortran: los kernels son `bind(C)`, sin E/S ni arreglos
/// asignables, y `-fno-backtrace` y `-fno-stack-protector` sacan las dos cosas
/// que podrían llamar a libgfortran o a libssp. El objeto enlaza igual con el
/// linker de MSVC que con el de GNU.
///
/// En x86-64 se compila para **x86-64-v3** (AVX2 y FMA): el que llama comprueba
/// la CPU antes de usarlo, igual que con los intrínsecos de Rust. En ARM64, para
/// la base, que ya trae NEON.
///
/// Qué gfortran: `COSXAMIX_GFORTRAN` si está; si no, en Windows el de MSYS2
/// (`C:\msys64\ucrt64\bin`), y si no el del PATH. Se comprueba que sea de la
/// arquitectura del target, porque en Windows es fácil tener a mano un MinGW
/// de 32 bits que compilaría sin quejarse algo que después no enlaza.
#[cfg(feature = "fortran")]
fn compilar_fortran() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let arq = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");
    let marcha: &[&str] = match arq.as_str() {
        "x86_64" => &["-march=x86-64-v3"],
        "aarch64" => &[],
        _ => return,
    };
    println!("cargo:rerun-if-env-changed=COSXAMIX_GFORTRAN");
    let msys = Path::new(r"C:\msys64\ucrt64\bin\gfortran.exe");
    let gfortran = std::env::var("COSXAMIX_GFORTRAN").unwrap_or_else(|_| {
        if cfg!(windows) && msys.exists() {
            msys.display().to_string()
        } else {
            "gfortran".into()
        }
    });

    // Su carpeta va primera en el PATH de lo que se lanza: el gfortran de
    // MSYS2 llamado desde fuera de su consola no encuentra sus DLL ni su `as`,
    // y falla con código 1 sin decir nada.
    let path = Path::new(&gfortran)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| {
            let mut v = vec![d.to_path_buf()];
            v.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            std::env::join_paths(v).expect("PATH")
        });
    let comando = |programa: &str| {
        let mut c = Command::new(programa);
        if let Some(p) = &path {
            c.env("PATH", p);
        }
        c
    };

    let maquina = comando(&gfortran)
        .arg("-dumpmachine")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "no pude ejecutar `{gfortran}`: {e}\n\
                 La feature `fortran` necesita gfortran (en Windows, el de MSYS2 en\n\
                 C:\\msys64\\ucrt64\\bin, o la ruta en COSXAMIX_GFORTRAN). Sin la feature\n\
                 la GEMM usa los intrínsecos de Rust."
            )
        });
    let maquina = String::from_utf8_lossy(&maquina.stdout).trim().to_string();
    if !maquina.starts_with(&arq) {
        panic!(
            "`{gfortran}` compila para {maquina} y el target es {arq}: \
             apuntá COSXAMIX_GFORTRAN a un gfortran de {arq}"
        );
    }

    let salida = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let objeto = salida.join("sgemm_micro_f90.o");
    let estado = comando(&gfortran)
        .args([
            "-c",
            "-O3",
            "-fno-backtrace",
            "-fno-stack-protector",
            "-fPIC",
        ])
        .args(marcha)
        .arg("-J")
        .arg(&salida)
        .arg("kernels/sgemm_micro.f90")
        .arg("-o")
        .arg(&objeto)
        .status();
    match estado {
        Ok(s) if s.success() => {}
        otro => panic!("gfortran no compiló kernels/sgemm_micro.f90: {otro:?}"),
    }

    // El `ar` de la misma instalación, si está al lado del compilador.
    let ar = Path::new(&gfortran)
        .parent()
        .map(|d| d.join(if cfg!(windows) { "ar.exe" } else { "ar" }))
        .filter(|p| p.exists())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "ar".into());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    let lib = salida.join(if target_os == "windows" {
        "cosxamix_fortran.lib"
    } else {
        "libcosxamix_fortran.a"
    });
    let _ = std::fs::remove_file(&lib);
    match comando(&ar).arg("rcs").arg(&lib).arg(&objeto).status() {
        Ok(s) if s.success() => {}
        otro => panic!("`{ar}` no armó la librería de Fortran: {otro:?}"),
    }
    println!("cargo:rustc-link-search=native={}", salida.display());
    println!("cargo:rustc-link-lib=static=cosxamix_fortran");
}

/// Compila las comprobaciones de SPARK (`ada/src/`) con el gcc de GNAT.
///
/// `-gnatp` quita los chequeos en tiempo de ejecución: GNATprove ya demostró
/// que ninguno puede fallar (desde `ada/`: `alr -n build` y `alr -n exec --
/// gnatprove -P cosxamix_checks.gpr --level=4 -j0 --checks-as-errors=on`). Un
/// paquete `Pure`, sin elaboración y sin divisiones de 128 bits no deja ningún
/// símbolo sin resolver, y el objeto enlaza también con el linker de MSVC.
///
/// Qué gcc: `COSXAMIX_GNAT_GCC`; si no, el de `gnat_native` en la caché de
/// Alire; si no, el del PATH, que es lo que deja `alr exec`.
#[cfg(feature = "ada")]
fn compilar_ada() {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    println!("cargo:rerun-if-changed=ada/src/");
    println!("cargo:rerun-if-env-changed=COSXAMIX_GNAT_GCC");
    let exe = if cfg!(windows) { "gcc.exe" } else { "gcc" };
    let de_alire = || -> Option<PathBuf> {
        let base = if cfg!(windows) {
            PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("alire")
        } else {
            PathBuf::from(std::env::var_os("HOME")?).join(".local/share/alire")
        };
        let mut candidatos: Vec<PathBuf> = std::fs::read_dir(base.join("cache/toolchains"))
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("gnat_native"))
            })
            .map(|p| p.join("bin").join(exe))
            .filter(|p| p.exists())
            .collect();
        candidatos.sort();
        candidatos.pop()
    };
    let gcc = std::env::var("COSXAMIX_GNAT_GCC")
        .ok()
        .or_else(|| de_alire().map(|p| p.display().to_string()))
        .unwrap_or_else(|| "gcc".into());

    // Como con gfortran: la carpeta del compilador primera en el PATH, para
    // que encuentre su `as` y sus DLL.
    let path = Path::new(&gcc)
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(|d| {
            let mut v = vec![d.to_path_buf()];
            v.extend(std::env::split_paths(
                &std::env::var_os("PATH").unwrap_or_default(),
            ));
            std::env::join_paths(v).expect("PATH")
        });
    let comando = |programa: &str| {
        let mut c = Command::new(programa);
        if let Some(p) = &path {
            c.env("PATH", p);
        }
        c
    };

    let salida = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let mut objetos = Vec::new();
    // Las unidades de `ada/src/`; una nueva se agrega acá.
    const UNIDADES: &[&str] = &["gguf_bounds"];
    for unidad in UNIDADES {
        let objeto = salida.join(format!("{unidad}.o"));
        let estado = comando(&gcc)
            .args(["-c", "-O2", "-gnatp", "-gnatn", "-gnat2022", "-fPIC"])
            .arg("-Iada/src")
            .arg(format!("ada/src/{unidad}.adb"))
            .arg("-o")
            .arg(&objeto)
            .current_dir(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
            .status();
        match estado {
            Ok(s) if s.success() => {}
            Ok(s) => panic!("GNAT no compiló ada/src/{unidad}.adb ({s:?})"),
            Err(e) => panic!(
                "no pude ejecutar `{gcc}`: {e}\n\
                 La feature `ada` necesita el gcc de GNAT: con Alire instalado se\n\
                 busca solo; si no, la ruta en COSXAMIX_GNAT_GCC. Sin la feature rige\n\
                 la misma regla escrita en Rust."
            ),
        }
        objetos.push(objeto);
    }

    let ar = Path::new(&gcc)
        .parent()
        .map(|d| d.join(if cfg!(windows) { "ar.exe" } else { "ar" }))
        .filter(|p| p.exists())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "ar".into());
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
    let lib = salida.join(if target_os == "windows" {
        "cosxamix_ada.lib"
    } else {
        "libcosxamix_ada.a"
    });
    let _ = std::fs::remove_file(&lib);
    match comando(&ar).arg("rcs").arg(&lib).args(&objetos).status() {
        Ok(s) if s.success() => {}
        otro => panic!("`{ar}` no armó la librería de Ada: {otro:?}"),
    }
    println!("cargo:rustc-link-search=native={}", salida.display());
    println!("cargo:rustc-link-lib=static=cosxamix_ada");
}

/// Cada plataforma nombra sus librerías estáticas a su manera, y `zig
/// build-lib` respeta esa convención. El linker de Rust pide el nombre
/// desnudo, pero el archivo en disco tiene que llamarse como toca.
///
/// Toma el sistema del **target**, no del host, por lo dicho arriba.
#[cfg(any(feature = "zig", feature = "c"))]
fn nombre_lib(base: &str, target_os: &str) -> String {
    if target_os == "windows" {
        format!("{base}.lib")
    } else {
        format!("lib{base}.a")
    }
}
