//! Compara todos los backends de un kernel, en la misma máquina.
//!
//! Esto es la **regla 3** hecha ejecutable: una implementación en otro lenguaje
//! se queda solo si le gana a la referencia en Rust, medido acá. Si no le gana,
//! se borra — no se mantiene un backend porque ya esté escrito ni porque el
//! lenguaje sea interesante.
//!
//!   cargo bench                     # referencia y AVX2
//!   cargo bench --features zig      # y además Zig
//!
//! Los tamaños no son redondos a propósito: 4096 cabe en caché y 1_000_003 no,
//! y un kernel que gana con el vector caliente puede perder cuando el cuello
//! pasa a ser la memoria. Medir solo el caso cómodo da una conclusión que no
//! se sostiene en producción.

use cosxamix::kernels;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn datos(n: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let b = (0..n)
        .map(|i| (i as f32 * 0.11).cos() * 2.5 - 1.0)
        .collect();
    (a, b)
}

fn dot(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot");

    for n in [1_024usize, 4_096, 65_536, 1_000_003] {
        let (a, b) = datos(n);
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("escalar", n), &n, |t, _| {
            t.iter(|| kernels::scalar::dot(black_box(&a), black_box(&b)))
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |t, _| {
                t.iter(|| unsafe { kernels::x86::dot_avx2(black_box(&a), black_box(&b)) })
            });
        }

        // El mismo bucle escrito a mano en ensamblador. La cola y la
        // reduccion son codigo compartido con la version de arriba, asi que
        // lo unico que separa a las dos barras es el bucle caliente.
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_with_input(BenchmarkId::new("asm", n), &n, |t, _| {
                t.iter(|| unsafe { kernels::x86_asm::dot_asm(black_box(&a), black_box(&b)) })
            });
        }

        #[cfg(feature = "zig")]
        g.bench_with_input(BenchmarkId::new("zig", n), &n, |t, _| {
            t.iter(|| kernels::zig::dot(black_box(&a), black_box(&b)))
        });
    }

    g.finish();
}

/// Tamaños realistas: la dimensión del modelo de un LLM chico va de 2048
/// (1B) a 4096 (7B). Medir rmsnorm con vectores de millones no diría nada,
/// porque en un transformer nunca se llama así.
fn rmsnorm(c: &mut Criterion) {
    let mut g = c.benchmark_group("rmsnorm");

    for n in [2_048usize, 4_096, 8_192] {
        let (entrada, peso) = datos(n);
        let mut salida = vec![0.0f32; n];
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("escalar", n), &n, |t, _| {
            t.iter(|| {
                kernels::scalar::rmsnorm(
                    black_box(&mut salida),
                    black_box(&entrada),
                    black_box(&peso),
                    1e-5,
                )
            })
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            g.bench_with_input(BenchmarkId::new("avx2", n), &n, |t, _| {
                t.iter(|| unsafe {
                    kernels::x86::rmsnorm_avx2(
                        black_box(&mut salida),
                        black_box(&entrada),
                        black_box(&peso),
                        1e-5,
                    )
                })
            });
        }

        #[cfg(feature = "zig")]
        g.bench_with_input(BenchmarkId::new("zig", n), &n, |t, _| {
            t.iter(|| {
                kernels::zig::rmsnorm(
                    black_box(&mut salida),
                    black_box(&entrada),
                    black_box(&peso),
                    1e-5,
                )
            })
        });
    }

    g.finish();
}

/// Cuánto cuesta cruzar la frontera, sin trabajo de por medio.
///
/// Se compara una función vacía en Zig (cruza la FFI) contra una vacía en
/// Rust marcada `#[inline(never)]` (no la cruza). La diferencia es el coste
/// de la llamada externa, que es el número que había que medir en vez de
/// suponerlo.
#[cfg(feature = "zig")]
fn ffi(c: &mut Criterion) {
    #[inline(never)]
    // Sin `black_box` ADENTRO: `#[inline(never)]` ya impide que desaparezca,
    // y meterlo dentro medía el black_box además de la llamada — con eso la
    // base de Rust salía 3x más cara que la FFI, que es absurdo.
    fn rust_noop(x: f32) -> f32 {
        x
    }

    let mut g = c.benchmark_group("ffi");
    g.bench_function("rust_noop", |t| t.iter(|| rust_noop(black_box(1.5))));
    g.bench_function("zig_noop", |t| {
        t.iter(|| kernels::zig::noop(black_box(1.5)))
    });
    g.finish();
}

#[cfg(not(feature = "zig"))]
fn ffi(_: &mut Criterion) {}

/// SiLU: la activacion del MLP, sobre los tamanos reales de `intermedio`.
///
/// Solo dos barras y no tres: no hay camino AVX2 porque vectorizar una
/// exponencial en Rust con intrinsecos pide escribir a mano una aproximacion
/// polinomica. Lo que se mide aca es cuanto vale el `@exp` vectorial de Zig
/// contra el `exp` escalar de la referencia.
fn silu(c: &mut Criterion) {
    let mut g = c.benchmark_group("silu");

    for n in [4_864usize, 11_008] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 4.0).collect();
        let mut salida = vec![0f32; n];
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("escalar", n), &n, |t, _| {
            t.iter(|| kernels::scalar::silu(black_box(&mut salida), black_box(&x)))
        });

        #[cfg(feature = "zig")]
        g.bench_with_input(BenchmarkId::new("zig", n), &n, |t, _| {
            t.iter(|| kernels::zig::silu(black_box(&mut salida), black_box(&x)))
        });
    }

    g.finish();
}

/// El producto punto en enteros contra el de f32, sobre el mismo largo.
///
/// Es la comparacion que decide el camino de la multiplicacion cuantizada:
/// AVX2 hace 32 int8 por instruccion contra 8 f32, asi que si la diferencia
/// medida no se acerca a eso, el cuello esta en otro lado.
fn dot_enteros(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot_i8");

    for n in [2_048usize, 11_008] {
        let a: Vec<i8> = (0..n)
            .map(|i| ((i * 37) % 251) as i32 as i8)
            .map(|v| v.max(-127))
            .collect();
        let b: Vec<i8> = (0..n)
            .map(|i| ((i * 53) % 251) as i32 as i8)
            .map(|v| v.max(-127))
            .collect();
        let af: Vec<f32> = a.iter().map(|v| *v as f32).collect();
        let bf: Vec<f32> = b.iter().map(|v| *v as f32).collect();
        g.throughput(Throughput::Elements(n as u64));

        g.bench_with_input(BenchmarkId::new("f32_avx2", n), &n, |t, _| {
            t.iter(|| kernels::dot(black_box(&af), black_box(&bf)))
        });
        g.bench_with_input(BenchmarkId::new("i8_escalar", n), &n, |t, _| {
            t.iter(|| kernels::scalar::dot_i8(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("i8_avx2", n), &n, |t, _| {
            t.iter(|| kernels::dot_i8(black_box(&a), black_box(&b)))
        });

        // El kernel por bloques, que es el que usa la multiplicacion: AVX2
        // con el rodeo del signo contra Zig ensanchando a i32.
        let nb = n / 32;
        let ea: Vec<f32> = (0..nb).map(|i| 0.01 + i as f32 * 0.0001).collect();
        let eb: Vec<f32> = (0..nb).map(|i| 0.02 - i as f32 * 0.00005).collect();
        g.bench_with_input(BenchmarkId::new("escalado_escalar", n), &n, |t, _| {
            t.iter(|| kernels::scalar::dot_i8_escalado(black_box(&a), black_box(&b), &ea, &eb, 32))
        });
        #[cfg(target_arch = "x86_64")]
        g.bench_with_input(BenchmarkId::new("escalado_avx2", n), &n, |t, _| {
            t.iter(|| unsafe {
                kernels::x86::dot_i8_escalado_avx2(black_box(&a), black_box(&b), &ea, &eb, 32)
            })
        });
        #[cfg(feature = "zig")]
        g.bench_with_input(BenchmarkId::new("escalado_zig", n), &n, |t, _| {
            t.iter(|| kernels::zig::dot_i8_escalado(black_box(&a), black_box(&b), &ea, &eb, 32))
        });
    }

    g.finish();
}

/// Lo que cuesta cuantizar la entrada a `i8`.
///
/// Se mide porque esta en el camino de CADA multiplicacion: la entrada se
/// cuantiza una vez por matmul, con los dos anchos que tiene el 3B — 2048,
/// que es el modelo, y 11008, que es el interior del FFN.
///
/// La pregunta original de este bench —si recuantizar los pesos explicaba que
/// `matmul_i8` fuera mas lenta que la de `f32`— quedo contestada y el rodeo
/// se elimino: ahora se desempaqueta directo del formato con
/// `quant::desempaquetar_i8`, sin pasar por `f32`.
fn cuantizar(c: &mut Criterion) {
    let mut g = c.benchmark_group("cuantizar_i8");
    for n in [2_048usize, 11_008] {
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin()).collect();
        let mut q = vec![0i8; n];
        let mut e = vec![0f32; n / 32];
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("entrada", n), &n, |t, _| {
            t.iter(|| {
                cosxamix::ops::cuantizar_i8(black_box(&x), black_box(&mut q), black_box(&mut e))
            })
        });
    }
    g.finish();
}

/// Generar e inicializar pesos, que es el camino de escribir un daiss.
///
/// Mide lo que el codigo puede dar sin el disco de por medio: en esta maquina
/// escribir un modelo va a 30 MB/s en el disco de F:, pero ese disco hace 41
/// MB/s y punto, asi que el numero que dice si el generador es lento o no es
/// este y no aquel.
fn inicializar(c: &mut Criterion) {
    use cosxamix::daiss::Azar;
    let mut g = c.benchmark_group("daiss_init");
    let n = 4_096usize;
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function(BenchmarkId::new("normales", n), |t| {
        t.iter(|| {
            let mut azar = Azar::con_semilla(black_box(7));
            let mut suma = 0f32;
            for _ in 0..n / 2 {
                let (a, b) = azar.normales(0.02);
                suma += a + b;
            }
            black_box(suma)
        })
    });
    g.finish();
}

/// El kernel fusionado de Q4_K en sus tres caminos.
///
/// Lo que decide si el de Zig vale la pena NO es si le gana a AVX2 —no le
/// gana, AVX2 tiene 32 carriles contra 8— sino si le gana al **escalar**, que
/// es lo unico que hay en ARM y en Termux. Ahi el de Rust usa
/// `core::arch::x86_64` y ni siquiera compila.
fn fusionado(c: &mut Criterion) {
    use cosxamix::kernels;

    let mut g = c.benchmark_group("dot_q4k_i8");
    let n = 2_048usize; // el ancho del 3B
    let pesos: Vec<u8> = (0..n / 256 * 144)
        .map(|i| ((i * 131 + 17) % 251) as u8)
        .collect();
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin()).collect();
    let mut xq = vec![0i8; n];
    let mut xe = vec![0f32; n / 32];
    cosxamix::ops::cuantizar_i8(&x, &mut xq, &mut xe);
    let xs: Vec<f32> = xq
        .chunks(32)
        .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
        .collect();

    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("escalar", |t| {
        t.iter(|| kernels::scalar::dot_q4k_i8(black_box(&pesos), black_box(&xq), &xe, &xs))
    });

    // Q3_K aparte: su sub-bloque es de 16, asi que tiene el doble de escalas,
    // y es el formato del modelo por defecto.
    let pesos3: Vec<u8> = (0..n / 256 * 110)
        .map(|i| ((i * 97 + 31) % 251) as u8)
        .collect();
    let mut xq3 = vec![0i8; n];
    let mut xe3 = vec![0f32; n / 16];
    cosxamix::ops::cuantizar_i8_en(&x, &mut xq3, &mut xe3, 16);
    let xs3: Vec<f32> = xq3
        .chunks(16)
        .map(|c| c.iter().map(|v| *v as i32).sum::<i32>() as f32)
        .collect();
    g.bench_function("q3k escalar", |t| {
        t.iter(|| kernels::scalar::dot_q3k_i8(black_box(&pesos3), black_box(&xq3), &xe3, &xs3))
    });
    #[cfg(target_arch = "x86_64")]
    g.bench_function("q3k avx2", |t| {
        t.iter(|| unsafe {
            kernels::x86::dot_q3k_i8_avx2(black_box(&pesos3), black_box(&xq3), &xe3, &xs3)
        })
    });
    #[cfg(all(feature = "asm_q3k", target_arch = "x86_64"))]
    g.bench_function("q3k ensamblador", |t| {
        t.iter(|| unsafe {
            kernels::q3k_asm::dot_q3k_i8(black_box(&pesos3), black_box(&xq3), &xe3, &xs3)
        })
    });
    #[cfg(all(feature = "c", target_arch = "x86_64"))]
    g.bench_function("q3k c", |t| {
        t.iter(|| unsafe {
            kernels::c::dot_q3k_i8(black_box(&pesos3), black_box(&xq3), &xe3, &xs3)
        })
    });
    #[cfg(target_arch = "x86_64")]
    g.bench_function("avx2", |t| {
        t.iter(|| unsafe {
            kernels::x86::dot_q4k_i8_avx2(black_box(&pesos), black_box(&xq), &xe, &xs)
        })
    });
    // Los dos bucles de ensamblador AISLADOS, con las escalas calculadas una
    // vez fuera de lo cronometrado: separan el costo del bucle del de lo que
    // lo rodea.
    #[cfg(all(feature = "asm_q4k", hay_q4k_asm, hay_q4k_fila, target_arch = "x86_64"))]
    {
        let mut esc = vec![0f32; n / 32];
        unsafe {
            let mut d = std::arch::x86_64::_mm256_setzero_ps();
            for bl in 0..n / 256 {
                let e = kernels::x86::escalas_q4k(
                    &pesos[bl * 144..bl * 144 + 144],
                    &xe[bl * 8..bl * 8 + 8],
                    &xs[bl * 8..bl * 8 + 8],
                    &mut d,
                );
                esc[bl * 8..bl * 8 + 8].copy_from_slice(&e);
            }
        }
        g.bench_function("solo bucle por bloque", |t| {
            t.iter(|| unsafe {
                let mut acc = [0f32; 8];
                kernels::q4k_asm::bucle_por_bloque(black_box(&pesos), &xq, &esc, &mut acc);
                acc
            })
        });
        g.bench_function("solo bucle por fila", |t| {
            t.iter(|| unsafe {
                let mut acc = [0f32; 8];
                kernels::q4k_asm::bucle_por_fila(black_box(&pesos), &xq, &esc, &mut acc);
                acc
            })
        });
    }
    #[cfg(all(feature = "asm_q4k", hay_q4k_asm, target_arch = "x86_64"))]
    g.bench_function("ensamblador por bloque", |t| {
        t.iter(|| unsafe {
            kernels::q4k_asm::dot_q4k_i8_asm(black_box(&pesos), black_box(&xq), &xe, &xs)
        })
    });
    #[cfg(all(feature = "asm_q4k", hay_q4k_completo, target_arch = "x86_64"))]
    g.bench_function("ensamblador completo", |t| {
        t.iter(|| unsafe {
            kernels::q4k_asm::dot_q4k_i8_asm_completo(black_box(&pesos), black_box(&xq), &xe, &xs)
        })
    });
    #[cfg(all(feature = "asm_q4k", hay_q4k_fila, target_arch = "x86_64"))]
    g.bench_function("ensamblador por fila", |t| {
        t.iter(|| unsafe {
            kernels::q4k_asm::dot_q4k_i8_asm_fila(black_box(&pesos), black_box(&xq), &xe, &xs)
        })
    });
    #[cfg(feature = "zig")]
    g.bench_function("zig", |t| {
        t.iter(|| kernels::zig::dot_q4k_i8(black_box(&pesos), black_box(&xq), &xe, &xs))
    });
    // Los de C con `sdot`, en ARM: contra el escalar y el de Zig de arriba.
    #[cfg(all(feature = "c", target_arch = "aarch64"))]
    if kernels::c::hay_dotprod() {
        g.bench_function("c con sdot", |t| {
            t.iter(|| unsafe {
                kernels::c::dot_q4k_i8_neon(black_box(&pesos), black_box(&xq), &xe, &xs)
            })
        });
        g.bench_function("q3k c con sdot", |t| {
            t.iter(|| unsafe {
                kernels::c::dot_q3k_i8_neon(black_box(&pesos3), black_box(&xq3), &xe3, &xs3)
            })
        });
    }
    // En ARM, contra el de Zig de justo arriba: es el camino vectorizado que
    // hay hoy en esa arquitectura y el que tiene que superar. Sin `dotprod` no
    // se mide, porque `sdot` seria una instruccion ilegal.
    #[cfg(all(feature = "asm_q4k", hay_q4k_neon, target_arch = "aarch64"))]
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        g.bench_function("neon con sdot", |t| {
            t.iter(|| unsafe {
                kernels::q4k_neon::dot_q4k_i8_neon(black_box(&pesos), black_box(&xq), &xe, &xs)
            })
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    dot,
    rmsnorm,
    silu,
    dot_enteros,
    cuantizar,
    inicializar,
    fusionado,
    ffi
);
criterion_main!(benches);
