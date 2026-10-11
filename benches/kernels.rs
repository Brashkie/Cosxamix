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

criterion_group!(benches, dot, rmsnorm, silu, ffi);
criterion_main!(benches);
