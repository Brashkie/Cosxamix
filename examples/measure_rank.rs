//! ¿Cuánta energía de un peso entrenado cabe en un rango bajo?
//!
//! Es la pregunta que decide si las redes de tensores —Tensor-Train, MPS,
//! Tucker— sirven acá, y se puede contestar sin implementar ninguna.
//!
//! Toda descomposición de rango bajo apuesta a lo mismo: que la matriz tenga
//! **rango efectivo** mucho menor que su rango nominal, o sea que casi toda su
//! norma viva en pocas direcciones. Si eso es cierto, tirar el resto cuesta
//! poco. Si no, la descomposición pierde información sin devolver nada.
//!
//! Lo que se mide es la fracción de energía de Frobenius que captura la mejor
//! aproximación de rango `k`:
//!
//! ```text
//!     energia(k) = ||Qᵀ A||²_F / ||A||²_F
//! ```
//!
//! donde `Q` es una base ortonormal del subespacio dominante de `A`. No hace
//! falta una SVD completa, que en 151.936 × 896 sería inviable acá: alcanza un
//! **buscador de rango aleatorizado** con iteraciones de potencia, que es el
//! método estándar (Halko, Martinsson, Tropp) y da una cota por debajo — así
//! que si dice que hay estructura, la hay.
//!
//! La comparación es controlada: el **mismo tensor y la misma forma** en un
//! modelo entrenado (Qwen2.5-0.5B) y en uno recién inicializado
//! (daiss-0.5b). Una matriz aleatoria es de rango completo por construcción,
//! así que su energía debería crecer casi lineal con `k`; la entrenada, si la
//! apuesta es correcta, mucho más rápido.
//!
//!     cargo run --release --example measure_rank

use std::io::Write;
use std::time::Instant;

use cosxamix::gguf::Gguf;
use rayon::prelude::*;

/// Una matriz `m × n` por filas.
struct Matriz {
    m: usize,
    n: usize,
    d: Vec<f32>,
}

impl Matriz {
    /// `Y = A · O`, con `O` de `n × k`.
    fn por(&self, o: &[f32], k: usize) -> Vec<f32> {
        let mut y = vec![0f32; self.m * k];
        y.par_chunks_mut(k)
            .zip(self.d.par_chunks(self.n))
            .for_each(|(dst, fila)| {
                for (j, a) in fila.iter().enumerate() {
                    let col = &o[j * k..(j + 1) * k];
                    for (d, c) in dst.iter_mut().zip(col) {
                        *d += a * c;
                    }
                }
            });
        y
    }

    /// `Z = Aᵀ · Y`, con `Y` de `m × k`. Sale de `n × k`.
    fn traspuesta_por(&self, y: &[f32], k: usize) -> Vec<f32> {
        // Se acumula por bloques de filas y se suman los parciales: el destino
        // se toca entero en cada fila, asi que no se puede repartir por filas
        // del destino sin que dos hilos escriban el mismo sitio.
        // Bloques grandes y no de 256 filas: con 256 habia una reserva de
        // `n*k` y una reduccion por cada bloque, 594 de cada una en la matriz
        // de salida del 0,5B.
        const BLOQUE: usize = 8192;
        self.d
            .par_chunks(self.n * BLOQUE)
            .zip(y.par_chunks(k * BLOQUE))
            .map(|(bloque, yb)| {
                let mut z = vec![0f32; self.n * k];
                for (fila, yf) in bloque.chunks(self.n).zip(yb.chunks(k)) {
                    for (j, a) in fila.iter().enumerate() {
                        let dst = &mut z[j * k..(j + 1) * k];
                        for (d, c) in dst.iter_mut().zip(yf) {
                            *d += a * c;
                        }
                    }
                }
                z
            })
            .reduce(
                || vec![0f32; self.n * k],
                |mut a, b| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    a
                },
            )
    }

    fn energia(&self) -> f64 {
        self.d
            .par_iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
    }
}

/// Pasa una matriz `filas × k` por filas a `k × filas` por columnas.
///
/// Hace falta porque Gram-Schmidt trabaja **columna contra columna**, y con el
/// dato por filas cada columna queda a zancadas de `k`: medido, esa versión
/// era la que se comía el tiempo entero, 10 mil millones de accesos salteados
/// en un solo hilo por cada llamada. Transponer son 39 millones de movimientos
/// y deja los dos bucles de Gram-Schmidt contiguos y vectorizables.
fn transponer(y: &[f32], filas: usize, k: usize) -> Vec<f32> {
    let mut t = vec![0f32; filas * k];
    t.par_chunks_mut(filas).enumerate().for_each(|(c, col)| {
        for (i, d) in col.iter_mut().enumerate() {
            *d = y[i * k + c];
        }
    });
    t
}

fn destransponer(t: &[f32], filas: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; filas * k];
    y.par_chunks_mut(k).enumerate().for_each(|(i, fila)| {
        for (c, d) in fila.iter_mut().enumerate() {
            *d = t[c * filas + i];
        }
    });
    y
}

/// Ortonormaliza las `k` columnas de una matriz guardada **por columnas**, con
/// Gram-Schmidt modificado, que es numéricamente estable a diferencia del
/// clásico.
///
/// Devuelve cuántas columnas sobrevivieron: una que queda en cero significa
/// que el rango real es menor que `k`, y eso es información, no un error.
fn ortonormalizar_col(t: &mut [f32], filas: usize, k: usize) -> usize {
    let mut vivas = 0usize;
    for c in 0..k {
        for p in 0..vivas {
            // Las dos columnas son contiguas, asi que el producto punto y la
            // resta van por rayon y se vectorizan.
            let (antes, resto) = t.split_at_mut(c * filas);
            let base = &antes[p * filas..(p + 1) * filas];
            let col = &mut resto[..filas];
            let punto: f32 = base
                .par_iter()
                .zip(col.par_iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum::<f64>() as f32;
            col.par_iter_mut()
                .zip(base.par_iter())
                .for_each(|(x, b)| *x -= punto * b);
        }
        let col = &mut t[c * filas..(c + 1) * filas];
        let norma = col
            .par_iter()
            .map(|v| {
                let v = *v as f64;
                v * v
            })
            .sum::<f64>()
            .sqrt() as f32;
        // 1e-6 y no cero: una columna con norma minuscula es ruido de
        // redondeo, y normalizarla metería basura amplificada en la base.
        if norma > 1e-6 {
            let inv = 1.0 / norma;
            col.par_iter_mut().for_each(|v| *v *= inv);
            if c != vivas {
                let (destino, origen) = t.split_at_mut(c * filas);
                destino[vivas * filas..(vivas + 1) * filas].copy_from_slice(&origen[..filas]);
            }
            vivas += 1;
        }
    }
    vivas
}

/// Ortonormaliza una matriz por filas, transponiendo a los dos lados.
fn ortonormalizar(y: &mut Vec<f32>, filas: usize, k: usize) -> usize {
    let mut t = transponer(y, filas, k);
    let vivas = ortonormalizar_col(&mut t, filas, k);
    *y = destransponer(&t, filas, k);
    vivas
}

/// Fracción de la energía de `a` que captura su mejor aproximación de rango
/// `k`, estimada con un buscador de rango aleatorizado.
///
/// `potencias` afila la estimación: cada iteración multiplica por `AᵀA`, que
/// eleva al cuadrado la separación entre valores singulares y hace que el
/// subespacio encontrado se parezca más al dominante. Dos alcanzan.
fn energia_de_rango(a: &Matriz, k: usize, total: f64, semilla: u64) -> f64 {
    let mut azar = cosxamix::daiss::Azar::con_semilla(semilla);
    let mut o = vec![0f32; a.n * k];
    let mut i = 0;
    while i + 1 < o.len() {
        let (x, y) = azar.normales(1.0);
        o[i] = x;
        o[i + 1] = y;
        i += 2;
    }

    let mut y = a.por(&o, k);
    const POTENCIAS: usize = 2;
    for _ in 0..POTENCIAS {
        ortonormalizar(&mut y, a.m, k);
        let z = a.traspuesta_por(&y, k);
        y = a.por(&z, k);
    }
    let vivas = ortonormalizar(&mut y, a.m, k);

    // ||Qᵀ A||²_F, que es la energía que el subespacio retiene.
    let qt_a = a.traspuesta_por(&y, k);
    let cap: f64 = (0..a.n)
        .map(|j| {
            (0..vivas)
                .map(|c| {
                    let v = qt_a[j * k + c] as f64;
                    v * v
                })
                .sum::<f64>()
        })
        .sum();
    (cap / total).min(1.0)
}

fn cargar(ruta: &std::path::Path, nombre: &str) -> Result<Matriz, Box<dyn std::error::Error>> {
    let g = Gguf::abrir(ruta)?;
    let info = g.tensor(nombre).ok_or_else(|| format!("falta {nombre}"))?;
    let bytes = g.datos_de(ruta, info)?;
    let (n, m) = (info.dims[0] as usize, info.dims[1] as usize);
    let mut d = vec![0f32; m * n];
    cosxamix::quant::descuantizar(info.tipo, &bytes, &mut d)?;
    Ok(Matriz { m, n, d })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::var("COSXAMIX_MODELOS").unwrap_or_else(|_| "modelos".into());
    let dir = std::path::Path::new(&dir);
    let entrenado = dir.join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    let aleatorio = dir.join("daiss-0.5b-q8_0.gguf");

    for r in [&entrenado, &aleatorio] {
        if !r.exists() {
            println!("  falta {}", r.display());
            if r == &aleatorio {
                println!("  generalo: COSXAMIX_DAISS=daiss-0.5b cargo run --release --example create_daiss");
            }
            return Ok(());
        }
    }

    println!();
    println!("  Energia capturada por la mejor aproximacion de rango k");
    println!();
    println!("  La apuesta de toda descomposicion de rango bajo es que una");
    println!("  matriz entrenada tenga rango efectivo mucho menor que el");
    println!("  nominal. Una matriz aleatoria es de rango completo por");
    println!("  construccion, asi que sirve de control.");
    println!();

    for tensor in ["blk.0.ffn_gate.weight", "output.weight"] {
        let a = cargar(&entrenado, tensor)?;
        let b = cargar(&aleatorio, tensor)?;
        if (a.m, a.n) != (b.m, b.n) {
            println!(
                "  {tensor}: formas distintas, {:?} vs {:?}",
                (a.m, a.n),
                (b.m, b.n)
            );
            continue;
        }
        let rango_max = a.m.min(a.n);
        let (ea, eb) = (a.energia(), b.energia());

        println!("  {tensor} — {} x {}, rango maximo {rango_max}", a.m, a.n);
        println!(
            "  {:>8}{:>8}{:>14}{:>14}{:>12}",
            "k", "k/max", "Qwen entren.", "daiss aleat.", "ventaja"
        );
        println!("  {:-<56}", "");

        for k in [8usize, 16, 32, 64, 128, 256] {
            if k > rango_max / 2 {
                break;
            }
            let t0 = Instant::now();
            let ca = energia_de_rango(&a, k, ea, 12345);
            let cb = energia_de_rango(&b, k, eb, 12345);
            println!(
                "  {:>8}{:>8}{:>14}{:>14}{:>12}   {:.1}s",
                k,
                format!("{:.1}%", k as f64 / rango_max as f64 * 100.0),
                format!("{:.1}%", ca * 100.0),
                format!("{:.1}%", cb * 100.0),
                format!("{:.2}x", ca / cb.max(1e-9)),
                t0.elapsed().as_secs_f64(),
            );
            // Se vacia a mano: redirigida a un archivo, la salida de Rust va
            // por bloques y no se ve nada hasta el final.
            let _ = std::io::stdout().flush();
        }
        println!();
    }

    // ── control: ¿cuanto de ese rango es ruido de cuantizacion? ─────────────
    //
    // Los pesos de Qwen vienen cuantizados, y el error de cuantizacion es de
    // rango completo por naturaleza, asi que infla el rango aparente. Si fuera
    // el que manda, el mismo tensor en Q2_K —mucho mas ruidoso— tendria una
    // curva bastante peor que en Q4_K. Si las tres curvas van juntas, el ruido
    // no explica nada y lo que se midio arriba es la matriz de verdad.
    let control = [
        ("qwen2.5-3b-instruct-q2_k.gguf", "Q2_K"),
        ("qwen2.5-3b-instruct-q3_k_m.gguf", "Q3_K_M"),
        ("qwen2.5-3b-instruct-q4_k_m.gguf", "Q4_K_M"),
    ];
    if control.iter().all(|(f, _)| dir.join(f).exists()) {
        println!("  Control: el MISMO tensor del 3B a tres niveles de ruido");
        println!("  (blk.0.ffn_gate.weight, 11008 x 2048)");
        println!();
        println!("  {:>8}{:>10}{:>10}{:>10}", "k", "Q2_K", "Q3_K_M", "Q4_K_M");
        println!("  {:-<38}", "");
        let mats: Vec<Matriz> = control
            .iter()
            .map(|(f, _)| cargar(&dir.join(f), "blk.0.ffn_gate.weight"))
            .collect::<Result<_, _>>()?;
        let energias: Vec<f64> = mats.iter().map(|m| m.energia()).collect();
        for k in [8usize, 32, 128] {
            let vals: Vec<String> = mats
                .iter()
                .zip(&energias)
                .map(|(m, e)| format!("{:.1}%", energia_de_rango(m, k, *e, 12345) * 100.0))
                .collect();
            println!("  {:>8}{:>10}{:>10}{:>10}", k, vals[0], vals[1], vals[2]);
            let _ = std::io::stdout().flush();
        }
        println!();
    }

    println!("  Como leerlo. Si la columna de Qwen sube mucho mas rapido que la");
    println!("  de daiss, la premisa de las redes de tensores es correcta Y la");
    println!("  conclusion tambien: sirven sobre pesos ENTRENADOS y no sobre los");
    println!("  nuestros, que todavia son ruido. Si las dos columnas van parejas,");
    println!("  la premisa no se cumple ni siquiera en un modelo entrenado y la");
    println!("  propuesta se cae por si sola.");
    println!();

    Ok(())
}
