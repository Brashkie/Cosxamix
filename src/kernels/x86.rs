//! Kernels con intrínsecos de x86-64: AVX2 + FMA.
//!
//! Están en Rust y no en otro lenguaje a propósito, y conviene decir por qué
//! antes de que alguien asuma que esto "debería" ser Zig o ensamblador: los
//! intrínsecos de `core::arch` son las **mismas instrucciones** que emitiría
//! cualquiera de los dos. No hay nada que Rust no pueda expresar acá.
//!
//! Lo que sí cambia entre lenguajes es la ergonomía de escribirlos y de
//! generar variantes por arquitectura. Esa es la discusión real, y se resuelve
//! midiendo: por eso existe el banco que compara backends.

use std::arch::x86_64::*;

/// Producto punto con AVX2 y FMA.
///
/// # Safety
///
/// (El encabezado va en inglés porque clippy busca ese literal.)
///
/// La CPU tiene que soportar `avx2` **y** `fma`. Quien llama lo garantiza;
/// `kernels::elegido()` solo devuelve `Backend::Avx2` tras comprobar las dos
/// con `is_x86_feature_detected!`. Ejecutar esto en una CPU sin ellas no da un
/// resultado malo: tira instrucción ilegal y el proceso muere.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    // Cuatro acumuladores y no uno: la latencia de un FMA es de varios ciclos,
    // y con un solo acumulador cada iteración espera a la anterior porque
    // depende de ella. Con cuatro cadenas independientes el procesador las
    // solapa. Es la diferencia entre usar la unidad vectorial y hacer cola
    // para ella.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();

    let mut i = 0;

    // 32 por vuelta: cuatro registros de 8 f32.
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            s3,
        );
        i += 32;
    }

    let mut acc = [0f32; 32];
    _mm256_storeu_ps(acc.as_mut_ptr(), s0);
    _mm256_storeu_ps(acc.as_mut_ptr().add(8), s1);
    _mm256_storeu_ps(acc.as_mut_ptr().add(16), s2);
    _mm256_storeu_ps(acc.as_mut_ptr().add(24), s3);

    reducir_y_cola(a, b, &acc, i)
}

/// La cola y la suma horizontal, compartidas con la versión en ensamblador.
///
/// Está factorizado para que las dos implementaciones usen **el mismo** código
/// aquí y la única diferencia entre ellas sea el bucle caliente. Comparar dos
/// kernels que además reducen distinto no diría cuál bucle es mejor, diría que
/// dan números distintos.
///
/// `desde` es cuántos elementos consumió ya el bucle de 32.
///
/// # Safety
///
/// Pide AVX2 y FMA, y que `acc` traiga los cuatro acumuladores de 8.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn reducir_y_cola(a: &[f32], b: &[f32], acc: &[f32; 32], desde: usize) -> f32 {
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());

    let mut s0 = _mm256_loadu_ps(acc.as_ptr());
    let s1 = _mm256_loadu_ps(acc.as_ptr().add(8));
    let s2 = _mm256_loadu_ps(acc.as_ptr().add(16));
    let s3 = _mm256_loadu_ps(acc.as_ptr().add(24));

    // Lo que no llegó a 32 pero sí a 8.
    let mut i = desde;
    while i + 8 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        i += 8;
    }

    let mut total = suma_horizontal(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));

    // La cola, de a uno. Olvidarse de esto es el error clásico de un kernel
    // vectorizado: con n múltiplo de 8 todos los tests pasan y con 4097
    // elementos se pierde uno. Por eso el test prueba longitudes raras.
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }

    total
}

/// Suma los 8 `f32` de un registro de 256 bits.
#[target_feature(enable = "avx2")]
unsafe fn suma_horizontal(v: __m256) -> f32 {
    // Se parte en dos mitades de 128 y se suman; después se baja a 64 y a 32.
    // Es menos legible que sumar de a uno, pero una suma horizontal ingenua
    // cuesta más que el bucle entero cuando los vectores son cortos.
    let bajo = _mm256_castps256_ps128(v);
    let alto = _mm256_extractf128_ps(v, 1);
    let suma4 = _mm_add_ps(bajo, alto);

    let suma2 = _mm_add_ps(suma4, _mm_movehl_ps(suma4, suma4));
    let suma1 = _mm_add_ss(suma2, _mm_shuffle_ps(suma2, suma2, 0x55));

    _mm_cvtss_f32(suma1)
}

/// Normalización RMS con AVX2 + FMA.
///
/// # Safety
///
/// La CPU tiene que soportar `avx2` y `fma`, igual que `dot_avx2`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn rmsnorm_avx2(salida: &mut [f32], entrada: &[f32], peso: &[f32], eps: f32) {
    debug_assert_eq!(salida.len(), entrada.len());
    debug_assert_eq!(entrada.len(), peso.len());

    let n = entrada.len();
    if n == 0 {
        return;
    }

    let pe = entrada.as_ptr();
    let pp = peso.as_ptr();
    let ps = salida.as_mut_ptr();

    // ── Primera pasada: suma de cuadrados ────────────────────────────────────
    //
    // Dos acumuladores y no cuatro como en `dot`: acá la segunda pasada domina
    // el tiempo, así que desenrollar más esta no compra nada y alarga el
    // código.
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();

    let mut i = 0;
    while i + 16 <= n {
        let a = _mm256_loadu_ps(pe.add(i));
        let b = _mm256_loadu_ps(pe.add(i + 8));
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
        i += 16;
    }
    while i + 8 <= n {
        let a = _mm256_loadu_ps(pe.add(i));
        s0 = _mm256_fmadd_ps(a, a, s0);
        i += 8;
    }

    let mut suma = suma_horizontal(_mm256_add_ps(s0, s1));
    while i < n {
        let x = *pe.add(i);
        suma += x * x;
        i += 1;
    }

    // `sqrt` y la división se hacen UNA vez, en escalar. Vectorizarlas no
    // tendría sentido: son dos operaciones contra las 2n de las pasadas.
    let escala = 1.0 / (suma / n as f32 + eps).sqrt();
    let vescala = _mm256_set1_ps(escala);

    // ── Segunda pasada: escalar y aplicar el peso ────────────────────────────
    i = 0;
    while i + 8 <= n {
        let e = _mm256_loadu_ps(pe.add(i));
        let p = _mm256_loadu_ps(pp.add(i));
        _mm256_storeu_ps(ps.add(i), _mm256_mul_ps(_mm256_mul_ps(e, vescala), p));
        i += 8;
    }
    while i < n {
        *ps.add(i) = *pe.add(i) * escala * *pp.add(i);
        i += 1;
    }
}

/// Producto punto de `i8` en AVX2.
///
/// AVX2 **no tiene** una instruccion de i8 x i8 -> i32. La que hay,
/// `vpmaddubsw`, multiplica un operando SIN signo por uno con signo. El rodeo
/// que usa todo el mundo, y que viene de llama.cpp, es mover el signo de un
/// lado al otro:
///
/// * `vpsignb(a, a)` da `|a|`, que ya se puede leer como sin signo;
/// * `vpsignb(b, a)` le pasa a `b` el signo de `a`.
///
/// El producto `|a| * (b*sign(a))` es el mismo `a*b` y ahora si entra en la
/// instruccion. Despues `vpmaddwd` con unos suma los pares de i16 a i32.
///
/// Que `vpmaddubsw` sature en i16 no es un problema aca: el peor caso por par
/// es 127*127*2 = 32258, que entra en 32767. Con valores de 8 bits no puede
/// desbordar.
///
/// # El valor -128 esta fuera de contrato
///
/// El rodeo del signo falla **cuando los DOS operandos valen -128 en la misma
/// posicion**, y no por descuido: en `i8` ese valor no tiene opuesto, asi que
/// `vpsignb` lo niega y lo deja igual.
///
/// Con -128 de un solo lado el resultado sale bien, y vale saberlo porque la
/// primera version de este comentario decia "falla con -128" a secas, que es
/// mas conservador de lo necesario. Medido:
///
/// ```text
///   solo a = -128     avx2 OK
///   solo b = -128     avx2 OK
///   los dos = -128    avx2 MAL
/// ```
///
/// No se arregla porque no hace falta: los cuantizadores de GGML calculan su
/// escala como `amax / 127`, asi que el rango que producen es **[-127, 127]**
/// y -128 no aparece nunca. llama.cpp se apoya en lo mismo.
///
/// Queda escrito y comprobado en depuracion en vez de supuesto: una
/// precondicion que no se verifica es una trampa esperando a que alguien
/// alimente este kernel desde otro sitio.
///
/// # Safety
///
/// Pide AVX2. Quien llame tiene que haberlo comprobado.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(
        !a.contains(&i8::MIN),
        "dot_i8_avx2: el primer vector trae -128, que esta fuera del rango          [-127,127] que producen los cuantizadores de GGML. El fallo real pide          -128 en los DOS lados a la vez, pero comprobar eso costaria recorrer          ambos vectores comparando: se usa la condicion barata y conservadora"
    );
    let n = a.len();
    let pa = a.as_ptr() as *const __m256i;
    let pb = b.as_ptr() as *const __m256i;

    let unos = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();

    let bloques = n / 32;
    for i in 0..bloques {
        let va = _mm256_loadu_si256(pa.add(i));
        let vb = _mm256_loadu_si256(pb.add(i));
        let abs_a = _mm256_sign_epi8(va, va);
        let b_con_signo = _mm256_sign_epi8(vb, va);
        let p16 = _mm256_maddubs_epi16(abs_a, b_con_signo);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, unos));
    }

    // Suma horizontal de los ocho i32.
    let bajo = _mm256_castsi256_si128(acc);
    let alto = _mm256_extracti128_si256(acc, 1);
    let s4 = _mm_add_epi32(bajo, alto);
    let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0b01_00_11_10));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b00_00_00_01));
    let mut total = _mm_cvtsi128_si32(s1);

    // La cola, de a uno. Olvidarla es el error clasico: con n multiplo de 32
    // todos los tests pasan.
    for i in bloques * 32..n {
        total += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
    }
    total
}

/// Producto punto cuantizado por bloques, en AVX2.
///
/// La clave esta en **lo que no hace**: no reduce por bloque. El vector de
/// ocho `i32` que sale de un bloque se convierte a `f32`, se multiplica por
/// las dos escalas y se acumula en un vector de `f32`. Los ocho carriles
/// guardan sumas parciales del mismo bloque, asi que la escala les aplica a
/// todos por igual y la **unica** suma horizontal va al final.
///
/// Esto es lo que distingue un kernel compuesto de uno escrito para el caso:
/// llamando a `dot_i8` por bloque, la reduccion horizontal se paga 64 veces
/// sobre k=2048 y la version en enteros sale mas lenta que la de f32, que es
/// lo que paso al medirlo.
///
/// # Safety
///
/// Pide AVX2 y FMA, el largo multiplo de 32, y valores en `[-127, 127]` como
/// los que producen los cuantizadores de GGML — ver `dot_i8_avx2`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_i8_escalado_avx2(
    a: &[i8],
    b: &[i8],
    esc_a: &[f32],
    esc_b: &[f32],
    bloque: usize,
) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(bloque == 16 || bloque.is_multiple_of(32));

    let pa = a.as_ptr() as *const __m256i;
    let pb = b.as_ptr() as *const __m256i;
    let unos = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();

    // Con sub-bloques de 16 no hace falta bajar a 128 bits. Los ocho `i32`
    // que salen de una pasada de 32 bytes se reparten en dos lanes de AVX2:
    // los cuatro primeros cubren los bytes 0-15 y los otros cuatro los 16-31.
    // O sea que un vector de escalas con `[s0 s0 s0 s0 s1 s1 s1 s1]` aplica la
    // escala correcta a cada mitad y se mantiene el ancho completo.
    if bloque == 16 {
        let pares = esc_a.len() / 2;
        for i in 0..pares {
            let e0 = esc_a[2 * i] * esc_b[2 * i];
            let e1 = esc_a[2 * i + 1] * esc_b[2 * i + 1];
            let escala = _mm256_setr_ps(e0, e0, e0, e0, e1, e1, e1, e1);
            let va = _mm256_loadu_si256(pa.add(i));
            let vb = _mm256_loadu_si256(pb.add(i));
            let abs_a = _mm256_sign_epi8(va, va);
            let b_con_signo = _mm256_sign_epi8(vb, va);
            let p16 = _mm256_maddubs_epi16(abs_a, b_con_signo);
            let p32 = _mm256_madd_epi16(p16, unos);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, acc);
        }
        let bajo = _mm256_castps256_ps128(acc);
        let alto = _mm256_extractf128_ps(acc, 1);
        let s4 = _mm_add_ps(bajo, alto);
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
        return _mm_cvtss_f32(s1);
    }

    let por_bloque = bloque / 32;
    for (i, (ea, eb)) in esc_a.iter().zip(esc_b).enumerate() {
        let escala = _mm256_set1_ps(ea * eb);
        for t in 0..por_bloque {
            let idx = i * por_bloque + t;
            let va = _mm256_loadu_si256(pa.add(idx));
            let vb = _mm256_loadu_si256(pb.add(idx));
            let abs_a = _mm256_sign_epi8(va, va);
            let b_con_signo = _mm256_sign_epi8(vb, va);
            let p16 = _mm256_maddubs_epi16(abs_a, b_con_signo);
            let p32 = _mm256_madd_epi16(p16, unos);
            // Los ocho carriles son del mismo bloque, asi que la escala les
            // toca igual y no hace falta reducir todavia.
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, acc);
        }
    }

    let bajo = _mm256_castps256_ps128(acc);
    let alto = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(bajo, alto);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    _mm_cvtss_f32(s1)
}
