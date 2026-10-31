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
/// Suma los ocho carriles `f32` de un acumulador.
///
/// Estaba escrita dos veces y ahora tres la usan. Baja por mitades —256 a 128,
/// a 64, a 32— en vez de sumar carril por carril: son tres instrucciones en
/// lugar de siete.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn reducir_f32(acc: __m256) -> f32 {
    let bajo = _mm256_castps256_ps128(acc);
    let alto = _mm256_extractf128_ps(acc, 1);
    let s4 = _mm_add_ps(bajo, alto);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    _mm_cvtss_f32(s1)
}

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
        return reducir_f32(acc);
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

    reducir_f32(acc)
}

/// Las ocho escalas de un bloque Q4_K, ya multiplicadas por las de la
/// entrada, y su aporte al término del desplazamiento.
///
/// Está aparte porque la usan **dos** kernels: [`dot_q4k_i8_avx2`] y el puente
/// del de ensamblador en `q4k_asm`. Que los dos calculen las escalas con el
/// mismo código es lo que hace que lo único distinto entre ellos sea el bucle
/// vectorial, que es lo que se quiere comparar. Con el puente calculándolas de
/// a una, como estaba, el ensamblador habría parecido más lento por culpa del
/// envoltorio.
///
/// El desempaquetado de los pares de 6 bits sigue siendo escalar: son ocho
/// bytes repartidos de una forma que no se vectoriza bien. Lo que viene
/// después sí: `cvtepu8_epi32` más `cvtepi32_ps` convierten ocho bytes en dos
/// instrucciones donde había dieciséis, y fue lo que dio el 1,90x.
///
/// `bloque` son los 144 bytes del bloque; `xe` y `xs`, los 8 suyos.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. El que llama lo garantiza.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn escalas_q4k(bloque: &[u8], xe: &[f32], xs: &[f32], desp: &mut __m256) -> [f32; 8] {
    // `d` y `dmin`, los dos f16 del encabezado, con UNA instruccion:
    // `vcvtph2ps`. Antes se convertian por software con `leer_f16`, que
    // desarma exponente y mantisa a mano. Lo encontro el ensamblador del
    // contrato completo, que usaba la instruccion y ganaba: dandosela tambien
    // a este camino, el kernel paso de ~97 a ~79 ns, sin tocar nada mas. Todo
    // f16 cabe exacto en un f32, asi que el resultado es el mismo bit a bit, y
    // lo comprueba el test de las escalas sin ramas.
    let h = _mm_cvtph_ps(_mm_cvtsi32_si128(i32::from_le_bytes(
        bloque[0..4]
            .try_into()
            .expect("un bloque Q4_K mide 144 bytes"),
    )));
    let d = _mm_cvtss_f32(h);
    let dmin = _mm_cvtss_f32(_mm_movehdup_ps(h));

    // Las 8 escalas y los 8 minimos de 6 bits, SIN RAMAS y de a cuatro.
    //
    // Antes eran 8 llamadas a `escala_y_minimo`, cada una con su `if j < 4` y
    // desempaquetando byte por byte. Medido aparte, calcular las escalas de
    // una fila costaba 70 ns: mas que el bucle vectorial entero. La forma de
    // los 12 bytes es fija, asi que se puede rearmar con mascaras sobre
    // palabras de 32 bits —es la tecnica que usa llama.cpp para estos mismos
    // bloques—:
    //
    //   - las escalas 0-3 son los 6 bits bajos de los bytes 0-3, y los
    //     minimos 0-3 los de los bytes 4-7;
    //   - las escalas 4-7 toman los 4 bits bajos de los bytes 8-11 y les ponen
    //     encima los 2 bits altos de los bytes 0-3; los minimos 4-7, los 4
    //     bits altos de 8-11 con los 2 altos de 4-7.
    //
    // Correr un `u32` a la derecha no cruza bytes para lo que importa: el
    // bit 6 y 7 de cada byte bajan al 0 y 1 del MISMO byte, y lo que entra
    // del byte vecino lo borra la mascara. Hay un test que lo compara contra
    // `escala_y_minimo` sobre miles de bloques.
    //
    // Los 12 bytes se toman como UN arreglo de tamano fijo, con una sola
    // comprobacion. La primera version leia `bloque[4 + i]` byte por byte:
    // doce comprobaciones de limites con sus caminos de panico, que agrandaron
    // la funcion lo suficiente para que LLVM dejara de ponerla en linea. Con
    // una llamada por bloque y el resultado por memoria, el kernel entero paso
    // de 90 a 178 ns: la tecnica estaba bien y lo que la hundio fue como
    // estaba escrita. Visto en el ensamblador generado.
    let q: &[u8; 12] = bloque[4..16]
        .try_into()
        .expect("un bloque Q4_K mide 144 bytes");
    let a = u32::from_le_bytes([q[0], q[1], q[2], q[3]]);
    let b = u32::from_le_bytes([q[4], q[5], q[6], q[7]]);
    let c = u32::from_le_bytes([q[8], q[9], q[10], q[11]]);
    const SEIS: u32 = 0x3F3F_3F3F;
    const CUATRO: u32 = 0x0F0F_0F0F;
    const DOS: u32 = 0x0303_0303;
    let sc = (a & SEIS) as u64 | (((c & CUATRO) | (((a >> 6) & DOS) << 4)) as u64) << 32;
    let mn = (b & SEIS) as u64 | ((((c >> 4) & CUATRO) | (((b >> 6) & DOS) << 4)) as u64) << 32;

    // Ocho bytes a ocho f32: `cvtepu8_epi32` ensancha y `cvtepi32_ps` convierte.
    let a_f32 = |v: u64| _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(v as i64)));
    let xe_v = _mm256_loadu_ps(xe.as_ptr());
    let xs_v = _mm256_loadu_ps(xs.as_ptr());

    let escalas_v = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(d), a_f32(sc)), xe_v);
    *desp = _mm256_fmadd_ps(
        _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(dmin), a_f32(mn)), xe_v),
        xs_v,
        *desp,
    );

    let mut escalas = [0f32; 8];
    _mm256_storeu_ps(escalas.as_mut_ptr(), escalas_v);
    escalas
}

/// [`super::scalar::dot_q4k_i8`] en AVX2, desempaquetando en registros.
///
/// Q4_K es el formato ideal para el primer kernel fusionado, y por un motivo
/// de la instrucción y no de popularidad: sus nibbles son **sin signo**, 0 a
/// 15, y la entrada cuantizada es **con signo**. Eso es literalmente la firma
/// de `vpmaddubsw` —`u8 × i8 → i16`—, así que no hace falta el rodeo del signo
/// con `vpsignb` que usa [`dot_i8_escalado_avx2`], y con él se va su único
/// caso límite: el del −128 en los dos operandos a la vez.
///
/// Tampoco puede saturar, y conviene dejar la cuenta escrita: el producto
/// máximo es 15 × 127 = 1905, `vpmaddubsw` suma dos de esos y da 3810, contra
/// los 32.767 que entran en un `i16`. Hay 8,6 veces de margen.
///
/// Acumula en `f32` con una sola reducción al final. Reducir por sub-bloque
/// fue el error que hizo que `matmul_i8` saliera más lenta que la de `f32`
/// pese a usar un kernel 4x más rápido.
///
/// ## Las escalas se calculan las ocho de una, y eso es la mitad del tiempo
///
/// La primera versión las calculaba de a una, intercaladas con el trabajo
/// vectorial. No se veía leyendo el código; se vio leyendo **el ensamblador
/// que genera el compilador para ella**, que tenía 40 `vmulss`, 16
/// `vcvtsi2ss` y 8 `vsubss` contra 8 `vpmaddubsw`, 8 `vpmaddwd` y 8
/// `vfmadd213ps`: más trabajo escalar que vectorial.
///
/// Un intento previo las **adelantó** fuera del bucle y no cambió nada
/// —adelantar las mueve, no las quita—. Calcularlas las ocho juntas sí:
/// `cvtepu8_epi32` más `cvtepi32_ps` convierten ocho bytes en dos
/// instrucciones donde había dieciséis, y `xe[bl*8 + j]` con `j` de 0 a 7 es
/// contiguo, igual que `xs`, así que entran de una carga. **Medido: 175,3 a
/// 92,2 ns, 1,90x.**
///
/// El desempaquetado de los pares de 6 bits sigue siendo escalar: ocho bytes
/// repartidos de una forma que no se vectoriza bien, y ya no son el cuello.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. El que llama lo garantiza.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q4k_i8_avx2(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    let unos = _mm256_set1_epi16(1);
    let nibble = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    let mut desp = _mm256_setzero_ps();

    for bl in 0..xq.len() / 256 {
        let o = bl * 144;
        let escalas = escalas_q4k(
            &bloques[o..o + 144],
            &xe[bl * 8..bl * 8 + 8],
            &xs[bl * 8..bl * 8 + 8],
            &mut desp,
        );
        let qs = bloques.as_ptr().add(o + 16);

        for par in 0..4 {
            let bytes = _mm256_loadu_si256(qs.add(par * 32) as *const __m256i);
            let bajos = _mm256_and_si256(bytes, nibble);
            let altos = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble);

            for (mitad, q) in [bajos, altos].into_iter().enumerate() {
                let x = _mm256_loadu_si256(
                    xq.as_ptr().add(bl * 256 + par * 64 + mitad * 32) as *const __m256i
                );
                let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                let escala = _mm256_set1_ps(escalas[par * 2 + mitad]);
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, acc);
            }
        }
    }

    reducir_f32(acc) - reducir_f32(desp)
}

/// [`super::scalar::dot_q3k_i8`] en AVX2.
///
/// Dos diferencias con [`dot_q4k_i8_avx2`], y las dos son del formato:
///
/// - **los valores llevan signo**, −4 a 3, así que se les suma 4 para entrar
///   en el rango sin signo que `vpmaddubsw` pide, y el 4 se descuenta después
///   con `Σ x·q = Σ x·(q+4) − 4·Σx`. La suma de la entrada ya está calculada;
/// - **el sub-bloque es de 16**, así que cada vector de 32 bytes cubre dos
///   escalas. Se resuelve con el reparto de carriles de AVX2: después de
///   `maddubs` y `madd`, los carriles `i32` 0-3 vienen de los bytes 0-15 y los
///   4-7 de los bytes 16-31, así que un vector `[s0 s0 s0 s0 s1 s1 s1 s1]`
///   aplica la escala correcta a cada mitad y el kernel sigue a ancho completo.
///
/// Tampoco satura: el valor sin signo llega a 7, por 127 son 889, `maddubs`
/// suma dos y da 1778 contra los 32.767 del `i16`.
///
/// ## Lo que lo llevó de 152 a 91 ns por fila de 2048
///
/// Cuatro cosas, ninguna de lenguaje, y por eso están acá y en el C y en el
/// ensamblador por igual:
///
/// 1. **Las dieciséis escalas de 6 bits, en vector.** `escalas_q3k` las
///    rearma en escalar, unas treinta operaciones de bytes por bloque. En
///    vector son ocho: los nibbles bajos con un `and` y un corrimiento, y los
///    dos bits altos de las cuatro palabras con **un** `vpsrlvd` de
///    corrimientos `[0 2 4 6]`.
/// 2. **Las escalas no pasan por la pila.** Antes se guardaban dieciséis `f32`
///    y se leían de a ocho por grupo; ahora viven en dos registros y cada
///    grupo saca las suyas con una permutación de un patrón constante.
/// 3. **El tercer bit, con dos operaciones y no cuatro.** El bit g de cada
///    byte de `hmask` se lleva al bit 2 con un corrimiento inmediato de 16
///    bits y se aísla con un `and`; lo que arrastra del byte vecino cae fuera.
///    Antes eran una constante por grupo, `and`, `cmpeq` y otro `and`. Con el
///    bucle desenrollado a mano, todos los corrimientos son inmediatos.
/// 4. **Dos acumuladores.** Las ocho FMA de un bloque iban en una sola cadena,
///    y cada una esperaba a la anterior.
///
/// El `d` en f16 se convierte con `vcvtph2ps`, como en Q4_K.
///
/// # Safety
///
/// Pide AVX2, FMA y F16C. El que llama lo garantiza.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn dot_q3k_i8_avx2(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));

    let unos = _mm256_set1_epi16(1);
    let dos_bits = _mm256_set1_epi8(0x03);
    let cuatro = _mm256_set1_epi8(4);
    let nibble = _mm_set1_epi8(0x0F);
    let tres = _mm_set1_epi8(3);
    let treinta_y_dos = _mm_set1_epi8(32);
    let corrimientos = _mm_setr_epi32(0, 2, 4, 6);
    // Los cuatro repartos de dos escalas en las dos mitades de los carriles.
    let r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    let r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    let r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut correccion = _mm256_setzero_ps();

    for bl in 0..xq.len() / 256 {
        let b = bloques.as_ptr().add(bl * 110);
        let base = bl * 256;

        // Las dieciséis escalas de 6 bits, en vector. Los cuatro bits bajos:
        // los nibbles bajos de los bytes 96-103 son las escalas 0-7, y los
        // altos, las 8-15. Los dos bits altos de la escala j están en el byte
        // 104 + j%4, corridos 2*(j/4): un `srlv` por palabra de 32 bits los
        // pone en su sitio de a cuatro.
        let lo8 = _mm_loadl_epi64(b.add(96) as *const __m128i);
        let bajos = _mm_unpacklo_epi64(
            _mm_and_si128(lo8, nibble),
            _mm_and_si128(_mm_srli_epi16(lo8, 4), nibble),
        );
        let t = (b.add(104) as *const i32).read_unaligned();
        let altos = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(t), corrimientos), tres);
        let esc8 = _mm_sub_epi8(_mm_or_si128(bajos, _mm_slli_epi16(altos, 4)), treinta_y_dos);

        let h = (b.add(108) as *const u16).read_unaligned();
        let d = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)));

        let xe_p = xe.as_ptr().add(bl * 16);
        let xs_p = xs.as_ptr().add(bl * 16);
        let e0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(esc8));
        let e1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(esc8, 8)));
        let v0 = _mm256_mul_ps(_mm256_mul_ps(d, e0), _mm256_loadu_ps(xe_p));
        let v1 = _mm256_mul_ps(_mm256_mul_ps(d, e1), _mm256_loadu_ps(xe_p.add(8)));
        correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
        correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p.add(8)), correccion);

        let hm = _mm256_loadu_si256(b as *const __m256i);
        let qs0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
        let qs1 = _mm256_loadu_si256(b.add(64) as *const __m256i);

        // Un grupo de 32 valores. Todos los corrimientos son inmediatos: con
        // el bucle desenrollado a mano no queda ninguno variable.
        //
        // El tercer bit: el bit g de cada byte de `hm`, llevado al bit 2 con
        // un corrimiento de 16 bits y aislado con un `and`. Lo que el
        // corrimiento arrastra del byte vecino cae fuera del bit 2.
        macro_rules! grupo {
            ($acc:ident, $g:literal, $qs:ident, $lo:literal, $alto:expr, $v:ident, $r:ident) => {{
                let bajos = _mm256_and_si256(_mm256_srli_epi16($qs, $lo), dos_bits);
                let q = _mm256_or_si256(bajos, _mm256_and_si256($alto, cuatro));
                let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32) as *const __m256i);
                let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                let escala = _mm256_permutevar8x32_ps($v, $r);
                $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, $acc);
            }};
        }
        grupo!(acc0, 0, qs0, 0, _mm256_slli_epi16(hm, 2), v0, r0);
        grupo!(acc1, 1, qs0, 2, _mm256_slli_epi16(hm, 1), v0, r1);
        grupo!(acc0, 2, qs0, 4, hm, v0, r2);
        grupo!(acc1, 3, qs0, 6, _mm256_srli_epi16(hm, 1), v0, r3);
        grupo!(acc0, 4, qs1, 0, _mm256_srli_epi16(hm, 2), v1, r0);
        grupo!(acc1, 5, qs1, 2, _mm256_srli_epi16(hm, 3), v1, r1);
        grupo!(acc0, 6, qs1, 4, _mm256_srli_epi16(hm, 4), v1, r2);
        grupo!(acc1, 7, qs1, 6, _mm256_srli_epi16(hm, 5), v1, r3);
    }

    reducir_f32(_mm256_add_ps(acc0, acc1)) - 4.0 * reducir_f32(correccion)
}

/// Producto de una fila Q6_K contra la entrada cuantizada, fundido con el
/// desempaquetado: el kernel del decode para la matriz de salida.
///
/// En el Q3_K_M del bot la matriz de salida —151.936 × 2048, 255 MB— va en
/// Q6_K, y sin kernel fusionado pasaba por el camino con buffer: desempaquetar
/// la fila entera a `i8` y después multiplicar. Medido con
/// `examples/measure_decode`, iba a 12,9 GB/s contra 18,6 del Q3_K y era el
/// 25% de cada token.
///
/// Misma forma que [`dot_q3k_i8_avx2`]: ocho grupos de 32 valores por bloque,
/// cada uno con dos escalas, que se reparten en las dos mitades de los
/// carriles con un `permutevar`. Los valores se arman sin signo (0 a 63) y el
/// 32 que hay que restarles se descuenta con la suma de la entrada. El layout
/// está en [`super::scalar::dot_q6k_i8`].
///
/// # Safety
///
/// AVX2, FMA y F16C, y `bloques` con `xq.len() / 256 · 210` bytes.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn dot_q6k_i8_avx2(bloques: &[u8], xq: &[i8], xe: &[f32], xs: &[f32]) -> f32 {
    debug_assert!(xq.len().is_multiple_of(256));

    let unos = _mm256_set1_epi16(1);
    let nibble = _mm256_set1_epi8(0x0F);
    let altos = _mm256_set1_epi8(0x30);
    let r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    let r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    let r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    let r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut correccion = _mm256_setzero_ps();

    for bl in 0..xq.len() / 256 {
        let b = bloques.as_ptr().add(bl * 210);
        let base = bl * 256;

        // Las dieciséis escalas son `i8` seguidos: sin el armado de bits de
        // Q3_K.
        let esc8 = _mm_loadu_si128(b.add(192) as *const __m128i);
        let h = (b.add(208) as *const u16).read_unaligned();
        let d = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32)));

        let xe_p = xe.as_ptr().add(bl * 16);
        let xs_p = xs.as_ptr().add(bl * 16);
        let e0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(esc8));
        let e1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(esc8, 8)));
        let v0 = _mm256_mul_ps(_mm256_mul_ps(d, e0), _mm256_loadu_ps(xe_p));
        let v1 = _mm256_mul_ps(_mm256_mul_ps(d, e1), _mm256_loadu_ps(xe_p.add(8)));
        correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
        correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p.add(8)), correccion);

        // Un grupo de 32: 4 bits de `ql` y 2 de `qh` llevados a los bits 4-5.
        // Los corrimientos de 16 bits arrastran bits del byte vecino, pero
        // caen fuera de la máscara.
        macro_rules! grupo {
            ($acc:ident, $g:literal, $bajos:expr, $alto:expr, $v:ident, $r:ident) => {{
                let q = _mm256_or_si256(
                    _mm256_and_si256($bajos, nibble),
                    _mm256_and_si256($alto, altos),
                );
                let x = _mm256_loadu_si256(xq.as_ptr().add(base + $g * 32) as *const __m256i);
                let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);
                let escala = _mm256_permutevar8x32_ps($v, $r);
                $acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, $acc);
            }};
        }
        let ql0 = _mm256_loadu_si256(b as *const __m256i);
        let ql1 = _mm256_loadu_si256(b.add(32) as *const __m256i);
        let qh = _mm256_loadu_si256(b.add(128) as *const __m256i);
        grupo!(acc0, 0, ql0, _mm256_slli_epi16(qh, 4), v0, r0);
        grupo!(acc1, 1, ql1, _mm256_slli_epi16(qh, 2), v0, r1);
        grupo!(acc0, 2, _mm256_srli_epi16(ql0, 4), qh, v0, r2);
        grupo!(
            acc1,
            3,
            _mm256_srli_epi16(ql1, 4),
            _mm256_srli_epi16(qh, 2),
            v0,
            r3
        );

        let ql0 = _mm256_loadu_si256(b.add(64) as *const __m256i);
        let ql1 = _mm256_loadu_si256(b.add(96) as *const __m256i);
        let qh = _mm256_loadu_si256(b.add(160) as *const __m256i);
        grupo!(acc0, 4, ql0, _mm256_slli_epi16(qh, 4), v1, r0);
        grupo!(acc1, 5, ql1, _mm256_slli_epi16(qh, 2), v1, r1);
        grupo!(acc0, 6, _mm256_srli_epi16(ql0, 4), qh, v1, r2);
        grupo!(
            acc1,
            7,
            _mm256_srli_epi16(ql1, 4),
            _mm256_srli_epi16(qh, 2),
            v1,
            r3
        );
    }

    reducir_f32(_mm256_add_ps(acc0, acc1)) - 32.0 * reducir_f32(correccion)
}

/// Una **tesela** de la matmul por bloques: `R` filas de la entrada contra
/// `C` columnas de pesos, todas a la vez, con los `R·C` acumuladores en
/// registros.
///
/// Es lo que cambia frente a [`dot_i8_escalado_avx2`], que hace un producto
/// por llamada: ahí cada trozo de 32 bytes que se trae se usa **una** vez. Acá
/// la columna cargada sirve para las `R` filas y la fila para las `C`
/// columnas, así que por cada byte traído de la caché se hacen `R` o `C`
/// veces más cuentas. Y la escala deja de armarse con `setr` en cada trozo
/// —que costaba más que el producto—: las dos vienen ya expandidas, un `f32`
/// por carril (ver `ops::expandir_escalas`).
///
/// `filas[r]` y `cols[c]` apuntan a `32·trozos` valores `i8`; `esc_filas[r]` y
/// `esc_cols[c]` a `8·trozos` escalas expandidas.
///
/// # Safety
///
/// La CPU tiene que tener AVX2 y FMA, y los punteros tienen que cubrir lo que
/// dice arriba. Los valores de `filas` tienen que estar en `-127..=127`: es
/// el rodeo de signo de `vpmaddubsw`, igual que en `dot_i8_escalado_avx2`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn tesela_i8_avx2<const R: usize, const C: usize>(
    filas: &[*const i8; R],
    esc_filas: &[*const f32; R],
    cols: &[*const i8; C],
    esc_cols: &[*const f32; C],
    trozos: usize,
) -> [[f32; C]; R] {
    let unos = _mm256_set1_epi16(1);
    let mut acc = [[_mm256_setzero_ps(); C]; R];
    for t in 0..trozos {
        let mut crudo = [_mm256_setzero_si256(); C];
        let mut abs = [_mm256_setzero_si256(); C];
        let mut ec = [_mm256_setzero_ps(); C];
        for c in 0..C {
            crudo[c] = _mm256_loadu_si256(cols[c].add(32 * t) as *const __m256i);
            abs[c] = _mm256_sign_epi8(crudo[c], crudo[c]);
            ec[c] = _mm256_loadu_ps(esc_cols[c].add(8 * t));
        }
        for r in 0..R {
            let x = _mm256_loadu_si256(filas[r].add(32 * t) as *const __m256i);
            let ef = _mm256_loadu_ps(esc_filas[r].add(8 * t));
            for c in 0..C {
                let p16 = _mm256_maddubs_epi16(abs[c], _mm256_sign_epi8(x, crudo[c]));
                let p32 = _mm256_madd_epi16(p16, unos);
                let v = _mm256_mul_ps(_mm256_cvtepi32_ps(p32), ec[c]);
                acc[r][c] = _mm256_fmadd_ps(v, ef, acc[r][c]);
            }
        }
    }
    let mut salida = [[0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            salida[r][c] = reducir_f32(acc[r][c]);
        }
    }
    salida
}

/// La misma tesela con los pesos **sin signo** (`0..=129`): ya no hace falta
/// el rodeo del signo, y por cada producto se ahorran dos instrucciones —el
/// valor absoluto de la columna y el signo pasado a la fila—.
///
/// Quien llama corre los pesos para que queden sin signo y descuenta el
/// corrimiento aparte (ver `ops::matmul_i8_bloques`). El tope de 129 es el de
/// `vpmaddubsw`, que suma dos productos en 16 bits con saturación:
/// `2 · 129 · 127 = 32766` todavía entra.
///
/// # Safety
///
/// Lo mismo que [`tesela_i8_avx2`], con `cols` en `0..=129`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn tesela_u8_avx2<const R: usize, const C: usize>(
    filas: &[*const i8; R],
    esc_filas: &[*const f32; R],
    cols: &[*const i8; C],
    esc_cols: &[*const f32; C],
    trozos: usize,
) -> [[f32; C]; R] {
    let unos = _mm256_set1_epi16(1);
    let mut acc = [[_mm256_setzero_ps(); C]; R];
    for t in 0..trozos {
        let mut col = [_mm256_setzero_si256(); C];
        let mut ec = [_mm256_setzero_ps(); C];
        for c in 0..C {
            col[c] = _mm256_loadu_si256(cols[c].add(32 * t) as *const __m256i);
            ec[c] = _mm256_loadu_ps(esc_cols[c].add(8 * t));
        }
        for r in 0..R {
            let x = _mm256_loadu_si256(filas[r].add(32 * t) as *const __m256i);
            let ef = _mm256_loadu_ps(esc_filas[r].add(8 * t));
            for c in 0..C {
                let p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(col[c], x), unos);
                let v = _mm256_mul_ps(_mm256_cvtepi32_ps(p32), ec[c]);
                acc[r][c] = _mm256_fmadd_ps(v, ef, acc[r][c]);
            }
        }
    }
    let mut salida = [[0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            salida[r][c] = reducir_f32(acc[r][c]);
        }
    }
    salida
}
