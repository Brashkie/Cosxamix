// El kernel fusionado de Q3_K en C, con intrínsecos de AVX2.
//
// El mismo algoritmo que `dot_q3k_i8_avx2` de `src/kernels/x86.rs`, línea por
// línea, para que lo que se mida sea el lenguaje y su compilador y no una idea
// distinta. La especificación ejecutable sigue siendo
// `scalar::dot_q3k_i8`, y los tests de `src/kernels/c.rs` comparan contra ella.
//
// Por qué C y no solo Rust o Zig: los intrínsecos son los mismos en los tres,
// pero C pone las extensiones de CPU **por función** con
// `__attribute__((target))`, como `#[target_feature]` de Rust y a diferencia
// de Zig, que las pone por archivo con `-mcpu`. Así este objeto se compila una
// sola vez, en baseline, y el kernel de AVX2 vive dentro sin que nada más del
// archivo lo pida: es seguro también en el binario para las CPU sin AVX2.
//
// Lo compila `zig cc` desde build.rs, con la feature `c`: no hace falta otro
// compilador que el que ya pide la feature `zig`. Con `-ffreestanding`, sin
// libc: no usa nada de ella, y así no hace falta el SDK de cada plataforma
// para cruzar (por eso `__builtin_memcpy` y no `memcpy`).

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define AVX2 __attribute__((target("avx2,fma,f16c")))

AVX2 static inline float reducir(__m256 v) {
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    return _mm_cvtss_f32(s1);
}

// Un grupo de 32 valores: el desempaquetado, el producto entero y la escala.
// Macro y no función para que los corrimientos sean inmediatos.
#define GRUPO(acc, g, qs, lo, alto, v, r)                                              \
    do {                                                                               \
        __m256i bajos = _mm256_and_si256(_mm256_srli_epi16(qs, lo), dos_bits);         \
        __m256i q = _mm256_or_si256(bajos, _mm256_and_si256(alto, cuatro));            \
        __m256i x = _mm256_loadu_si256((const __m256i *)(xq + base + (g) * 32));       \
        __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), unos);             \
        __m256 escala = _mm256_permutevar8x32_ps(v, r);                                \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, acc);                   \
    } while (0)

AVX2 float cosxamix_q3k_c(const uint8_t *bloques, const int8_t *xq, const float *xe,
                          const float *xs, size_t n_bloques) {
    const __m256i unos = _mm256_set1_epi16(1);
    const __m256i dos_bits = _mm256_set1_epi8(0x03);
    const __m256i cuatro = _mm256_set1_epi8(4);
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m128i tres = _mm_set1_epi8(3);
    const __m128i treinta_y_dos = _mm_set1_epi8(32);
    const __m128i corrimientos = _mm_setr_epi32(0, 2, 4, 6);
    const __m256i r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    const __m256i r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    const __m256i r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    const __m256i r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 correccion = _mm256_setzero_ps();

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 110;
        const size_t base = bl * 256;

        // Las dieciséis escalas de 6 bits, en vector (ver la versión de Rust).
        __m128i lo8 = _mm_loadl_epi64((const __m128i *)(b + 96));
        __m128i bajos = _mm_unpacklo_epi64(_mm_and_si128(lo8, nibble),
                                           _mm_and_si128(_mm_srli_epi16(lo8, 4), nibble));
        int32_t t;
        __builtin_memcpy(&t, b + 104, 4);
        __m128i altos = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(t), corrimientos), tres);
        __m128i esc8 = _mm_sub_epi8(_mm_or_si128(bajos, _mm_slli_epi16(altos, 4)), treinta_y_dos);

        uint16_t h;
        __builtin_memcpy(&h, b + 108, 2);
        __m256 d = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));

        const float *xe_p = xe + bl * 16;
        const float *xs_p = xs + bl * 16;
        __m256 e0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(esc8));
        __m256 e1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(esc8, 8)));
        __m256 v0 = _mm256_mul_ps(_mm256_mul_ps(d, e0), _mm256_loadu_ps(xe_p));
        __m256 v1 = _mm256_mul_ps(_mm256_mul_ps(d, e1), _mm256_loadu_ps(xe_p + 8));
        correccion = _mm256_fmadd_ps(v0, _mm256_loadu_ps(xs_p), correccion);
        correccion = _mm256_fmadd_ps(v1, _mm256_loadu_ps(xs_p + 8), correccion);

        __m256i hm = _mm256_loadu_si256((const __m256i *)b);
        __m256i qs0 = _mm256_loadu_si256((const __m256i *)(b + 32));
        __m256i qs1 = _mm256_loadu_si256((const __m256i *)(b + 64));

        GRUPO(acc0, 0, qs0, 0, _mm256_slli_epi16(hm, 2), v0, r0);
        GRUPO(acc1, 1, qs0, 2, _mm256_slli_epi16(hm, 1), v0, r1);
        GRUPO(acc0, 2, qs0, 4, hm, v0, r2);
        GRUPO(acc1, 3, qs0, 6, _mm256_srli_epi16(hm, 1), v0, r3);
        GRUPO(acc0, 4, qs1, 0, _mm256_srli_epi16(hm, 2), v1, r0);
        GRUPO(acc1, 5, qs1, 2, _mm256_srli_epi16(hm, 3), v1, r1);
        GRUPO(acc0, 6, qs1, 4, _mm256_srli_epi16(hm, 4), v1, r2);
        GRUPO(acc1, 7, qs1, 6, _mm256_srli_epi16(hm, 5), v1, r3);
    }

    return reducir(_mm256_add_ps(acc0, acc1)) - 4.0f * reducir(correccion);
}
