// El kernel fusionado de Q2_K en C, con intrínsecos de AVX2.
//
// El mismo algoritmo que `avx2` en `src/kernels/q2k.rs`, para que lo que se mida
// sea el lenguaje y su compilador. La especificación ejecutable es `escalar` en
// ese mismo archivo.

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

AVX2 static inline __m256 f16_en(const uint8_t *p) {
    uint16_t h;
    __builtin_memcpy(&h, p, 2);
    return _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}

AVX2 static inline __m256 f8(__m128i v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v));
}

AVX2 float cosxamix_q2k_c(const uint8_t *bloques, const int8_t *xq, const float *xe,
                          const float *xs, size_t n_bloques) {
    const __m256i unos = _mm256_set1_epi16(1);
    const __m256i dos_bits = _mm256_set1_epi8(3);
    const __m128i nibble = _mm_set1_epi8(0x0F);
    const __m256i r0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);
    const __m256i r1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    const __m256i r2 = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);
    const __m256i r3 = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 correccion = _mm256_setzero_ps();

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 84;
        const __m128i esc = _mm_loadu_si128((const __m128i *)b);
        const __m128i bajos = _mm_and_si128(esc, nibble);
        const __m128i altos = _mm_and_si128(_mm_srli_epi16(esc, 4), nibble);
        const __m256 d = f16_en(b + 80);
        const __m256 dm = f16_en(b + 82);
        const float *e = xe + bl * 16;
        const float *s = xs + bl * 16;
        const __m256 e0 = _mm256_loadu_ps(e);
        const __m256 e1 = _mm256_loadu_ps(e + 8);
        const __m256 v0 = _mm256_mul_ps(_mm256_mul_ps(d, f8(bajos)), e0);
        const __m256 v1 = _mm256_mul_ps(_mm256_mul_ps(d, f8(_mm_srli_si128(bajos, 8))), e1);
        const __m256 m0 = _mm256_mul_ps(_mm256_mul_ps(dm, f8(altos)), e0);
        const __m256 m1 = _mm256_mul_ps(_mm256_mul_ps(dm, f8(_mm_srli_si128(altos, 8))), e1);
        correccion = _mm256_fmadd_ps(m0, _mm256_loadu_ps(s), correccion);
        correccion = _mm256_fmadd_ps(m1, _mm256_loadu_ps(s + 8), correccion);

        const __m256i qs0 = _mm256_loadu_si256((const __m256i *)(b + 16));
        const __m256i qs1 = _mm256_loadu_si256((const __m256i *)(b + 48));
        const int8_t *x = xq + bl * 256;

#define GRUPO(acc, g, qs, lo, v, r)                                                     \
    do {                                                                                \
        const __m256i q = _mm256_and_si256(_mm256_srli_epi16(qs, lo), dos_bits);        \
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + (g) * 32));         \
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(q, xv), unos);       \
        const __m256 escala = _mm256_permutevar8x32_ps(v, r);                           \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), escala, acc);                    \
    } while (0)

        GRUPO(acc0, 0, qs0, 0, v0, r0);
        GRUPO(acc1, 1, qs0, 2, v0, r1);
        GRUPO(acc0, 2, qs0, 4, v0, r2);
        GRUPO(acc1, 3, qs0, 6, v0, r3);
        GRUPO(acc0, 4, qs1, 0, v1, r0);
        GRUPO(acc1, 5, qs1, 2, v1, r1);
        GRUPO(acc0, 6, qs1, 4, v1, r2);
        GRUPO(acc1, 7, qs1, 6, v1, r3);
#undef GRUPO
    }
    return reducir(_mm256_add_ps(acc0, acc1)) - reducir(correccion);
}
