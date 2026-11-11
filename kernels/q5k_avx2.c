// El kernel fusionado de Q5_K en C, con intrínsecos de AVX2.
//
// El algoritmo es el del ensamblador (`q5k_avx2_*.s`): el quinto bit de cada
// valor se lleva a la posición 4 corriendo `qh` y se aísla con un `and 0x10`.
// La especificación ejecutable es `scalar::dot_q5k_i8` en `src/kernels/scalar.rs`.

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

// Ocho bytes a ocho f32.
AVX2 static inline __m256 f8(uint64_t v) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)v)));
}

static inline uint32_t leer32(const uint8_t *p) {
    uint32_t v;
    __builtin_memcpy(&v, p, 4);
    return v;
}

AVX2 float cosxamix_q5k_c(const uint8_t *bloques, const int8_t *xq, const float *xe,
                          const float *xs, size_t n_bloques) {
    const __m256i unos = _mm256_set1_epi16(1);
    const __m256i nibble = _mm256_set1_epi8(0x0F);
    const __m256i bit5 = _mm256_set1_epi8(0x10);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 correccion = _mm256_setzero_ps();

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 176;
        // Las escalas y los mínimos de 6 bits, sin ramas (ver `escalas_q4k`).
        const uint32_t a = leer32(b + 4), m = leer32(b + 8), c = leer32(b + 12);
        const uint32_t seis = 0x3F3F3F3F, cuatro = 0x0F0F0F0F, dos = 0x03030303;
        const uint64_t sc =
            (a & seis) | ((uint64_t)((c & cuatro) | (((a >> 6) & dos) << 4)) << 32);
        const uint64_t mn =
            (m & seis) | ((uint64_t)(((c >> 4) & cuatro) | (((m >> 6) & dos) << 4)) << 32);

        const __m256 e = _mm256_loadu_ps(xe + bl * 8);
        const __m256 escalas = _mm256_mul_ps(_mm256_mul_ps(f16_en(b), f8(sc)), e);
        correccion = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_mul_ps(f16_en(b + 2), f8(mn)), e),
                                     _mm256_loadu_ps(xs + bl * 8), correccion);

        const __m256i qh = _mm256_loadu_si256((const __m256i *)(b + 16));
        const uint8_t *qs = b + 48;
        const int8_t *x = xq + bl * 256;

// Un sub-bloque: `q` son sus nibbles, `alto` su quinto bit ya en la posición 4.
#define SUB(acc, j, q, alto)                                                            \
    do {                                                                                \
        const __m256i v = _mm256_or_si256(q, _mm256_and_si256(alto, bit5));             \
        const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + (j) * 32));         \
        const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(v, xv), unos);       \
        const __m256 esc = _mm256_permutevar8x32_ps(escalas, _mm256_set1_epi32(j));     \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), esc, acc);                       \
    } while (0)

#define PAR(p, alto_par, alto_impar)                                                    \
    do {                                                                                \
        const __m256i bytes = _mm256_loadu_si256((const __m256i *)(qs + (p) * 32));      \
        SUB(acc0, 2 * (p), _mm256_and_si256(bytes, nibble), alto_par);                  \
        SUB(acc1, 2 * (p) + 1, _mm256_and_si256(_mm256_srli_epi16(bytes, 4), nibble),   \
            alto_impar);                                                                \
    } while (0)

        PAR(0, _mm256_slli_epi16(qh, 4), _mm256_slli_epi16(qh, 3));
        PAR(1, _mm256_slli_epi16(qh, 2), _mm256_slli_epi16(qh, 1));
        PAR(2, qh, _mm256_srli_epi16(qh, 1));
        PAR(3, _mm256_srli_epi16(qh, 2), _mm256_srli_epi16(qh, 3));
#undef PAR
#undef SUB
    }
    return reducir(_mm256_add_ps(acc0, acc1)) - reducir(correccion);
}
