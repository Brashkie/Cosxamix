// Clasificar bytes de a 32: letras ASCII, dígitos y bytes altos, en AVX2.
//
// El mismo algoritmo que `kernels/classify.zig` y que los `.s`. La
// especificación ejecutable es `escalar` en `src/kernels/classify.rs`.

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

__attribute__((target("avx2")))
void cosxamix_clasificar_c(const uint8_t *p, size_t bloques, uint32_t *letras,
                           uint32_t *digitos, uint32_t *altos) {
    const __m256i bit5 = _mm256_set1_epi8(0x20);
    const __m256i a = _mm256_set1_epi8('a');
    const __m256i veinticinco = _mm256_set1_epi8(25);
    const __m256i cero = _mm256_set1_epi8('0');
    const __m256i nueve = _mm256_set1_epi8(9);
    for (size_t i = 0; i < bloques; i++) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(p + i * 32));
        // Sin signo, `x <= k` es `min(x, k) == x`: AVX2 no compara sin signo.
        const __m256i l = _mm256_sub_epi8(_mm256_or_si256(v, bit5), a);
        letras[i] = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_min_epu8(l, veinticinco), l));
        const __m256i d = _mm256_sub_epi8(v, cero);
        digitos[i] = (uint32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(_mm256_min_epu8(d, nueve), d));
        altos[i] = (uint32_t)_mm256_movemask_epi8(v);
    }
}
