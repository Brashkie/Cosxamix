// SPDX-License-Identifier: GPL-3.0-or-later
//
// La tesela del prefill por grupos de 128 valores, en C con intrínsecos de
// AVX2.
//
// El mismo algoritmo que `tesela_u8_grupos_avx2` de `src/kernels/x86.rs`, con
// la forma fija de 3 filas × 2 columnas que usa `ops::matmul_i8_bloques`, para
// que lo que se mida sea el lenguaje y su compilador y no una idea distinta:
//
//   - por cada par fila × columna y cada trozo de 32 valores, `maddubs` y
//     `madd` hasta enteros de 32 bits;
//   - los cuatro trozos del grupo se suman en enteros con dos niveles de
//     `hadd`, que deja en el carril j la suma de un sub-bloque de 16;
//   - una conversión a float, un producto de escalas y una FMA por grupo.
//
// Lo compila `zig cc` desde build.rs con la feature `c`, sin libc. Los tests
// de `src/kernels/c.rs` lo comparan con el de Rust.

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#define AVX2 __attribute__((target("avx2,fma")))

AVX2 static inline float reducir(__m256 v) {
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    return _mm_cvtss_f32(s1);
}

// filas: 3 punteros a `128 * grupos` valores en -127..=127;
// cols: 2 punteros a `128 * grupos` valores en 0..=129;
// esc_filas, esc_cols: 8 escalas por grupo, en el orden de los carriles;
// salida: 3 × 2, fila por fila.
AVX2 void cosxamix_tesela_grupos_3x2_c(const int8_t *const filas[3],
                                       const float *const esc_filas[3],
                                       const uint8_t *const cols[2],
                                       const float *const esc_cols[2],
                                       size_t grupos, float salida[6]) {
    const __m256i unos = _mm256_set1_epi16(1);
    __m256 acc[3][2];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            acc[r][c] = _mm256_setzero_ps();

    for (size_t g = 0; g < grupos; g++) {
        const size_t base = 128 * g;
        __m256i mitad[2][3][2];
        for (int m = 0; m < 2; m++) {
            const size_t q = base + 64 * (size_t)m;
            __m256i c0[2], c1[2];
            for (int c = 0; c < 2; c++) {
                c0[c] = _mm256_loadu_si256((const __m256i *)(cols[c] + q));
                c1[c] = _mm256_loadu_si256((const __m256i *)(cols[c] + q + 32));
            }
            for (int r = 0; r < 3; r++) {
                const __m256i x0 = _mm256_loadu_si256((const __m256i *)(filas[r] + q));
                const __m256i x1 = _mm256_loadu_si256((const __m256i *)(filas[r] + q + 32));
                for (int c = 0; c < 2; c++) {
                    const __m256i a = _mm256_madd_epi16(_mm256_maddubs_epi16(c0[c], x0), unos);
                    const __m256i b = _mm256_madd_epi16(_mm256_maddubs_epi16(c1[c], x1), unos);
                    mitad[m][r][c] = _mm256_hadd_epi32(a, b);
                }
            }
        }
        for (int c = 0; c < 2; c++) {
            const __m256 ec = _mm256_loadu_ps(esc_cols[c] + 8 * g);
            for (int r = 0; r < 3; r++) {
                const __m256 ef = _mm256_loadu_ps(esc_filas[r] + 8 * g);
                const __m256i h = _mm256_hadd_epi32(mitad[0][r][c], mitad[1][r][c]);
                acc[r][c] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h), _mm256_mul_ps(ef, ec), acc[r][c]);
            }
        }
    }
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 2; c++)
            salida[r * 2 + c] = reducir(acc[r][c]);
}
