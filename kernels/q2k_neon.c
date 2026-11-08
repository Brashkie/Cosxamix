// El kernel fusionado de Q2_K para ARM64, con `sdot`.
//
// Por qué en C, lo mismo que `neon_dotprod.c`: Rust estable no expone `sdot` y Zig
// no la genera. Con `target("dotprod")` por función, y quien llama comprueba que
// la CPU la tenga. La especificación ejecutable es `escalar` en
// `src/kernels/q2k.rs`, y los tests comparan contra ella en el runner ARM de CI.
//
// Los valores van de 0 a 3 y `sdot` multiplica con signo: entran como `i8`
// positivos sin ningún rodeo.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#define DOTPROD __attribute__((target("dotprod")))

static inline float f16_en(const uint8_t *p) {
    __fp16 h;
    __builtin_memcpy(&h, p, 2);
    return (float)h;
}

DOTPROD float cosxamix_q2k_neon(const uint8_t *bloques, const int8_t *xq, const float *xe,
                                const float *xs, size_t n_bloques) {
    const uint8x16_t tres = vdupq_n_u8(3);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float correccion = 0.0f;

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 84;
        const float d = f16_en(b + 80);
        const float dm = f16_en(b + 82);
        const int8_t *x = xq + bl * 256;
        const float *e = xe + bl * 16;
        const float *s = xs + bl * 16;

        for (int g = 0; g < 8; g++) {
            const uint8_t *qsg = b + 16 + (g / 4) * 32;
            // Corrimientos negativos de `vshlq_u8` corren a la derecha.
            const int8x16_t corr = vdupq_n_s8((int8_t)(-2 * (g % 4)));
            const uint8x16_t q0 = vandq_u8(vshlq_u8(vld1q_u8(qsg), corr), tres);
            const uint8x16_t q1 = vandq_u8(vshlq_u8(vld1q_u8(qsg + 16), corr), tres);
            const int32x4_t p0 =
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q0), vld1q_s8(x + g * 32));
            const int32x4_t p1 =
                vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q1), vld1q_s8(x + g * 32 + 16));

            const uint8_t sc0 = b[2 * g];
            const uint8_t sc1 = b[2 * g + 1];
            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), d * (float)(sc0 & 0xF) * e[2 * g]);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1), d * (float)(sc1 & 0xF) * e[2 * g + 1]);
            correccion += dm * (float)(sc0 >> 4) * e[2 * g] * s[2 * g];
            correccion += dm * (float)(sc1 >> 4) * e[2 * g + 1] * s[2 * g + 1];
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) - correccion;
}
