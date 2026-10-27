// Los kernels fusionados de Q4_K y Q3_K para ARM64, con `sdot`.
//
// Por qué en C: `sdot` (`vdotq_s32`) hace el producto punto de cuatro `i8` en
// un `i32`, dieciséis productos por instrucción, y es la forma de que ARM haga
// lo que AVX2 hace con `vpmaddubsw`. Rust estable no la expone —sigue detrás de
// `stdarch_neon_dotprod`— y Zig no la genera: su kernel para un Cortex-A76 sale
// con `smull` y ningún `sdot` (DECISIONES §5). En C es `arm_neon.h`, estable,
// y `zig cc` la emite tal cual. Era la parte del andamio de
// `src/kernels/q4k_neon.rs` que faltaba escribir en ensamblador.
//
// `sdot` no está en todos los ARM: no la tienen el Cortex-A53 ni el A72 de la
// Raspberry Pi 4. Por eso va con `__attribute__((target("dotprod")))` por
// función, el objeto se compila para ARMv8.0 básico, y quien llama comprueba
// `is_aarch64_feature_detected!("dotprod")` antes (src/kernels/c.rs).
//
// La especificación ejecutable son `scalar::dot_q4k_i8` y
// `scalar::dot_q3k_i8`, y los tests comparan contra ellas. Esta máquina de
// desarrollo es x86-64: los ejecuta el runner ARM de CI.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#define DOTPROD __attribute__((target("dotprod")))

static inline float f16(const uint8_t *p) {
    __fp16 h;
    __builtin_memcpy(&h, p, 2);
    return (float)h;
}

// ─── Q4_K ────────────────────────────────────────────────────────────────────
//
// 144 bytes por bloque: `d` y `dmin` en f16, las 8 escalas y los 8 mínimos de
// 6 bits en 12 bytes, y 128 bytes de nibbles: 4 pares de 32 bytes, los bajos el
// sub-bloque par y los altos el impar.
//
// Los nibbles van de 0 a 15 y `sdot` multiplica con signo: entran como `i8`
// positivos sin ningún rodeo.

// `escala_y_minimo` de src/quant.rs: los cuatro primeros directos, los otros
// cuatro con dos bits del byte de cuatro posiciones antes.
static inline void escala_y_minimo(int j, const uint8_t *q, uint8_t *sc, uint8_t *mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

DOTPROD float cosxamix_q4k_neon(const uint8_t *bloques, const int8_t *xq, const float *xe,
                                const float *xs, size_t n_bloques) {
    const uint8x16_t nibble = vdupq_n_u8(0x0F);
    // Dos acumuladores: las FMA de un bloque no esperan cada una a la anterior.
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float desplazamiento = 0.0f;

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 144;
        const float d = f16(b);
        const float dmin = f16(b + 2);
        const uint8_t *esc = b + 4;
        const uint8_t *qs = b + 16;
        const int8_t *x = xq + bl * 256;

        for (int par = 0; par < 4; par++) {
            const uint8x16_t q0 = vld1q_u8(qs + par * 32);
            const uint8x16_t q1 = vld1q_u8(qs + par * 32 + 16);
            const int8_t *xp = x + par * 64;

            // Sub-bloque par: los nibbles bajos contra los primeros 32.
            int32x4_t p = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(q0, nibble)),
                                    vld1q_s8(xp));
            p = vdotq_s32(p, vreinterpretq_s8_u8(vandq_u8(q1, nibble)), vld1q_s8(xp + 16));
            // Sub-bloque impar: los altos contra los 32 siguientes.
            int32x4_t r = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)),
                                    vld1q_s8(xp + 32));
            r = vdotq_s32(r, vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)), vld1q_s8(xp + 48));

            for (int mitad = 0; mitad < 2; mitad++) {
                const int j = par * 2 + mitad;
                uint8_t sc, mn;
                escala_y_minimo(j, esc, &sc, &mn);
                const size_t sub = bl * 8 + j;
                const float escala = d * (float)sc * xe[sub];
                if (mitad == 0) {
                    acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p), escala);
                } else {
                    acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(r), escala);
                }
                desplazamiento += xe[sub] * dmin * (float)mn * xs[sub];
            }
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) - desplazamiento;
}

// ─── Q3_K ────────────────────────────────────────────────────────────────────
//
// 110 bytes por bloque: 32 de `hmask` (el tercer bit), 64 de `qs` (los dos
// bits bajos, cuatro valores por byte), 12 de escalas de 6 bits y `d` en f16.
// Sub-bloques de 16.
//
// En x86 los valores, de −4 a 3, se suben a 0-7 porque `vpmaddubsw` quiere un
// operando sin signo, y el 4 se descuenta después con la suma de la entrada.
// `sdot` multiplica con signo los dos: acá se resta el 4 en el byte y no hace
// falta ninguna corrección.

// `escalas_q3k` de src/quant.rs.
static inline void escalas_q3k(const uint8_t *b, int8_t *esc) {
    uint32_t aux[4];
    __builtin_memcpy(aux, b + 96, 12);
    const uint32_t m1 = 0x03030303u, m2 = 0x0F0F0F0Fu;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
    aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
    aux[0] = (aux[0] & m2) | ((tmp & m1) << 4);
    aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
    __builtin_memcpy(esc, aux, 16);
}

DOTPROD float cosxamix_q3k_neon(const uint8_t *bloques, const int8_t *xq, const float *xe,
                                const float *xs, size_t n_bloques) {
    (void)xs; // con `sdot` no hace falta: ver arriba
    const uint8x16_t tres = vdupq_n_u8(3);
    const uint8x16_t uno = vdupq_n_u8(1);
    const int8x16_t cuatro = vdupq_n_s8(4);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    for (size_t bl = 0; bl < n_bloques; bl++) {
        const uint8_t *b = bloques + bl * 110;
        const float d = f16(b + 108);
        int8_t esc[16];
        escalas_q3k(b, esc);
        const uint8x16_t hm0 = vld1q_u8(b);
        const uint8x16_t hm1 = vld1q_u8(b + 16);
        const int8_t *x = xq + bl * 256;
        const float *e = xe + bl * 16;

        for (int g = 0; g < 8; g++) {
            const uint8_t *qsg = b + 32 + (g / 4) * 32;
            // Corrimientos negativos de `vshlq_u8` corren a la derecha. Con el
            // bucle de largo fijo, el compilador los deja inmediatos.
            const int8x16_t bajo = vdupq_n_s8((int8_t)(-2 * (g % 4)));
            const int8x16_t alto = vdupq_n_s8((int8_t)(-g));

            // valor = (dos bits bajos | tercer bit << 2) - 4, de -4 a 3.
            uint8x16_t l0 = vandq_u8(vshlq_u8(vld1q_u8(qsg), bajo), tres);
            uint8x16_t l1 = vandq_u8(vshlq_u8(vld1q_u8(qsg + 16), bajo), tres);
            uint8x16_t h0 = vshlq_n_u8(vandq_u8(vshlq_u8(hm0, alto), uno), 2);
            uint8x16_t h1 = vshlq_n_u8(vandq_u8(vshlq_u8(hm1, alto), uno), 2);
            int8x16_t v0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l0, h0)), cuatro);
            int8x16_t v1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(l1, h1)), cuatro);

            int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(x + g * 32));
            int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(x + g * 32 + 16));

            acc0 = vfmaq_n_f32(acc0, vcvtq_f32_s32(p0), d * (float)(esc[2 * g] - 32) * e[2 * g]);
            acc1 = vfmaq_n_f32(acc1, vcvtq_f32_s32(p1),
                               d * (float)(esc[2 * g + 1] - 32) * e[2 * g + 1]);
        }
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
}
