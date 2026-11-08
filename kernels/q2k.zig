//! El kernel fusionado de Q2_K en Zig, con `@Vector`: AVX2 en x86-64-v3, SSE2 en
//! baseline y NEON en ARM64, también en los ARM sin `sdot`.
//!
//! Suma cada sub-bloque con `@reduce`, a 16 bits. Rinde 1,14× sobre el camino con
//! buffer en x86, muy lejos del 6× del AVX2 escrito con intrínsecos. Se probaron
//! otras dos formas, y salieron peor: reducir de a pares con `@shuffle` hasta las 8
//! sumas de un grupo (1,08×), y escribir la forma de `vpmaddwd` para que el
//! compilador la reconociera (0,75×). Lo que falta es medirlo en un ARM sin `sdot`,
//! que es donde se usa: en x86 no se elige nunca (DECISIONES §30).
//!
//! La especificación ejecutable es `escalar` en `src/kernels/q2k.rs`.

const V16b = @Vector(16, u8);
const V16w = @Vector(16, i16);
const V8f = @Vector(8, f32);

inline fn leer_f16(p: [*]const u8) f32 {
    const bits = @as(u16, p[0]) | (@as(u16, p[1]) << 8);
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

pub fn cosxamix_q2k_zig(
    bloques: [*]const u8,
    xq: [*]const i8,
    xe: [*]const f32,
    xs: [*]const f32,
    n_bloques: usize,
) callconv(.C) f32 {
    var enteros: [16]i32 = undefined;
    const dos_bits: V16b = @splat(3);
    var acc: V8f = @splat(0.0);
    var correccion: V8f = @splat(0.0);

    var bl: usize = 0;
    while (bl < n_bloques) : (bl += 1) {
        const b = bloques + bl * 84;
        const d = leer_f16(b + 80);
        const dm = leer_f16(b + 82);
        const x = xq + bl * 256;

        var s: usize = 0;
        while (s < 16) : (s += 1) {
            const g = s / 2;
            const medio = s % 2;
            const qs: [*]const u8 = b + 16 + (g / 4) * 32 + medio * 16;
            const corr: @Vector(16, u3) = @splat(@intCast(2 * (g % 4)));
            const crudo: V16b = qs[0..16].*;
            const q: V16w = (crudo >> corr) & dos_bits;
            const entrada: @Vector(16, i8) = (x + g * 32 + medio * 16)[0..16].*;
            const xv: V16w = entrada;
            // Cada producto entra en i16: 3 * 128 = 384, y los 16 suman ±6144.
            enteros[s] = @reduce(.Add, q *% xv);
        }

        // Los términos de cada sub-bloque, de a 8.
        var t: usize = 0;
        while (t < 16) : (t += 8) {
            var ve: V8f = undefined;
            var vm: V8f = undefined;
            var ent: V8f = undefined;
            var i: usize = 0;
            while (i < 8) : (i += 1) {
                const sc = b[t + i];
                ve[i] = d * @as(f32, @floatFromInt(sc & 0xF));
                vm[i] = dm * @as(f32, @floatFromInt(sc >> 4));
                ent[i] = @floatFromInt(enteros[t + i]);
            }
            const e: V8f = (xe + bl * 16 + t)[0..8].*;
            const su: V8f = (xs + bl * 16 + t)[0..8].*;
            acc += e * ve * ent;
            correccion += e * vm * su;
        }
    }
    return @reduce(.Add, acc) - @reduce(.Add, correccion);
}
