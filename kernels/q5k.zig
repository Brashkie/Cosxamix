//! El kernel fusionado de Q5_K en Zig, con `@Vector`: AVX2 en x86-64-v3, SSE2 en
//! baseline y NEON en ARM64.
//!
//! Escrito simple, como el de Q2_K: cada sub-bloque de 32 se multiplica a 16 bits
//! y se suma con `@reduce`. En ARM sin `sdot` es el único camino vectorizado.
//!
//! La especificación ejecutable es `scalar::dot_q5k_i8` en `src/kernels/scalar.rs`.

const V32b = @Vector(32, u8);
const V32w = @Vector(32, i16);
const V8f = @Vector(8, f32);

inline fn leer_f16(p: [*]const u8) f32 {
    const bits = @as(u16, p[0]) | (@as(u16, p[1]) << 8);
    const h: f16 = @bitCast(bits);
    return @floatCast(h);
}

inline fn leer32(p: [*]const u8) u32 {
    return @as(u32, p[0]) | (@as(u32, p[1]) << 8) | (@as(u32, p[2]) << 16) | (@as(u32, p[3]) << 24);
}

pub fn cosxamix_q5k_zig(
    bloques: [*]const u8,
    xq: [*]const i8,
    xe: [*]const f32,
    xs: [*]const f32,
    n_bloques: usize,
) callconv(.C) f32 {
    const nibble: V32b = @splat(0x0F);
    const uno: V32b = @splat(1);
    const cuatro_bits: @Vector(32, u3) = @splat(4);
    var acc: V8f = @splat(0.0);
    var correccion: V8f = @splat(0.0);

    var bl: usize = 0;
    while (bl < n_bloques) : (bl += 1) {
        const b = bloques + bl * 176;
        const d = leer_f16(b);
        const dmin = leer_f16(b + 2);

        // Las escalas y los mínimos de 6 bits, sin ramas (ver `escalas_q4k`).
        const a = leer32(b + 4);
        const m = leer32(b + 8);
        const c = leer32(b + 12);
        const seis: u32 = 0x3F3F3F3F;
        const cuatro: u32 = 0x0F0F0F0F;
        const dos: u32 = 0x03030303;
        const sc: u64 = @as(u64, a & seis) | (@as(u64, (c & cuatro) | (((a >> 6) & dos) << 4)) << 32);
        const mn: u64 = @as(u64, m & seis) | (@as(u64, ((c >> 4) & cuatro) | (((m >> 6) & dos) << 4)) << 32);

        const qh: V32b = (b + 16)[0..32].*;
        var enteros: V8f = undefined;
        var p: usize = 0;
        while (p < 4) : (p += 1) {
            const bytes: V32b = (b + 48 + p * 32)[0..32].*;
            var mitad: usize = 0;
            while (mitad < 2) : (mitad += 1) {
                const j = p * 2 + mitad;
                const q4 = if (mitad == 0) bytes & nibble else (bytes >> cuatro_bits) & nibble;
                const corr: @Vector(32, u3) = @splat(@intCast(j));
                const alto = ((qh >> corr) & uno) << cuatro_bits;
                const q: V32w = q4 | alto;
                const entrada: @Vector(32, i8) = (xq + bl * 256 + j * 32)[0..32].*;
                const xv: V32w = entrada;
                // Cada producto entra en i16 (31 * 128 = 3968); la suma, en i32.
                const prod: @Vector(32, i32) = q *% xv;
                enteros[j] = @floatFromInt(@reduce(.Add, prod));
            }
        }

        var ve: V8f = undefined;
        var vm: V8f = undefined;
        var i: usize = 0;
        while (i < 8) : (i += 1) {
            const s: u8 = @truncate(sc >> @intCast(i * 8));
            const n: u8 = @truncate(mn >> @intCast(i * 8));
            ve[i] = d * @as(f32, @floatFromInt(s));
            vm[i] = dmin * @as(f32, @floatFromInt(n));
        }
        const e: V8f = (xe + bl * 8)[0..8].*;
        const su: V8f = (xs + bl * 8)[0..8].*;
        acc += e * ve * enteros;
        correccion += e * vm * su;
    }
    return @reduce(.Add, acc) - @reduce(.Add, correccion);
}
