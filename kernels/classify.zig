//! Clasificar bytes de a 32: letras ASCII, dígitos y bytes altos.
//!
//! Es lo que pre-tokenizar mira un byte a la vez, y acá salen tres máscaras de
//! 32 bits por bloque. Con `@Vector` el mismo código da AVX2 en x86-64-v3, dos
//! registros de SSE2 en baseline y NEON en ARM64.
//!
//! La especificación ejecutable es `escalar` en `src/kernels/classify.rs`.

const V = @Vector(32, u8);

inline fn mascara(v: @Vector(32, bool)) u32 {
    return @bitCast(v);
}

pub fn cosxamix_clasificar(
    p: [*]const u8,
    bloques: usize,
    letras: [*]u32,
    digitos: [*]u32,
    altos: [*]u32,
) callconv(.C) void {
    const bit5: V = @splat(0x20);
    const a: V = @splat('a');
    const cero: V = @splat('0');
    const veinticinco: V = @splat(25);
    const nueve: V = @splat(9);
    const alto: V = @splat(0x80);
    var i: usize = 0;
    while (i < bloques) : (i += 1) {
        const v: V = (p + i * 32)[0..32].*;
        letras[i] = mascara(((v | bit5) -% a) <= veinticinco);
        digitos[i] = mascara((v -% cero) <= nueve);
        altos[i] = mascara(v >= alto);
    }
}
