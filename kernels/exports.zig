//! Los símbolos que ve Rust, con un sufijo.
//!
//! En x86-64 los kernels se compilan dos veces: `lib.zig` en baseline (SSE2,
//! corre en cualquier PC de 64 bits) sin sufijo, y `lib_v3.zig` para
//! x86-64-v3 (AVX2 + FMA) con `_v3`. Rust elige al arrancar según la CPU
//! (`src/kernels/zig.rs`). Es el mismo código fuente: lo que cambia son las
//! instrucciones que el compilador puede usar. En SSE2 un `@Vector(8, f32)`
//! son dos registros de 4.
//!
//! Antes había una sola copia, para x86-64-v3: en una PC sin AVX2 —un Celeron,
//! un Pentium, un Atom, justo el hardware débil al que apunta Cosxamix— el
//! binario con la feature `zig` moría con "instrucción ilegal" en el primer
//! kernel.
//!
//! En ARM hay una sola copia (baseline): NEON es parte de la base de ARMv8-A.

const dot = @import("dot.zig");

pub fn exportar(comptime sufijo: []const u8) void {
    @export(&dot.cosxamix_dot_f32, .{ .name = "cosxamix_dot_f32" ++ sufijo });
    @export(&dot.cosxamix_rmsnorm_f32, .{ .name = "cosxamix_rmsnorm_f32" ++ sufijo });
    @export(&dot.cosxamix_noop, .{ .name = "cosxamix_noop" ++ sufijo });
    @export(&dot.cosxamix_silu_f32, .{ .name = "cosxamix_silu_f32" ++ sufijo });
    @export(&dot.cosxamix_dot_i8_escalado_f32, .{ .name = "cosxamix_dot_i8_escalado_f32" ++ sufijo });
    @export(&dot.cosxamix_dot_q4k_i8_f32, .{ .name = "cosxamix_dot_q4k_i8_f32" ++ sufijo });
}
