//! Raíz de los kernels de Zig: los de `dot.zig` con sus nombres de siempre.
//! En x86-64 se compila en baseline; la copia para AVX2 es `lib_v3.zig`.

comptime {
    @import("exports.zig").exportar("");
}
