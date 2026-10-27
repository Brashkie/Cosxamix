//! Raíz de los kernels de Zig para x86-64-v3 (AVX2 + FMA): los mismos, con el
//! sufijo `_v3`. build.rs la compila aparte como objeto y la mete en la misma
//! librería; Rust la usa solo si la CPU tiene lo que pide (src/kernels/zig.rs).

comptime {
    @import("exports.zig").exportar("_v3");
}
