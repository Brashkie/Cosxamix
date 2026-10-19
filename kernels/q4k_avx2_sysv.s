// Bucle vectorial del kernel fusionado de Q4_K en AVX2 -- convencion de
// llamada System V (Linux y macOS).
//
// Este archivo y `q4k_avx2_win64.s` son el MISMO kernel; solo cambian los
// registros en los que llegan los argumentos:
//
//   Windows x64:  rcx  rdx  r8   r9   [rsp+40]
//   System V:     rdi  rsi  rdx  rcx  r8
//
// Firma, igual en los dos (el contrato completo esta en src/kernels/q4k_asm.rs):
//   void cosxamix_q4k_pares(const uint8_t *qs, const int8_t *xq,
//                           const float *esc, size_t pares, float *acc)
//
// Por cada par: 32 bytes de `qs` con dos sub-bloques (nibbles bajos el par,
// altos el impar), cada uno contra sus 32 valores de `xq` y acumulado
// escalado por su `esc`. `acc` entra y sale con los 8 f32 del acumulador.
//
// Generado con `rustc --emit asm -C opt-level=3 -C target-feature=+avx2,+fma`
// a partir del bucle de `dot_q4k_i8_avx2` en src/kernels/x86.rs, y limpiado a
// mano para que no quede nada que dependa del formato de objeto:
//
//   - las dos constantes (0x0F y 1) se arman con inmediatos en vez de leerse
//     de una seccion de datos, que se declara distinto en cada formato
//     (`.rdata` con COMDAT en COFF, `.rodata` en ELF, `__TEXT,__const` en
//     Mach-O);
//   - etiquetas numericas locales (`1:`, `1b`) y no las `.LBB0_2` del
//     compilador: dentro de `global_asm!` comparten objeto con el codigo que
//     genera rustc, y una etiqueta con nombre puede chocar;
//   - fuera `@feat.00`, `.file`, `_fltused`, `.cfi_*`, `.size` e `.ident`.
//
// Un solo archivo sirve para ELF y Mach-O, como `dot_avx2_sysv.s`: no lleva
// `.type sym, @function` (de ELF; Mach-O la rechaza, y eso rompio un release
// de macOS una vez) y el simbolo se define con y sin guion bajo, porque en
// Mach-O una funcion de C se llama `_foo` a nivel de enlazador y en ELF `foo`.
//
// Solo usa ymm0-ymm5, que son volatiles en las dos convenciones: no hace falta
// salvar xmm6-xmm15 en Windows ni prologo ni epilogo.
//
// Sintaxis AT&T, declarada con `options(att_syntax)` en el `global_asm!`.

    .text
    .globl cosxamix_q4k_pares
    .globl _cosxamix_q4k_pares
    .p2align 4
cosxamix_q4k_pares:
_cosxamix_q4k_pares:
    vmovups (%r8), %ymm0                // el acumulador entra
    testq   %rcx, %rcx
    je      2f                          // pares == 0: acc queda como entro

    movl    $0x0F0F0F0F, %eax
    vmovd   %eax, %xmm1
    vpbroadcastd %xmm1, %ymm1           // mascara de nibble, 0x0F por byte
    movl    $0x00010001, %eax
    vmovd   %eax, %xmm2
    vpbroadcastd %xmm2, %ymm2           // unos en i16, para vpmaddwd
    xorl    %eax, %eax                  // rax = par*8: qs avanza x4, xq x8, esc x1

    .p2align 4
1:
    vmovdqu (%rdi,%rax,4), %ymm3        // 32 bytes de qs, dos sub-bloques
    vpand   %ymm1, %ymm3, %ymm4         // nibbles bajos: el sub-bloque par
    vpmaddubsw (%rsi,%rax,8), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3         // nibbles altos: el impar
    vpmaddubsw 32(%rsi,%rax,8), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4        // pares de i16 a i32
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss (%rdx,%rax), %ymm5     // esc[par*2]
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5     // ymm5 = p0*esc + acc
    vbroadcastss 4(%rdx,%rax), %ymm0    // esc[par*2 + 1]
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0     // acc = p1*esc + ymm5
    addq    $8, %rax
    decq    %rcx
    jne     1b

2:
    vmovups %ymm0, (%r8)                // y sale
    vzeroupper
    retq

// Contrato por fila: el mismo bucle de pares dentro de uno de bloques, con el
// acumulador y las dos constantes armados UNA vez por fila y no por bloque.
//
//   void cosxamix_q4k_fila(const uint8_t *bloques, const int8_t *xq,
//                          const float *esc, size_t n_bloques, float *acc)
//
// `xq` y `esc` son contiguos de punta a punta, asi que rax sigue siendo el
// indice global (par*8) y no se reinicia entre bloques. `qs` no: dentro de
// cada bloque empieza 16 bytes despues del inicio y de un bloque al siguiente
// hay 144, asi que rdi pasa a ser el puntero al `qs` del bloque actual y r11
// el desplazamiento dentro de el (0, 32, 64, 96). r11 es volatil en System V.

    .globl cosxamix_q4k_fila
    .globl _cosxamix_q4k_fila
    .p2align 4
cosxamix_q4k_fila:
_cosxamix_q4k_fila:
    vmovups (%r8), %ymm0                // el acumulador entra
    testq   %rcx, %rcx
    je      5f                          // n_bloques == 0: acc queda como entro

    movl    $0x0F0F0F0F, %eax
    vmovd   %eax, %xmm1
    vpbroadcastd %xmm1, %ymm1           // mascara de nibble, 0x0F por byte
    movl    $0x00010001, %eax
    vmovd   %eax, %xmm2
    vpbroadcastd %xmm2, %ymm2           // unos en i16, para vpmaddwd
    xorl    %eax, %eax                  // rax = par*8 global: xq x8, esc x1
    addq    $16, %rdi                   // rdi = qs del primer bloque

3:
    xorl    %r11d, %r11d                // r11 = par*32 dentro del bloque

    .p2align 4
4:
    vmovdqu (%rdi,%r11), %ymm3          // 32 bytes de qs, dos sub-bloques
    vpand   %ymm1, %ymm3, %ymm4         // nibbles bajos: el sub-bloque par
    vpmaddubsw (%rsi,%rax,8), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3         // nibbles altos: el impar
    vpmaddubsw 32(%rsi,%rax,8), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4        // pares de i16 a i32
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss (%rdx,%rax), %ymm5     // esc[par*2]
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5     // ymm5 = p0*esc + acc
    vbroadcastss 4(%rdx,%rax), %ymm0    // esc[par*2 + 1]
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0     // acc = p1*esc + ymm5
    addq    $8, %rax
    addq    $32, %r11
    cmpq    $128, %r11                  // 4 pares por bloque
    jne     4b

    addq    $144, %rdi                  // qs del bloque siguiente
    decq    %rcx
    jne     3b

5:
    vmovups %ymm0, (%r8)                // y sale
    vzeroupper
    retq

// Contrato completo: escalas, desplazamiento y bucle en una sola pasada.
//
//   void cosxamix_q4k_completo(const uint8_t *bloques, const int8_t *xq,
//                              const float *xe, const float *xs,
//                              size_t n_bloques, float *acc)
//
// Por bloque, lo mismo que `escalas_q4k` y el bucle de `dot_q4k_i8_avx2` en
// src/kernels/x86.rs, que son la especificacion legible:
//
//   - `vcvtph2ps` convierte d y dmin, bytes 0-3, en una instruccion;
//   - las 8 escalas y los 8 minimos de 6 bits salen de los bytes 4-15 con el
//     desempaquetado sin ramas: con a, b, c las palabras de 4, 8 y 12,
//       sc = (a & 3F) | ((c & 0F) | ((a >> 2) & 30)) << 32
//       mn = (b & 3F) | (((c >> 4) & 0F) | ((b >> 2) & 30)) << 32
//     donde `(a >> 2) & 30` es `((a >> 6) & 03) << 4` en una mascara menos.
//     Las tres palabras se releen del bloque en vez de tenerlas en registros:
//     estan en L1, en la misma linea, y asi alcanza con dos temporales;
//   - escala[j] = d * sc[j] * xe[j], en el mismo orden que los intrinsecos;
//   - acc[8..16] += dmin * mn[j] * xe[j] * xs[j], en positivo; Rust lo resta.
//
// Los 4 pares van desenrollados: con desplazamientos fijos el bucle interior
// no necesita contador.
//
// En System V sobran registros: rdi bloque, rsi xq, rdx xe, rcx xs, r8
// contador, r9 acc, y rax/r11 temporales. Las 8 escalas del bloque van en la
// zona roja, -32(%rsp), los 128 bytes bajo rsp que una funcion hoja puede
// usar sin moverlo, en Linux y en macOS. El resto es instruccion por
// instruccion el de Windows, incluido acumular el desplazamiento sobre
// acc[8..16] en memoria.

    .globl cosxamix_q4k_completo
    .globl _cosxamix_q4k_completo
    .p2align 4
cosxamix_q4k_completo:
_cosxamix_q4k_completo:
    vmovups (%r9), %ymm0                // el acumulador de productos entra
    testq   %r8, %r8
    je      9f                          // n_bloques == 0: acc queda como entro

    movl    $0x0F0F0F0F, %eax
    vmovd   %eax, %xmm1
    vpbroadcastd %xmm1, %ymm1           // mascara de nibble, 0x0F por byte
    movl    $0x00010001, %eax
    vmovd   %eax, %xmm2
    vpbroadcastd %xmm2, %ymm2           // unos en i16, para vpmaddwd

    .p2align 4
6:
    // --- escalas: d * sc[j] * xe[j] ---
    movl    12(%rdi), %r11d
    andl    $0x0F0F0F0F, %r11d          // c & 0F
    movl    4(%rdi), %eax
    shrl    $2, %eax
    andl    $0x30303030, %eax           // 2 bits altos de a, ya en su lugar
    orl     %eax, %r11d                 // sc[4..8]
    movl    4(%rdi), %eax
    andl    $0x3F3F3F3F, %eax           // sc[0..4]
    shlq    $32, %r11
    orq     %r11, %rax                  // los 8 sc, un byte cada uno
    vmovq   %rax, %xmm3
    vpmovzxbd %xmm3, %ymm3
    vcvtdq2ps %ymm3, %ymm3
    vcvtph2ps (%rdi), %xmm5              // [d, dmin, basura, basura] en f32
    vbroadcastss %xmm5, %ymm4           // d
    vmulps  %ymm4, %ymm3, %ymm3         // d * sc
    vmulps  (%rdx), %ymm3, %ymm3          // * xe
    vmovups %ymm3, -32(%rsp)              // las 8 escalas, para vbroadcastss

    // --- desplazamiento: acc[8..16] += dmin * mn[j] * xe[j] * xs[j] ---
    movl    12(%rdi), %r11d
    shrl    $4, %r11d
    andl    $0x0F0F0F0F, %r11d          // (c >> 4) & 0F
    movl    8(%rdi), %eax
    shrl    $2, %eax
    andl    $0x30303030, %eax           // 2 bits altos de b
    orl     %eax, %r11d                 // mn[4..8]
    movl    8(%rdi), %eax
    andl    $0x3F3F3F3F, %eax           // mn[0..4]
    shlq    $32, %r11
    orq     %r11, %rax                  // los 8 mn
    vmovq   %rax, %xmm3
    vpmovzxbd %xmm3, %ymm3
    vcvtdq2ps %ymm3, %ymm3
    vmovshdup %xmm5, %xmm4
    vbroadcastss %xmm4, %ymm4           // dmin
    vmulps  %ymm4, %ymm3, %ymm3         // dmin * mn
    vmulps  (%rdx), %ymm3, %ymm3          // * xe
    vmovups 32(%r9), %ymm4
    vfmadd231ps (%rcx), %ymm3, %ymm4      // * xs + desplazamiento
    vmovups %ymm4, 32(%r9)

    // --- los 4 pares, como el bucle de siempre ---
    vmovdqu 16(%rdi), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 0(%rsi), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 32(%rsi), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss -32(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss -28(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 48(%rdi), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 64(%rsi), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 96(%rsi), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss -24(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss -20(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 80(%rdi), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 128(%rsi), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 160(%rsi), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss -16(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss -12(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 112(%rdi), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 192(%rsi), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 224(%rsi), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss -8(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss -4(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    addq    $144, %rdi                  // siguiente bloque
    addq    $256, %rsi
    addq    $32, %rdx
    addq    $32, %rcx
    decq    %r8
    jne     6b

9:
    vmovups %ymm0, (%r9)                // y sale
    vzeroupper
    retq
