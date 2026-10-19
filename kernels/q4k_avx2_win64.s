// Bucle vectorial del kernel fusionado de Q4_K en AVX2 -- convencion de
// llamada de Windows x64.
//
// Este archivo y `q4k_avx2_sysv.s` son el MISMO kernel; solo cambian los
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
//   - lleva los `.def`/`.scl`/`.endef` de COFF, que en ELF y Mach-O no existen.
//
// Solo usa ymm0-ymm5, que son volatiles en las dos convenciones: no hace falta
// salvar xmm6-xmm15 en Windows ni prologo ni epilogo.
//
// Sintaxis AT&T, declarada con `options(att_syntax)` en el `global_asm!`.

    .text
    .globl cosxamix_q4k_pares
    .def cosxamix_q4k_pares
    .scl 2
    .type 32
    .endef
    .p2align 4
cosxamix_q4k_pares:
    movq    40(%rsp), %rax              // acc: el quinto argumento va por la pila
    vmovups (%rax), %ymm0               // el acumulador entra
    testq   %r9, %r9
    je      2f                          // pares == 0: acc queda como entro

    movl    $0x0F0F0F0F, %r10d
    vmovd   %r10d, %xmm1
    vpbroadcastd %xmm1, %ymm1           // mascara de nibble, 0x0F por byte
    movl    $0x00010001, %r10d
    vmovd   %r10d, %xmm2
    vpbroadcastd %xmm2, %ymm2           // unos en i16, para vpmaddwd
    xorl    %r10d, %r10d                // r10 = par*8: qs avanza x4, xq x8, esc x1

    .p2align 4
1:
    vmovdqu (%rcx,%r10,4), %ymm3        // 32 bytes de qs, dos sub-bloques
    vpand   %ymm1, %ymm3, %ymm4         // nibbles bajos: el sub-bloque par
    vpmaddubsw (%rdx,%r10,8), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3         // nibbles altos: el impar
    vpmaddubsw 32(%rdx,%r10,8), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4        // pares de i16 a i32
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss (%r8,%r10), %ymm5      // esc[par*2]
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5     // ymm5 = p0*esc + acc
    vbroadcastss 4(%r8,%r10), %ymm0     // esc[par*2 + 1]
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0     // acc = p1*esc + ymm5
    addq    $8, %r10
    decq    %r9
    jne     1b

2:
    vmovups %ymm0, (%rax)               // y sale
    vzeroupper
    retq

// Contrato por fila: el mismo bucle de pares dentro de uno de bloques, con el
// acumulador y las dos constantes armados UNA vez por fila y no por bloque.
//
//   void cosxamix_q4k_fila(const uint8_t *bloques, const int8_t *xq,
//                          const float *esc, size_t n_bloques, float *acc)
//
// `xq` y `esc` son contiguos de punta a punta, asi que r10 sigue siendo el
// indice global (par*8) y no se reinicia entre bloques. `qs` no: dentro de
// cada bloque empieza 16 bytes despues del inicio y de un bloque al siguiente
// hay 144, asi que rcx pasa a ser el puntero al `qs` del bloque actual y r11
// el desplazamiento dentro de el (0, 32, 64, 96). r11 es volatil en Windows
// x64, no hay que salvarlo.

    .globl cosxamix_q4k_fila
    .def cosxamix_q4k_fila
    .scl 2
    .type 32
    .endef
    .p2align 4
cosxamix_q4k_fila:
    movq    40(%rsp), %rax              // acc: el quinto argumento va por la pila
    vmovups (%rax), %ymm0               // el acumulador entra
    testq   %r9, %r9
    je      5f                          // n_bloques == 0: acc queda como entro

    movl    $0x0F0F0F0F, %r10d
    vmovd   %r10d, %xmm1
    vpbroadcastd %xmm1, %ymm1           // mascara de nibble, 0x0F por byte
    movl    $0x00010001, %r10d
    vmovd   %r10d, %xmm2
    vpbroadcastd %xmm2, %ymm2           // unos en i16, para vpmaddwd
    xorl    %r10d, %r10d                // r10 = par*8 global: xq x8, esc x1
    addq    $16, %rcx                   // rcx = qs del primer bloque

3:
    xorl    %r11d, %r11d                // r11 = par*32 dentro del bloque

    .p2align 4
4:
    vmovdqu (%rcx,%r11), %ymm3          // 32 bytes de qs, dos sub-bloques
    vpand   %ymm1, %ymm3, %ymm4         // nibbles bajos: el sub-bloque par
    vpmaddubsw (%rdx,%r10,8), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3         // nibbles altos: el impar
    vpmaddubsw 32(%rdx,%r10,8), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4        // pares de i16 a i32
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss (%r8,%r10), %ymm5      // esc[par*2]
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5     // ymm5 = p0*esc + acc
    vbroadcastss 4(%r8,%r10), %ymm0     // esc[par*2 + 1]
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0     // acc = p1*esc + ymm5
    addq    $8, %r10
    addq    $32, %r11
    cmpq    $128, %r11                  // 4 pares por bloque
    jne     4b

    addq    $144, %rcx                  // qs del bloque siguiente
    decq    %r9
    jne     3b

5:
    vmovups %ymm0, (%rax)               // y sale
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
// Windows x64 no deja tocar rbx, rsi, rdi ni xmm6-xmm15 sin declarar el
// prologo con directivas `.seh_*` para el desenrollado, asi que todo cabe en
// los siete volatiles -- rcx bloque, rdx xq, r8 xe, r9 xs, r10 contador y
// rax/r11 temporales -- y en ymm0-ymm5, sin prologo ni epilogo como las otras
// dos. De ahi:
//
//   - el puntero `acc` (sexto argumento, [rsp+48]) se relee de la pila cada
//     vez que hace falta, que es una vez por bloque;
//   - las 8 escalas del bloque se guardan en los 32 bytes del espacio de
//     sombra, 8(%rsp)..40(%rsp), que son del llamado; de ahi las lee
//     `vbroadcastss` como el bucle de siempre;
//   - el desplazamiento se acumula sobre acc[8..16] en memoria: no queda un
//     ymm libre sin tocar xmm6.

    .globl cosxamix_q4k_completo
    .def cosxamix_q4k_completo
    .scl 2
    .type 32
    .endef
    .p2align 4
cosxamix_q4k_completo:
    movq    48(%rsp), %rax              // acc, sexto argumento
    vmovups (%rax), %ymm0               // el acumulador de productos entra
    movq    40(%rsp), %r10              // n_bloques, quinto argumento
    testq   %r10, %r10
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
    movl    12(%rcx), %r11d
    andl    $0x0F0F0F0F, %r11d          // c & 0F
    movl    4(%rcx), %eax
    shrl    $2, %eax
    andl    $0x30303030, %eax           // 2 bits altos de a, ya en su lugar
    orl     %eax, %r11d                 // sc[4..8]
    movl    4(%rcx), %eax
    andl    $0x3F3F3F3F, %eax           // sc[0..4]
    shlq    $32, %r11
    orq     %r11, %rax                  // los 8 sc, un byte cada uno
    vmovq   %rax, %xmm3
    vpmovzxbd %xmm3, %ymm3
    vcvtdq2ps %ymm3, %ymm3
    vcvtph2ps (%rcx), %xmm5              // [d, dmin, basura, basura] en f32
    vbroadcastss %xmm5, %ymm4           // d
    vmulps  %ymm4, %ymm3, %ymm3         // d * sc
    vmulps  (%r8), %ymm3, %ymm3          // * xe
    vmovups %ymm3, 8(%rsp)              // las 8 escalas, para vbroadcastss

    // --- desplazamiento: acc[8..16] += dmin * mn[j] * xe[j] * xs[j] ---
    movl    12(%rcx), %r11d
    shrl    $4, %r11d
    andl    $0x0F0F0F0F, %r11d          // (c >> 4) & 0F
    movl    8(%rcx), %eax
    shrl    $2, %eax
    andl    $0x30303030, %eax           // 2 bits altos de b
    orl     %eax, %r11d                 // mn[4..8]
    movl    8(%rcx), %eax
    andl    $0x3F3F3F3F, %eax           // mn[0..4]
    shlq    $32, %r11
    orq     %r11, %rax                  // los 8 mn
    vmovq   %rax, %xmm3
    vpmovzxbd %xmm3, %ymm3
    vcvtdq2ps %ymm3, %ymm3
    vmovshdup %xmm5, %xmm4
    vbroadcastss %xmm4, %ymm4           // dmin
    vmulps  %ymm4, %ymm3, %ymm3         // dmin * mn
    vmulps  (%r8), %ymm3, %ymm3          // * xe
    movq    48(%rsp), %rax              // acc otra vez: no hay registro para tenerlo
    vmovups 32(%rax), %ymm4
    vfmadd231ps (%r9), %ymm3, %ymm4      // * xs + desplazamiento
    vmovups %ymm4, 32(%rax)

    // --- los 4 pares, como el bucle de siempre ---
    vmovdqu 16(%rcx), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 0(%rdx), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 32(%rdx), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss 8(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss 12(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 48(%rcx), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 64(%rdx), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 96(%rdx), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss 16(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss 20(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 80(%rcx), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 128(%rdx), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 160(%rdx), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss 24(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss 28(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    vmovdqu 112(%rcx), %ymm3
    vpand   %ymm1, %ymm3, %ymm4
    vpmaddubsw 192(%rdx), %ymm4, %ymm4
    vpsrlw  $4, %ymm3, %ymm3
    vpand   %ymm1, %ymm3, %ymm3
    vpmaddubsw 224(%rdx), %ymm3, %ymm3
    vpmaddwd %ymm2, %ymm4, %ymm4
    vpmaddwd %ymm2, %ymm3, %ymm3
    vbroadcastss 32(%rsp), %ymm5
    vcvtdq2ps %ymm4, %ymm4
    vfmadd213ps %ymm0, %ymm4, %ymm5
    vbroadcastss 36(%rsp), %ymm0
    vcvtdq2ps %ymm3, %ymm3
    vfmadd213ps %ymm5, %ymm3, %ymm0

    addq    $144, %rcx                  // siguiente bloque
    addq    $256, %rdx
    addq    $32, %r8
    addq    $32, %r9
    decq    %r10
    jne     6b

9:
    movq    48(%rsp), %rax
    vmovups %ymm0, (%rax)                // y sale
    vzeroupper
    retq
