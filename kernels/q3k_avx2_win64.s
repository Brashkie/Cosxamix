// El kernel fusionado de Q3_K en AVX2 -- convencion de llamada de Windows x64.
//
// Este archivo y `q3k_avx2_sysv.s` son el MISMO kernel; cambian los
// registros de los argumentos, y en Windows el guardado de xmm6-xmm15:
//
//   Windows x64:  rcx  rdx  r8   r9   [rsp+40]
//   System V:     rdi  rsi  rdx  rcx  r8
//
// Firma (el contrato esta en src/kernels/q3k_asm.rs):
//   float cosxamix_q3k_completo(const uint8_t *bloques,  // 110 bytes por bloque
//                               const int8_t  *xq,       // 256 por bloque
//                               const float   *xe,       // 16 por bloque
//                               const float   *xs,       // 16 por bloque
//                               size_t         n_bloques);
//
// Devuelve el producto en xmm0, ya reducido: a diferencia del de Q4_K, no hay
// acumulador que entre y salga, porque la fila se recorre en una llamada.
//
// El algoritmo es el de `dot_q3k_i8_avx2` en src/kernels/x86.rs, el mismo que
// el de kernels/q3k_avx2.c: las escalas en vector, en registros y no en la
// pila, el tercer bit con un corrimiento inmediato y un `and`, y dos
// acumuladores. Lo propio del ensamblador es el reparto de registros: las
// siete constantes del bucle viven en ymm9-ymm15 todo el tiempo, y las tres
// que solo usan las escalas van en la pila y entran como operando de memoria.
//
// Escrito a mano siguiendo las reglas de q4k_avx2_*.s: constantes armadas con
// inmediatos y no en una seccion de datos, etiquetas numericas locales, nada de
// `.cfi_*` ni `.file`. Sintaxis AT&T.

    .text
    .globl cosxamix_q3k_completo
    .def cosxamix_q3k_completo
    .scl 2
    .type 32
    .endef
    .p2align 4
cosxamix_q3k_completo:
    // Marco: xmm6-xmm15 son del llamador en Windows (sus 128 bits bajos)
    // y aca se usan los dieciseis ymm, asi que se guardan. 216 bytes:
    // 160 de registros y 48 de constantes; con los 8 del retorno, la
    // pila queda alineada a 16.
    subq    $216, %rsp
    vmovaps %xmm6, 0(%rsp)
    vmovaps %xmm7, 16(%rsp)
    vmovaps %xmm8, 32(%rsp)
    vmovaps %xmm9, 48(%rsp)
    vmovaps %xmm10, 64(%rsp)
    vmovaps %xmm11, 80(%rsp)
    vmovaps %xmm12, 96(%rsp)
    vmovaps %xmm13, 112(%rsp)
    vmovaps %xmm14, 128(%rsp)
    vmovaps %xmm15, 144(%rsp)
    movq    256(%rsp), %r10               // n_bloques, quinto argumento (40 + 216)

    // --- constantes de las escalas, en la pila ---
    movl    $0x0F0F0F0F, %eax
    vmovd   %eax, %xmm0
    vpbroadcastd %xmm0, %xmm0
    vmovdqa %xmm0, 160(%rsp)              // 0x0F por byte
    movabsq $0x0000000200000000, %rax
    vmovq   %rax, %xmm0
    movabsq $0x0000000600000004, %rax
    vmovq   %rax, %xmm1
    vpunpcklqdq %xmm1, %xmm0, %xmm0
    vmovdqa %xmm0, 176(%rsp)              // corrimientos [0 2 4 6]
    movl    $0x20202020, %eax
    vmovd   %eax, %xmm0
    vpbroadcastd %xmm0, %xmm0
    vmovdqa %xmm0, 192(%rsp)              // 32 por byte

    // --- constantes del bucle, en registros ---
    movl    $0x00010001, %eax
    vmovd   %eax, %xmm15
    vpbroadcastd %xmm15, %ymm15          // unos en i16, para vpmaddwd
    movl    $0x03030303, %eax
    vmovd   %eax, %xmm14
    vpbroadcastd %xmm14, %ymm14          // dos bits
    movl    $0x04040404, %eax
    vmovd   %eax, %xmm13
    vpbroadcastd %xmm13, %ymm13          // el tercer bit, ya en su lugar
    // Los cuatro repartos [k k k k k+1 k+1 k+1 k+1], k = 0 2 4 6.
    movl    $1, %eax
    vmovd   %eax, %xmm0
    vpbroadcastd %xmm0, %xmm0
    vpxor   %xmm12, %xmm12, %xmm12
    vinserti128 $1, %xmm0, %ymm12, %ymm12   // [0 0 0 0 1 1 1 1]
    movl    $2, %eax
    vmovd   %eax, %xmm0
    vpbroadcastd %xmm0, %ymm0
    vpaddd  %ymm0, %ymm12, %ymm11          // [2 2 2 2 3 3 3 3]
    vpaddd  %ymm0, %ymm11, %ymm10          // [4 ...]
    vpaddd  %ymm0, %ymm10, %ymm9           // [6 ...]

    vxorps  %xmm8, %xmm8, %xmm8            // acumulador de los grupos pares
    vxorps  %xmm7, %xmm7, %xmm7            // de los impares
    vxorps  %xmm6, %xmm6, %xmm6            // correccion del +4
    testq   %r10, %r10
    je      9f

    .p2align 4
1:
    // --- las dieciseis escalas: d * (sc - 32) * xe ---
    vmovq   96(%rcx), %xmm0                // 8 bytes de nibbles
    vpsrlw  $4, %xmm0, %xmm1
    vpunpcklqdq %xmm1, %xmm0, %xmm0       // [bytes, bytes >> 4]
    vpand   160(%rsp), %xmm0, %xmm0       // los 16 nibbles bajos, en orden
    vpbroadcastd 104(%rcx), %xmm1          // los dos bits altos, en las cuatro palabras
    vpsrlvd 176(%rsp), %xmm1, %xmm1       // corridas 0 2 4 6
    vpand   %xmm14, %xmm1, %xmm1
    vpsllw  $4, %xmm1, %xmm1
    vpor    %xmm1, %xmm0, %xmm0
    vpsubb  192(%rsp), %xmm0, %xmm0       // las 16 escalas, con signo
    vpmovsxbd %xmm0, %ymm1                 // 0-7
    vpsrldq $8, %xmm0, %xmm0
    vpmovsxbd %xmm0, %ymm0                 // 8-15
    vcvtdq2ps %ymm1, %ymm1
    vcvtdq2ps %ymm0, %ymm0
    movzwl  108(%rcx), %eax
    vmovd   %eax, %xmm2
    vcvtph2ps %xmm2, %xmm2
    vbroadcastss %xmm2, %ymm2             // d
    vmulps  %ymm1, %ymm2, %ymm1
    vmulps  %ymm0, %ymm2, %ymm0
    vmulps  (%r8), %ymm1, %ymm5            // v0: escalas 0-7
    vmulps  32(%r8), %ymm0, %ymm4          // v1: 8-15
    vfmadd231ps (%r9), %ymm5, %ymm6
    vfmadd231ps 32(%r9), %ymm4, %ymm6     // correccion += v * xs
    vmovdqu (%rcx), %ymm3                  // hmask, el tercer bit
    vmovdqu 32(%rcx), %ymm2                // qs de los grupos 0-3

    // grupo 0
    vpand   %ymm14, %ymm2, %ymm0
    vpsllw  $2, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 0(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm5, %ymm12, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm8

    // grupo 1
    vpsrlw  $2, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpsllw  $1, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 32(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm5, %ymm11, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm7

    // grupo 2
    vpsrlw  $4, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpand   %ymm13, %ymm3, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 64(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm5, %ymm10, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm8

    // grupo 3
    vpsrlw  $6, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpsrlw  $1, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 96(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm5, %ymm9, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm7

    vmovdqu 64(%rcx), %ymm2                // qs de los grupos 4-7

    // grupo 4
    vpand   %ymm14, %ymm2, %ymm0
    vpsrlw  $2, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 128(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm4, %ymm12, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm8

    // grupo 5
    vpsrlw  $2, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpsrlw  $3, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 160(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm4, %ymm11, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm7

    // grupo 6
    vpsrlw  $4, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpsrlw  $4, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 192(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm4, %ymm10, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm8

    // grupo 7
    vpsrlw  $6, %ymm2, %ymm0
    vpand   %ymm14, %ymm0, %ymm0
    vpsrlw  $5, %ymm3, %ymm1
    vpand   %ymm13, %ymm1, %ymm1
    vpor    %ymm1, %ymm0, %ymm0
    vpmaddubsw 224(%rdx), %ymm0, %ymm0
    vpmaddwd %ymm15, %ymm0, %ymm0
    vcvtdq2ps %ymm0, %ymm0
    vpermps %ymm4, %ymm9, %ymm1
    vfmadd231ps %ymm1, %ymm0, %ymm7

    addq    $110, %rcx
    addq    $256, %rdx
    addq    $64, %r8
    addq    $64, %r9
    decq    %r10
    jne     1b

9:
    // reducir(acc0 + acc1) - 4 * reducir(correccion), como en Rust
    vaddps  %ymm7, %ymm8, %ymm0
    vextractf128 $1, %ymm0, %xmm2
    vaddps  %xmm2, %xmm0, %xmm0
    vmovhlps %xmm0, %xmm0, %xmm2
    vaddps  %xmm2, %xmm0, %xmm0
    vmovshdup %xmm0, %xmm2
    vaddss  %xmm2, %xmm0, %xmm0
    vextractf128 $1, %ymm6, %xmm2
    vaddps  %xmm2, %xmm6, %xmm1
    vmovhlps %xmm1, %xmm1, %xmm2
    vaddps  %xmm2, %xmm1, %xmm1
    vmovshdup %xmm1, %xmm2
    vaddss  %xmm2, %xmm1, %xmm1
    movl    $0x40800000, %eax              // 4.0
    vmovd   %eax, %xmm2
    vmulss  %xmm2, %xmm1, %xmm1
    vsubss  %xmm1, %xmm0, %xmm0
    vmovaps 0(%rsp), %xmm6
    vmovaps 16(%rsp), %xmm7
    vmovaps 32(%rsp), %xmm8
    vmovaps 48(%rsp), %xmm9
    vmovaps 64(%rsp), %xmm10
    vmovaps 80(%rsp), %xmm11
    vmovaps 96(%rsp), %xmm12
    vmovaps 112(%rsp), %xmm13
    vmovaps 128(%rsp), %xmm14
    vmovaps 144(%rsp), %xmm15
    addq    $216, %rsp
    vzeroupper
    retq
