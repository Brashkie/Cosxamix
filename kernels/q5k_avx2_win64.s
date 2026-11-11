// El kernel fusionado de Q5_K en AVX2 -- convencion de llamada de Windows x64.
//
//   float cosxamix_q5k_asm(const uint8_t *bloques,  // 176 bytes por bloque
//                          const int8_t  *xq,       // 256 por bloque
//                          const float   *xe,       // 8 por bloque
//                          const float   *xs,       // 8 por bloque
//                          size_t         n_bloques);
//
//   bloques = rcx, xq = rdx, xe = r8, xs = r9, n_bloques = [rsp + 40]
//
// Este archivo y `q5k_avx2_sysv.s` son el mismo kernel. Aca xmm6-xmm15 son del
// llamador: se guardan en la pila y se restauran.
//
// Devuelve en xmm0. El algoritmo es `dot_q5k_i8_avx2` de `src/kernels/x86.rs`
// con una diferencia, que es lo que se mide aca: el quinto bit no se arma con
// `and`, `cmpeq` y `and`, sino corriendo `qh` hasta que el bit `j` quede en la
// posicion 4 y aislandolo con un solo `and 0x10` (en el sub-bloque 4, ni el
// corrimiento). La especificacion ejecutable es `scalar::dot_q5k_i8`.
//
// Las escalas de 6 bits se desarman en registros generales con la tecnica sin
// ramas de `escalas_q4k`, usando solo rax y r11, que son volatiles en las dos
// convenciones.
//
// Registros: ymm13-ymm15 y ymm6 son constantes; ymm10-ymm12, los acumuladores;
// ymm9, el `qh` del bloque; ymm3, sus escalas; ymm7, el indice que las reparte.
//
// Generado por `scripts/gen_q5k_asm.py`: el cuerpo es el mismo en las dos
// convenciones y solo cambian los registros de los argumentos.

    .text
    .globl cosxamix_q5k_asm
    .p2align 4
cosxamix_q5k_asm:
    mov r10, [rsp + 40]
    sub rsp, 168
    vmovups xmmword ptr [rsp + 0], xmm6
    vmovups xmmword ptr [rsp + 16], xmm7
    vmovups xmmword ptr [rsp + 32], xmm8
    vmovups xmmword ptr [rsp + 48], xmm9
    vmovups xmmword ptr [rsp + 64], xmm10
    vmovups xmmword ptr [rsp + 80], xmm11
    vmovups xmmword ptr [rsp + 96], xmm12
    vmovups xmmword ptr [rsp + 112], xmm13
    vmovups xmmword ptr [rsp + 128], xmm14
    vmovups xmmword ptr [rsp + 144], xmm15

    // Constantes del bucle, armadas con inmediatos: sin seccion de datos.
    mov eax, 0x00010001
    vmovd xmm15, eax
    vpbroadcastd ymm15, xmm15       // unos, de 16 bits
    mov eax, 0x0F0F0F0F
    vmovd xmm14, eax
    vpbroadcastd ymm14, xmm14       // un nibble
    mov eax, 0x10101010
    vmovd xmm13, eax
    vpbroadcastd ymm13, xmm13       // el quinto bit, ya en su lugar
    mov eax, 1
    vmovd xmm6, eax
    vpbroadcastd ymm6, xmm6         // lo que avanza el indice de escalas

    vxorps xmm10, xmm10, xmm10      // acc0, sub-bloques pares
    vxorps xmm11, xmm11, xmm11      // acc1, impares
    vxorps xmm12, xmm12, xmm12      // correccion

    test r10, r10
    jz 3f

    .p2align 4
1:
    // Las 8 escalas y los 8 minimos de 6 bits, en registros generales.
    mov eax, dword ptr [rcx + 12]     // c
    and eax, 0x0F0F0F0F
    mov r11d, dword ptr [rcx + 4]
    shr r11d, 6
    and r11d, 0x03030303
    shl r11d, 4
    or eax, r11d                    // las cuatro altas
    shl rax, 32
    mov r11d, dword ptr [rcx + 4]
    and r11d, 0x3F3F3F3F            // las cuatro bajas
    or rax, r11
    vmovq xmm1, rax
    mov eax, dword ptr [rcx + 12]     // c
    shr eax, 4
    and eax, 0x0F0F0F0F
    mov r11d, dword ptr [rcx + 8]
    shr r11d, 6
    and r11d, 0x03030303
    shl r11d, 4
    or eax, r11d                    // las cuatro altas
    shl rax, 32
    mov r11d, dword ptr [rcx + 8]
    and r11d, 0x3F3F3F3F            // las cuatro bajas
    or rax, r11
    vmovq xmm2, rax

    // El termino de los minimos no toca los valores: va directo a la correccion.
    movzx eax, word ptr [rcx + 2]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // dmin
    vpmovzxbd ymm5, xmm2
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm8
    vmulps ymm5, ymm5, ymmword ptr [r8]
    vfmadd231ps ymm12, ymm5, ymmword ptr [r9]

    // Las escalas de cada sub-bloque, multiplicadas por las de la entrada.
    movzx eax, word ptr [rcx]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // d
    vpmovzxbd ymm3, xmm1
    vcvtdq2ps ymm3, ymm3
    vmulps ymm3, ymm3, ymm8
    vmulps ymm3, ymm3, ymmword ptr [r8]

    vmovdqu ymm9, ymmword ptr [rcx + 16]  // qh: el quinto bit de cada valor
    vpxor xmm7, xmm7, xmm7          // el indice de escalas, desde 0
    vmovdqu ymm0, ymmword ptr [rcx + 48]
    vpand ymm1, ymm0, ymm14
    vpsllw ymm2, ymm9, 4
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 0]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm10, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm1, ymm0, 4
    vpand ymm1, ymm1, ymm14
    vpsllw ymm2, ymm9, 3
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 32]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm11, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vmovdqu ymm0, ymmword ptr [rcx + 80]
    vpand ymm1, ymm0, ymm14
    vpsllw ymm2, ymm9, 2
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 64]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm10, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm1, ymm0, 4
    vpand ymm1, ymm1, ymm14
    vpsllw ymm2, ymm9, 1
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 96]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm11, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vmovdqu ymm0, ymmword ptr [rcx + 112]
    vpand ymm1, ymm0, ymm14
    vpand ymm2, ymm9, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 128]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm10, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm1, ymm0, 4
    vpand ymm1, ymm1, ymm14
    vpsrlw ymm2, ymm9, 1
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 160]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm11, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vmovdqu ymm0, ymmword ptr [rcx + 144]
    vpand ymm1, ymm0, ymm14
    vpsrlw ymm2, ymm9, 2
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 192]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm10, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm1, ymm0, 4
    vpand ymm1, ymm1, ymm14
    vpsrlw ymm2, ymm9, 3
    vpand ymm2, ymm2, ymm13
    vpor ymm1, ymm1, ymm2
    vpmaddubsw ymm1, ymm1, ymmword ptr [rdx + 224]
    vpmaddwd ymm1, ymm1, ymm15
    vcvtdq2ps ymm1, ymm1
    vpermps ymm2, ymm7, ymm3
    vfmadd231ps ymm11, ymm1, ymm2
    vpaddd ymm7, ymm7, ymm6
    add rcx, 176
    add rdx, 256
    add r8, 32
    add r9, 32
    dec r10
    jnz 1b

3:
    // acc0 + acc1, reducido, menos la correccion reducida.
    vaddps ymm10, ymm10, ymm11
    vextractf128 xmm0, ymm10, 1
    vaddps xmm0, xmm0, xmm10
    vmovhlps xmm1, xmm1, xmm0
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 0x55
    vaddss xmm0, xmm0, xmm1

    vextractf128 xmm2, ymm12, 1
    vaddps xmm2, xmm2, xmm12
    vmovhlps xmm1, xmm1, xmm2
    vaddps xmm2, xmm2, xmm1
    vshufps xmm1, xmm2, xmm2, 0x55
    vaddss xmm2, xmm2, xmm1
    vsubss xmm0, xmm0, xmm2
    vmovups xmm6, xmmword ptr [rsp + 0]
    vmovups xmm7, xmmword ptr [rsp + 16]
    vmovups xmm8, xmmword ptr [rsp + 32]
    vmovups xmm9, xmmword ptr [rsp + 48]
    vmovups xmm10, xmmword ptr [rsp + 64]
    vmovups xmm11, xmmword ptr [rsp + 80]
    vmovups xmm12, xmmword ptr [rsp + 96]
    vmovups xmm13, xmmword ptr [rsp + 112]
    vmovups xmm14, xmmword ptr [rsp + 128]
    vmovups xmm15, xmmword ptr [rsp + 144]
    add rsp, 168
    vzeroupper
    ret
