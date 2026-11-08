// El kernel fusionado de Q2_K en AVX2 -- convencion de llamada de Windows x64.
//
//   float cosxamix_q2k_asm(const uint8_t *bloques,  // 84 bytes por bloque
//                          const int8_t  *xq,       // 256 por bloque
//                          const float   *xe,       // 16 por bloque
//                          const float   *xs,       // 16 por bloque
//                          size_t         n_bloques);
//
//   bloques = rcx, xq = rdx, xe = r8, xs = r9, n_bloques = [rsp + 40]
//
// Este archivo y `q2k_avx2_sysv.s` son el mismo kernel. Aca xmm6-xmm15 son del
// llamador: se guardan en la pila y se restauran.
//
// Devuelve en xmm0. El algoritmo es `avx2` de `src/kernels/q2k.rs`; la
// especificacion ejecutable es `escalar`, en el mismo archivo.
//
// El reparto de registros, que es lo propio del ensamblador: ymm13-ymm15 y ymm6,
// ymm9 son constantes de todo el bucle; ymm10-ymm12, los acumuladores; ymm3 y ymm4,
// las escalas del bloque; ymm0 y ymm1, los valores.

    .text
    .globl cosxamix_q2k_asm
    .p2align 4
cosxamix_q2k_asm:
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
    mov eax, 0x03030303
    vmovd xmm14, eax
    vpbroadcastd ymm14, xmm14       // dos bits
    mov eax, 0x0F0F0F0F
    vmovd xmm13, eax
    vpbroadcastd ymm13, xmm13       // un nibble
    mov eax, 2
    vmovd xmm6, eax
    vpbroadcastd ymm6, xmm6         // lo que avanza el indice de escalas
    // El primer reparto de escalas: 0 en la mitad baja y 1 en la alta.
    vpxor xmm9, xmm9, xmm9
    mov eax, 1
    vmovd xmm7, eax
    vpbroadcastd xmm7, xmm7
    vinserti128 ymm9, ymm9, xmm7, 1

    vxorps xmm10, xmm10, xmm10      // acc0
    vxorps xmm11, xmm11, xmm11      // acc1
    vxorps xmm12, xmm12, xmm12      // correccion

    test r10, r10
    jz 3f

    .p2align 4
1:
    // Las escalas y los minimos de los 16 sub-bloques.
    vmovdqu xmm0, xmmword ptr [rcx]
    vpand xmm1, xmm0, xmm13         // escalas
    vpsrlw xmm2, xmm0, 4
    vpand xmm2, xmm2, xmm13         // minimos

    // El termino de los minimos no toca los valores: va directo a la correccion.
    movzx eax, word ptr [rcx + 82]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // dmin
    vpmovzxbd ymm5, xmm2
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm8
    vmulps ymm5, ymm5, ymmword ptr [r8]
    vfmadd231ps ymm12, ymm5, ymmword ptr [r9]
    vpsrldq xmm2, xmm2, 8
    vpmovzxbd ymm5, xmm2
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm8
    vmulps ymm5, ymm5, ymmword ptr [r8 + 32]
    vfmadd231ps ymm12, ymm5, ymmword ptr [r9 + 32]

    // Las escalas de cada sub-bloque, multiplicadas por las de la entrada.
    movzx eax, word ptr [rcx + 80]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // d
    vpmovzxbd ymm3, xmm1
    vcvtdq2ps ymm3, ymm3
    vmulps ymm3, ymm3, ymm8
    vmulps ymm3, ymm3, ymmword ptr [r8]
    vpsrldq xmm1, xmm1, 8
    vpmovzxbd ymm4, xmm1
    vcvtdq2ps ymm4, ymm4
    vmulps ymm4, ymm4, ymm8
    vmulps ymm4, ymm4, ymmword ptr [r8 + 32]

    vmovdqu ymm0, ymmword ptr [rcx + 16]
    vmovdqu ymm1, ymmword ptr [rcx + 48]
    vmovdqa ymm7, ymm9              // el indice de escalas
    vpand ymm2, ymm0, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 0]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm3
    vfmadd231ps ymm10, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm0, 2
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 32]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm3
    vfmadd231ps ymm11, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm0, 4
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 64]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm3
    vfmadd231ps ymm10, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm0, 6
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 96]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm3
    vfmadd231ps ymm11, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vmovdqa ymm7, ymm9
    vpand ymm2, ymm1, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 128]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm4
    vfmadd231ps ymm10, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm1, 2
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 160]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm4
    vfmadd231ps ymm11, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm1, 4
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 192]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm4
    vfmadd231ps ymm10, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    vpsrlw ymm2, ymm1, 6
    vpand ymm2, ymm2, ymm14
    vpmaddubsw ymm2, ymm2, ymmword ptr [rdx + 224]
    vpmaddwd ymm2, ymm2, ymm15
    vcvtdq2ps ymm2, ymm2
    vpermps ymm8, ymm7, ymm4
    vfmadd231ps ymm11, ymm2, ymm8
    vpaddd ymm7, ymm7, ymm6
    add rcx, 84
    add rdx, 256
    add r8, 64
    add r9, 64
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
