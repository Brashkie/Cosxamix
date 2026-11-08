// Clasificar bytes de a 32 en AVX2 -- convencion de llamada de Windows x64.
//
//   void cosxamix_clasificar_asm(const uint8_t *p, size_t bloques,
//                                uint32_t *letras, uint32_t *digitos,
//                                uint32_t *altos)
//
//   p = rcx, bloques = rdx, letras = r8, digitos = r9, altos = [rsp + 40]
//
// Este archivo y `classify_avx2_sysv.s` son el mismo kernel: solo cambian los
// registros de entrada. Hacen falta dos registros mas de los volatiles, asi
// que xmm6 y xmm7 se guardan en el espacio de sombra, que es del que recibe la
// llamada y alcanza justo para los dos sin mover la pila.
//
// La especificacion ejecutable es `escalar` en `src/kernels/classify.rs`.

    .text
    .globl cosxamix_clasificar_asm
    .p2align 4
cosxamix_clasificar_asm:
    test rdx, rdx
    jz 2f
    mov r10, [rsp + 40]
    vmovups xmmword ptr [rsp + 8], xmm6
    vmovups xmmword ptr [rsp + 24], xmm7

    mov eax, 0x20202020
    vmovd xmm2, eax
    vpbroadcastd ymm2, xmm2
    mov eax, 0x61616161
    vmovd xmm3, eax
    vpbroadcastd ymm3, xmm3
    mov eax, 0x19191919
    vmovd xmm4, eax
    vpbroadcastd ymm4, xmm4
    mov eax, 0x30303030
    vmovd xmm5, eax
    vpbroadcastd ymm5, xmm5
    mov eax, 0x09090909
    vmovd xmm6, eax
    vpbroadcastd ymm6, xmm6

    .p2align 4
1:
    vmovdqu ymm0, ymmword ptr [rcx]
    vpor ymm1, ymm0, ymm2
    vpsubb ymm1, ymm1, ymm3
    vpminub ymm7, ymm1, ymm4
    vpcmpeqb ymm1, ymm1, ymm7
    vpmovmskb eax, ymm1
    mov dword ptr [r8], eax

    vpsubb ymm1, ymm0, ymm5
    vpminub ymm7, ymm1, ymm6
    vpcmpeqb ymm1, ymm1, ymm7
    vpmovmskb eax, ymm1
    mov dword ptr [r9], eax

    vpmovmskb eax, ymm0
    mov dword ptr [r10], eax

    add rcx, 32
    add r8, 4
    add r9, 4
    add r10, 4
    dec rdx
    jnz 1b

    vmovups xmm6, xmmword ptr [rsp + 8]
    vmovups xmm7, xmmword ptr [rsp + 24]
    vzeroupper
2:
    ret
