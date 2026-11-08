// Clasificar bytes de a 32 en AVX2 -- convencion de llamada System V.
//
//   void cosxamix_clasificar_asm(const uint8_t *p, size_t bloques,
//                                uint32_t *letras, uint32_t *digitos,
//                                uint32_t *altos)
//
//   p = rdi, bloques = rsi, letras = rdx, digitos = rcx, altos = r8
//
// Este archivo y `classify_avx2_win64.s` son el mismo kernel. Aca los dieciseis
// ymm son volatiles, asi que no hay nada que guardar.
//
// La especificacion ejecutable es `escalar` en `src/kernels/classify.rs`.

    .text
    // Con y sin guion bajo: en Mach-O (macOS) el enlazador busca `_cosxamix_clasificar_asm`, y en
    // ELF `cosxamix_clasificar_asm`. Como en `dot_avx2_sysv.s`: faltaba, y rompio el build de macOS.
    .globl cosxamix_clasificar_asm
    .globl _cosxamix_clasificar_asm
    .p2align 4
cosxamix_clasificar_asm:
_cosxamix_clasificar_asm:
    test rsi, rsi
    jz 2f

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
    vmovdqu ymm0, ymmword ptr [rdi]
    vpor ymm1, ymm0, ymm2
    vpsubb ymm1, ymm1, ymm3
    vpminub ymm7, ymm1, ymm4
    vpcmpeqb ymm1, ymm1, ymm7
    vpmovmskb eax, ymm1
    mov dword ptr [rdx], eax

    vpsubb ymm1, ymm0, ymm5
    vpminub ymm7, ymm1, ymm6
    vpcmpeqb ymm1, ymm1, ymm7
    vpmovmskb eax, ymm1
    mov dword ptr [rcx], eax

    vpmovmskb eax, ymm0
    mov dword ptr [r8], eax

    add rdi, 32
    add rdx, 4
    add rcx, 4
    add r8, 4
    dec rsi
    jnz 1b

    vzeroupper
2:
    ret
