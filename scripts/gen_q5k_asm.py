"""Genera kernels/q5k_avx2_sysv.s y kernels/q5k_avx2_win64.s:

    python scripts/gen_q5k_asm.py

El cuerpo es uno solo, con los registros de los argumentos como marcadores:
B (bloques), X (xq), E (xe), S (xs), N (n_bloques).
"""
import os

# Desde la raiz del proyecto, sea donde sea que se llame.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def escalas(origen_bajo, dest_xmm):
    """Los 8 bytes de escalas (o de minimos) en `dest_xmm`, con la tecnica sin
    ramas de `escalas_q4k`: rax y r11 son los unicos registros generales que se
    tocan, libres en las dos convenciones."""
    # origen_bajo: 4 para las escalas (bytes 4-7 = a), 8 para los minimos (b).
    alto_corr = '' if origen_bajo == 4 else '    shr eax, 4\n'
    return f'''    mov eax, dword ptr [B + 12]     // c
{alto_corr}    and eax, 0x0F0F0F0F
    mov r11d, dword ptr [B + {origen_bajo}]
    shr r11d, 6
    and r11d, 0x03030303
    shl r11d, 4
    or eax, r11d                    // las cuatro altas
    shl rax, 32
    mov r11d, dword ptr [B + {origen_bajo}]
    and r11d, 0x3F3F3F3F            // las cuatro bajas
    or rax, r11
    vmovq {dest_xmm}, rax
'''


def sub_bloque(par, mitad):
    j = par * 2 + mitad
    acc = 'ymm10' if mitad == 0 else 'ymm11'
    s = []
    if mitad == 0:
        s.append('    vpand ymm1, ymm0, ymm14')
    else:
        s.append('    vpsrlw ymm1, ymm0, 4')
        s.append('    vpand ymm1, ymm1, ymm14')
    # El bit j de cada byte de qh, a la posicion 4: un corrimiento de 16 bits
    # no mezcla nada que el `and` con 0x10 no borre.
    if j < 4:
        s.append(f'    vpsllw ymm2, ymm9, {4 - j}')
        s.append('    vpand ymm2, ymm2, ymm13')
    elif j == 4:
        s.append('    vpand ymm2, ymm9, ymm13')
    else:
        s.append(f'    vpsrlw ymm2, ymm9, {j - 4}')
        s.append('    vpand ymm2, ymm2, ymm13')
    s.append('    vpor ymm1, ymm1, ymm2')
    s.append(f'    vpmaddubsw ymm1, ymm1, ymmword ptr [X + {par * 64 + mitad * 32}]')
    s.append('    vpmaddwd ymm1, ymm1, ymm15')
    s.append('    vcvtdq2ps ymm1, ymm1')
    s.append('    vpermps ymm2, ymm7, ymm3')
    s.append(f'    vfmadd231ps {acc}, ymm1, ymm2')
    s.append('    vpaddd ymm7, ymm7, ymm6')
    return '\n'.join(s) + '\n'


cuerpo = '''
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

    test N, N
    jz 3f

    .p2align 4
1:
    // Las 8 escalas y los 8 minimos de 6 bits, en registros generales.
''' + escalas(4, 'xmm1') + escalas(8, 'xmm2') + '''
    // El termino de los minimos no toca los valores: va directo a la correccion.
    movzx eax, word ptr [B + 2]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // dmin
    vpmovzxbd ymm5, xmm2
    vcvtdq2ps ymm5, ymm5
    vmulps ymm5, ymm5, ymm8
    vmulps ymm5, ymm5, ymmword ptr [E]
    vfmadd231ps ymm12, ymm5, ymmword ptr [S]

    // Las escalas de cada sub-bloque, multiplicadas por las de la entrada.
    movzx eax, word ptr [B]
    vmovd xmm8, eax
    vcvtph2ps xmm8, xmm8
    vbroadcastss ymm8, xmm8         // d
    vpmovzxbd ymm3, xmm1
    vcvtdq2ps ymm3, ymm3
    vmulps ymm3, ymm3, ymm8
    vmulps ymm3, ymm3, ymmword ptr [E]

    vmovdqu ymm9, ymmword ptr [B + 16]  // qh: el quinto bit de cada valor
    vpxor xmm7, xmm7, xmm7          // el indice de escalas, desde 0
'''
for par in range(4):
    cuerpo += f'    vmovdqu ymm0, ymmword ptr [B + {48 + par * 32}]\n'
    cuerpo += sub_bloque(par, 0) + sub_bloque(par, 1)
cuerpo += '''    add B, 176
    add X, 256
    add E, 32
    add S, 32
    dec N
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
'''

FIRMA = '''//   float cosxamix_q5k_asm(const uint8_t *bloques,  // 176 bytes por bloque
//                          const int8_t  *xq,       // 256 por bloque
//                          const float   *xe,       // 8 por bloque
//                          const float   *xs,       // 8 por bloque
//                          size_t         n_bloques);
'''
COMUN = '''// Devuelve en xmm0. El algoritmo es `dot_q5k_i8_avx2` de `src/kernels/x86.rs`
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
'''

sysv = '''// El kernel fusionado de Q5_K en AVX2 -- convencion de llamada System V.
//
''' + FIRMA + '''//
//   bloques = rdi, xq = rsi, xe = rdx, xs = rcx, n_bloques = r8
//
// Este archivo y `q5k_avx2_win64.s` son el mismo kernel. Aca los dieciseis ymm
// son volatiles: no hay nada que guardar.
//
''' + COMUN + '''
    .text
    // Con y sin guion bajo: en Mach-O (macOS) el enlazador busca
    // `_cosxamix_q5k_asm`, y en ELF `cosxamix_q5k_asm`.
    .globl cosxamix_q5k_asm
    .globl _cosxamix_q5k_asm
    .p2align 4
cosxamix_q5k_asm:
_cosxamix_q5k_asm:
''' + cuerpo + '''
    vzeroupper
    ret
'''
for a, b in [('[B', '[rdi'), ('B,', 'rdi,'), ('[X', '[rsi'), ('X,', 'rsi,'), ('[E]', '[rdx]'), ('E,', 'rdx,'),
             ('[S]', '[rcx]'), ('S,', 'rcx,'), ('N, N', 'r8, r8'), ('dec N', 'dec r8')]:
    sysv = sysv.replace(a, b)

GUARDAR = ''.join(f'    vmovups xmmword ptr [rsp + {i * 16}], xmm{6 + i}\n' for i in range(10))
RESTAURAR = ''.join(f'    vmovups xmm{6 + i}, xmmword ptr [rsp + {i * 16}]\n' for i in range(10))
win = '''// El kernel fusionado de Q5_K en AVX2 -- convencion de llamada de Windows x64.
//
''' + FIRMA + '''//
//   bloques = rcx, xq = rdx, xe = r8, xs = r9, n_bloques = [rsp + 40]
//
// Este archivo y `q5k_avx2_sysv.s` son el mismo kernel. Aca xmm6-xmm15 son del
// llamador: se guardan en la pila y se restauran.
//
''' + COMUN + '''
    .text
    .globl cosxamix_q5k_asm
    .p2align 4
cosxamix_q5k_asm:
    mov r10, [rsp + 40]
    sub rsp, 168
''' + GUARDAR + cuerpo + RESTAURAR + '''    add rsp, 168
    vzeroupper
    ret
'''
for a, b in [('[B', '[rcx'), ('B,', 'rcx,'), ('[X', '[rdx'), ('X,', 'rdx,'), ('[E]', '[r8]'), ('E,', 'r8,'),
             ('[S]', '[r9]'), ('S,', 'r9,'), ('N, N', 'r10, r10'), ('dec N', 'dec r10')]:
    win = win.replace(a, b)

for nombre, texto in [('kernels/q5k_avx2_sysv.s', sysv), ('kernels/q5k_avx2_win64.s', win)]:
    open(nombre, 'w', newline='\n').write(texto)
print('kernels/q5k_avx2_sysv.s y kernels/q5k_avx2_win64.s generados')
