// Producto punto f32 en AVX2 — convencion de llamada System V (Linux, macOS).
//
// Este archivo y `dot_avx2_sysv.s` son el MISMO kernel. Lo unico que cambia
// son los registros en los que llegan los argumentos, y eso es una propiedad
// del ensamblador que ningun otro lenguaje de esta frontera tiene: Rust, Zig
// y Fortran no se enteran de la ABI, aca hay que escribirla.
//
//   Windows x64:  rcx  rdx  r8   r9
//   System V:     rdi  rsi  rdx  rcx
//
// Firma, igual en los dos:
//   void cosxamix_dot_bloques(const float *a, const float *b,
//                             float *acc, size_t bloques)
//
// Consume `bloques * 32` elementos y deja cuatro acumuladores de 8 en `acc`.
// La cola y la suma horizontal las hace Rust, con el MISMO codigo que la
// version de intrinsecos: asi lo unico que separa a las dos es este bucle.
//
// Solo se usan ymm0-ymm5 a proposito. En Windows x64 los registros xmm6 a
// xmm15 son no volatiles y habria que salvarlos y restaurarlos; usando solo
// los volatiles el kernel no necesita prologo ni epilogo.

// Sintaxis Intel, sin declararla: `global_asm!` de Rust ya la selecciona, y
// repetir la directiva dispara el lint `bad_asm_style` -- que busca el texto,
// asi que ni siquiera se puede nombrar en un comentario. Para ensamblar este
// archivo con GNU `as` por fuera de cargo hacen falta `-msyntax=intel` y
// `-mnaked-reg`, porque el suyo por defecto es AT&T.

// Un solo archivo para Linux y macOS, que comparten la convencion de llamada
// pero NO el formato de objeto. De ahi las dos rarezas de aca abajo, las dos
// aprendidas rompiendo el build de macOS en CI:
//
//   1. No lleva `.type sym, @function`. Esa directiva es de ELF y el
//      ensamblador de Mach-O la rechaza con "unknown directive". No hace
//      falta para enlazar, solo marca el simbolo como funcion.
//
//   2. El simbolo se define DOS veces, con y sin guion bajo. En Mach-O una
//      funcion de C se llama `_foo` a nivel de enlazador y en ELF `foo`, asi
//      que definir los dos nombres evita partir el archivo en dos por una
//      letra. El que sobra es una entrada mas en la tabla de simbolos.

    .text
    .globl cosxamix_dot_bloques
    .globl _cosxamix_dot_bloques

cosxamix_dot_bloques:
_cosxamix_dot_bloques:
    // Cuatro cadenas independientes: un FMA tarda varios ciclos y con un
    // solo acumulador cada vuelta espera a la anterior.
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3

    test    rcx, rcx
    jz      2f

1:
    // vfmadd231ps acepta el ultimo operando en memoria, asi que solo hace
    // falta cargar uno de los dos vectores.
    vmovups ymm4, [rdi]
    vfmadd231ps ymm0, ymm4, [rsi]
    vmovups ymm5, [rdi + 32]
    vfmadd231ps ymm1, ymm5, [rsi + 32]
    vmovups ymm4, [rdi + 64]
    vfmadd231ps ymm2, ymm4, [rsi + 64]
    vmovups ymm5, [rdi + 96]
    vfmadd231ps ymm3, ymm5, [rsi + 96]

    add     rdi, 128
    add     rsi, 128
    dec     rcx
    jnz     1b

2:
    vmovups [rdx], ymm0
    vmovups [rdx + 32], ymm1
    vmovups [rdx + 64], ymm2
    vmovups [rdx + 96], ymm3

    // Obligatorio antes de volver a codigo que pueda usar SSE: sin esto se
    // paga la penalizacion de transicion AVX-SSE en cada llamada.
    vzeroupper
    ret
