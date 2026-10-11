"""Comprueba que la libreria de kernels de Zig sea de la arquitectura pedida.

Que `cargo build --target X --features zig` termine bien NO prueba que el
kernel sea de X. Paso de verdad: `build.rs` pasaba `-target=aarch64-linux-gnu`
en un solo argumento, Zig lo quiere en dos, y como el flag no se reconocia
compilaba para el host. El .a resultante era un archivador perfectamente
valido, con los simbolos correctos, lleno de codigo x86-64.

Se mira la cabecera del primer objeto de dentro del archivador, que es el unico
sitio donde la arquitectura esta escrita y no deducida. Son tres formatos y
hacen falta los tres, uno por plataforma de la matriz de release:

    ELF     Linux y Android
    Mach-O  macOS          <- el que faltaba, y tiro el primer release
    COFF    Windows

    python scripts/check_kernel_arch.py <target-de-rust> <arquitectura>

donde la arquitectura es `aarch64`, `x86_64` o `arm`.
"""

import glob
import os
import struct
import sys

# ELF: e_machine, u16 en el offset 18.
ELF_MAQUINA = {0x03: "x86", 0x28: "arm", 0x3E: "x86_64", 0xB7: "aarch64"}
# Mach-O: cputype, u32 en el offset 4. El bit 0x01000000 marca los de 64 bits.
MACHO_MAQUINA = {7: "x86", 12: "arm", 0x01000007: "x86_64", 0x0100000C: "aarch64"}
# COFF: Machine, u16 en el offset 0 del objeto.
COFF_MAQUINA = {0x014C: "x86", 0x01C0: "arm", 0x8664: "x86_64", 0xAA64: "aarch64"}

# Miembros del archivador que no son objetos: tablas de simbolos y de nombres.
NO_OBJETOS = ("/", "//", "__.SYMDEF", "__.SYMDEF SORTED")


def primer_objeto(datos):
    """Devuelve el contenido del primer miembro real de un archivador `ar`.

    Se parsea el formato en vez de buscar la magia por todo el archivo. Buscar
    parece mas simple y es peor: un escaneo que acepta el primer valor
    plausible puede encontrarlo por casualidad dentro de los datos y reportar
    con total confianza una arquitectura equivocada, que es peor que no
    reportar nada.

    Cabecera de 60 bytes por miembro: nombre[16] fecha[12] uid[6] gid[6]
    modo[8] tamano[10] fin[2], y los datos van alineados a par.
    """
    if not datos.startswith(b"!<arch>\n"):
        return datos  # no es un archivador: ya es el objeto suelto

    pos = 8
    while pos + 60 <= len(datos):
        cabecera = datos[pos:pos + 60]
        if cabecera[58:60] != b"`\n":
            return None
        nombre = cabecera[0:16].decode("ascii", "replace").strip()
        try:
            tamano = int(cabecera[48:58].decode("ascii", "replace").strip())
        except ValueError:
            return None

        inicio = pos + 60
        # Estilo BSD, que es el que usa macOS: `#1/N` dice que los primeros N
        # bytes de los datos son el nombre largo, no el contenido.
        if nombre.startswith("#1/"):
            try:
                largo = int(nombre[3:])
            except ValueError:
                return None
            real = datos[inicio:inicio + largo].rstrip(b"\x00").decode("ascii", "replace")
            inicio += largo
            tamano -= largo
            nombre = real

        if nombre.rstrip("/") not in [n.rstrip("/") for n in NO_OBJETOS]:
            return datos[inicio:inicio + tamano]

        pos = inicio + tamano
        pos += pos % 2  # los miembros empiezan en offset par

    return None


def arquitectura(ruta):
    """Lee la arquitectura del primer objeto dentro del archivador."""
    objeto = primer_objeto(open(ruta, "rb").read())
    if not objeto or len(objeto) < 20:
        return None

    magia = objeto[:4]

    if magia == b"\x7fELF":
        orden = "<" if objeto[5] == 1 else ">"
        maquina = struct.unpack_from(orden + "H", objeto, 18)[0]
        return ELF_MAQUINA.get(maquina, f"ELF desconocido (0x{maquina:04x})")

    if magia in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        cputype = struct.unpack_from("<I", objeto, 4)[0]
        return MACHO_MAQUINA.get(cputype, f"Mach-O desconocido (0x{cputype:08x})")

    if magia in (b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"):
        cputype = struct.unpack_from(">I", objeto, 4)[0]
        return MACHO_MAQUINA.get(cputype, f"Mach-O desconocido (0x{cputype:08x})")

    maquina = struct.unpack_from("<H", objeto, 0)[0]
    if maquina in COFF_MAQUINA:
        return COFF_MAQUINA[maquina]

    return f"formato desconocido, primeros bytes {objeto[:8].hex()}"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    target, esperada = sys.argv[1], sys.argv[2]

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Un target vacio (o "native") es la compilacion para la propia maquina,
    # que cargo deja en `target/<perfil>/` sin el nivel del triple.
    base = "target" if target in ("", "native") else f"target/{target}"
    patrones = [
        f"{base}/{perfil}/build/*/out/{lib}"
        for perfil in ("release", "debug")
        for lib in ("libcosxamix_kernels.a", "cosxamix_kernels.lib")
    ]
    encontrados = []
    for p in patrones:
        encontrados += glob.glob(os.path.join(raiz, p))

    if not encontrados:
        print(f"  no encontre la libreria de kernels para {target}")
        print("  compilaste con --features zig?")
        return 1

    ruta = encontrados[0]
    obtenida = arquitectura(ruta)
    rel = os.path.relpath(ruta, raiz)

    for simbolo in (b"cosxamix_dot_f32", b"cosxamix_rmsnorm_f32"):
        if simbolo not in open(ruta, "rb").read():
            print(f"  FALTA el simbolo {simbolo.decode()} en {rel}")
            return 1

    if obtenida != esperada:
        print(f"  {rel}")
        print(f"  arquitectura esperada {esperada}, encontrada {obtenida}")
        print("  el kernel de Zig NO es del target: revisa `-target` en build.rs")
        return 1

    print(f"  {rel}")
    print(f"  arquitectura {obtenida}, con los dos simbolos exportados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
