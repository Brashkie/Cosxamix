"""Comprueba que la libreria de kernels de Zig sea de la arquitectura pedida.

Que `cargo build --target X --features zig` termine bien NO prueba que el
kernel sea de X. Paso de verdad: `build.rs` pasaba `-target=aarch64-linux-gnu`
en un solo argumento, Zig lo quiere en dos, y como el flag no se reconocia
compilaba para el host. El .a resultante era un archivador perfectamente
valido, con los simbolos correctos, lleno de codigo x86-64.

Se mira la cabecera ELF o COFF del primer objeto de dentro, que es el unico
sitio donde la arquitectura esta escrita y no deducida.

    python scripts/check_kernel_arch.py <target-de-rust> <arquitectura>

donde la arquitectura es `aarch64`, `x86_64` o `arm`.
"""

import glob
import os
import struct
import sys

# ELF: e_machine, en el offset 18 de la cabecera.
ELF_MAQUINA = {0x03: "x86", 0x28: "arm", 0x3E: "x86_64", 0xB7: "aarch64"}
# COFF (Windows): Machine, en el offset 0 de la cabecera del objeto.
COFF_MAQUINA = {0x014C: "x86", 0x8664: "x86_64", 0xAA64: "aarch64"}


def arquitectura(ruta):
    """Lee la arquitectura del primer objeto dentro del archivador."""
    datos = open(ruta, "rb").read()

    i = datos.find(b"\x7fELF")
    if i >= 0:
        maquina = struct.unpack_from("<H", datos, i + 18)[0]
        return ELF_MAQUINA.get(maquina, f"desconocida (0x{maquina:02x})")

    # En Windows el .lib es un archivador de objetos COFF. El primer miembro
    # suele ser el indice de simbolos, asi que se busca una cabecera cuya
    # maquina sea conocida en vez de leer el offset fijo.
    for off in range(0, min(len(datos) - 2, 65536)):
        maquina = struct.unpack_from("<H", datos, off)[0]
        if maquina in COFF_MAQUINA:
            # Comprobacion barata de que es una cabecera y no ruido: el numero
            # de secciones de un objeto asi es bajo.
            secciones = struct.unpack_from("<H", datos, off + 2)[0]
            if 0 < secciones < 64:
                return COFF_MAQUINA[maquina]

    return None


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
