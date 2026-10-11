"""Revisa la documentacion: tablas, enlaces y nombres viejos.

Existe porque los tres fallos que aparecieron de verdad en este proyecto fueron
estos, y ninguno lo ve el compilador:

  1. una tabla de Markdown con una fila de distinto numero de columnas que su
     cabecera se renderiza partida en GitHub, y en el editor local no;
  2. un enlace relativo a un archivo renombrado apunta a la nada, y el
     renombrado de un .md es exactamente cuando pasa;
  3. un nombre viejo del proyecto sobrevive en algun parrafo despues de un
     renombrado masivo.

Se corre igual en local y en CI:

    python scripts/check_docs.py

Sale con codigo 1 si encuentra algo, que es lo que hace que Actions lo marque.
"""

import io
import os
import re
import sys

# La consola de Windows en espanol usa cp1252 por defecto, y ahi un print con
# cualquier caracter fuera de ASCII --un acento, una flecha-- lanza
# UnicodeEncodeError y mata el script antes de decir nada util. Pasa en local y
# no en CI, que es la peor combinacion: el que lo rompe no lo ve.
for flujo in (sys.stdout, sys.stderr):
    try:
        flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Nombres que el proyecto tuvo antes. Se permiten solo donde se documenta la
# historia del renombrado a proposito.
NOMBRES_VIEJOS = ("xoryn", "duzanix")
PERMITIDOS = {"DECISIONES.md"}


def columnas(linea):
    """Cuantas celdas tiene una fila de tabla de Markdown."""
    t = linea.strip()
    if t.startswith("|"):
        t = t[1:]
    if t.endswith("|"):
        t = t[:-1]
    return len(t.split("|"))


def revisar(ruta, raiz):
    fallos = []
    texto = io.open(ruta, encoding="utf-8").read()
    lineas = texto.split("\n")

    # ── tablas: cada fila contra SU cabecera ────────────────────────────────
    # No contra un numero fijo: este proyecto tiene tablas de 2, 3 y 4
    # columnas, y dar por bueno que todas son de 2 fue un error real.
    dentro = False
    esperadas = 0
    inicio = 0
    for i, linea in enumerate(lineas, 1):
        if linea.strip().startswith("|"):
            if not dentro:
                dentro, esperadas, inicio = True, columnas(linea), i
            else:
                c = columnas(linea)
                if c != esperadas:
                    fallos.append(
                        f"{ruta}:{i}: la fila tiene {c} columnas y la cabecera "
                        f"de la linea {inicio} tiene {esperadas}"
                    )
        else:
            dentro = False

    # ── enlaces relativos ───────────────────────────────────────────────────
    for destino in re.findall(r"\]\((?!https?:|#|mailto:)([^)]+)\)", texto):
        archivo = destino.split("#")[0]
        if not archivo:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(ruta), archivo)):
            fallos.append(f"{ruta}: enlace roto a {archivo}")

    # ── nombres viejos ──────────────────────────────────────────────────────
    if os.path.basename(ruta) not in PERMITIDOS:
        for viejo in NOMBRES_VIEJOS:
            for m in re.finditer(viejo, texto, re.IGNORECASE):
                linea = texto[: m.start()].count("\n") + 1
                fallos.append(
                    f"{ruta}:{linea}: quedo el nombre viejo '{m.group(0)}'"
                )

    return fallos


def niveles(ruta):
    """La secuencia de niveles de encabezado de un documento: [2, 2, 3, 2, ...].

    Se comparan los NIVELES y no los textos porque los dos README estan en
    idiomas distintos. Alcanza igual: si alguien agrega una seccion a uno y se
    olvida del otro, la secuencia deja de coincidir.
    """
    salida = []
    dentro_de_codigo = False
    for linea in io.open(ruta, encoding="utf-8"):
        if linea.startswith("```"):
            dentro_de_codigo = not dentro_de_codigo
        elif not dentro_de_codigo:
            m = re.match(r"^(#{2,6}) ", linea)
            if m:
                salida.append(len(m.group(1)))
    return salida


def revisar_traduccion(raiz):
    """Que los dos README tengan la misma estructura de secciones."""
    a = os.path.join(raiz, "README.md")
    b = os.path.join(raiz, "README.es.md")
    if not (os.path.exists(a) and os.path.exists(b)):
        return []

    na, nb = niveles(a), niveles(b)
    if na == nb:
        return []

    fallos = [
        f"README.md tiene {len(na)} secciones y README.es.md tiene {len(nb)}: "
        "uno de los dos se quedo atras"
    ]
    for i, (x, y) in enumerate(zip(na, nb)):
        if x != y:
            fallos.append(
                f"la seccion {i + 1} es de nivel {x} en README.md y {y} en README.es.md"
            )
            break
    return fallos


def main():
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs = sorted(
        os.path.join(raiz, n) for n in os.listdir(raiz) if n.endswith(".md")
    )
    if not docs:
        print("no encontre ningun .md en la raiz", file=sys.stderr)
        return 1

    todos = []
    for d in docs:
        f = revisar(d, raiz)
        estado = "OK" if not f else f"{len(f)} problema(s)"
        print(f"  {os.path.basename(d):<34} {estado}")
        todos += f

    t = revisar_traduccion(raiz)
    print(f"  {'README.md vs README.es.md':<34} "
          f"{'misma estructura' if not t else f'{len(t)} problema(s)'}")
    todos += t

    if todos:
        print()
        for f in todos:
            print(f"  {f}")
        print(f"\n  {len(todos)} problema(s) en la documentacion")
        return 1

    print(f"\n  {len(docs)} documentos sin problemas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
