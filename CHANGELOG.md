# Changelog

Todas las versiones de Cosxamix. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el versionado es
[SemVer](https://semver.org/lang/es/).

Las decisiones de diseño y sus mediciones viven en [DECISIONES.md](DECISIONES.md);
acá va qué cambió, no por qué.

---

## [0.1.0] - 2026-10-09

Primera versión. Cosxamix sirve un modelo de lenguaje cuantizado por HTTP hablando
el dialecto de Ollama, y WinsiBot ya responde a través de él.

### Añadido

**Capa HTTP** — servidor `axum` en el puerto `11435`, el de Ollama +1 para que
los dos puedan convivir mientras se migra.

| ruta | qué hace |
|---|---|
| `GET /health` | estado, modelo cargado, sesiones vivas |
| `GET /api/tags` | catálogo de modelos, formato Ollama |
| `POST /v1/chat/completions` | generación, formato OpenAI/Ollama |

Extensión propia: `cosxamix_sesion` en el cuerpo de la petición identifica la
conversación y selecciona su caché de atención. Ollama ignora el campo, así que
el mismo cliente sirve para los dos.

**Motor de inferencia** (`--features candle`) — `MotorCandle` sobre
`candle-core` 0.9.2, Rust puro, sin cadena de herramientas de C++. Carga GGUF
cuantizado, mantiene el modelo caliente entre peticiones y expone el rasgo
`Motor` para que el backend sea reemplazable. Por defecto no se compila: la capa
HTTP sola son 6 crates, con el motor son 155 y ~85 s.

**Caché de atención por sesión** — mapa de sesiones con expulsión LRU
(`MAX_SESIONES = 16`, `MAX_CACHE_TOKENS = 4096`). Cambiar de conversación mueve
punteros, no recalcula. Reaprovecha el **prefijo común más largo** entre lo
cacheado y el prompt nuevo, no solo el caso en que el prompt extiende al
anterior — que es el único que se da cuando el cliente manda `system + pregunta`
sin historial.

**Frontera de kernels** (`src/kernels/`) — tres implementaciones de `dot` y
`rmsnorm` detrás de una misma firma, elegidas en tiempo de ejecución:

| backend | cuándo |
|---|---|
| `Avx2` | x86-64 con AVX2+FMA (el objetivo principal) |
| `Zig` | con `--features zig`; única ruta vectorizada en ARM y Android |
| `Escalar` | referencia y respaldo; acumula en f64 |

`COSXAMIX_KERNEL=escalar|avx2|zig` fuerza uno, para comparar sin recompilar.
Benchmarks de Criterion en `benches/kernels.rs`.

**Kernels en Zig** (`kernels/dot.zig`) — `@Vector(8, f32)` portable, compilados
por `build.rs` con `zig build-lib -O ReleaseFast -mcpu=x86_64_v3`. El mismo
archivo cruza a `aarch64-linux-gnu` y `aarch64-linux-android` sin tocarlo.

**`.cargo/config.toml`** — `target-cpu=x86-64-v3` para los tres targets de
x86_64. No es afinado fino: candle cierra sus kernels fusionados tras
`#[cfg(target_feature = "avx2")]`, que es una condición de **compilación**, así
que sin esto el motor cae al camino escalar y va **4x más lento** sin decir nada.

**Modelo propio** (`src/model/qwen2.rs`) — `quantized_qwen2` de
candle-transformers traído al repositorio (MIT OR Apache-2.0) con todos los
cambios marcados `// COSXAMIX:`. Agrega `tomar_cache`, `poner_cache`,
`limpiar_cache`, `truncar_cache` y `forward_todas`, nada de lo cual expone el
crate.

**Pool de hilos fijado a núcleos físicos** — `num_cpus::get_physical()`, porque
`available_parallelism()` devuelve 12 en una CPU de 6 núcleos y los dos hilos de
un núcleo pelean por la misma unidad vectorial.

**Once ejemplos de medición** — `compare_models`, `measure_memory`,
`measure_weights`, `measure_threads`, `measure_projection`, `measure_interleaved`,
`check_cache`, `check_sessions`, `check_embeddings`, `compare_logits`,
`check_engine`. Cada cifra de la documentación sale de uno de estos; ninguna es
estimada.

**Integración continua y releases** (`.github/`) — dos workflows, y los dos
comprueban cosas que el compilador no ve.

`ci.yml` corre en cuatro frentes: la matriz de Linux, Windows y macOS por las
dos combinaciones de features que tienen que seguir compilando solas; los
kernels de Zig con sus tests de equivalencia contra la referencia escalar; el
**cruce a `aarch64`**, que no está por completitud sino porque es la única
razón por la que Zig está en el proyecto, y si dejara de funcionar los binarios
de ARM saldrían con kernels escalares sin que nada avisara; y la
documentación.

`release.yml` publica **6 targets** al etiquetar, con `fail-fast: false` porque
en WinsiBot tres targets fallaron en un mismo release por tres causas
distintas y con fail-fast solo se habría visto la primera:

| plataforma | kernels | por qué |
|---|---|---|
| Linux x86-64, Windows x86-64, macOS Intel | AVX2 de Rust | Zig pierde 3-8% ahí, medido |
| Linux ARM64, Android ARM64, macOS Apple Silicon | NEON por Zig | es la única ruta vectorizada que existe |

macOS Intel se **cruza desde el runner ARM** en vez de pedir un runner Intel,
que es el que GitHub fue retirando. Android va aparte de `aarch64-unknown-linux-gnu`
porque usa Bionic en vez de glibc y un binario del otro baja bien y falla al
ejecutarse.

**Tres scripts de comprobación** que corren igual en local y en CI:

* `scripts/check_kernel_arch.py` — lee la cabecera ELF o COFF de la librería de
  Zig y comprueba que sea de la arquitectura del target. Hace falta porque que
  compile no prueba nada: con un `-target` mal pasado, Zig produce un
  archivador válido, con los símbolos correctos, lleno de código del host.
* `scripts/check_docs.py` — tablas de Markdown contra su propia cabecera,
  enlaces relativos, y nombres viejos del proyecto sobrevividos a un
  renombrado. Los tres son fallos que pasaron de verdad.
* `.github/actions/instalar-zig/` — baja Zig 0.14.1 del sitio oficial con el
  SHA-256 verificado, sin una action de terceros, porque es lo que decide qué
  compilador entra en el binario que se publica.

**Licencia MIT** (`LICENSE`) y [THIRD-PARTY.md](THIRD-PARTY.md), que registra el
único archivo copiado de otro proyecto: `src/model/qwen2.rs`, de candle, bajo
MIT OR Apache-2.0 — compatible. El `LICENSE-MIT` de candle **no lleva línea de
copyright**, empieza directo en "Permission is hereby granted", así que se
enlaza a los textos originales en vez de inventar un titular.

**README en dos idiomas**: [README.md](README.md) en inglés como entrada, y
[README.es.md](README.es.md) en español. Los otros cuatro documentos siguen
solo en español a propósito — son 11.500 palabras de mediciones con su
contexto, y mantener dos copias de un número es la forma fiable de terminar con
dos números distintos.

Para que los dos README no se desincronicen en silencio, `check_docs.py`
compara la **secuencia de niveles de encabezado** entre ambos. No compara
textos, que están en idiomas distintos; alcanza con la estructura: si alguien
agrega una sección a uno y se olvida del otro, CI falla.

### Cambiado

- **Modelo por defecto: Qwen2.5-3B-Instruct Q3_K_M**, sobre 0,5B y 1,5B. En 15
  preguntas de respuesta verificable: 3B acierta 10, el 1,5B ocho y el 0,5B
  siete. El 1,5B cuesta 1,7x de velocidad y gana una pregunta de quince.
- **Cuantizado Q3_K_M**, sobre Q4_K_M: mismos 10/15 con 363 MB menos y 10% más
  rápido. Q2_K baja a 6/15 y su manera de fallar es mala — responde
  *"canossa"* por Canberra y mete acentos franceses en español.

### Corregido

- **Embeddings: 938 → 506 MB residentes.** `from_gguf` descuantizaba la tabla
  entera a f32 (519 MB, más que el modelo completo) cuando un forward solo
  necesita las filas de los tokens que entran, 3,5 KB cada una. Relación
  RAM/disco de x2,0 a **x1,03**.
- **Máscara causal con caché.** La original era cuadrada `(t, t)` y solo valía
  con el caché vacío; impedía procesar el delta por lotes. Ahora construye
  `(t, cacheados + t)`. El banco de intercalado bajó de 5422 a **4772 ms**.
- **Uso de memoria liberada en candle** (`as_t_slice`) — toma el `Cow` por valor
  y devuelve un slice hacia él, así que con `Cow::Owned` el `Vec` se libera al
  salir. Síntoma: de siete filas pedidas, solo se corrompía la primera, la que
  pisa el asignador con su lista de libres. Se evita pasando `Cow::Borrowed`.
- **`tokenizers` con `default-features = false`** — el feature `onig` por
  defecto arrastra Oniguruma, una librería de C, y rompe el cruce a Android con
  un error que no nombra ni Android ni el feature.
- **Compilación cruzada de los kernels de Zig** — `-femit-bin=<ruta>` se pasaba
  como dos argumentos, así que Zig escribía con el nombre por defecto y el
  enlazador no encontraba nada.
- **Alineación en los kernels de Zig** — las cargas usaban `@alignCast` sobre
  slices que vienen de Rust, que no garantizan alineación a 32 bytes: era
  comportamiento indefinido latente. Ahora `align(@alignOf(f32))`.

- **`build.rs` no podía compilar los kernels para otro target**, con dos fallos
  independientes que solo aparecen al cruzar, así que ninguno se veía
  compilando a diario:
  1. nombraba la librería con `cfg!(target_os = "windows")`, y un build script
     se compila para el **host**: sus `cfg!` describen la máquina que
     construye, no la que va a ejecutar. Cruzando de Linux a Windows pedía
     `libcosxamix_kernels.a` mientras Zig escribía un `.lib`, y el linker no
     encontraba nada por un motivo que no se parece a la causa. Ahora sale de
     `CARGO_CFG_TARGET_OS`.
  2. no pasaba `-target` a Zig, así que compilaba siempre para el host. Al
     agregarlo apareció el segundo: **Zig quiere `-target X` en dos
     argumentos**, al revés que `-femit-bin=X`, que va en uno — y el error que
     da, `unrecognized parameter`, no insinúa que el problema sea la forma de
     pasarlo. Verificado: la librería cruzada es ELF AArch64 con los dos
     símbolos exportados.

  Sin esto, publicar binarios de ARM con `--features zig` habría hecho falsa la
  afirmación de que Zig está para cubrir ARM y Android.

- **Formato con `cargo fmt`** en los 20 archivos que no lo cumplían, y la
  comprobación puesta en CI. La mitad de las diferencias las causó el
  renombrado del crate: `cosxamix` ordena antes que `criterion`
  alfabéticamente, donde el nombre anterior iba después, así que rustfmt quería
  invertir los `use`.

- **`.gitignore`**: le faltaban las cachés de Zig (`.zig-cache/`,
  `zig-cache/`, `zig-out/`), que Zig deja en la raíz del crate porque es donde
  se invoca a `build.rs`, y las salidas de medición (`perf.data`,
  `flamegraph.svg`, `*.log`). Queda escrito en el propio archivo por qué
  `Cargo.lock` **sí** va al repositorio.

### Eliminado

- **Decodificación especulativa** (`src/motor_especulativo.rs`) — borrador 0,5B
  verificado por el 3B. Implementada entera y medida: **1,91x más lenta**
  (46.045 ms contra 24.128). La técnica supone que procesar K posiciones cuesta
  casi lo mismo que una; medido, 5 posiciones cuestan 3,74x, o sea que el motor
  está limitado por cálculo y no por memoria, que es lo único que la especulativa
  acelera. Se fue por la regla 3.

  Lo que dejó: la máscara causal arreglada y `truncar_cache`, sin la cual el
  caché por prefijo común no habría sido posible.

### Notas

- `cargo test` corre con y sin features; los ejemplos llevan
  `required-features` para que `cargo test` a secas no falle intentando
  compilarlos.
- Verificado en esta versión: **22 tests**, 0 avisos, clippy limpio en las tres
  combinaciones (por defecto, `candle`, `zig,candle`).

[0.1.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.1.0
