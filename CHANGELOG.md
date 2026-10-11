# Changelog

Todas las versiones de Cosxamix. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el versionado es
[SemVer](https://semver.org/lang/es/).

Las decisiones de diseño y sus mediciones viven en [DECISIONES.md](DECISIONES.md);
acá va qué cambió, no por qué.

---

## [0.2.0] - sin publicar

### Averiguado

**Por qué de 6 núcleos salen 2,6x**, que era la pregunta abierta del ROADMAP.
Nuevo `examples/measure_matmul.rs`, que mide cada matriz por separado a
distintos hilos en vez de medir el forward entero, donde todo se mezcla.

Lo que entra en los 12 MB de L3 escala **7,03x**, superlineal; lo que no,
entre 1,76x y 2,82x. El FFN y la salida son el 89% de los pesos, así que el
forward hereda su techo: la predicción ponderada da **2,47x** contra los
**2,60x** medidos. Es el muro de memoria. Más hilos no lo arreglan.

**Si entrenar en esta maquina es viable**, que era la pregunta que bloqueaba el
diseno. Nuevo `examples/measure_lora.rs`.

Primero, lo que decide la arquitectura: **`QMatMul` no implementa `bwd`**, asi
que el gradiente no atraviesa una multiplicacion cuantizada. El backward pide
`dL/dx = dL/dy · W` —multiplicar por `W` sin transponer— y candle solo sabe
hacer `x · Wt`. Entrenar sobre el modelo cuantizado pide un kernel que no
existe.

La alternativa medible es descuantizar la base a f32 y dejar que el autograd
haga su trabajo. Cuesta memoria y eso acota el alcance: el 0,5B en f32 son
~2 GB y entra en los 7,3 GB libres; el 3B serian ~12 GB y no.

Medido sobre una capa del 0,5B con LoRA r=8 en q y v, 256 tokens, y proyectado
a las 24 capas:

| | |
|---|---:|
| forward, una capa | 27,5 ms |
| backward + optimizador | 107,8 ms |
| paso completo, una capa | 135,2 ms |
| **paso del modelo entero** | **~3,3 s** |
| 1000 pasos | **~55 min** |
| parametros entrenables | 540.672, el 0,1% del modelo |

**Ajustar el 0,5B es cosa de una hora, no de dias.** La cifra es un piso: deja
fuera atencion, RoPE y normalizaciones, y la capa de prueba tiene un `cat` que
el modelo real no tiene.

Tambien quedo medido **cuanto del backward es trabajo tirado**. `backprop.rs`
calcula los dos gradientes de cada multiplicacion sin preguntar si hacen falta:

```rust
let lhs_grad = grad.matmul(&rhs.t()?)?;   // dL/dx, se necesita
let rhs_grad = lhs.t()?.matmul(&grad)?;   // dL/dW, con W congelada no
```

Los dos cuestan lo mismo en FLOPs, asi que con la base congelada sobra la
mitad del backward *de las multiplicaciones*. En el paso completo son **23,1 ms
por capa de 135**, o sea un **17%** — real, pero bastante menos que el 50% que
hacia esperar el conteo de operaciones, porque el paso tiene mucho mas que las
siete matrices.

### Añadido

**Gestor de varios modelos** (`src/engine_manager.rs`). Hasta acá Cosxamix
servía **uno**, fijado por `COSXAMIX_GGUF`; pedir otro devolvía
`ModeloDesconocido` aunque el archivo estuviera ahí al lado.

Ahora registra todos los `.gguf` del directorio, carga bajo demanda y desaloja
el menos usado cuando no entra en `COSXAMIX_RAM_MB`. Verificado con un
presupuesto de 2500 MB y cinco modelos:

| | en RAM después |
|---|---|
| inicio | 0,5B = 469 MB |
| pido el 3B Q3 | 0,5B + 3B = 2113 MB |
| pido el 1,5B | solo 1,5B = 1066 MB — desalojó los dos |
| pido el 0,5B | 0,5B + 1,5B = 1534 MB, en 2,1 s |

No reescribe nada: **compone**. Sostiene un `MotorCandle` por GGUF y decide
cuáles están en RAM. La inferencia y el caché por sesión, que es la parte
difícil y la que está medida, no se tocan. Que las sesiones vivan dentro de
cada motor es lo correcto: un caché de atención son estados internos de unos
pesos concretos, y restaurárselo a otro modelo no daría error, daría basura.

El presupuesto va en **bytes y no en cantidad de modelos** porque no pesan
parecido: de 468 a 2007 MB en este directorio, así que un tope de "tres
modelos" dejaría entrar 1,4 GB o 6 GB según cuáles.

Fuera `COSXAMIX_TOKENIZER`: una variable no puede servir a cinco modelos. Cada
GGUF se empareja solo —`<modelo>.tokenizer.json`, después
`tokenizer-<tamaño>.json`, después `tokenizer.json`— y el emparejamiento se
registra en el log, porque un tokenizador equivocado no da error: da texto sin
sentido.

**Respuestas en *streaming*** con `"stream": true`, por SSE y en el dialecto de
OpenAI: un `data:` por trozo con `choices[0].delta`, y un `data: [DONE]` al
final.

Con respuestas de varios segundos esto no es un adorno. Medido con el 3B: el
primer trozo llega a los **2,14 s** —el tiempo del prefill— y el último a los
5,82, o sea **3,68 s de texto apareciendo** en vez de pantalla en blanco.

Un token **no** es un trozo de texto: puede ser media palabra o medio carácter
UTF-8. Así que se decodifica todo lo generado en cada paso y se emite lo que
creció desde la vez anterior, comprobando que el nuevo decodificado empiece por
el anterior. Cuesta O(n²) en el largo de la respuesta y no importa: decodificar
200 tokens son microsegundos al lado de los ~100 ms que cuesta el forward que
produce cada uno.

**Si el cliente corta, la generación se detiene.** El trait `Motor` gana
`generar_en_partes`, cuyo emisor devuelve si hay que seguir; `blocking_send`
falla cuando el receptor ya no está. Verificado: cerrando la conexión a los
1,87 s sobre una petición de 120 tokens, el servidor paró en 5. Sin eso habría
seguido calculando unos doce segundos para nadie.

El método trae implementación por defecto —generar todo y emitir una vez— para
que un motor que no sepa transmitir siga cumpliendo el contrato.

**Entrenamiento LoRA** (`src/training/`, feature `training`). Era la mitad del
objetivo del proyecto y no existía nada: hasta acá Cosxamix solo infería.

```bash
cargo run --release --features training --example train_lora -- corpus.txt 1000
```

| | |
|---|---:|
| parámetros entrenables | 540.672 de 494 millones, el 0,1% |
| paso completo, 0,5B, 256 tokens | ~5,2 s |
| 1000 pasos | ~1,4 h |
| adaptador resultante | **2,1 MB**, 96 tensores |

Verificado entrenando de verdad: la pérdida bajó de 3,9098 a 3,7888.

El modelo de entrenamiento (`src/training/model.rs`) es una reimplementación en
f32 del de inferencia, y **que sean el mismo se comprueba** con
`examples/check_training_model.rs`: compara los logits de los dos caminos sobre
un prompt real y exige que elijan el mismo token. Hace falta porque un error en
RoPE, en el repetido de cabezas KV o en la escala de la atención haría bajar la
pérdida igual, optimizando contra un modelo que no es el que después se sirve.
Medido: 0,984 de diferencia máxima sobre un rango de 30,42 —el 3,2%— y el mismo
token.

Los adaptadores se guardan y se cargan solos, en safetensors, con tres tests
que cubren lo que importa: que un adaptador recién creado **no cambie la
salida** (`B` arranca en cero a propósito), que solo el adaptador sea
entrenable, y que el viaje al disco y de vuelta conserve la salida exacta.

### La frontera de kernels deja de ser solo Rust y Zig

**Ensamblador en archivos propios**: `kernels/dot_avx2_win64.s` y
`kernels/dot_avx2_sysv.s`, incluidos con `global_asm!`. Son **dos** porque el
ensamblador es el único lenguaje de la frontera que tiene que saber la
convención de llamada — Windows pasa en `rcx rdx r8 r9` y System V en
`rdi rsi rdx rcx`. Usan solo registros volátiles, así que no necesitan prólogo
ni epílogo. Los dos se comprueban: el de Windows con los tests, el de System V
con `cargo check --target x86_64-unknown-linux-gnu`.

**Tercer kernel en la frontera: `silu`**, la activación del MLP, que corre
sobre 11008 elementos por capa y por token.

| n | escalar | Zig | |
|---:|---:|---:|---|
| 4.864 | 21,23 µs | **16,88 µs** | 1,26x |
| 11.008 | 47,96 µs | **38,76 µs** | 1,24x |

Tiene dos implementaciones y no tres, y ahí está el dato: **Rust no puede
vectorizar una exponencial sin escribirla a mano** —`f32::exp` es escalar y
aplicarla a un `__m256` pide una aproximación polinómica con reducción de
rango— mientras que el `@exp` de Zig acepta un vector. Es la primera vez en
este proyecto que la ventaja de Zig no es cobertura de plataforma sino
velocidad.

Con una salvedad dicha: **1,25x no es 8x**. Si se vectorizara de verdad sobre
los ocho carriles habría que ver mucho más, así que lo probable es que LLVM lo
expanda a ocho llamadas escalares ahorrando solo el bucle.

### Probado de verdad, no razonado

**Un `dot` con el bucle caliente en ensamblador** (`src/kernels/x86_asm.rs`).
El proyecto venía afirmando que escribir a mano no compraba nada **sin haberlo
escrito**. Ahora está escrito, con el mismo orden de acumulación y la cola
compartida, así que da resultado **bit a bit idéntico** y lo único que separa a
las dos versiones es el bucle.

| n | AVX2 (intrínsecos) | ensamblador | |
|---:|---:|---:|---|
| 1.024 | **49,4 ns** | 50,3 ns | +1,9% |
| 4.096 | **216,0 ns** | 217,1 ns | +0,5% |
| 65.536 | 7,96 µs | **7,79 µs** | −2,1% |
| 1.000.003 | **146,5 µs** | 183,4 µs | **+25%** |

No gana. Queda en el banco pero **fuera del despachador**, porque el resultado
depende de la microarquitectura y hay un caso donde podría darse vuelta:
candle no usa AVX-512 ni SVE, y esta máquina no tiene AVX-512.

**Fortran y Ada no se pudieron probar, y por qué.** El `gfortran` instalado es
MinGW de **32 bits** (`-dumpmachine` = `mingw32`, objeto COFF i386) contra un
Rust de 64, así que los objetos no enlazan; lo desbloquea un gfortran de
`mingw-w64` o `flang`. De Ada no hay compilador instalado. Queda escrito como
"sin comprobar", que no es lo mismo que descartado.

### Medido y NO construido

**Persistir el caché de sesión en disco.** Estaba en el ROADMAP y se midió
antes de escribirlo, que es lo que corresponde después de haber borrado dos
kernels el mismo día por no superar a la referencia.

Ahorra el prefill del *system prompt* en el primer mensaje de cada conversación
tras un reinicio: **~1 s**, medido contra WinsiBot (4905 ms a 3891 ms). Un
reinicio cuesta **45-48 s** de cargar el modelo. Es optimizar un segundo de
cada cincuenta.

Y cuesta: 16 sesiones de 1000 tokens del 3B son **1,1 GB** que escribir al
apagar, un formato en disco que mantener, y un riesgo que no avisa — restaurar
un caché contra otros pesos no da error, da basura.

Queda escrito en el ROADMAP con la condición que lo haría valer la pena: que
reiniciar deje de costar 45 s.

### Corregido

- **El caché de sesión pesa tres veces lo que decía el README.** Son 24,0 KB
  por token en el 0,5B pero **72,0 KB en el 3B**, que es el modelo por defecto:
  36 capas contra 24 y 128 de dimensión por cabeza contra 64. La cifra estaba
  escrita sin decir de qué modelo, así que subestimaba por tres la memoria de
  quien dimensionara un despliegue leyéndola. Nuevo
  `examples/measure_cache_disk.rs` y `CacheAtencion::bytes()`.

### Corregido en los tests

- **Un test que filtraba estado y mentía sobre qué se rompió.** El de
  emparejamiento de tokenizadores usaba `Path::ends_with`, que compara
  **componentes** de ruta y no sufijos de texto, así que una de sus
  aserciones no podía ser cierta nunca. Al fallar ahí no llegaba a limpiar el
  directorio temporal, y la corrida siguiente fallaba en la **primera**
  aserción por culpa de esa basura — señalando el sitio equivocado. Ahora cada
  test borra su directorio **al empezar**, no solo al terminar.

### Probado y borrado, por la regla 3

Dos kernels de multiplicación cuantizada, escritos y medidos contra el de
candle:

| intento | resultado |
|---|---|
| los bucles al revés, reusando la columna de pesos entre filas del lote | 0,91x–1,17x, dentro del ruido |
| descuantizar cada columna una vez y hacer `m` productos en f32 | 0,04x–0,20x |

Los dos pasaban los tests de equivalencia —el primero daba resultado **bit a
bit idéntico** a candle— y los dos se borraron igual.

El segundo explica al primero: `vec_dot_q3k_q8k` desempaqueta el k-quant dentro
de los registros, fundido con el producto, sin materializar nunca los f32.
Separarlo cuesta 8 KB de ida y vuelta por columna y pierde el AVX2 escrito a
mano.

### Corregido en la documentación

- **Una corrección que estaba mal y duró medio día.** Al ver que candle relee
  la matriz de pesos por cada fila del lote se escribió en `DECISIONES.md` §10
  que el motor está limitado por memoria y que la conclusión anterior era
  falsa. Los dos kernels la desmintieron: hay **dos regímenes**, y confundirlos
  fue el error. Con una fila manda la memoria —y eso explica el escalado de
  hilos—; con más de una manda el coste de descuantizar, que es lo que decía la
  conclusión original. Queda escrito el error, no solo el resultado.
- **Una advertencia de método**: los milisegundos absolutos no son estables
  entre corridas, porque el FFN gate pesa 9,7 MB contra 12 MB de L3. La misma
  multiplicación dio 1,10 ms con 150 MB de tensores cargados y 0,24 ms con uno
  solo. Comprobado que no eran ni los pesos reales contra sintéticos ni la
  entrada en cero contra datos.

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

**Doce ejemplos de medición** — `compare_models`, `measure_memory`, `measure_matmul`,
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

- **Dos fallos que solo existen fuera de Windows**, encontrados por el primer CI
  y que ninguna compilación local podía haber encontrado, porque el código que
  rompe es justo el que en el host no se compila:
  1. `examples/measure_memory.rs` define `memoria_mb` dos veces, una por
     plataforma, y a la de Windows **le faltaba su `#[cfg(windows)]`**. En
     Windows compila —allí la que queda fuera es la otra— y en Linux y macOS
     quedan las dos definidas: error `E0428`, no un aviso.
  2. El `mut` de `kernels::disponibles()` no hace falta en un ARM sin la feature
     `zig`, porque ahí no se empuja ningún backend. Con `-D warnings` eso es un
     fallo, y es lo que tiró los dos jobs de macOS, que corren en Apple Silicon.

  De los dos salió una comprobación nueva en CI: el job del cruce ahora corre
  **clippy para `aarch64`** antes de compilar, en las dos combinaciones de
  features. Es más rápido que esperar al runner de macOS y el mensaje es mejor.

- **`check_kernel_arch.py` no entendia Mach-O**, y tiro el primer intento de
  release: de los 6 targets compilaron 5 —Android ARM64 incluido— y
  `aarch64-apple-darwin` cayo en el paso de verificacion con "arquitectura
  encontrada None". El kernel estaba bien; el script solo leia ELF y COFF, y
  macOS usa un tercer formato.

  Reproducido en local cruzando con el mismo Zig: el `.a` de macOS es un
  archivador con objetos Mach-O de 64 bits, `cputype 0x0100000C` para ARM64 y
  `0x01000007` para x86-64.

  De paso se corrigio algo peor que el fallo: la deteccion de COFF **escaneaba**
  el archivo aceptando el primer valor plausible, y eso puede encontrarlo por
  casualidad dentro de los datos y reportar con total confianza una
  arquitectura equivocada. Ahora se parsea el archivador `ar` —incluidos los
  nombres largos al estilo BSD que usa macOS— y se lee la cabecera del primer
  miembro real. Probado contra los cuatro artefactos de verdad: COFF de
  Windows, ELF de ARM, y Mach-O de ARM y de x86-64.

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
