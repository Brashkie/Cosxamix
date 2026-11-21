# Cosxamix

*[English](README.md) · **Español***

Runtime de IA propio. Sirve modelos de lenguaje por HTTP, hablando el mismo
dialecto que Ollama para poder reemplazarlo sin tocar a quien lo llama.

> **Estado: inferencia real andando, y WinsiBot ya responde a través de ella.**
> Sirve un modelo GGUF por HTTP con el dialecto de Ollama, precalentado al
> arrancar, con caché de atención por conversación que reutiliza el prefijo
> común, respuestas en *streaming*, varios modelos a la vez con desalojo por
> uso, y ajuste fino con LoRA.

## Los documentos

| | |
|---|---|
| este archivo | cómo funciona, cómo se usa, y las mediciones del motor |
| [README.md](README.md) | lo mismo en inglés |
| [DECISIONES.md](DECISIONES.md) | por qué cada cosa es como es, con la medición que lo sostiene — incluidos los resultados negativos y las cifras que hubo que corregir |
| [CHANGELOG.md](CHANGELOG.md) | qué cambió en cada versión |
| [ROADMAP.md](ROADMAP.md) | qué falta, en qué orden, y qué quedó cerrado |
| [cosxamix-deep-theory-guide.md](cosxamix-deep-theory-guide.md) | la teoría de fondo; cada sección marca si está implementada o es diseño previsto |
| [THIRD-PARTY.md](THIRD-PARTY.md) | el único archivo copiado de otro proyecto, de dónde sale y por qué |

---

## Por qué existe, y qué NO es

Hay un malentendido que conviene sacar del medio antes de escribir una línea:
**Ollama no implementa inferencia.** Ollama es una capa de servicio —registro
de modelos, carga y descarga, una API HTTP— montada sobre `llama.cpp`, que es
quien hace el trabajo numérico.

Eso define qué tiene sentido construir acá y qué sería tirar años a la basura:

| capa | quién la hace | ¿propia? |
|---|---|---|
| API HTTP, gestión de modelos, caché, planificador | **Cosxamix** | **sí** |
| Ejecución del grafo, atención, cuantización, imágenes, entrenamiento | **Cosxamix** — sin `candle` desde la 0.11.0 | **sí** |
| Kernels sueltos del camino caliente | Cosxamix, **solo donde el perfilado lo justifique** | puede |

**Esa segunda fila decía "no" hasta la 0.3.0, y lo que la cambió fueron
mediciones, no ambición.** Apoyarse en candle salía barato y resultó ser un
techo en cinco sitios a la vez, todos comprobados en este repositorio:

| | |
|---|---|
| `QMatMul` no implementa `bwd` | el gradiente no atraviesa una multiplicación cuantizada, así que entrenar el 3B pide 12 GB en f32 contra 7,3 libres |
| su `matmul` relee la matriz entera por cada fila del lote | 8 filas cuestan 8,55x una, y eso es lo que mató a la decodificación especulativa |
| no usa AVX-512 ni SVE | `avx.rs` es de 256 bits y `neon.rs` de 128 |
| no soporta los formatos IQ | su enumeración llega a los k-quants clásicos |
| su backprop calcula `dL/dW` de capas congeladas | 17% del paso de entrenamiento, tirado |

Ninguno de los cinco se arregla desde afuera. Y la superficie que de verdad se
usa es estrecha —`Tensor`, `QMatMul`, el lector de GGUF y una docena de
funciones sueltas—, así que reemplazarla es un trabajo acotado y no una
reescritura de `llama.cpp`.

Se hizo **pieza por pieza, con candle de oráculo**: cada parte propia se
comparó contra la suya sobre los modelos reales antes de reemplazarla. En la
0.11.0 salió del todo, también de los tests: lo que calculó candle para ellos
quedó guardado (`tests/fixtures/ops_candle.json`), y el entrenamiento se
comprueba contra diferencias finitas, que no necesitan a nadie.

La capa de servicio sigue siendo donde están las decisiones que más mueven la
latencia percibida: que el modelo no se descargue, que el prompt no se
recalcule entero, que una petición no espere detrás de otra.

**Crear modelos sí está en el alcance; crear uno grande, no.** Un modelo de
mil millones de parámetros son miles de horas de GPU y cientos de GB de texto,
y eso no sale de una máquina sin GPU. Uno **chico y especializado** sí, y es
coherente con el resto del proyecto: si el objetivo es que ande en un teléfono,
un modelo de 8 millones de parámetros entrenado para una tarea le gana a uno de
3000 millones que no entra.

Hay un detalle que decide el orden de ese trabajo, y es aritmética, no opinión.
Con el vocabulario de Qwen —151.936 entradas— un modelo de 256 de ancho y 6
capas sería **87% tabla de embeddings**: 38,9 M de 44,6 M. No sería un modelo,
sería una tabla de búsqueda con un transformer de adorno. Con un vocabulario
propio de 8.192, la misma forma son 7,8 M con el embedding al 27%.

O sea que **crear un modelo empieza por crear un tokenizador**. Está en el
[ROADMAP](ROADMAP.md) §8 con las cifras.

---

## El contrato, que es el primer hito

Cosxamix nació para que WinsiBot deje de depender de un servicio externo. WinsiBot
le habla a Ollama por **exactamente dos endpoints**, comprobado leyendo su
`ai_chat.rs`:

```
GET  /api/tags              → {"models":[{"name":"..."}]}
POST /v1/chat/completions   → formato OpenAI
```

Así que ese es el objetivo mínimo: implementar esos dos y que WinsiBot funcione
**sin cambiar una línea**, solo apuntando su `OLLAMA_URL` a Cosxamix.

Se eligió imitar el dialecto ajeno en vez de inventar uno propio por una razón
concreta: permite cambiar el motor de abajo —o volver a Ollama— sin tocar al
que consume. Un formato propio obligaría a modificar cada cliente y a
mantenerlo para siempre, a cambio de nada.

```bash
# En WinsiBot, sin tocar código:
OLLAMA_URL=http://127.0.0.1:11435
```

---

## Uso

```bash
cargo run --release
```

```bash
curl http://127.0.0.1:11435/api/tags

curl -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"cosxamix-demo","messages":[{"role":"user","content":"hola"}]}'
```

Con `"stream": true` la respuesta va por SSE en el dialecto de OpenAI, un
`data:` por trozo y un `data: [DONE]` al final. Medido con el 3B: el primer
trozo a los **2,14 s** y el último a los 5,82, o sea **3,68 s de texto
apareciendo** en vez de pantalla en blanco. Y si el cliente corta, la
generación se detiene: comprobado cerrando a los 1,87 s sobre una petición de
120 tokens, el servidor paró en 5.

```bash
curl -N -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hola"}]}'
```

Variables de entorno:

| | por defecto | |
|---|---|---|
| `COSXAMIX_PORT` | `11435` | Uno distinto al 11434 de Ollama, para que convivan |
| `COSXAMIX_MODELOS` | `./modelos` | Dónde buscar los archivos de modelo |
| `COSXAMIX_GGUF` | `qwen2.5-3b-instruct-q3_k_m.gguf` | Cuál se precalienta al arrancar |
| `COSXAMIX_RAM_MB` | `4096` | Presupuesto de memoria para los modelos cargados |
| `RAYON_NUM_THREADS` | núcleos físicos | Hilos del cálculo matricial |
| `COSXAMIX_SIN_BLOQUES`, `COSXAMIX_SIN_Q6K_FUSIONADO`, `COSXAMIX_SIN_VARIAS`, `COSXAMIX_SIN_PRESTAMO` | sin definir | Cada una apaga una optimización de la 0.7.0 —la matmul por bloques del prefill, el kernel de Q6_K, q-k-v en una pasada, el prefijo entre conversaciones—, para medir con y sin ella ([DECISIONES §24-26](DECISIONES.md)) |
| `COSXAMIX_PRECARGA` | `2` | Cuántas filas por delante pide el decode a la caché mientras calcula una; `0` la apaga ([DECISIONES §33](DECISIONES.md)) |
| `COSXAMIX_SIN_Q5K_FUSIONADO` | sin definir | Apaga el kernel fusionado de Q5_K, para medir con y sin él ([DECISIONES §34](DECISIONES.md)) |
| `COSXAMIX_IMAGEN_LIBERAR_MIN` | `10` | Minutos sin pedidos de imagen antes de liberar el modelo (2,6 GB de pico) |

**Todos los `.gguf` del directorio se registran**, no solo el de
`COSXAMIX_GGUF`: esa variable dice cuál se precalienta, no cuál es el único
que existe. Pedir otro por `model` lo carga, desalojando el menos usado si no
entra en el presupuesto.

El presupuesto va en **bytes y no en cantidad de modelos** porque no pesan
parecido: en un directorio normal van de 468 a 2007 MB, y un tope de "tres
modelos" dejaría entrar 1,4 GB o 6 GB según cuáles sean. Se mide por el tamaño
del archivo, que desde que los embeddings dejaron de descuantizarse es buena
aproximación: la relación RAM/disco quedó en x1,02-x1,08.

Ya no hay `COSXAMIX_TOKENIZER`: una variable no puede servir a cinco modelos.
Cada GGUF se empareja solo, buscando primero `<modelo>.tokenizer.json`,
después `tokenizer-<tamaño>.json` y por último `tokenizer.json`. El
emparejamiento **se registra en el log al arrancar**, porque un tokenizador
equivocado no da error: da texto sin sentido.

---

## Compilar, probar y publicar

```bash
cargo run --release --features zig,c                # texto e imágenes, con los kernels de Zig y C: 80 crates
```

Lo que comprueba cada cosa, y se corre igual en local que en CI:

```bash
cargo fmt --all -- --check
cargo clippy --features zig,c,training --all-targets -- -D warnings
cargo test --features zig,c,training                 # ninguno necesita un modelo
python scripts/check_docs.py                         # tablas, enlaces y nombres viejos
python scripts/check_kernel_arch.py "" x86_64        # que el kernel sea del target
```

**`RUSTFLAGS` no se usa para nada acá, y es a propósito.** Esa variable de
entorno *reemplaza* a los `rustflags` de `[target.*]` de `.cargo/config.toml`
en vez de sumarse: un `RUSTFLAGS=-D warnings` borraría el
`target-cpu=x86-64-v3`, que son [4x medidos](#el-hallazgo-más-grande-target-cpu),
y nada lo diría — compila, pasa los tests, y va cuatro veces más lento. Los
avisos se deniegan pasándoselos a clippy como argumento suyo.

### Los dos workflows

`ci.yml` corre en cada push: la matriz de Linux, Windows y macOS por las dos
combinaciones de features, los kernels de Zig con sus tests de equivalencia,
**el cruce a `aarch64`**, y la documentación.

El job del cruce no está por completitud. Zig está en el proyecto porque es la
única ruta vectorizada en ARM y Android; si cruzar dejara de funcionar, los
binarios de ARM saldrían con kernels escalares —19 veces más lentos— sin que
nada avisara. Ya atrapó un fallo real, que `build.rs` pasaba `-target=X` en un
argumento y Zig lo quiere en dos.

`release.yml` publica 6 targets al etiquetar:

| plataforma | kernels | por qué |
|---|---|---|
| Linux x86-64 · Windows x86-64 · macOS Intel | AVX2 de Rust | Zig pierde 3-8% ahí, [medido](#lo-que-midió-el-primer-kernel) |
| Linux ARM64 · Android ARM64 · macOS Apple Silicon | NEON por Zig | es la única ruta vectorizada que existe |

Un archivo por plataforma, sin DLL ni `.so` al lado, cada uno con su
`.sha256`. macOS Intel se cruza desde el runner ARM en vez de pedir un runner
Intel, que es el que GitHub fue retirando. Android va aparte de
`aarch64-unknown-linux-gnu` porque usa Bionic en vez de glibc, y un binario del
otro baja bien y falla al ejecutarse.

---

## Lo que sigue

Está en **[ROADMAP.md](ROADMAP.md)**, con el estado medido arriba y el orden
puesto por cuánto mueve la latencia, no por cuán interesante es de programar.

En resumen: **leer el prompt** ya va 1,7x más rápido que en la 0.6.0, y el
comienzo del *system* se lee una sola vez para todos los chats (0.7.0).
**Generar** está al **~81-87% del techo de memoria** medido (~30 GB/s,
DDR4-2133), con la precarga de la 0.9.0 ([DECISIONES §33](DECISIONES.md)); el
51% que se decía antes se calculaba contra un techo que esta RAM no puede dar. Lo
que queda es poco y está en los kernels. Que de 6 núcleos salgan 2,6x en vez de
6x **ya tiene explicación**: un hilo solo lee 19 GB/s y con 4 la RAM se satura.
La decodificación especulativa sigue fuera: con el motor cerca del techo, la
cuenta todavía no da (ROADMAP §5).

Más adelante, en un proyecto aparte, va a nacer **QCML** (`.qcml`, *Quantum
Compression for Machine Learning*): un formato propio que quiere **superar a GGUF
en todos los aspectos**, medido aspecto por aspecto. Guarda las redes de tensores
—la matemática de los estados cuánticos, sin cómputo cuántico— como núcleos y no
como matrices densas, para que el ahorro de la 0.8.0 —tablas 30,6× más chicas—
llegue también al archivo (ROADMAP §11).

---

## daiss: los modelos propios

Cosxamix no solo sirve modelos ajenos. **daiss** es la familia propia, cinco
tamaños que el proyecto define, inicializa y escribe como GGUF:

| modelo | parámetros | Q8_0 | caché/token |
|---|---:|---:|---:|
| daiss-mini | 0,08 B | 0,08 GB | 6,0 KB |
| daiss-0.5b | 0,63 B | 0,62 GB | 24,0 KB |
| daiss-3b | 3,40 B | 3,36 GB | 72,0 KB |
| daiss-7b | 7,61 B | 7,54 GB | 112,0 KB |
| daiss-14b | 14,77 B | 14,62 GB | 384,0 KB |

```
cargo run --release --example create_daiss
COSXAMIX_DAISS=daiss-7b cargo run --release --example create_daiss
```

Se generan en flujo y en paralelo, con el pico de memoria acotado a 16 MB sea
cual sea el tamaño, y con semilla: la misma semilla da el mismo archivo byte
por byte. El daiss-7b está creado y candle lo lee con cero diferencias.

Y con el tokenizador propio —`src/tokenizer.rs`, un BPE a nivel de bytes
entrenado por el proyecto— el vocabulario baja de 151.936 piezas a 8.192 a
cambio de un **3,4% más de tokens** medido sobre texto no visto, y daiss-mini
pasa de **84 MB a 9 MB**:

```
cargo run --release --example train_tokenizer
COSXAMIX_BPE=modelos/daiss-8192.bpe cargo run --release --example create_daiss
```

El tokenizador viaja dentro del archivo del modelo.

**Chavín** es el modelo grande: `chavin-30b`, 29,5 B de parámetros, medido para una
máquina de 16 GB. Ahora los modelos se pueden escribir en k-quants, y él va en
Q2_K, 9,02 GB:

```
COSXAMIX_DAISS=chavin-30b cargo run --release --example create_daiss
COSXAMIX_TIPO=q3k COSXAMIX_DAISS=daiss-3b cargo run --release --example create_daiss
```

Corre acá a 2,38 tok/s, el 57% de lo que la memoria permite, gracias a un kernel
fusionado de Q2_K escrito en Rust, ensamblador, C y Zig.

Y ahora se pueden entrenar desde cero, en esta CPU: `examples/train_daiss.rs` entrena
todos los parámetros, y el GGUF exportado se sirve por la API con su propio
vocabulario, sin JSON. Como no tiene plantilla de chat, los mensajes le llegan uno
detrás de otro y los continúa. Un daiss-mini llega a 2,03 bits por byte sobre texto no
visto en 10 minutos, por debajo de `xz` (2,15), y a 1,77 en 30:

```
COSXAMIX_VOCAB=8192 cargo run --release --features training --example train_daiss -- corpus.txt
```

Escribe frases que parecen español y todavía no quieren decir nada.

**Un daiss o un Chavín recién escrito no está entrenado**: sus pesos son ruido con la
desviación correcta y genera texto sin sentido. Lo que existe y está
verificado es el camino entero —definir, inicializar, cuantizar, escribir,
leer, cargar—, que es la condición previa de cualquier entrenamiento. Lo que
falta está en [ROADMAP.md](ROADMAP.md) §8.

---

## El motor: propio, y de dónde viene

Cosxamix nació sobre **candle** (de Hugging Face), elegido midiendo: Rust puro,
sin el toolchain de C++ que pide llama.cpp y que en Termux es justo el requisito
que se quería quitar. Desde la 0.3.0 se fue reemplazando pieza por pieza —el
lector de GGUF, la cuantización, las multiplicaciones, la atención, el forward—,
siempre comparando contra candle hasta dar lo mismo. En la 0.6.0 el motor propio
pasó a ser el de por defecto: 11 de 15 en la batería de preguntas contra 10, y
1,6-1,9x más rápido por mensaje ([DECISIONES §23](DECISIONES.md)). **En la
0.11.0 candle salió del todo**: el texto, las imágenes (CLIP, el UNet de SD y
TAESD sobre una GEMM propia, 2,3 veces más rápido y con 1,6 GB menos) y el
entrenamiento (el backward escrito a mano, 1,9 veces más rápido en LoRA) son
propios. El binario pesa 3,25 MB en vez de 7,3, con las imágenes adentro, con 80 crates en vez de 139.

Las mediciones de esa época —el caché de atención, la tabla de embeddings, la
especulativa que se borró— están en [DECISIONES.md](DECISIONES.md) y en el
[CHANGELOG](CHANGELOG.md), de la 0.1.0 a la 0.10.0. Lo que sigue valiendo:

### El hallazgo más grande: `target-cpu`

La primera medición dio **1,8 tokens/s** con un modelo de 0,5B. **Rust compila
por defecto para el x86-64 básico de 2003**, sin AVX2. Mismo código, mismo
modelo, solo cambia la bandera:

| | por respuesta | tokens/s |
|---|---:|---:|
| genérico (por defecto) | 4428 ms | 1,8 |
| **`x86-64-v3`** (AVX2+FMA) | **1094 ms** | **6,1** |
| `native` | 1066 ms | 6,3 |

Está fijado `x86-64-v3` en `.cargo/config.toml`, y no `native`, que ata el
binario a la CPU que compila. **El coste**: no arranca en un x86 anterior a
2015. Para esos:

```bash
# Para una CPU sin AVX2 (Celeron, Pentium, Atom...). RUSTFLAGS reemplaza al config.
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release --features zig
```

Con `zig`, esa compilación lleva los kernels de Zig dos veces —en SSE2 y en
AVX2— y elige al arrancar: en una CPU vieja usa la de SSE2, de **2,4x a 12x**
más rápida que la referencia escalar ([DECISIONES §20](DECISIONES.md)).

### Qué modelo y qué cuantización

El 3B en Q3_K_M, medido con las mismas 15 preguntas de respuesta verificable
(`examples/compare_models.rs`):

| | disco | aciertos |
|---|---:|---:|
| 0,5B Q4_K_M | 504 MB | 7/15 |
| 1,5B Q4_K_M | 1108 MB | 8/15 |
| 3B Q2_K | 1313 MB | 6/15, se rompe |
| **3B Q3_K_M** | **1644 MB** | **10/15** |
| 3B Q4_K_M | 2007 MB | 10/15 |

Q3_K_M acierta lo mismo que Q4_K_M con 363 MB menos, y menos bytes por peso son
menos bytes que leer de la RAM en cada token. El 3B es la **Qwen Research
License** (no comercial); el 0,5B y el 1,5B, Apache-2.0
([THIRD-PARTY.md](THIRD-PARTY.md)).

### Hilos, caché y memoria

- **Hilos: los núcleos físicos, no los lógicos.** De 6 a 12 hilos en un
  i5-10400 el decode gana un 2% por 54% más de CPU: los dos hilos de un núcleo
  pelean por la misma unidad AVX2. `RAYON_NUM_THREADS` lo cambia.
- **Caché de atención por sesión**: cada conversación guarda el suyo, y el
  delta de un mensaje nuevo entra sin recalcular lo anterior.
- **Memoria ≈ el archivo**: los pesos se mapean y la tabla de embeddings se lee
  fila por fila; un modelo ocupa en RAM 1,02-1,08x lo que pesa en disco.
- **La decodificación especulativa** se implementó, se midió y se borró en la
  época de candle; cuándo vuelve a tener sentido está en el
  [ROADMAP](ROADMAP.md) §5.

## Varios lenguajes, una sola frontera

Cosxamix no se escribe solo en Rust. Rust es el principal, y la frontera está
abierta al que gane la medición:

| | qué aporta | qué falta |
|---|---|---|
| **Zig** | una sola fuente que cubre x86-64, ARM64 y Android | nada: ya está dentro |
| **Ensamblador** | el kernel fusionado de Q4_K en x86-64, primero en el despachador con `asm_q4k` (1,04x-1,08x sobre los intrínsecos). El de Q3_K (`asm_q3k`) empata con los intrínsecos, igual bit a bit | el cuerpo NEON, `kernels/q4k_neon.s`, y medirlo en ARM |
| **Fortran** | arreglos sin aliasing, que es por lo que BLAS y LAPACK siguen siendo Fortran | un compilador en cada plataforma, y Android es justo el problema |
| **Ada / SPARK** | demostrar que un kernel no desborda, en vez de comprobarlo con tests | un kernel lo bastante crítico como para que valga el toolchain |
| **C** | dentro con la feature `c`, compilado con `zig cc` y sin libc. Extensiones de CPU por función (`__attribute__((target))`), y en ARM `vdotq_s32` —`sdot`— estable, que Rust no expone y Zig no genera. En x86 su Q3_K pierde un 4% con Rust ([DECISIONES §21](DECISIONES.md)) | la cifra de un teléfono: en el runner ARM de CI los fusionados con `sdot` ganan 2,4x (Q4_K) y 5,4x (Q3_K), y desde la 0.5.1 están en el despachador |

El riesgo de eso no son los lenguajes: es tener **N sitios distintos** por donde
cada uno se mete. Por eso hay una sola frontera, en `src/kernels/`, y tres
reglas que la gobiernan.

1. **Todo kernel tiene una implementación de referencia en Rust portable.**
   Define qué resultado es el correcto y corre cuando la CPU no tiene nada
   mejor. Nunca se borra.
2. **Toda implementación se compara contra la referencia**, en los tests. Un
   kernel que va más rápido y da otro número no es más rápido: está mal.
3. **Toda implementación tiene que justificarse midiendo**: ganarle a la
   referencia en velocidad, **o** habilitar una plataforma donde no había
   aceleración. Si no hace ninguna de las dos, se borra — no se mantiene un
   backend porque ya esté escrito.

El backend se elige **en tiempo de ejecución**, mirando lo que la CPU soporta.
Un binario compilado donde hay AVX-512 y ejecutado donde no lo hay se cae con
instrucción ilegal; detectar al arrancar es lo que permite publicar **un solo
binario** que anda en todas y usa lo mejor de cada una. `COSXAMIX_KERNEL` fuerza
uno, para poder medir uno contra otro en la misma máquina.

```bash
cargo test                      # referencia y AVX2
cargo test  --features zig      # y además Zig
cargo bench --features zig      # la regla 3, ejecutable
cargo test  --features asm_q4k  # los kernels escritos en ensamblador
cargo test  --features asm_q3k  # el Q3_K en ensamblador
cargo test  --features c        # los kernels en C (pide Zig, por `zig cc`)
```

### Lo que midió el primer kernel

`dot` (producto punto `f32`), i5 con AVX2+FMA sin AVX-512, mediana de Criterion:

| n | escalar | AVX2 (Rust) | Zig | Zig vs AVX2 |
|---|---:|---:|---:|---:|
| 1.024 | 1,039 µs | **54,1 ns** | 58,2 ns | +7,5% |
| 4.096 | 4,631 µs | **233,5 ns** | 252,3 ns | +8,0% |
| 65.536 | 75,9 µs | 8,53 µs | **8,50 µs** | empate |
| 1.000.003 | 1,633 ms | **933 µs** | 1,010 ms | +8,2% |

Dos conclusiones, y la segunda es la que importa:

**En x86-64, Zig no gana.** Queda un 8% por detrás de los intrínsecos AVX2
escritos a mano, y empata en un solo tamaño. Si el criterio fuera solo
velocidad en esta máquina, la regla 3 lo borraría. Pero la regla contempla la
cobertura, y por eso se queda:

**Pero el kernel de Zig es UNA fuente que cubre x86-64, ARM64 y Android.**
Comprobado: el mismo `kernels/dot.zig`, sin tocar una línea, compila para
`aarch64-linux-gnu` y para `aarch64-linux-android` —el target de Termux—. El
camino de Rust es `core::arch::x86_64`: **solo x86**. En un teléfono, hoy, cae
a la referencia escalar, que es 19 veces más lenta.

Y desde la 0.11.0, sin candle, eso vale también para la multiplicación de
matrices cuantizadas, que es donde está el 99% del trabajo: en ARM la hacen el
kernel de C con `sdot` (feature `c`) y el de Zig (feature `zig`). Hasta la
0.10.0 la cubría el `neon.rs` de candle; hoy, un ARM compilado sin `zig` ni `c`
cae a la referencia escalar en todo. Por eso los binarios de ARM se publican
con `zig,c`.

O sea que lo que Zig aporta acá no es velocidad: es **cobertura**. Un kernel
portable que rinde casi como el específico es mejor que un kernel específico
que en media flota no existe. Esa es la razón por la que se queda — y está
escrita para que, si alguien escribe el camino NEON en Rust, se vuelva a medir
y se decida otra vez.

### Segundo kernel: `rmsnorm`

Entró para contestar una pregunta que el primero no podía: **¿la frontera
generaliza?** Una que funciona para un caso puede no servir para otro.

`rmsnorm` tiene otra forma a propósito — dos pasadas sobre los datos, una raíz
cuadrada, tres vectores en vez de dos, y escribe en un buffer de salida en vez
de devolver un escalar. **No hubo que cambiar nada del mecanismo de
selección**: misma enumeración, mismo despacho, misma FFI de punteros y
longitud. Eso era lo que se quería saber.

Tamaños realistas: la dimensión de un LLM chico va de 2048 a 8192. Medirlo con
vectores de millones no diría nada, porque en un transformer nunca se llama así.

| n | escalar | AVX2 (Rust) | Zig | Zig vs AVX2 |
|---|---:|---:|---:|---:|
| 2.048 | 2,874 µs | **294,6 ns** | 375,4 ns | +27% |
| 4.096 | 6,152 µs | **759,4 ns** | 799,1 ns | +5,2% |
| 8.192 | 12,185 µs | **1,970 µs** | 2,033 µs | +3,2% |

Mismo patrón que en `dot`: AVX2 adelante, Zig a un 3-27% según el tamaño, y
los dos muy por encima de la referencia.

**Una advertencia sobre esa última columna**, para no sacar más de lo que dice:
la referencia acumula la suma de cuadrados en `f64`, a propósito, porque su
trabajo es ser correcta y en `f32` pierde precisión con vectores largos. Así
que el 6-10x contra ella **no es solo vectorización**: parte es la diferencia
de precisión. La comparación limpia entre lenguajes es la de las dos últimas
columnas, que sí acumulan igual.

### Tercer kernel: `silu`, y la primera vez que Zig gana por velocidad

`silu` es la activación del MLP —`x * sigmoid(x)`— y el MLP es el 79% de los
pesos, así que corre sobre 11008 elementos por capa y por token.

Tiene **dos** implementaciones y no tres, y esa es la parte interesante:

| n | escalar | Zig | |
|---:|---:|---:|---|
| 4.864 | 21,23 µs | **16,88 µs** | 1,26x |
| 11.008 | 47,96 µs | **38,76 µs** | 1,24x |

No hay camino AVX2 porque **Rust no puede vectorizar una exponencial sin
escribirla a mano**: `f32::exp` es escalar, y aplicarla a un `__m256` obliga a
una aproximación polinómica con reducción de rango — unas cuarenta líneas
delicadas donde un coeficiente mal puesto no rompe nada, solo da respuestas un
poco peores. El `@exp` de Zig acepta un vector y baja al intrínseco de LLVM.

Es la primera vez que la ventaja de Zig acá no es "corre en ARM" sino **el
lenguaje trae la herramienta**.

Dicho eso, 1,25x no es 8x. Si `@exp` se vectorizara de verdad sobre los ocho
carriles habría que ver mucho más, así que lo más probable es que LLVM lo
expanda a ocho llamadas escalares ahorrando solo el bucle. Es un kernel que
vale la pena volver a medir cuando cambie la versión de Zig.

### Una trampa que costó una medición

La primera comparación daba a Zig perdiendo entre un 3% y un **26%**. El
motivo no era el lenguaje: `build.rs` invocaba a `zig build-lib` sin `-mcpu`,
así que compilaba para el x86-64 básico de 2003 y su `@Vector(8, f32)` se
partía en operaciones de 128 bits. Con `-mcpu=x86_64_v3` —que es exactamente
AVX2+FMA, lo mismo que habilita el `#[target_feature]` de Rust— la diferencia
bajó a un 8%. Comparar sin igualar el target es comparar dos targets, no dos
lenguajes.

---

## Entrenar sin GPU

Cosxamix ya no solo infiere. Con la feature `training` ajusta un modelo con
**LoRA**: entrena el 0,1% de los parametros en vez del 100%.

```bash
cargo run --release --features training --example train_lora -- corpus.txt 1000
```

Sale un `adaptador.safetensors` de **2,1 MB**. El modelo base no se toca, asi
que el adaptador se distribuye solo.

Medido en un i5-10400 sin GPU, Qwen2.5-0,5B, secuencias de 256 tokens:

| | |
|---|---:|
| parametros entrenables | 540.672 de 494 millones |
| paso completo | ~2,8 s (con candle, ~5,2) |
| 1000 pasos | ~47 min |
| adaptador resultante | 2,1 MB |

El forward y el backward son propios (`src/training/transformer.rs`), cada
gradiente comprobado contra diferencias finitas. Con la base congelada no se
calcula `dL/dW` ni el gradiente de las tablas de vocabulario, que es buena
parte de por qué el paso bajó de 5,2 s a 2,8.

**La base va en f32 y no cuantizada**, y eso acota el alcance: el backward pide
`dL/dy · W` —multiplicar por `W` sin transponer— y los kernels cuantizados solo
hacen `x · Wt`. El 0,5B en f32 son ~2 GB y entra; el 3B serian ~12 GB y no.
Entrenar el 3B pide ese kernel, y esta en el [ROADMAP](ROADMAP.md) §7.

El modelo de entrenamiento es una reimplementacion en f32 del de inferencia, y
que sean el mismo **se comprueba**, porque un error ahi —el orden de RoPE, el
repetido de cabezas KV, la escala de la atencion— haria bajar la perdida
igual, optimizando contra un modelo que no es el que despues se sirve:

```bash
cargo run --release --features training --example check_training_model
```

Compara los logits de los dos caminos sobre un prompt real. No exige igualdad
—uno descuantiza a f32 y el otro multiplica en enteros— sino que **elijan el
mismo token**. Medido: diferencia maxima del 2,6% del rango, mismo token. Contra
el Qwen2 de candle en f32, antes de quitarlo, la diferencia fue de 4e-5: el
2,6% es lo que cuantiza el motor, no un error del modelo de entrenamiento.

---

## Licencia

**GPL-3.0-or-later** — ver [LICENSE](LICENSE): se puede usar, estudiar,
modificar y redistribuir, y lo que se distribuya modificado sigue siendo GPL,
con su código. Hasta la 0.10.0 Cosxamix se publicó bajo MIT, y esas versiones
siguen siendo MIT para quien las tenga; desde la 0.11.0, GPL.

Los modelos que usa y no vienen en el repositorio, y sus licencias, están en
[THIRD-PARTY.md](THIRD-PARTY.md).
