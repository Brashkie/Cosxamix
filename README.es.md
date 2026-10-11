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
| Ejecución del grafo, atención, cuantización | motor existente (`llama.cpp`, ONNX Runtime, `candle`) | no |
| Kernels sueltos del camino caliente | Cosxamix, **solo donde el perfilado lo justifique** | puede |

Escribir un motor de inferencia desde cero compite con años de trabajo de mucha
gente en `llama.cpp`. Escribir la capa de servicio es un proyecto abarcable y
es donde están las decisiones que de verdad cambian la latencia percibida: que
el modelo no se descargue, que el prompt no se recalcule entero, que una
petición no espere detrás de otra.

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
cargo run --release                                  # solo la capa HTTP, 6 crates
cargo run --release --features candle                # el motor de verdad, 155 crates
cargo build --release --features zig,candle          # con los kernels de Zig
```

Lo que comprueba cada cosa, y se corre igual en local que en CI:

```bash
cargo fmt --all -- --check
cargo clippy --features zig,candle --all-targets -- -D warnings
cargo test --features zig,candle                     # 22 tests, ninguno necesita un modelo
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

En resumen: el motor está al **51% del techo de ancho de banda** de esta
máquina, o sea que queda menos de 2x por ganar acá, y lo que lo bloquea es que
de 6 núcleos se obtienen 2,6x en vez de 6x — y esa brecha todavía **no tiene
explicación**. Hasta cerrarla, cualquier optimización del lado de la memoria
rinde menos de lo que promete, que es exactamente lo que mató a la
decodificación especulativa.

---

## El motor: candle, y por qué

La decisión se tomó midiendo, no leyendo comparativas.

| | candle | llama.cpp |
|---|---|---|
| dependencias | 155 crates, **Rust puro** | + toolchain de C++ por plataforma |
| build release | 84 s | minutos, y hay que tener compilador |
| cruza a `aarch64-linux-android` | **sí** | con NDK y pelea |

La última fila es la que decide. Termux es un objetivo del proyecto, y un motor
que necesita un compilador de C++ en el teléfono reintroduce el requisito que se
estaba quitando — movido de sitio, pero el mismo.

**Una trampa que solo se ve probando**: el crate `tokenizers` trae por defecto el
feature `onig`, que es Oniguruma —una librería de C— y rompe el cruce a Android
con `failed to run custom build command for onig_sys`. Se desactiva con
`default-features = false` y el camino alternativo usa `fancy-regex`, Rust puro.
Está anotado en `Cargo.toml` porque es la clase de dependencia que se vuelve a
colar sin que nadie la vea.

### El hallazgo más grande: `target-cpu`

Con el motor andando, la primera medición dio **1,8 tokens/s** — inusable para un
modelo de 0,5B. El motivo no era candle: **Rust compila por defecto para el
x86-64 básico de 2003**, así que las multiplicaciones de matrices no usaban AVX2.

Es exactamente la misma trampa que nos había pasado con Zig y `-mcpu`, ahora en
el otro lenguaje. Mismo código, mismo modelo, solo cambia la bandera:

| | por respuesta | tokens/s |
|---|---:|---:|
| genérico (por defecto) | 4428 ms | 1,8 |
| **`x86-64-v3`** (AVX2+FMA) | **1094 ms** | **6,1** |
| `native` | 1066 ms | 6,3 |

Se eligió `x86-64-v3` y no `native`, y está fijado en `.cargo/config.toml`.
`native` mira la CPU que compila y ata el binario a ella: el que salga de un
servidor con AVX-512 se cae con instrucción ilegal en una máquina sin él.
`x86-64-v3` es AVX2+FMA —cualquier CPU de 2015 en adelante— y da el 97% del
rendimiento.

**El coste, dicho claro**: el binario no arranca en un x86 anterior a 2015. Para
esos hace falta compilar sin esa línea, y correrá 4 veces más lento. Los kernels
de `src/kernels/` sí eligen en tiempo de ejecución, pero el interior de candle no.

### Medido de punta a punta

Qwen2.5-0.5B-Instruct Q4_K_M (491 MB), CPU, sin GPU:

```
carga del modelo     ~1,9 s   (una sola vez, al arrancar)
primera respuesta     3,0 s   frío
respuesta típica      1,1 s   caliente
velocidad             6,1 tokens/s
```

Cosxamix **precalienta al arrancar**: el usuario no paga los 2 s de carga en su
primera pregunta. Esa es la razón de ser del proyecto — Ollama descarga el modelo
a los 5 minutos de inactividad y un bot con mensajes espaciados paga esa recarga
casi siempre.

Sin modelo en `modelos/`, o compilado sin `--features candle`, arranca con el
motor de prueba y el contrato HTTP sigue en pie.

```bash
cargo run --release --features candle
```
### Reutilización del caché de atención

El prefill domina la latencia: generar **un** token costaba 857 ms, de los cuales
~817 eran procesar el prompt. Y en una conversación ese trabajo se repite casi
entero en cada turno, porque el prompt del turno N empieza exactamente igual que
el del turno N-1.

Cosxamix reutiliza el **prefijo común más largo** entre el prompt nuevo y lo que
hay en el caché, recorta el caché a esa parte y procesa solo el resto.

La primera versión exigía que el prompt nuevo empezara con *todo* lo cacheado.
Funcionaba en una conversación que crece, pero no en el caso real de WinsiBot,
que manda `system + pregunta` sin historial: el turno 2 comparte el system con
el 1 y después diverge. Medido contra el bot de verdad, daba **`cacheados=0`
siempre**. Con el prefijo común pasa a **131**, que es el system entero —
perfiles de estilo y catálogo de comandos que antes se reprocesaban en cada
mensaje.

Eso solo se pudo hacer porque `truncar_cache` ya existía, del intento de
decodificación especulativa.

| turno 2 de una conversación | |
|---|---:|
| sin caché | 2454 ms |
| **con caché** (53 de 69 tokens reaprovechados) | **1562 ms** |
| ahorro | 892 ms · **1,6x** |

La ventaja crece con la conversación: cuanto más largo el historial, mayor es la
parte que no hay que reprocesar.

#### Una limitación de candle, y por qué el delta va de a un token

`ModelWeights::forward` arma la máscara causal como **seq_len × seq_len**, que
solo es correcta con el caché vacío: con caché, la atención es
seq_len × (cacheados + seq_len) y las formas no cuadran — el intento da
`cannot broadcast [16, 16] to [1, 14, 16, 69]`. Candle soporta el prefill completo
o decodificar de a un token, no un bloque con caché de por medio.

De a uno no hay máscara (`seq_len == 1` la omite) y funciona. Cuesta el doble por
token que el prefill en bloque —~40 ms contra ~21— pero se saltea todo el prefijo
cacheado. La cuenta: reutilizar C tokens y alimentar D de a uno gana mientras
**D < 1,1·C**, que a partir del segundo turno se cumple de sobra.

#### Por qué la respuesta cambia, y por qué eso NO es un bug

Con el caché puesto, el mismo prompt da un texto distinto. La primera lectura es
que el caché está roto. No lo está, y distinguirlo costó aislar la variable.

Comparar textos no sirve: con argmax, una diferencia de un bit en los logits
cambia un token y de ahí todo diverge, así que dos salidas distintas no
distinguen un caché roto de ruido de punto flotante. Hicieron falta dos pruebas.

**La primera** fija el prompt en 69 tokens y compara tres caminos:

| | caché | prefill | salida |
|---|---|---|---|
| A | no | en bloque | *…enviar mensajes, **controlar chats** y mucho más* |
| B | no | de a un token | *…enviar mensajes, **controlar tu conversación** y mucho más* |
| C | **sí** | de a un token | *…enviar mensajes, **y mucho más*** |

**A y B difieren, y ninguno usa caché.** Lo único que cambia entre ellos es la
forma de la matriz, porque `gemm` acumula en otro orden según el bloque. Eso
descarta al caché como causa. El interruptor `COSXAMIX_PREFILL_INCREMENTAL=1` existe
para poder correr B; no sirve en producción, solo para aislar esta variable.

**La segunda** compara los logits, que es la magnitud que el modelo produce de
verdad, con un prompt real —no con ids inventados, que dan una distribución plana
donde el token más probable lo decide el ruido:

```
rango de los logits  33,42
diferencia maxima     0,68   (2,0% del rango)
token mas probable   68012 sin cache | 68012 con cache  ->  IGUAL
```

Un caché que atendiera a tokens equivocados no daría logits parecidos: daría
otros, y el máximo se movería. Elige el mismo token.

```bash
cargo run --release --features candle --example compare_logits   # corrección
cargo run --release --features candle --example check_cache      # rendimiento
```
#### El límite real: el caché es uno solo

El caché vive dentro del modelo, así que hay **uno**. Mientras una conversación
avanza sola funciona muy bien; cuando varias llegan intercaladas —lo normal en
WhatsApp— cada una invalida la de la otra.

Medido con dos conversaciones de tres turnos, las mismas seis peticiones, lo
único que cambia es el orden:

| | tiempo | tokens reaprovechados |
|---|---:|---:|
| secuencial (A completa, después B) | 5541 ms | 244 |
| intercalado (A1 B1 A2 B2 A3 B3) | 9770 ms | **0** |
| | **+76%** | |

En el caso bueno se ve crecer la ventaja: el turno 3 reaprovechó **93 de 107**
tokens y tardó 871 ms, *menos* que el turno 2 pese a tener un prompt más largo.
En el intercalado no acierta ni una vez.

**El caché es frágil a que el historial no coincida.** Reutiliza solo si los
tokens nuevos extienden exactamente a los cacheados, y los cacheados incluyen lo
que el modelo generó. Si quien llama devuelve un texto distinto del generado —un
recorte, un reemplazo, un resumen— el prefijo no encaja y no se reaprovecha nada.
La primera versión de esta medición armaba el historial con texto inventado y
daba cero reutilización **también en el caso secuencial**; parecía que el caché
no servía, y lo que no servía era la medición.

#### Qué modelo: medido, no estimado

Tres modelos de la misma familia, mismo cuantizado (Q4_K_M), misma máquina.
Velocidad como mediana de 3 repeticiones —una sola corrida llegó a dar 21,6 y
32,3 tok/s para el mismo modelo— y calidad sobre 15 preguntas con respuesta
verificable: aritmética encadenada, seguir un formato pedido, conocimiento,
preguntas con trampa y negaciones.

| | 0,5B | 1,5B | 3B |
|---|---:|---:|---:|
| disco | 468 MB | 1066 MB | 2007 MB |
| residente | 504 MB | 1108 MB | 2051 MB |
| relación RAM/disco | x1,08 | x1,04 | x1,02 |
| decode | ~32 tok/s | 19,3 | 10,0 |
| prefill | 846 ms | 814 ms | 1561 ms |
| respuesta de 40 tokens | ~2,1 s | ~2,9 s | ~5,6 s |
| aciertos | 7/15 | 8/15 | **10/15** |

**El 1,5B no es el punto dulce.** Cuesta 1,7x de velocidad y gana una pregunta
de quince, que es ruido. El salto real está en el 3B: acierta las que los otros
dos fallan —el kilo de plomo contra el de plumas, los 206 huesos, Canberra— y
contesta "pingüino" donde el 0,5B dice "girafas" y el 1,5B "urálico".

La elección es entre **0,5B y 3B**, según si pesa más la latencia o el acierto.
Ninguno de los dos es bueno: 10 de 15 en preguntas básicas sigue siendo flojo.

La relación RAM/disco bajó a **x1,02-x1,08** después de quitar la descuantización
de los embeddings. Antes era x2, y en el 3B eso habrían sido 4 GB en vez de 2.

```bash
cargo run --release --features candle --example compare_models
cargo run --release --features candle --example measure_memory -- <gguf> <tokenizer>
```

#### Cuantización: Q3_K_M, no IQ3

**candle no soporta los formatos IQ.** Su enumeración de tipos llega hasta los
k-quants clásicos —`Q2K Q3K Q4K Q5K Q6K Q8K`— y los IQ (IQ3_M, IQ2_XXS) son de
llama.cpp. Son mejores por bit, pero no están disponibles acá.

Q3K es lo más parecido que sí está. Mismo modelo de 3B, mismas 15 preguntas:

| | disco | residente | decode | 40 tokens | aciertos |
|---|---:|---:|---:|---:|---:|
| Q2_K | 1313 MB | — | 13,7 tok/s | ~4,0 s | **6/15** |
| **Q3_K_M** | **1644 MB** | **1687 MB** | **11,9 tok/s** | **~4,7 s** | **10/15** |
| Q4_K_M | 2007 MB | 2051 MB | 10,8 tok/s | ~5,2 s | 10/15 |

**Q3 acierta lo mismo que Q4 con 363 MB menos y un 10% más rápido.** Menos bytes
por peso son menos bytes que leer de la RAM en cada token, que es exactamente el
cuello de la generación en CPU.

Q2 ya no: el daño se ve a simple vista. Dice *"canossa"* por Canberra, *1972*
por 1969, *"20 mandile"* donde debería ir un número, y mete acentos franceses
—*"debido à su arquitectura"*— en medio del español. Así se rompe un modelo
sobrecuantizado: no falla con un error, responde cosas que casi parecen bien.

Por eso el defecto es **Qwen2.5-3B Q3_K_M**: 1687 MB residentes y ~4,7 s por
respuesta de 40 tokens.

```bash
cargo run --release --features candle --example compare_models -- --cuant
```

#### Paralelismo: cuántos núcleos usa de verdad

El techo teórico en CPU es `ancho de banda de RAM / tamaño del modelo`. Con un
i5-10400 y DDR4 de doble canal son ~40 GB/s sobre 0,47 GB: **~85 tokens/s**.

Medido separando prefill de decode —dos corridas con el mismo prompt y distinta
cantidad de tokens, para que el prefill se cancele al restar:

| hilos | prefill | decode | núcleos efectivos | CPU consumida |
|---:|---:|---:|---:|---:|
| 1 | 3173 ms | 9,86 tok/s | 0,98 | 32,9 s |
| 2 | 1639 ms | 16,94 | 1,76 | 33,8 s |
| 4 | 1060 ms | 23,65 | 2,95 | 41,8 s |
| **6** | 933 ms | **25,58** | 4,06 | **53,2 s** |
| 12 | 854 ms | 26,08 | 6,23 | 81,8 s |

**De 6 a 12 hilos se gana un 2% y se gasta un 54% más de CPU.** Los dos hilos de
un mismo núcleo compiten por la misma unidad vectorial, y en un trabajo que es
casi todo AVX2 el segundo no aporta. En un teléfono o un portátil eso es calor y
batería a cambio de nada.

Por eso Cosxamix fija el pool a los núcleos **físicos** al arrancar. `num_cpus`
hace falta porque `available_parallelism()` devuelve 12 en una máquina que tiene
6. `RAYON_NUM_THREADS` sigue mandando si alguien lo fija.

##### Una cifra que estaba mal y hay que corregir

Antes se reportó **6,1 tokens/s**, y sobre eso se concluyó que el motor estaba al
7% del techo y que no valía la pena pensar en decodificación especulativa.

Esa cifra era un artefacto: salía de dividir tokens generados por el tiempo
**total**, en respuestas de uno y dos tokens donde el prefill se lleva casi todo.
No medía la velocidad de generación. El número real es **~25 tok/s**, o sea el
**30% del techo**, no el 7%.

Cambia la conclusión: queda unas 3x de margen antes de chocar con la memoria, así
que la decodificación especulativa —que cambia lecturas de pesos por cálculo— sí
tiene espacio para rendir. Menos del que tendría al 95% del techo, pero espacio.

```bash
cargo run --release --features candle --example measure_threads   # prefill y decode
```

#### 938 MB para un modelo de 469: la tabla de embeddings

El §4.1 de la guía culpa a la carga convencional de "duplicar el consumo de RAM
física" y propone `mmap`. Se midió antes de aceptarlo, y la causa era otra:

| | |
|---|---:|
| archivo en disco | 468,6 MB |
| tensores cuantizados | 463,0 MB |
| **embedding pasada a f32** | **519,3 MB** |
| estimado | 982,3 MB |
| medido | 938,1 MB |

`from_gguf` hacía `tok_embeddings.dequantize()` sobre la tabla entera: 151936 x
896 en f32 pesan **más que el modelo completo**. Y un forward solo necesita las
filas de los tokens que entran — 3,5 KB por token.

Ahora la tabla queda cuantizada y se descuantizan solo esas filas. La tabla está
guardada por filas y cada una ocupa un número entero de bloques (896 elementos en
bloques de 32 son 28), así que los bytes se recortan sin desarmar nada.

**938 MB → 506 MB.** El residente queda cerca del archivo, que es lo correcto.
`mmap` habría ahorrado mucho menos; el problema no era la copia.

##### Un bug de candle que apareció en el camino

La primera versión generaba basura. Aislando: de siete filas pedidas, seis salían
**exactas** y solo fallaba la primera del recorte — el mismo índice 0 estaba bien
si no iba primero. No era el índice: era la posición.

La causa está en `candle-core`:

```rust
fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {   // toma el Cow POR VALOR
    let ptr = data.as_ptr();
    unsafe { std::slice::from_raw_parts(ptr as *const T, ...) }
}                                                  // y aquí libera el Vec
```

Con `Cow::Owned` el `Vec` lo posee la variable local y se libera al salir: el
slice devuelto queda colgando y el `.to_vec()` de quien llama lee memoria
liberada. Casi toda sigue intacta —por eso seis filas de siete salían bien— menos
los primeros bytes, que el asignador pisa con su lista de libres.

Upstream no lo nota porque siempre pasa `Cow::Borrowed`. Acá se hace lo mismo, y
la diferencia contra descuantizar la tabla entera pasó a ser **exactamente cero**.

```bash
cargo run --release --features candle --example measure_weights        # el diagnostico
cargo run --release --features candle --example check_embeddings  # fila por fila
```

#### Decodificación especulativa: implementada, medida y borrada

La idea: el 0,5B propone K tokens y el 3B los verifica **en una sola pasada**,
leyendo sus pesos una vez en vez de K veces. Promete 2-3x.

Se implementó entera —con el borrador, la verificación y el recorte del caché—
y salió **1,91x más lenta**. La regla 3 dice que lo que no gana se borra, así
que se borró. Queda esto escrito para que nadie la reintente a ciegas.

##### Por qué no puede funcionar acá

La técnica supone que procesar K posiciones cuesta casi lo mismo que una: los
pesos son los mismos y se leen una sola vez. Medido sobre el 3B:

| posiciones | ms | vs 1 |
|---:|---:|---:|
| 1 | 112 | 1,00x |
| 2 | 202 | 1,80x |
| 4 | 368 | 3,28x |
| 5 | 420 | 3,74x |
| 8 | 663 | 5,91x |

**Escala casi lineal.** La primera sospecha fue la capa de salida —el
vocabulario de Qwen son 151936 tokens— pero resultó ser solo el **15%** del
modelo, y el 15% no explica un 3,74x.

Lo que lo explica: si la pasada estuviera limitada por leer pesos de la RAM, 5
posiciones costarían ~1x. Que cuesten 3,74x significa que está limitada por
**cálculo**. Y la especulativa solo rinde cuando el cuello es la memoria —
convierte lecturas de pesos en aritmética, y acá la aritmética ya es el cuello.

Encaja con el techo de ancho de banda medido aparte: el motor va al **30%** de
ese techo. Hasta llegar cerca del 100%, batear no compra nada.

A eso se suma que el 0,5B solo coincide con el 3B en el **28-50%** de los
tokens, así que de cada 4 propuestas sobreviven 1,4.

**Vuelve a tener sentido** si alguna vez el motor se acerca al techo de memoria
—por ejemplo con kernels de multiplicación mejores— o con un borrador que
coincida mucho más. Antes, no.

##### Lo que sí quedó de ese trabajo

Dos piezas que hicieron falta para intentarlo y que valen por sí solas:

* **La máscara causal contempla el caché.** La del original era cuadrada y solo
  valía con el caché vacío. Arreglada, el delta de una conversación entra por
  lotes en vez de token a token: el banco de intercalado pasó de 5422 a
  **4772 ms**, un 12%, con turnos individuales que bajaron a la mitad.
* **`truncar_cache`.** Candle no deja recortar el caché de atención. Ahora sí,
  y es lo que permitiría cualquier esquema que procese tokens tentativos.

```bash
cargo run --release --features candle --example measure_projection
```

#### Caché por sesión

El caché vive dentro del modelo, así que de fábrica hay **uno**. Con varias
conversaciones intercaladas —lo normal en WhatsApp— cada una invalidaba la de la
otra. Medido antes de arreglarlo, con dos conversaciones de tres turnos y las
mismas seis peticiones, cambiando solo el orden:

| | tiempo | tokens reaprovechados |
|---|---:|---:|
| secuencial | 5541 ms | 244 |
| intercalado | 9770 ms | **0** |
| | **+76%** | |

Las dos salidas obvias se midieron y las dos se descartaron. **Cargar el modelo
por sesión**: 931 MB cada instancia, el doble del GGUF — diez chats serían 9,1 GB.
**Usar la API del crate**: `ModelWeights` no es `Clone` y su `kv_cache` es privado
dentro de `LayerWeights`; `quantized_qwen2` solo expone `from_gguf` y `forward`.

Así que `quantized_qwen2.rs` está **en el repositorio**, en `src/model/`, con
tres métodos agregados —`tomar_cache`, `poner_cache`, `limpiar_cache`— y el resto
intacto. El modelo tiene un caché activo; los de las otras conversaciones se
guardan aparte y se intercambian al entrar, moviendo punteros y sin copiar datos.

| después del cambio | tiempo | tokens reaprovechados |
|---|---:|---:|
| secuencial | 5422 ms | 244 |
| intercalado | 5357 ms | **244** |
| | sin diferencia | |

El aislamiento se verifica aparte, porque es el fallo peligroso: si se restaurara
el caché equivocado el modelo respondería condicionado por otro chat, sin error y
con una respuesta que suena bien. `check_sessions` corre la conversación A sola
y después intercalada con B, y compara: **los tres turnos salen idénticos**.

La sesión viaja en el cuerpo de la petición como `cosxamix_sesion`, fuera del
dialecto de OpenAI —que no tiene el concepto, ahí cada llamada es independiente—.
Un cliente que no la mande funciona igual, solo que reprocesa el prompt entero.

Tope de 16 sesiones con desalojo de la menos usada, y 4096 tokens por
conversación. Lo que ocupa **depende del modelo**, y medirlo corrigió una cifra
que este documento daba mal:

| | KB por token | 16 sesiones de 1000 tokens |
|---|---:|---:|
| 0,5B | 24,0 | 0,4 GB |
| **3B Q3** (el de por defecto) | **72,0** | **1,1 GB** |

Son 36 capas contra 24, y 128 de dimensión por cabeza contra 64: tres veces más
por token. Este README decía "~24 KB por token, 16 sesiones son ~380 MB" sin
aclarar de qué modelo, y con el que viene por defecto eso subestima la memoria
por **tres**. Medido con `examples/measure_cache_disk.rs`.

Aun así el negocio sigue siendo bueno al lado de los 931 MB que cuesta **cada**
instancia del modelo, que es la alternativa que se descartó.

#### El coste de tener el modelo en el repositorio

Son 338 líneas de código ajeno bajo nuestro mantenimiento, que no reciben los
arreglos de upstream. Al actualizar candle hay que compararlo con el del crate.
Por eso el archivo empieza diciendo de dónde sale, bajo qué licencia
—MIT OR Apache-2.0, de huggingface/candle— y cada cambio propio está marcado con
`// COSXAMIX:`: sin esa marca, en seis meses nadie distingue un cambio nuestro de un
arreglo que falta.

```bash
cargo run --release --features candle --example measure_interleaved  # rendimiento
cargo run --release --features candle --example check_sessions    # aislamiento
cargo run --release --features candle --example measure_memory      # los 931 MB
```
---

## Varios lenguajes, una sola frontera

Cosxamix no se escribe solo en Rust. Rust es el principal, y la frontera está
abierta al que gane la medición:

| | qué aporta | qué falta |
|---|---|---|
| **Zig** | una sola fuente que cubre x86-64, ARM64 y Android | nada: ya está dentro |
| **Ensamblador** | control total donde el compilador emite algo redundante | un bucle concreto señalado por el perfilado, o una instrucción que candle no use |
| **Fortran** | arreglos sin aliasing, que es por lo que BLAS y LAPACK siguen siendo Fortran | un compilador en cada plataforma, y Android es justo el problema |
| **Ada / SPARK** | demostrar que un kernel no desborda, en vez de comprobarlo con tests | un kernel lo bastante crítico como para que valga el toolchain |
| **C** | el denominador común, y lo que candle ya usa por debajo | nada que no cubran los otros |

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

Conviene no leer eso de más: lo que cae a escalar en ARM son **estos** kernels,
`dot` y `rmsnorm`. La multiplicación de matrices cuantizadas, que es donde está
el 99% del trabajo, la cubre candle con su propio `neon.rs` escrito a mano. El
motor en un teléfono no es escalar; estos dos kernels sueltos sí lo serían sin
Zig.

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
| paso completo | ~5,2 s |
| 1000 pasos | ~1,4 h |
| adaptador resultante | 2,1 MB |

**La base va en f32 y no cuantizada**, y eso acota el alcance. `QMatMul` de
candle no implementa `bwd`: el gradiente no atraviesa una multiplicacion
cuantizada, porque el backward pide `dL/dy · W` —multiplicar por `W` sin
transponer— y candle solo sabe hacer `x · Wt`. El 0,5B en f32 son ~2 GB y
entra; el 3B serian ~12 GB y no. Entrenar el 3B pide un kernel que todavia no
existe, y esta en el [ROADMAP](ROADMAP.md) §7.

El modelo de entrenamiento es una reimplementacion en f32 del de inferencia, y
que sean el mismo **se comprueba**, porque un error ahi —el orden de RoPE, el
repetido de cabezas KV, la escala de la atencion— haria bajar la perdida
igual, optimizando contra un modelo que no es el que despues se sirve:

```bash
cargo run --release --features training --example check_training_model
```

Compara los logits de los dos caminos sobre un prompt real. No exige igualdad
—uno descuantiza a f32 y el otro multiplica en enteros— sino que **elijan el
mismo token**. Medido: diferencia maxima del 3,2% del rango, mismo token.

---

## Licencia

**MIT** — ver [LICENSE](LICENSE).

Un archivo de este repositorio viene de otro proyecto: `src/model/qwen2.rs`,
copiado de [candle](https://github.com/huggingface/candle) bajo MIT OR
Apache-2.0, con todos sus cambios marcados `// COSXAMIX:`. El detalle y el
motivo de tenerlo copiado en vez de usar el del crate están en
[THIRD-PARTY.md](THIRD-PARTY.md).
