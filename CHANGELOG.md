# Changelog

Todas las versiones de Cosxamix. El formato sigue
[Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el versionado es
[SemVer](https://semver.org/lang/es/).

Las decisiones de diseño y sus mediciones viven en [DECISIONES.md](DECISIONES.md);
acá va qué cambió, no por qué.

---

## [0.7.0] - 2026-10-30

Responder un mensaje cuesta mucho menos donde más se notaba: **leer el
prompt**, que en el primer mensaje de un chat era el 72% del tiempo.

- Leer el prompt es **1,70 veces más rápido** de punta a punta: con el 3B,
  333 tokens pasan de 11,2 s a 6,6 s, con la misma precisión y el mismo texto
  generado.
- Un chat nuevo **copia de otra conversación el comienzo que comparten**: el
  primer mensaje de un grupo, como lo usa WinsiBot desde su 9.1.0, baja de
  6,0 a 3,5 s.
- Cada token generado, **3 a 5% más rápido**, con el resultado idéntico bit a
  bit.

Todo con el texto generado igual que antes, y cada mejora con su variable
`COSXAMIX_SIN_*` para apagarla y comparar.

### Añadido

- **`ops::matmul_i8_bloques`, la matmul por bloques del prefill.** Hace las
  mismas cuentas que `matmul_i8_directo` en otro orden. Allá cada producto
  punto era una llamada, y por cada columna se recorría la entrada entera:
  con 333 filas son 680 KB por columna, que no entran en L2, así que la
  multiplicación esperaba a la caché. Acá cada hilo desempaqueta su tramo de
  columnas una vez, recorre la entrada en franjas que entran en L2 junto con
  el tramo, y adentro calcula una **tesela de 4 filas × 2 columnas en
  registros** (`kernels::x86::tesela_u8_avx2` y `tesela_i8_avx2`): cada
  trozo de columna se usa 4 veces y cada trozo de fila 2. Las escalas vienen
  ya expandidas a una por carril, en vez de armarse con `setr` en cada trozo,
  y los pesos se corren para quedar **sin signo** —todos los k-quants entran
  en `0..=129`— con el corrimiento descontado en el término del
  desplazamiento; Q8_0, que puede ir de -127 a 127, sigue con signo.
  Cuantizar la entrada y transponer la salida van **en paralelo**: medido
  por etapas, en un hilo eran entre el 24% y el 64% de cada multiplicación,
  y eran el freno de verdad. `matmul_auto` lo usa desde 24 filas
  (`BLOQUES_DESDE`). Medido sobre todas las matrices de dos capas del 3B
  Q3_K_M (`examples/measure_blocked_matmul`):

  | Filas | Antes | Por bloques | |
  |---:|---:|---:|---:|
  | 8 | 19,1 ms | 24,2 ms | 0,79x |
  | 16 | 32,5 ms | 30,2 ms | 1,07x |
  | 24 | 47,6 ms | 35,4 ms | 1,34x |
  | 32 | 61,4 ms | 41,6 ms | 1,48x |
  | 128 | 249 ms | 127 ms | 1,95x |
  | 333 | 643 ms | 323 ms | 1,99x |

  La diferencia con el camino de antes es el redondeo de sumar en otro
  orden, y en los logits del modelo entero la pérdida es la misma (KL
  0,00109 contra 0,00102). 2 tests, y el del despacho prueba ahora los dos
  lados del umbral.
- **`examples/measure_precision`**: cuánto se aparta un camino de la matmul
  de la referencia que no cuantiza la entrada, en la distribución entera del
  próximo token (divergencia KL) sobre diez prompts de 32 a 320 tokens. Una
  batería de 15 preguntas no alcanza para decidir entre caminos que difieren
  en el redondeo: un acierto de más o de menos puede ser una pregunta en el
  límite.
- **Kernel fusionado de Q6_K** (`kernels::dot_q6k_i8`, en AVX2 y con su
  referencia escalar). En el Q3_K_M del bot la matriz de salida —255 MB— va
  en Q6_K, y sin kernel propio pasaba por el camino con buffer. Medida dentro
  del forward, la proyección de salida baja de 11,3 a 9,7 ms por token (26
  GB/s, casi lo que da la RAM). Solo con AVX2: en otras arquitecturas sigue
  el camino con buffer. `COSXAMIX_SIN_Q6K_FUSIONADO` lo apaga. 1 test.
- **`ops::matmul_varias_una_fila`: q, k y v —y gate y up— en una sola
  pasada paralela** en el decode. Medido dentro del forward, q, k y v iban a
  12 GB/s contra 21 de las ffn: `attn_k` y `attn_v` son tan chicas (0,2 MB)
  que los ~10 µs fijos de repartir y juntar los hilos en cada llamada eran
  casi todo su costo, y había 252 llamadas por token. Ahora son 144. Mismos
  kernels y misma entrada, así que el resultado es **idéntico bit a bit**
  (lo comprueba un test, mezclando formatos). `COSXAMIX_SIN_VARIAS` lo apaga.
  Con las dos cosas, en 8 corridas alternadas el token baja de 124,3 a 120,3
  ms de mediana (3 a 5%, con la máquina cargada por otros programas).
- **`examples/measure_decode`**: el decode por formato y forma de matriz,
  en GB/s contra lo que da la RAM. Ojo con una trampa que encontró: aislada,
  la matriz de salida por el camino con buffer parecía costar 19,7 ms, y
  dentro del forward cuesta 11,3. **Las mejoras del decode se confirman
  midiendo el forward**, no la matmul suelta.
- **El prefijo común entre conversaciones distintas.** Hasta acá el caché
  de atención se reaprovechaba solo dentro de una misma sesión, así que un
  chat nuevo —o un grupo cuyo *system* cambió— leía todo desde cero aunque
  empezara igual que otro. Ahora, si otra sesión comparte un prefijo más
  largo, se copia ese tramo (`Cache::prefijo`): el caché de un token depende
  solo de los anteriores, así que es el mismo. Copiar 130 tokens del 3B son
  ~10 MB y unos milisegundos. Medido con `examples/measure_shared_prefix`,
  el primer mensaje del grupo B después del A:

  | | Tokens copiados | Primer mensaje |
  |---|---:|---:|
  | Antes | 0 | 6,0 s |
  | Con el préstamo | 68 | 4,9 s |
  | Con el préstamo y el *system* de WinsiBot 9.1.0 | 130 | **3,5 s** |

  La respuesta sale igual en los tres. WinsiBot 9.1.0 pasó sus instrucciones
  fijas antes del perfil del grupo, que es lo que alarga el tramo común.
  `COSXAMIX_SIN_PRESTAMO` lo apaga. 1 test.
- **`examples/check_sessions` prueba los dos motores**: que intercalar dos
  conversaciones no cambie ni una respuesta, ahora también en el propio y con
  el préstamo de prefijo.
- **`COSXAMIX_SIN_BLOQUES`** apaga la matmul por bloques, para comparar.
  Con y sin ella, `examples/generate` da el mismo texto.

- **`examples/measure_response`**: dónde se va el tiempo de una respuesta,
  usada como la usa WinsiBot (un *system* largo, el primer mensaje, la
  respuesta y un segundo mensaje), y qué tan lejos está del límite de la
  máquina. Separa leer el prompt (GFLOP/s) de generar (GB/s), y compara el
  decode con lo que da la RAM recorriendo los mismos bytes con todos los
  núcleos. Con el 3B Q3_K_M en la PC de desarrollo (12 hilos):

  | Parte | Tokens | Tiempo | Ritmo |
  |---|---|---|---|
  | Leer el prompt, sin caché | 333 | 11,2 s | 29,7 tok/s, 183 GFLOP/s |
  | … con la matmul por bloques | 333 | 6,6 s | 50,5 tok/s, 312 GFLOP/s |
  | Generar | 48 | 4,4 s | 10,8 tok/s, 18,6 GB/s |
  | Segundo mensaje, *system* en caché | 18 | 0,6 s | |
  | Segundo mensaje, *system* cambiado al principio | 286 | 10,2 s | |

  La RAM da 27,7 GB/s: el decode usa el **67%**, y su techo es ~16 tok/s.
  En el primer mensaje, **72% del tiempo es leer el prompt**. La última fila
  era el caso normal de WinsiBot hasta su 9.1.0 —su *system* cambiaba con
  cada mensaje del grupo—, y se arregló allá, sin tocar el motor.

### Corregido

- **Clippy 1.99 rompía el CI en todas las plataformas.** Trae un aviso nuevo,
  `chunks_exact_to_as_chunks`, y el CI niega los avisos: los
  `chunks_exact(8)` y `chunks_exact(210)` de tamaño fijo pasan a
  `as_chunks::<N>()`, que deja el tamaño en el tipo.
- **En ARM64 no compilaba con `-D warnings`.** `transponer_en_paralelo` y
  `expandir_escalas` solo las usa el camino de AVX2, y fuera de x86-64
  quedaban sin uso. Ahora se compilan solo ahí, igual que ese camino.

### Probado y descartado

- **Escalas enteras en los k-quants**, como llama.cpp: la entrada con una
  escala cada 256 valores (el Q8_K de candle) y el producto de un bloque
  entero acumulado en enteros, con la escala del sub-bloque dentro del
  `vpmaddwd`. Son 3 instrucciones por trozo en vez de 5, pero el freno no
  eran las cuentas. Medido, con escala cada 256 / 128 / 64 / 32 valores:
  2,04x / 1,82x / 1,76x / 1,62x contra el camino de antes, y KL 0,0071 /
  0,0034 / 0,0027 / 0,0015 contra 0,0011 del camino que quedó. Ganaba un 15%
  y perdía de 3 a 7 veces más información; en la batería, 10 de 15 contra
  11. Se sacó.
- **Tajadas a lo largo de `k`** para que la franja y el tramo de columnas
  entraran juntos en los 256 KB de L2: con 512, 1024 y 2048 valores, 1,67x,
  1,93x y 2,11x contra 2,05x sin partir. Se sacó.

---

## [0.6.0] - 2026-10-27

El motor propio llega al servidor, y es el de por defecto. El forward de la
casa —y con él todos los kernels de `src/kernels/`: Zig, C, ensamblador—
sirve las peticiones de verdad, con sesiones y caché de atención. Usado como
lo usa WinsiBot, responde **1,6 a 1,9 veces más rápido** que candle por
mensaje, y en la batería de 15 preguntas acierta **11 contra 10**.
`COSXAMIX_MOTOR=candle` vuelve al de antes.

### Añadido

- **`MotorPropio`** (`src/engine_own.rs`): el motor propio detrás del mismo
  trait `Motor` que candle. Copia del de candle lo que decide el resultado —la
  plantilla de chat, el muestreo (semilla 42, top-p 0,9), el tope de 16
  sesiones y 4096 tokens por sesión—, para que elegir uno u otro sea cambiar
  una variable y no de comportamiento. Reaprovecha el **prefijo común** entre
  dos prompts de la misma sesión, transmite el texto en partes y para en
  `<|im_end|>`. Es más simple que el de candle en una cosa: el caché es un
  valor aparte que `forward` recibe, así que cada sesión tiene el suyo en vez
  de turnarse el único que tiene el modelo adentro. Al cargar hace un forward
  de un token, porque los pesos van mapeados y sin eso la primera pregunta
  pagaría leer el modelo del disco.
- **`COSXAMIX_MOTOR=propio|candle`, con el propio por defecto.** El log de
  arranque dice cuál se usa.
- `examples/compare_models -- --motores`: la batería de 15 preguntas sobre el
  mismo modelo con los dos motores.
- **`forward::Cache::truncar`**, para quedarse con el prefijo común.
- El gestor de modelos sirve cualquiera de los dos, por el trait
  `MotorCargable` (precalentar, descargar, si está caliente).

### Cambiado

- La plantilla de chat de Qwen pasa a `engine::formatear`, una sola copia para
  los dos motores: si armaran prompts distintos, compararlos no diría nada.

### Medido

Qwen2.5-3B Q3_K_M —el modelo del bot—, el servidor de verdad y como lo usa
WinsiBot: el mismo *system* de ~145 tokens, una pregunta distinta por mensaje,
la misma sesión, temperatura 0,7. Dos tandas de ocho mensajes por motor:

| | candle | propio |
|---|---:|---:|
| carga del modelo | 2,7 – 3,6 s | **1,1 s** |
| primer mensaje (nada en caché, ~155 tokens) | 8,3 – 12,9 s | **6,2 – 9,4 s** |
| mensajes 2 a 8, media por mensaje | 2528 – 3727 ms | **1570 – 1940 ms** |
| velocidad contando todo | 3,3 – 4,9 tok/s | **5,7 – 7,1 tok/s** |
| tokens del *system* reaprovechados | 142 – 144 | 142 – 144 |

En `examples/generate`, que mide un prompt sin caché y 48 tokens, el propio
procesa el prompt **1,8x más rápido** (1907 contra 3405 ms) y escribe **un 8%
más lento** (6,9 contra 7,5 tok/s). En el bot pesa lo primero: con el
*system* en caché, cada mensaje es procesar la pregunta y escribir una
respuesta corta.

Las respuestas no salen idénticas: los dos cuantizan la entrada distinto y un
empate cercano cae de otro lado —sobre el modelo en F32 coinciden enteras,
ROADMAP §9—. Por eso decidió la batería de 15 preguntas verificables con la que
se eligió el 3B, en modo voraz, sobre el mismo archivo:

| | candle | propio |
|---|---:|---:|
| aciertos | 10/15 | **11/15** |
| decode, mediana de 3 | 10,5 tok/s | **11,3 tok/s** |
| prefill | 1782 ms | **1279 ms** |
| carga, desde el disco | 43,7 s | **39,2 s** |

candle saca otra vez el 10/15 con el que se eligió el 3B, así que la batería
reproduce. En 13 de las 15 los dos responden lo mismo; el propio acierta además
"100 menos 37, dividido por 7" (9; candle dijo 25). Su respuesta del barco es
confusa y cuenta solo porque nombra el agua —la de candle tampoco era buena—, así
que con rigor es 10 contra 10 u 11 contra 10: igual o mejor, nunca peor.

---

## [0.5.1] - 2026-10-26

Los kernels con `sdot` de la 0.5.0 se midieron en ARM de verdad, ganaron, y
entran al despachador.

### Cambiado

- **`sdot` en el despachador de ARM64.** Con la feature `c` y una CPU con
  `dotprod`, `kernels::dot_q4k_i8` y `kernels::dot_q3k_i8` usan los de
  `kernels/neon_dotprod.c`. En un ARM sin la extensión —Cortex-A53, A72— sigue
  el de Zig en Q4_K y la referencia en Q3_K. `COSXAMIX_KERNEL=escalar` sigue
  forzando la referencia.
- **Los binarios de ARM de la release llevan `c`**: Linux, Android (Termux) y
  macOS. Los compila el mismo Zig que ya llevaban, con `zig cc`.

### Medido

En el runner ARM64 de CI (Neoverse N2), fila de 2048:

| | antes | con `sdot` | |
|---|---:|---:|---:|
| Q4_K | 382 ns (Zig) | **162 ns** | **2,4x** |
| Q3_K | 1998 ns (referencia: no había camino vectorizado) | **367 ns** | **5,4x** |

Un runner compartido no es un teléfono, pero la diferencia es de 2,4x y 5,4x,
no de un 5%: ningún ruido la explica.

---

## [0.5.0] - 2026-10-26

La compilación para las CPU sin AVX2 —la que `.cargo/config.toml` dice que hace
falta para un x86 anterior a 2015— por fin anda con la feature `zig`, y con los
kernels de Zig vectorizados en SSE2 en vez de caer a la referencia escalar. El
kernel de Q3_K, el formato del modelo por defecto, pasa de 154 a 91 ns por fila.
Y entra C, el sexto lenguaje: en x86 con el mismo kernel escrito también en
ensamblador para comparar los tres, y en ARM64 con lo que ningún otro lenguaje
del proyecto alcanzaba, `sdot`, en los fusionados de Q4_K y Q3_K.

### Corregido

- **Con `zig`, la compilación sin `target-cpu` moría con "instrucción ilegal"
  en una CPU sin AVX2, por dos lados.** `build.rs` compilaba Zig siempre para
  `x86_64_v3`, aunque Rust fuera para el x86-64 básico; y seis caminos de
  `src/kernels/mod.rs` llamaban a los kernels de Rust con AVX2 cuando el
  backend era Zig, dando por hecho que "Zig" quería decir "CPU con AVX2". Ahora
  lo pregunta `usar_avx2()`, que mira la CPU una vez y lo guarda.

### Añadido

- **Zig sigue a lo que se le pidió a Rust.** Con Rust en `x86-64-v3` —lo de
  siempre— hay una sola copia de los kernels de Zig, también en v3: el binario
  igual no arranca sin AVX2 y una segunda copia sería peso muerto. Con Rust en
  el x86-64 básico, la librería lleva **dos copias** del mismo fuente: en
  baseline (SSE2) y en `x86_64_v3` con sufijo `_v3`
  (`kernels/exports.zig`, `lib.zig`, `lib_v3.zig`), y se elige una al primer
  uso. Así ese binario anda en un Celeron de 2012 y rinde en un i5 de 2020.
- **Tres variables para medir y probar:** `COSXAMIX_ZIG=baseline` fuerza la
  copia baseline, `COSXAMIX_SIN_AVX2=1` hace como si la CPU no tuviera AVX2
  —para correr en una máquina moderna el camino de una vieja— y
  `COSXAMIX_SIN_V3=1` compila sin la copia `_v3`.
- **El log de arranque dice qué copia de Zig se usa** (`kernels de Zig
  copia=...`).
- **CI prueba el camino de una CPU sin AVX2**: compila sin `target-cpu` y corre
  los tests con `COSXAMIX_SIN_AVX2=1`.

- **El kernel fusionado de Q3_K, 1,65x: de 154 a 91 ns por fila de 2048.**
  Cuatro cambios de algoritmo, ninguno de lenguaje: las dieciséis escalas de 6
  bits se desempaquetan en vector (un `vpsrlvd` con corrimientos `[0 2 4 6]`
  hace el trabajo de treinta operaciones escalares), viven en registros en vez
  de ir y volver de la pila, el tercer bit de cada valor sale con un
  corrimiento inmediato y un `and` en vez de cuatro operaciones, y las FMA van
  en dos cadenas y no en una. Es el que usa el despachador.
- **C entra al proyecto, compilado con `zig cc`.** Feature `c`, sin otro
  compilador que Zig y sin libc (`-ffreestanding`), así que cruza a Linux y
  macOS sin el SDK del otro lado. El primer kernel es el mismo Q3_K, línea por
  línea (`kernels/q3k_avx2.c`). Lo que C trae que Zig no: las extensiones de
  CPU **por función**, con `__attribute__((target))`, así que el objeto se
  compila en baseline y es seguro en cualquier x86-64.
- **El mismo Q3_K en ensamblador**, `kernels/q3k_avx2_win64.s` y
  `_sysv.s`, detrás de la feature `asm_q3k`. Usa los dieciséis `ymm`, con las
  siete constantes del bucle fijas en registros, y devuelve el resultado ya
  reducido. Da **igual bit a bit** que los intrínsecos en 1, 2, 8, 17 y 43
  bloques: mismo algoritmo, mismo orden.
- CI corre el ensamblador de Q3_K en Linux y Windows, lo ensambla para Mach-O, y
  prueba el de C con el job de Zig.
- **`sdot` en ARM64, desde C: los fusionados de Q4_K y Q3_K**
  (`kernels/neon_dotprod.c`). Dieciséis productos de `i8` por instrucción, que
  ni Rust estable ni Zig pueden emitir desde este proyecto; `zig cc` los saca
  tal cual —seis `sdot` y ningún `smull` en el código generado—. En Q3_K
  sobra el rodeo del +4 que pide x86: `sdot` multiplica con signo los dos
  operandos y los valores de −4 a 3 entran como son. Con
  `__attribute__((target("dotprod")))` por función, el objeto va para
  ARMv8.0 y quien llama pregunta `is_aarch64_feature_detected!("dotprod")`.
  Cruza a Linux, Android y macOS. **Fuera del despachador hasta medirlo en
  ARM**: esta máquina no puede ejecutarlo, y lo corren los tests y el bench del
  runner ARM64 de CI.

### Medido

**Rust, C y ensamblador con el mismo algoritmo de Q3_K**, tres tandas, fila de
2048 (8 bloques):

| | ns |
|---|---:|
| Rust, intrínsecos | 90,1 – 91,5 |
| ensamblador | 90,1 – 92,2 |
| C | 94,4 – 95,4 |
| (Rust, el algoritmo de antes) | 150 – 157 |

El ensamblador **empata** y C **pierde un 4%**: por la regla 3 ninguno entra al
despachador, y quedan detrás de sus features para el bench, como los contratos
de Q4_K que perdieron. Probado también con **cuatro acumuladores** en vez de
dos: 3-4% más lento en Rust, porque el compilador se queda sin registros. Se
sacó.

En la compilación sin `target-cpu`, lo que corre una CPU sin AVX2 contra lo que
corría antes (la referencia en Rust, o el cuelgue). i5-10400 ejecutando código
SSE2: en una CPU vieja de verdad los tiempos son otros, las proporciones dan
una idea.

| kernel | Rust escalar | Zig baseline (SSE2) | |
|---|---:|---:|---:|
| `dot` n=1024 | 994 ns | 85 ns | **11,7x** |
| `dot` n=4096 | 4,07 µs | 331 ns | **12,3x** |
| `dot` n=1.000.003 | 1,00 ms | 162 µs | **6,2x** |
| `rmsnorm` n=2048 | 2,56 µs | 401 ns | **6,4x** |
| `rmsnorm` n=8192 | 10,3 µs | 1,91 µs | **5,4x** |
| `dot_q4k_i8` (fila de 256) | 746 ns | 308 ns | **2,4x** |
| `dot_i8_escalado` n=2048 | 576 ns | 382 ns | **1,5x** |
| `silu` n=11008 | 37,2 µs | 35,2 µs | 1,06x |

La copia `_v3` en ese mismo binario, sobre la baseline: `rmsnorm` 401 → 349 ns,
`dot_i8_escalado` 382 → 243 ns. Por eso viaja también.

En la compilación de siempre no cambia nada: una copia, la misma que antes.

---

## [0.4.0] - 2026-10-19

El motor propio genera texto de punta a punta sin candle, con su propio
tokenizador y su propia familia de modelos, daiss, y el primer ensamblador del
proyecto entra al camino caliente midiendo. En ARM64 queda puesto el andamio del
kernel NEON con `sdot`, la única forma de usar esa instrucción desde aquí.

### Añadido

- **El primer ensamblador en el camino caliente: `cosxamix_q4k_completo`.**
  Calcula `d` y `dmin` con `vcvtph2ps`, las escalas sin ramas, el bucle con los
  pares desenrollados y el desplazamiento, en una pasada y una llamada por fila.
  Contra los intrínsecos de entonces ganaba 1,28x a 1,32x, y antes de
  atribuirle esa ganancia se separó: **la mayor parte era la instrucción**. Con
  la misma instrucción de los dos lados gana **1,04x a 1,08x en tres pasadas de
  tres**, y por la regla 3 entra primero en el despachador cuando `asm_q4k` está
  encendida. Los contratos por bloque y por fila siguen en los `.s` y en el
  bench, fuera del despachador porque los dos pierden. CI pasa a ejecutarlo con
  `asm_q4k` en la matriz, que es la única forma de correr el de System V.

- **El andamio del kernel NEON de Q4_K con `sdot`, para ARM64.** En ARM el
  ensamblador no compite por un 4%: es **la única forma de usar `sdot`** desde
  este proyecto, porque Rust estable no la expone —`vdotq_s32` sigue detrás de
  `stdarch_neon_dotprod`— y Zig no la genera —su kernel para un Cortex-A76 sale
  con 26 `smull` y ningún `sdot`—. Queda puesto todo menos el cuerpo:
  `src/kernels/q4k_neon.rs` con el mismo contrato `completo` en AAPCS64, tres
  tests contra la referencia y contra Zig, detección en `build.rs` por la
  etiqueta de `kernels/q4k_neon.s`, entrada en el bench, un job de CI que
  **ejecuta** en un runner ARM64 nativo (`ubuntu-24.04-arm`, con Zig fijado
  para Linux-ARM64) y el ensamblado del `.s` para ELF y Mach-O de ARM. Fuera
  del despachador hasta medirlo en ARM; cuando entre, preguntando
  `is_aarch64_feature_detected!("dotprod")`, porque el A53 y el A72 no la
  tienen.

- **El contrato completo del kernel en ensamblador, del lado de Rust:
  `cosxamix_q4k_completo`.** Medido aislado, el bucle por fila del ensamblador
  ganaba; de punta a punta perdía porque Rust le calculaba las escalas en una
  pasada aparte y se las entregaba por un buffer. Este contrato deja de
  entregárselas: recibe los bloques enteros con su encabezado, más `xe` y
  `xs`, y calcula él mismo `d` y `dmin` —con `vcvtph2ps`, que el Rust de hoy no
  usa—, las escalas con el desempaquetado sin ramas de `escalas_q4k`, el bucle
  y el desplazamiento, en una sola pasada. Seis argumentos, el sexto en
  `[rsp+48]` en Windows y en `r9` en System V; `acc` pasa a ser de 16 `f32`, el
  acumulador y el desplazamiento, y Rust solo reduce los dos. El contrato está
  documentado en `src/kernels/q4k_asm.rs` con la tabla de bytes del bloque.
  `build.rs` detecta las tres funciones por su etiqueta y conviven: el bench
  compara las que haya. El camino se verificó con el chequeo de tipos y clippy
  forzando su configuración, sin enlazar, porque todavía no tiene cuerpo, y
  también solo, sin los otros dos, para que no queden importaciones sin uso el
  día que se borren. Los tests piden F16C además de AVX2 y FMA.

- **`cosxamix_q4k_fila` en los dos `.s`, y los bucles medidos aislados.** El
  contrato por fila está escrito, pasa los tres tests y da bit a bit lo mismo
  que el por bloque. Aislado, con las escalas fuera de lo cronometrado, su bucle
  es **14% más rápido** —61–62 ns contra 71–73— y cuesta menos que el kernel
  entero de intrínsecos. De punta a punta pierde, y la causa está aislada: el
  envoltorio de Rust calcula las escalas en una pasada aparte y las copia a un
  buffer; con tanda 1, que hace una llamada por bloque como el contrato por
  bloque, mide 130 ns contra 108, así que la diferencia es el envoltorio y no el
  ensamblador. Los dos bucles aislados quedan en el bench como diagnóstico. El
  paso que haría ganar al ensamblador es que calcule las escalas él mismo,
  intercaladas por bloque, donde además puede usar `vcvtph2ps` para los `f16`.

- **Las escalas de Q4_K sin ramas**, con máscaras sobre tres palabras de 32 bits
  en lugar de 8 llamadas a `escala_y_minimo`. Comparadas bit a bit contra la
  versión legible sobre 5.000 bloques. Aisladas, 23% más rápidas; en los
  kernels no se nota, porque ese trabajo ya se superponía con el vectorial.

- **El contrato por fila del kernel en ensamblador, del lado de Rust.** El
  contrato por bloque perdía el 17% de sus instrucciones en armar y desarmar la
  llamada, ocho veces por fila. `cosxamix_q4k_fila` recorre la fila entera en
  una llamada, con los **mismos cinco argumentos en las mismas posiciones** para
  que la entrada de la ABI no cambie: `bloques` apunta a los bloques con su
  encabezado —`qs` a 16 de cada comienzo, 144 entre uno y otro, el único
  puntero no contiguo—, mientras `xq` y `esc` siguen contiguos de punta a
  punta. Rust calcula las escalas de a 64 bloques y llama una vez por tanda.
  Conviven los dos contratos: `build.rs` detecta cada uno por su **etiqueta**,
  con los dos puntos, porque buscar el nombre suelto lo encontraba en los
  comentarios de cabecera y daba por presente una función ausente. El test de
  filas largas pasa a cubrir 80 bloques, más de una tanda, para que el
  acumulador tenga que pasar de una llamada a la siguiente.

- **El primer ensamblador del camino caliente: `kernels/q4k_avx2_win64.s` y
  `kernels/q4k_avx2_sysv.s`**, el bucle vectorial del kernel fusionado de
  Q4_K, partiendo del que genera el compilador y limpiado a mano para no
  depender del formato de objeto. Es correcto —los tres tests pasan— y mide
  **107–110 ns**: 6,4x sobre el escalar (690) y 2,8x sobre Zig (296–301), y
  1,16x detrás de los intrínsecos (92–93). La causa está contada: de las 82
  instrucciones por bloque, 14 son de armar y desarmar la llamada, el 17%,
  porque el contrato es una llamada por bloque y los intrínsecos arman todo una
  vez por fila. No es el bucle. El despachador sigue eligiendo los intrínsecos
  por la regla 3; el ensamblador queda compilado con `asm_q4k`, con sus tests y
  en el bench. El andamio pasa a **dos archivos**, uno por convención de
  llamada como `dot_avx2_*.s`, porque el de System V está limpio a mano y sirve
  a ELF y Mach-O a la vez. Nuevo `x86::escalas_q4k`, compartido por los dos
  caminos para que lo único que se compare sea el bucle. Razonado en
  [DECISIONES](DECISIONES.md) §5.

- **El motor propio genera texto: `src/sampler.rs` y `Modelo::generar`.** El
  muestreador reproduce la semántica del `LogitsProcessor` de candle
  —temperatura y núcleo de top-p— para que los dos motores sean comparables,
  sin su generador de números: con temperatura las secuencias divergen y eso
  no dice nada, con temperatura cero las dos eligen el máximo y tienen que
  coincidir. Para el núcleo **no ordena el vocabulario entero**: separa los
  1024 más probables por selección parcial y solo ordena esos, y si no
  alcanzan la masa pedida —distribuciones muy planas— cae al orden completo,
  así que el atajo nunca cambia el resultado. Cinco tests, entre ellos que las
  frecuencias sigan a las probabilidades en 30.000 muestras, que es el error de
  muestreo que no da ningún fallo visible. `generar` hace el prefill de una
  vez, después va de a un token, para en el `eos` del archivo y antes de
  pasarse del contexto, y llama a un cierre con cada token para poder mandarlo
  en streaming y cortar si el cliente se fue. Nuevo `examples/generate.rs`,
  con el mismo prompt de chat que arma el bot, sobre el 0,5B: el motor propio
  responde *"Un número primo es un número natural mayor que 1 que es divisible
  solo por sí mismo y por 1. Por ejemplo, 2, 3, 5, 7 son números primos."*, y
  la secuencia es **idéntica a la de candle en los 42 tokens**. Prefill
  **303–315 ms contra 864–891, 2,8x**; decode a la par, 23 tok/s los dos;
  **1,26x de punta a punta**. La primera medición daba el prefill 9x peor y era
  el disco: el mapeo lee los pesos en el primer toque, 0,45 GB a 41 MB/s,
  mientras candle los lee en `from_gguf` fuera de su cronómetro. Ahora los dos
  calientan antes de medir.

- **El andamio para el kernel fusionado de Q4_K escrito a mano**, detrás de la
  feature `asm_q4k`, apagada por defecto. El cuerpo del kernel lo escribe una
  persona en `kernels/q4k_avx2_win64.s` y `kernels/q4k_avx2_sysv.s`; todo lo
  que lo rodea ya está: el puente `extern "C"`, el `global_asm!` por sistema
  operativo, el envoltorio seguro, la rama del despachador, el bench y **tres
  tests** —contra la referencia escalar, que es la especificación; contra los
  intrínsecos, que es contra quien compite; y con una fila de un solo bloque,
  porque olvidarse de que el bucle puede dar una sola vuelta es el error
  clásico y con 2048 valores no se nota—. El contrato está documentado en
  `src/kernels/q4k_asm.rs`: la firma, qué deja en cada posición de la salida
  —la reducción final la hace Rust, con el mismo código que los intrínsecos,
  así que lo único que separa a las dos es el bucle— y las tres cosas que
  dependen de la plataforma: dónde llegan los argumentos, qué registros hay
  que preservar y cómo se declara el símbolo en cada formato de objeto, que es
  lo que rompió un release de macOS una vez. `build.rs` **detecta si los dos
  archivos existen** y solo entonces compila el módulo: sin eso, encender la
  feature rompía `--all-features`, que es el comando con el que se verifica
  todo lo demás, y un hueco a medio hacer no puede tirar abajo la verificación
  del resto. Si la feature está encendida y los archivos faltan, avisa en vez
  de fallar. Y va **un archivo por target** —`win64`, `sysv` para Linux y
  `macos`— con cada plataforma pidiendo solo el suyo, por dos motivos: se puede
  tener el de Windows y probarlo sin los otros, y si los archivos se generan
  compilando Rust, cada target trae las directivas de su formato de objeto, y
  la de ELF (`.type`) no ensambla en Mach-O, que es lo que rompió un release.
  `build.rs` exige además que el archivo nombre a `cosxamix_q4k_pares`: uno
  vacío dejaba pasar la compilación y reventaba en el enlazador con un
  *"símbolo externo sin resolver"* que no dice qué falta.

- **Las escalas del kernel fusionado, calculadas las ocho de una: 1,90x.**
  Volcando el ensamblador que genera el compilador para `dot_q4k_i8_avx2`
  aparecieron **40 `vmulss`, 16 `vcvtsi2ss` y 8 `vsubss` contra 8
  `vpmaddubsw`, 8 `vpmaddwd` y 8 `vfmadd213ps`**: más trabajo escalar que
  vectorial en el bucle caliente, y es todo el cálculo de las escalas, hecho
  de a una. No se ve leyendo el código; se ve leyendo la salida del
  compilador. Calcularlas juntas —`cvtepu8_epi32` más `cvtepi32_ps` convierten
  ocho bytes en dos instrucciones donde había dieciséis, y `xe[bl*8 + j]` es
  contiguo— lo lleva de **175,3 a 92,2 ns** sin tocar una instrucción del
  producto punto. A Q3_K le tocó lo mismo más otra cosa: armaba su vector de
  escalas con `_mm256_setr_ps` de valores sueltos, ocho inserciones por grupo
  y ocho grupos por bloque, y ahora sale de una carga y un
  `permutevar8x32_ps` con el patrón `[0 0 0 0 1 1 1 1]`, que reparte dos
  escalas en las dos mitades de los carriles. En el forward entero: prefill
  **72 ms contra 149 de candle, 2,07x**. Razonado en
  [DECISIONES](DECISIONES.md) §11.

- **`examples/make_f32_model.rs`, que zanja si el forward propio es correcto.**
  Convierte un GGUF cuantizado a `F32` con el escritor propio, tensor por
  tensor —así que un modelo más grande que la RAM se convierte igual— y la
  conversión es exacta: cero diferencias sobre los 136.134.656 valores de
  `token_embd`. Con los pesos en `F32` ninguno de los dos caminos cuantiza
  nada, así que tienen que coincidir, y **coinciden**: diferencia de logits
  **0,0000** en prefill y en decode, y la generación voraz **idéntica en los
  24 tokens** en vez de 13. Eso prueba lo que antes solo se podía inferir — el
  forward propio es correcto, y el 1,88-2,19% que se medía sobre el modelo
  cuantizado era **íntegramente cuantización**, de los dos lados. No recupera
  información: los pesos ya venían cuantizados. Lo que quita es la
  cuantización de las *activaciones*, que era lo que estaba en discusión.

- **El kernel fusionado, también en Zig, que es lo único que tiene ARM.**
  `dot_q4k_i8` se había escrito solo en intrínsecos de AVX2, que no existen
  fuera de x86-64, así que en ARM y en Termux el despachador caía a la
  referencia escalar — byte a byte, en el bucle más caliente del motor. Nuevo
  `cosxamix_dot_q4k_i8_f32` en `kernels/dot.zig`, con su puente y su rama del
  despachador: **303 ns contra 683 del escalar, 2,25x**, y 179 de AVX2, que
  sigue eligiéndose primero en x86-64 porque `vpmaddubsw` le da 32 carriles
  contra los 8 que da ensanchar a `i32` con NEON. Entra por el segundo brazo
  de la regla 3. Verificado que cruza: el objeto de
  `aarch64-unknown-linux-gnu` contiene el símbolo y `clippy` para ese target
  pasa limpio. Con test propio, que hace falta porque en x86-64 el despachador
  elige AVX2 y nunca lo llamaría: sin él, el único camino que tiene ARM se
  quedaría sin probar en la máquina donde se desarrolla.

- **`src/forward.rs`: el forward de Qwen2 sobre la pila propia, sin candle.**
  Es la fase 3 del motor propio, y el momento en que las piezas anteriores
  dejan de ser ejercicios: lector de GGUF mapeado, descuantización,
  `matmul_auto`, RoPE, atención y softmax encadenados hasta los logits, con
  caché de claves y valores que crece por tokens. Nuevo
  `examples/compare_forward.rs`, que compara **los logits** y no el texto
  —dos implementaciones que difieran en el último bit eligen tokens distintos
  y divergen, y comparar texto no diría nada—. Sobre el 0,5B en Q4_K:
  **mismo argmax que candle** en prefill y en decode, con 1,88% y 1,35% de
  diferencia sobre el rango de los logits; prefill **68 ms contra 130** y
  decode 36 contra 30, o sea 2x a favor en lote y a la par en decode, que es
  exactamente lo que ya decían las mediciones de la matmul. Y la prueba que de
  verdad decide, porque coincidir en un argmax puede ser suerte: generando con
  el mismo criterio voraz, las dos producen **los mismos 13 tokens** y divergen
  en el catorceavo. La primera medición daba 9x a nuestro favor y era falsa:
  era la primera llamada de cada camino, con los fallos de página del mapeo de
  un lado y la inicialización perezosa del otro — ahora hay calentamiento y
  mediana de cuatro pasadas, que es lo que §15 obliga.

- **`COSXAMIX_MATMUL=f32`, para separar dos causas que se confunden.** La
  hipótesis obvia sobre esa diferencia del 1,88% era nuestra cuantización de la
  entrada, así que se agregó la forma de forzar el camino que no cuantiza nada.
  El resultado **la descarta**: con `f32` la diferencia sube a 2,19% y la
  coincidencia sube a 16 tokens. Nuestro camino de `f32` es en principio el más
  preciso y aun así se aparta un 2%, lo que apunta a la cuantización de
  activaciones de candle a Q8_K —una escala cada 256 contra una cada 32— a lo
  largo de 24 capas. Lo que no se puede afirmar es cuál está más cerca de la
  verdad: para eso hace falta un modelo sin cuantizar, y queda como medición
  pendiente y no como conclusión.


- **`ops::atencion`: atención con máscara causal y cabezas agrupadas**, que
  coincide con candle valor por valor. Tres cosas que se equivocan en silencio
  y que esta función tiene que acertar, dichas en su documentación porque son
  el verdadero contenido: **la máscara causal con caché caliente** —el token
  `t` de la tanda está en la posición absoluta `total − tokens + t`, y usar `t`
  deja al primer token nuevo viendo una sola clave cuando debería ver todo el
  prefijo, con lo que el modelo responde como si hubiera perdido la
  conversación—; **el reparto de GQA**, donde la cabeza `h` usa la de clave y
  valor `h / (cabezas / cabezas_kv)` y repartir con `%` compila igual y mezcla
  las cabezas; y **la escala** de `1/sqrt(dim_cabeza)`, sin la cual con 128 de
  dimensión los logits salen 11 veces más grandes y el softmax se vuelve casi
  un máximo, o sea que la atención deja de promediar y copia un valor. La
  máscara no se aplica rellenando con `-inf`: solo se calculan los puntos
  visibles, que además es menos trabajo. Dos tests: uno de propiedades que no
  depende de nadie —la salida es una combinación convexa de los valores, y
  cambiar un valor futuro no mueve la salida de un token anterior— y el que de
  verdad decide, contra candle, porque las propiedades no distinguen un reparto
  de GQA con `%` de uno con `/`.

- **`ops::rope`, `ops::frecuencias_rope` y `ops::softmax`, que cierran la fase
  2 del motor propio.** Los dos coinciden con candle **valor por valor**, con
  un test que lo comprueba bajo la feature `candle`. De RoPE lo que importaba
  no era la velocidad sino acertar la convención: Qwen2 usa la de **mitades
  partidas** —`candle_nn::rotary_emb::rope`, no `rope_i`— y elegir la
  intercalada **no da ningún error**, el modelo carga, genera y dice cosas sin
  sentido; es el tipo de bug que solo aparece comparando contra algo que ya
  funciona. Las tablas de ángulos se calculan en `f64` y se guardan en `f32`
  porque con 32.768 de contexto el ángulo llega a 32.768 radianes, donde un
  `f32` tiene unos 0,002 de resolución y el seno saldría con error visible.
  `softmax` resta el máximo antes de exponenciar, que no es un refinamiento:
  con 151.936 logits que llegan a 30, `exp(30)` ya es 10^13 y una fila con 90
  desbordaría a infinito dejando la fila entera en `NaN`; y una fila entera
  enmascarada en `-inf`, que es lo que deja una máscara causal, devuelve
  distribución uniforme en vez de `0/0`. Además de la comparación contra
  candle hay tests de propiedades que no dependen de nadie: que softmax sume 1
  y sea invariante a sumar una constante, y que RoPE **conserve la norma de
  cada par** —es una rotación— y sea la identidad en la posición cero.


- **Tokenizador propio: `src/tokenizer.rs`, un BPE a nivel de bytes con
  entrenamiento.** Era el punto que bloqueaba los tamaños chicos de daiss, y
  nuevo `examples/train_tokenizer.rs` lo contesta con un número: entrenado con
  los documentos del proyecto y los comentarios de su código —382 KB de español
  técnico— y medido sobre el **20% final que no vio**, el nuestro de 8.192
  necesita **3,4% más tokens** que el de Qwen2.5 de 151.936 (23.433 contra
  22.659, 3,34 contra 3,45 bytes por token). Y eso lleva a daiss-mini de **94%
  tabla de búsqueda y 82,5 M de parámetros a 47% y 8,9 M**: nueve veces más
  chico. Dos salvedades, dichas: el entrenamiento **se satura en 7.185
  fusiones** porque en 306 KB no queda ningún par repetido —así que los
  objetivos de 16.384 y 32.768 dan exactamente lo mismo— y el texto de prueba
  es del mismo dominio, de modo que parte del 3,4% es especialización y no
  eficiencia. A nivel de bytes, con los 256 como base, así que **no existe un
  token desconocido**: hay un test que codifica y recupera secuencias de bytes
  al azar, incluidas las que no son UTF-8 válido, que para un bot de WhatsApp
  no es teoría. Codificar aplica siempre la fusión de menor rango primero, que
  es lo que lo hace determinista; entrenar desempata por el par más chico,
  porque `HashMap` no tiene orden y sin eso el tokenizador cambiaría en cada
  corrida, con un test que entrena dos veces y compara. El formato guardado son
  solo las fusiones, 8 bytes cada una, y el de 8.192 pesa 57 KB; al cargar se
  comprueba que cada fusión se apoye en tokens anteriores. Siete tests, entre
  ellos que el entrenamiento se detenga solo en vez de inventar tokens de un
  solo uso. **Y daiss lo incrusta**, porque un tokenizador que no usa nadie es
  un adorno: nuevos `daiss::escribir_con` y `daiss::tokenizador_de`, con el
  vocabulario del tokenizador mandando sobre el de la configuración — un modelo
  cuya tabla de embeddings no coincide con su tokenizador está roto en
  silencio, así que las dos cifras no pueden discrepar. Medido de punta a
  punta: daiss-mini pasa de **84 MB con el vocabulario de Qwen a 9 MB con el
  propio**, el tokenizador viaja dentro del GGUF y el que sale codifica igual
  que el que entró, y candle sigue construyendo el modelo entero con cero
  diferencias. Las fusiones van bajo `cosxamix.tokenizer.*` y no bajo
  `tokenizer.ggml.*`: esa convención guarda las piezas como cadenas y las
  nuestras son bytes, que para ser texto necesitan la tabla de byte-a-unicode
  de GPT-2, y esa tabla no está escrita ni verificada — usar la clave estándar
  sin ella haría que otro runtime leyera un vocabulario equivocado creyendo que
  lo entiende. Razonado en [DECISIONES](DECISIONES.md) §19.


- **`kernels::dot_q3k_i8` y `ops::matmul_auto`: el motor elige el camino.**
  Q3_K es el formato dominante del modelo por defecto, así que era el siguiente
  fusionado, y necesita un rodeo que Q4_K no: sus valores van de −4 a 3, con
  signo, así que se les suma 4 para entrar en el rango sin signo que
  `vpmaddubsw` pide y se descuenta con `Σ x·q = Σ x·(q+4) − 4·Σx`, donde `Σx`
  por sub-bloque ya está calculado. Su sub-bloque es de 16, resuelto con el
  reparto de carriles de AVX2. Pero medirlo destapó algo que el fusionado no
  puede arreglar: **el fusionado desempaqueta una vez por fila de salida y el
  camino con buffer una vez por columna**, así que con una fila gana y con lote
  paga `m` desempaquetados contra uno — en Q3_K, que desempaqueta caro, con 32
  filas llega a ser **2,5x peor** (22,9 contra 9,1 ms). Y aparece un tercero
  que no estaba en el plan: **con lote grande en Q4_K gana el camino de `f32`**
  (2,9 contra 3,2 ms con 8 filas, 9,6 contra 12,6 con 32), porque la
  multiplicación deja de estar limitada por el desempaquetado y los productos
  en `f32` tienen FMA de 8 carriles sin recuantizar. Ninguno de los tres gana
  siempre, así que nuevo `ops::matmul_auto` que despacha por formato y por
  número de filas, con los umbrales medidos: fusionado hasta 2 filas en Q4_K y
  solo 1 en Q3_K, `f32` desde 8 en Q4_K, y el camino con buffer para el resto,
  que cubre los siete formatos y nunca es el peor de los tres. Los umbrales son
  un valor por defecto **medido en un tensor y esta máquina**, no una ley, y se
  dice así en el código. Cierra el punto del ROADMAP que decía que había que
  elegir a mano. De paso, el rearmado de las 16 escalas de 6 bits de Q3_K
  estaba escrito dos veces idéntico y ahora es `quant::escalas_q3k`, que usan
  tres sitios: verificado con `check_quant`, 27 tensores y cero diferencias.
  Ampliado en [DECISIONES](DECISIONES.md) §11.


- **El kernel fusionado, que una decisión escrita decía que no valía la pena.**
  Nuevos `kernels::dot_q4k_i8` —referencia escalar y versión AVX2— y
  `ops::matmul_i8_fusionado`: multiplican una fila en Q4_K contra la entrada
  cuantizada **sin desempaquetar los pesos a ningún buffer**, leyendo los
  nibbles a registros vectoriales y acumulando al vuelo. Sobre el `ffn_gate`
  del 3B, todo en el mismo proceso: con una fila 1,079 ms contra 1,426 del
  camino con buffer y 0,912 de candle; con ocho, 3,252 contra 3,937 y 6,279.
  En tres pasadas le gana al camino con buffer entre **1,16x y 1,44x** en todas
  las filas, y contra candle va de 0,82-0,92x con una fila a **1,93-2,39x** con
  ocho. **El caso de una fila, que estaba en 0,55x, queda en 0,85x**: candle
  sigue adelante por un 15% y no por un 80%, y era el punto que el ROADMAP
  tenía marcado como el que importa porque es el decode. Exactitud sin cambios,
  0,26% a nuestro favor. Lo que hace que Q4_K sea el formato ideal para el
  primer fusionado no es su popularidad sino la instrucción: sus nibbles son
  enteros **sin signo**, 0 a 15, y la entrada es con signo, que es literalmente
  la firma de `vpmaddubsw` —`u8 × i8 → i16`—, así que no hace falta el rodeo
  del signo con `vpsignb` y con él se va su único caso límite, el del −128 en
  los dos operandos a la vez; y no puede saturar, porque 15 × 127 = 1905, dos
  de esos son 3810 y en un `i16` entran 32.767. Los otros seis formatos caen a
  `matmul_i8_directo`, que los cubre, porque todos restan algo —32 en Q6_K, 16
  en Q5_0, hasta 4 en Q3_K según una máscara— y restar rompe el rango sin
  signo. Con un test que compara contra desempaquetar-y-multiplicar, que es el
  camino ya verificado contra candle, y que exige además que la referencia
  escalar y la vectorizada coincidan entre sí. De paso, la reducción de los
  ocho carriles `f32` estaba escrita dos veces en `x86.rs` y ahora es una
  función que usan tres. Razonado en [DECISIONES](DECISIONES.md) §11, que
  decía lo contrario y queda corregida.


- **`gguf::Pesos`: los pesos mapeados en memoria virtual, sin copias.**
  `datos_de` abría el archivo de nuevo por cada tensor y copiaba sus bytes a un
  `Vec` — 339 aperturas y 7,5 GB al montículo en el daiss-7b. Nuevo
  `examples/measure_mmap.rs` que compara los tres caminos antes de elegir uno,
  sobre el 3B en Q4_K: reabriendo 2352 ms, con un solo descriptor 1924
  (**1,22x**, y gratis), mapeado 1820 (**1,29x**; 1,21x a 1,29x en tres
  corridas). Y una segunda medición, esta en el mismo proceso, para el otro
  régimen —releer un tensor de 12,1 MB muchas veces, que es lo que hace el
  motor—: mapeado contra copiado da 1,17x, 0,93x y 1,04x, o sea **ninguna
  diferencia**, así que se usa la porción mapeada tal cual en los dos casos. O sea que el 1,22x no cuesta
  ninguna dependencia y lo que compra `memmap2` encima es un 5% de tiempo más
  **1,95 GB de montículo que no se reservan**, 7,5 GB en el daiss-7b; en una
  máquina donde ese modelo no entra en RAM, eso es el argumento y no el 5%. Y
  habilita lo que el reloj no muestra: `Pesos::de` devuelve una porción que
  apunta al archivo, y `descuantizar`, `desempaquetar_i8` y las dos `matmul` ya
  reciben `&[u8]`, así que el camino del disco al producto punto queda sin una
  sola copia. El test del mapeo encontró un bug en la primera corrida:
  `inicio_datos + offset` se desbordaba **antes** de la comprobación de
  límites, y en release eso da un índice envuelto sobre un mapeo, donde una
  porción fuera de rango no devuelve `Err` sino una violación de acceso. Las
  tres cuentas van comprobadas. Razonado en [DECISIONES](DECISIONES.md) §18.

- **`examples/measure_rank.rs`: cuánta energía de un peso cabe en rango bajo.**
  Contesta si las redes de tensores —Tensor-Train, MPS, Tucker— sirven acá sin
  implementar ninguna, estimando la fracción de energía de Frobenius que
  captura la mejor aproximación de rango `k` con un buscador de rango
  aleatorizado e iteraciones de potencia, que da una cota por debajo. La
  comparación es controlada: el mismo tensor y la misma forma en Qwen2.5-0.5B
  entrenado y en daiss-0.5b recién inicializado. Tres resultados. Uno: el
  aleatorio captura exactamente `k / rango_máximo`, cero estructura, así que
  descomponer daiss no sirve — y que el estimador reproduzca la predicción
  teórica al décimo de punto valida el estimador. Dos: `output.weight` tiene la
  señal más fuerte con `k` chico, **22,1x** sobre el aleatorio con k=8 contra
  7,58x del FFN, lo que confirma que es el objetivo correcto. Tres: las cifras
  de 50-70% de reducción con más del 98% de precisión **no se ven**, porque
  llegar al 47,5% de la energía pide el 28,6% del rango. Con el control de
  ruido de cuantización hecho: entre Q2_K y Q4_K_M del mismo tensor hay entre
  0,3 y 1,2 puntos, así que el ruido no infla el rango aparente. Ampliado en
  [DECISIONES](DECISIONES.md) §12.


- **daiss, la familia de modelos propia**, de 0,08 a 14,77 mil millones de
  parámetros en cinco tamaños. Nuevo `src/daiss.rs` con la definición, la
  inicialización y la escritura; nuevo `examples/create_daiss.rs` que imprime
  la familia con sus cuentas y crea el que se le pida. **El daiss-7b está
  creado y comprobado**: 339 tensores, 7,61 B de parámetros, 7,54 GB, y candle
  —que no comparte una línea con nuestro escritor— lo lee con **cero
  diferencias** sobre los 544.538.624 valores de `output.weight` y encuentra
  los 339 tensores que el cargador pide, con su forma. Se genera en flujo y en
  paralelo, fila por fila, con el pico de memoria acotado a 16 MB sea cual sea
  el modelo: escribir el 14b no pide 14 GB. Y con semilla, con la semilla
  mezclada con el nombre del tensor y el número de fila, que es lo que permite
  generar en paralelo sin que el resultado dependa del orden en que termine
  cada hilo — un test escribe el mini dos veces y compara los bytes. La
  inicialización es `normal(0, 0.02)`, con `attn_output` y `ffn_down` —las dos
  proyecciones que escriben en el residual— divididas por `sqrt(2 * capas)`,
  sin lo cual la varianza crece con la profundidad. No entrena: los pesos son
  ruido con la desviación correcta. Lo que aporta es el camino completo
  verificado. Razonado en [DECISIONES](DECISIONES.md) §16.

- **`gguf_write`: escribir GGUF, el camino de vuelta del lector.** Declara
  primero la forma de cada tensor, calcula con eso los offsets de la cabecera,
  y después pide los bytes de cada uno por separado, así que el pico de
  memoria es el del tensor más grande y no el del modelo. Escribe a un archivo
  temporal y renombra al final, porque un GGUF cortado a mitad tendría
  cabecera válida y datos truncados, y fallaría al usarlo y no al escribirlo.
  Comprueba que la fuente escriba exactamente los bytes que declaró —si no,
  todos los tensores siguientes quedan corridos y el síntoma aparece lejos— y
  rechaza antes de escribir nada una forma cuya primera dimensión no cuadre
  con el bloque del formato, que es el error silencioso del formato. Tres
  tests, incluido el de ida y vuelta completo contra nuestro propio lector.

- **`quant::cuantizar_bloques` y `quant::f32_a_f16`**, que son lo que faltaba
  para poder escribir pesos: F32, F16 y Q8_0. El `f16` con subnormales y
  **redondeo al par**, no truncando, porque truncar sesga cada escala hacia
  cero y una escala sesgada desvía los 32 valores que cuelgan de ella; y Q8_0
  cuantiza contra la escala **ya redondeada a `f16`**, que es la que se va a
  leer de vuelta, lo que quita un paso entero de error. Los k-quants no están
  a propósito: su cuantizador busca escala y mínimo con una búsqueda
  iterativa, no con una fórmula, y escribirlo sin tener contra qué compararlo
  valor por valor sería justo lo que este proyecto no hace. Tres tests: ida y
  vuelta exacta en `f16` sobre los once valores que rompen una conversión mal
  escrita, error acotado por la cota teórica del formato, y un bloque entero
  en cero que no puede dar NaN.

### Corregido

- **Los `f16` de cada bloque Q4_K se convertían por software**, desarmando
  exponente y mantisa a mano, en el bucle más caliente del motor. Lo encontró
  el ensamblador, que usaba `vcvtph2ps` y ganaba. Dándole la misma instrucción a
  los intrínsecos, sin tocar nada más, el kernel pasó de ~97 a ~79 ns, 1,24x,
  con resultado idéntico bit a bit —todo `f16` cabe exacto en un `f32`, y lo
  comprueba el test de las escalas—. En la matmul del 3B, el caso del decode
  pasa de perder contra candle 0,55x–0,85x a quedar parejo, 0,81x–1,09x. F16C
  entra a la detección del camino AVX2: todo procesador con AVX2 la tiene, pero
  una máquina virtual puede esconder una extensión y dejar la otra.

- **El envoltorio del contrato por fila ponía en cero 2 KB en cada llamada**
  —`[0f32; TANDA * 8]`, aunque una fila de 8 bloques usa 256 bytes— y copiaba
  con comprobación de límites. La hipótesis vino del lado del ensamblador sin
  medir y era correcta: de 139–141 a 120–121 ns.

- **Una regresión de 2x introducida y corregida en la misma tanda.** Las
  escalas sin ramas leían los 12 bytes uno por uno con `bloque[4 + i]`, y las
  doce comprobaciones de límites agrandaron la función lo suficiente para que
  LLVM dejara de ponerla en línea: el kernel de intrínsecos pasó de 90 a 178 ns
  con el escalar de control quieto. Visto en el ensamblador generado, que
  mostraba la llamada; leídos como un solo arreglo de tamaño fijo, vuelve a 92.

- **Los tres umbrales de `matmul_auto` quedaron obsoletos** al duplicarse la
  velocidad del kernel fusionado. La tabla anterior mandaba el lote de Q4_K al
  camino de `f32` y el de Q3_K al buffer desde dos filas, y era correcta para
  el kernel de entonces: ahora el fusionado gana en Q4_K en **todo el rango
  medido**, de 1 a 32 filas, y en Q3_K hasta 4. Rehecha y vuelta a medir con
  dos pasadas por formato. Es el riesgo de escribir umbrales medidos, y queda
  dicho en el código: hay que volver a medirlos cuando cambia lo que miden.

- **Una conclusión anterior estaba mal cerrada.** Se había medido que
  *adelantar* el cálculo de las escalas fuera del bucle vectorial no cambia
  nada —cierto, 188,9 contra 186,2 ns— y de ahí se dio por cerrado que no
  había nada que ganar ahí. Adelantar las mueve; no las quita.
  **Vectorizarlas da 1,90x.**

- **`matmul_auto` no sabía qué hacer con los formatos sin cuantizar.** Mandaba
  todo lo que no fusionaba al camino de enteros, y `F32` y `F16` no tienen
  desempaquetado a `i8` porque no tienen nada que desempaquetar: fallaba con
  *"F32: sin desempaquetado directo a i8"*. Nunca había aparecido porque todos
  los modelos vienen cuantizados, y lo encontró el experimento del modelo
  convertido a `F32` — que es el argumento para hacer el experimento. Ahora
  esos formatos van directo al camino de `f32`, que es su único camino
  posible, y el test del despacho los cubre. El test también tuvo que
  arreglarse: generaba los pesos con bytes al azar, que para un cuantizado es
  válido pero para `F16` da infinitos y NaN, y entonces el rango de la
  referencia salía `-inf` y la tolerancia no significaba nada.

- **Escribir modelos no era lento por el generador sino por el disco.**
  Dos hipótesis seguidas sobre el código —primero que faltaba paralelizar,
  después que las reservas de memoria y las 316.000 escrituras por tensor
  costaban— llevaron de 24 a 34 y de 34 a 35 MB/s. La segunda no movió la
  aguja un 3%. Lo que faltaba medir era el disco: **F: hace 41 MB/s** de
  escritura secuencial y C: hace 165, o sea que estábamos al 85% del techo
  físico. Escribiendo el mismo modelo en C: da **324 MB/s**. La paralelización
  se queda porque se midió que gana —a 24 MB/s estaba por debajo del techo del
  disco y ahora choca contra él—, no por si acaso. Y queda anotado lo que esto
  cambia fuera de daiss: cualquier medición de arranque en frío sobre este
  disco está midiendo el disco, no el motor. Ver [DECISIONES](DECISIONES.md)
  §17.

- **El bench `cuantizar` recorría una lista de un solo elemento** y su
  comentario describía una pregunta ya contestada —si recuantizar explicaba
  que `matmul_i8` fuera más lenta que la de `f32`—, cuando ese rodeo se
  eliminó hace dos tandas. Reescrito con los dos anchos reales del 3B, 2048 y
  11008, y agregado un `inicializar` que mide la generación de pesos sin el
  disco de por medio, que es el número que dice si el generador es lento.

### Medido y NO construido

- **El buffer reutilizable para las asignaciones de `matmul`.** Cada llamada
  reserva cuatro vectores —la entrada cuantizada, sus escalas, sus sumas y la
  salida transpuesta— y un forward del 3B hace **253 multiplicaciones**, así
  que tener un `Espacio` que el motor conserve entre llamadas parecía evidente.
  Nuevo `examples/measure_allocations.rs`, sobre las formas reales del 3B a una
  fila: **1,52%** del forward, 2,618 ms de 172,4. El peor caso es
  `ffn_gate`/`ffn_up` con 2,63%, porque su salida transpuesta son 44 KB de
  ceros. Eso está por debajo de la variación de esta máquina entre corridas,
  que es 1,8x, así que no es ni observable: no se escribe, porque haría que
  cada `matmul` recibiera un `&mut Espacio` a cambio de nada medible. De paso,
  el total de la medición da un dato útil: la suma de nuestras matmuls a una
  fila es **172 ms**, o sea 5,8 tok/s, en línea con los 6,1-6,3 medidos de
  punta a punta — las multiplicaciones son prácticamente todo el forward, que
  es lo que justifica haber puesto el trabajo ahí.

### Probado y borrado, por la regla 3

- **El desplazamiento en un registro en vez de en memoria.** El ensamblador lo
  acumula en `acc[8..16]` con una lectura y una escritura por bloque; tenerlo en
  un registro pediría salvar `xmm6`, que en Windows obliga a declarar el prólogo
  con `.seh_savexmm`. Se probó lo inverso en los intrínsecos —una variante que lo
  pasa por memoria en cada bloque, con accesos volátiles, verificado en el
  ensamblador generado—, y en tres pasadas la diferencia cambió de signo sin
  pasar del 3%: el procesador esconde la ida y vuelta porque esa cadena no
  depende de la del acumulador. La forma del ensamblador era la correcta.

- **Varios acumuladores en el kernel fusionado.** La hipótesis era que los dos
  `fmadd` encadenados por par imponían un piso de 8 ciclos. Probado en los
  intrínsecos con 1, 2 y 4 acumuladores, en cada pasada ganó uno distinto
  dentro del ruido: la cadena no es el cuello, porque cada par tiene trabajo
  independiente de sobra para cubrir la espera. Se probó en Rust antes de pedir
  el cambio en el ensamblador, que es justo lo que evitó hacerlo para nada.

- **El microkernel con bloqueo de registros**, que DECISIONES §5 señalaba
  desde hace versiones como *la primera oportunidad real del ensamblador*. Se
  escribió `dot_q4k_i8_x4_avx2` —cuatro columnas de una vez, cuatro
  acumuladores vivos, la entrada cargada una sola vez— y se ejecutaron los dos
  pasos del criterio. Uno: LLVM subió a **12 de 16 registros ymm y no volcó
  nada**, igual que los otros tres kernels calientes, que van en 6, 8 y 8 con
  cero volcados. Dos, que ni hacía falta preguntar: es **20% más lento**, 855
  ns contra 712 de cuatro llamadas sueltas. Y el motivo no tiene nada que ver
  con registros, que es lo interesante: cuatro llamadas sueltas recorren una
  fila contigua cada una, acceso secuencial perfecto, mientras la bloqueada
  salta entre cuatro filas separadas por el ancho de una fila entera — cuatro
  flujos entrelazados. Ahorrar una carga que está en L1 cuesta menos que eso.
  El ensamblador no arregla un patrón de acceso. Con esto se cierra la
  pregunta que §5 dejaba abierta: **no hay ninguna apertura para el
  ensamblador en este código hoy**, medida y no supuesta. Y apareció lo
  contrario de lo buscado: el bloqueo que gana ya estaba puesto, del otro
  lado — `matmul_i8_fusionado` recorre una columna de pesos y la reutiliza
  para las `m` filas de entrada, un solo flujo contiguo, que es justo lo que
  bloquear por la salida rompía.

---

## [0.3.0] - 2026-10-17

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

- **`quant::desempaquetar_i8` cubre los siete formatos cuantizados** —Q8_0,
  Q5_0, Q2_K, Q3_K, Q4_K, Q5_K y Q6_K—, desempaquetando a `i8` con una escala y
  un desplazamiento por sub-bloque **sin pasar por `f32`**. Los de sub-bloque de
  16 pedían además que el kernel aceptara ese tamaño: `dot_i8_escalado_avx2` lo
  hace a ancho completo aprovechando que, después de `maddubs` y `madd`, los
  carriles `i32` 0-3 cubren los bytes 0-15 y los 4-7 los bytes 16-31, así que un
  vector de escalas `[s0 s0 s0 s0 s1 s1 s1 s1]` aplica la correcta a cada mitad
  — no hace falta bajar a 128 bits. Medido, el camino de 16 resultó **más**
  rápido que el de 32, 1,07 contra 1,52 ms, al revés de lo que se suponía.
  Nueva `quant::sub_bloque`, que es de dónde sale el tamaño: los dos lados del
  producto tienen que usar el mismo, así que `matmul_i8_directo` cuantiza la
  entrada con el del formato de los pesos y no con una constante, y
  `cuantizar_i8_en` lo recibe. Un test reconstruye `escala * q + desplazamiento`
  y exige igualdad **exacta** contra `descuantizar` en los siete formatos: es
  exacta y no aproximada porque no hay recuantización de por medio.

- **`examples/profile_i8.rs`**, que mide el desempaquetado y el kernel por
  separado sobre tres formatos con el mismo número de valores. Hizo falta porque
  la ventaja medida sobre Q4_K no se sostuvo en Q3_K y había dos sospechosos;
  dijo que el desempaquetado era el 94% del trabajo de un hilo y que el kernel
  no tenía nada que ver.


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
convención de llamada. Usan solo registros volátiles, así que no necesitan
prólogo ni epílogo.

Resultó depender de **tres** cosas de la plataforma y no de una, y las dos
últimas se aprendieron rompiendo el build de macOS en CI:

| | |
|---|---|
| convención de llamada | Windows `rcx rdx r8 r9`, System V `rdi rsi rdx rcx` |
| formato de objeto | `.type sym, @function` es de ELF y Mach-O lo rechaza |
| decoración de símbolos | Mach-O llama `_foo` a lo que ELF llama `foo` |

Nuevo job de CI que ensambla para ELF y Mach-O en segundos, en vez de
descubrirlo tras compilar 155 crates. Es `cargo build --lib` y **no**
`cargo check`: `check` no genera código, así que nunca ensambla el
`global_asm!` y pasa aunque el archivo esté roto — comprobado reintroduciendo
el bug a propósito. `--lib` evita necesitar un SDK, porque compilar a rlib no
enlaza.

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

- **La comprobación de límites costaba entre 4 y 12 veces el kernel entero.**
  Los diecinueve bucles de descuantización y desempaquetado escribían por
  índice calculado —`q[i * 256 + par * 64 + j] = ...`—, así que cada valor
  llevaba su comprobación de límites, y una comprobación dentro del bucle le
  prohíbe vectorizar a LLVM: iba a 0,43 ns por valor, unos 1,3 ciclos, que es
  un bucle escalar byte a byte en el camino más caliente del motor. Ahora la
  ventana se pide una vez con su longitud fija —`&mut q[k..k + 32]`, o
  `split_at_mut` cuando el formato intercala mitades— y adentro va un par de
  iteradores de longitud conocida. Desempaquetado a `i8` de los 22,5 millones
  de valores del `ffn_gate`, un hilo: Q4_K de 10,733 a 2,364 ms (4,5x), Q3_K de
  31,616 a 2,623 (12,1x), Q2_K de 18,114 a 4,192 (4,3x). La matmul en `f32`
  sobre Q3_K con una fila, de 5,827 a 1,126 ms; `matmul_i8_directo`, de 5,989 a
  0,880. Sigue siendo exacto: 21 tensores y 8 formatos **idénticos** a candle,
  `output.weight` de 311.164.928 valores en Q6_K incluido. Aparte, en Q2_K y
  Q3_K el bucle interno iba de 16 en 16 porque el formato tiene sub-bloques de
  16, pero los dos sub-bloques seguidos comparten corrimiento y máscara y son
  contiguos en el origen y en el destino, así que quedaba a **medio registro**
  de AVX2 sin motivo; fusionados a 32. Razonado en
  [DECISIONES](DECISIONES.md) §15.

- **El banco estaba sesgado a favor de quien fuera más rápido.**
  `measure_own_matmul` fijaba 20 repeticiones por muestra, así que una llamada
  de 0,2 ms daba una muestra de 4 ms que el ruido del reloj y del planificador
  se comía: dos pasadas seguidas del mismo binario, sin tocar una línea,
  midieron **0,205 y 0,361 ms** para candle con una fila, un 76% de diferencia,
  mientras lo nuestro —diez veces más lento— variaba un 4%. Una medición que
  depende de a quién mide no decide nada. Ahora las repeticiones se calibran
  para juntar 60 ms por muestra, con cinco muestras y mediana: esa celda pasó
  de ±76% a ±3%. Y queda anotado que en trabajo que recorre memoria esta
  máquina da hasta 1,8x entre pasadas, así que solo se comparan cifras tomadas
  en el mismo proceso.

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

[0.7.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.7.0
[0.6.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.6.0
[0.5.1]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.5.1
[0.5.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.5.0
[0.4.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.4.0
[0.3.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.3.0
[0.1.0]: https://github.com/Brashkie/Cosxamix/releases/tag/v0.1.0
