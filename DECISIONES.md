# Decisiones, con la medición que las sostiene

Este archivo existe porque las decisiones sin su medición se olvidan y se
revierten. Cada entrada dice **qué se probó, qué dio y qué se decidió** — y las
que salieron mal están acá igual, que son las que más ahorran tiempo.

Regla de la casa: nada entra por parecer buena idea. Entra por medir mejor que
lo que había, o por habilitar una plataforma donde no había nada.

---

## Índice

| | decisión | veredicto |
|---|---|---|
| 1 | Imitar el dialecto de Ollama en vez de inventar uno | adoptado |
| 2 | candle como motor, no llama.cpp | adoptado |
| 3 | Frontera de kernels con Zig | adoptado, por cobertura |
| 4 | D en el camino crítico | descartado |
| 5 | ASM, Fortran y Ada en la frontera | qué falta para que entren |
| 6 | Embeddings sin descuantizar | adoptado, −431 MB |
| 7 | Caché de atención por sesión | adoptado, −76% al intercalar |
| 8 | Qwen2.5-3B en vez de 0,5B o 1,5B | adoptado |
| 9 | Q3_K_M en vez de Q4_K_M o IQ3 | adoptado, −363 MB |
| 10 | Decodificación especulativa | **implementada y borrada** |
| 11 | Kernel fusionado propio de Q3_K | **no escrito: ya existe** |
| 12 | Redes de tensores (MPS) | aplazado, ataca el lado equivocado |
| 13 | QNLP / DisCoCat | fuera de alcance |
| 14 | El nombre: Cosxamix | adoptado, verificado en 9 fuentes |
| A-D | [las cuatro investigaciones recibidas](#apéndice-las-investigaciones-recibidas-una-por-una) | qué se tomó de cada una |

---

## 1. Imitar el dialecto de Ollama

WinsiBot le habla a Ollama por **exactamente dos endpoints**: `GET /api/tags` y
`POST /v1/chat/completions`. Implementándolos, Cosxamix es un reemplazo directo sin
tocar una línea del bot — solo apuntar `OLLAMA_URL`.

Se eligió eso sobre inventar un formato propio porque permite cambiar el motor
de abajo, o volver a Ollama, sin tocar a quien consume. Un formato propio
obligaría a modificar cada cliente y mantenerlo para siempre, a cambio de nada.

La única extensión es `cosxamix_sesion` en el cuerpo, que Ollama ignora sin
quejarse.

---

## 2. candle, no llama.cpp

| | candle | llama.cpp |
|---|---|---|
| dependencias | 155 crates, Rust puro | + toolchain de C++ por plataforma |
| build release | 84 s | minutos, con compilador instalado |
| cruza a `aarch64-linux-android` | **sí** | con NDK y pelea |

Decide la última fila: Termux es un objetivo, y un motor que pide compilador de
C++ en el teléfono reintroduce el requisito que se quería quitar.

**Trampa que solo se vio probando**: `tokenizers` trae por defecto el feature
`onig` — Oniguruma, una librería de C — y rompe el cruce a Android con
`failed to run custom build command for onig_sys`. Con
`default-features = false` usa `fancy-regex`, Rust puro, y toda la pila cruza.

---

## 3. Zig: se queda, pero no por lo que uno esperaría

Dos kernels implementados por triplicado (referencia Rust, Rust+AVX2, Zig):

**`dot`**, mediana de Criterion:

| n | escalar | AVX2 | Zig | Zig vs AVX2 |
|---|---:|---:|---:|---:|
| 1.024 | 1,039 µs | **54,1 ns** | 58,2 ns | +7,5% |
| 65.536 | 75,9 µs | 8,53 µs | **8,50 µs** | empate |
| 1.000.003 | 1,633 ms | **933 µs** | 1,010 ms | +8,2% |

**`rmsnorm`**:

| n | escalar | AVX2 | Zig |
|---|---:|---:|---:|
| 2.048 | 2,874 µs | **294,6 ns** | 375,4 ns |
| 8.192 | 12,185 µs | **1,970 µs** | 2,033 µs |

**En x86-64 Zig pierde por 3-8%.** Por velocidad sola, la regla 3 lo borraría.

Se queda por **cobertura**: el mismo `dot.zig`, sin tocar una línea, compila
para `aarch64-linux-gnu` y `aarch64-linux-android`. El camino de Rust es
`core::arch::x86_64`, solo x86 — en un teléfono cae a la referencia escalar, 19
veces más lenta. Un kernel portable que rinde casi como el específico vale más
que uno específico que en media flota no existe.

La regla 3 se reformuló por esto: *ganarle a la referencia **o** habilitar una
plataforma donde no había aceleración*.

**Coste de cruzar la FFI**, medido con funciones vacías: **~0,8 ns**. Para `dot`
con n=1024 es el 1,5% del total. Antes la guía afirmaba "menos de 1,5 ns" sin
respaldo; acertaba, pero no estaba medido.

---

## 4. D: fuera del camino crítico

`ldc2` usa LLVM, el mismo backend que Rust y Zig, así que en álgebra pura no
puede ser intrínsecamente más rápido. Y su runtime y GC no cruzan bien una FFI
de este tipo: habría que usar BetterC, y ahí pierde casi todo lo que lo hace D.

Súmese que Zig cruza a Termux sin cadena de herramientas y D no.

---

## 5. Los lenguajes que todavía no entraron, y qué los haría entrar

La frontera de `src/kernels/` no es "Rust y Zig": es Rust y **el lenguaje que
gane la medición**. Zig está porque cubre ARM, no porque sea el único
candidato. Lo que sigue es qué aporta cada uno y qué falta para que entre,
escrito para que la pregunta no haya que rehacerla.

### Ensamblador: escrito y medido

Hasta la 0.3.0 este punto decía "el método es bueno, el momento no" y se
apoyaba en un razonamiento: que el bucle que importa ya está escrito a mano en
candle. Razonar no es medir. Así que se escribió.

El kernel vive en **dos archivos de ensamblador sueltos**,
`kernels/dot_avx2_win64.s` y `kernels/dot_avx2_sysv.s`, no en `asm!` metido
entre Rust.

**El ensamblador es el único lenguaje de esta frontera que tiene que saber de
la plataforma, y resultó depender de tres cosas y no de una.** Rust, Zig y
Fortran no se enteran de ninguna:

| | |
|---|---|
| **convención de llamada** | Windows pasa en `rcx rdx r8 r9`, System V en `rdi rsi rdx rcx` |
| **formato de objeto** | `.type sym, @function` es de ELF; el de Mach-O lo rechaza con *unknown directive* |
| **decoración de símbolos** | en Mach-O una función de C se llama `_foo` para el enlazador, en ELF `foo` |

Las dos últimas se aprendieron rompiendo el build de macOS en CI, después de
compilar 155 crates. El archivo de System V sirve ahora para Linux y macOS
porque no lleva la directiva de ELF y define el símbolo con y sin guion bajo.

Y una trampa de método que costó una afirmación falsa: **`cargo check` no
sirve para comprobar esto**. No genera código, así que nunca ensambla el
`global_asm!` y pasa aunque el archivo esté roto. Hace falta
`cargo build --lib`, que sí compila y no necesita SDK porque no enlaza.
Comprobado reintroduciendo el bug a propósito: con `check` pasa, con
`build --lib` da el mismo *unknown directive* que CI.

Se ensamblan con LLVM por `global_asm!`, así que no hace falta ningún
ensamblador externo. Cuatro acumuladores, `vfmadd231ps` con el segundo operando
en memoria, 32 elementos por vuelta, y solo registros volátiles —`ymm0` a
`ymm5`— para no tener que salvar `xmm6`-`xmm15`, que en Windows x64 son no
volátiles. La cola y la suma horizontal son
**el mismo código** que la versión de intrínsecos, factorizado en
`x86::reducir_y_cola`, para que lo único que separe a las dos sea el bucle. El
resultado sale **bit a bit idéntico**, comprobado en largos de 0 a 4097.

Criterion, mediana:

| n | AVX2 (intrínsecos) | ensamblador | |
|---:|---:|---:|---|
| 1.024 | **49,4 ns** | 50,3 ns | +1,9% |
| 4.096 | **216,0 ns** | 217,1 ns | +0,5% |
| 65.536 | 7,96 µs | **7,79 µs** | −2,1% |
| 1.000.003 | **146,5 µs** | 183,4 µs | **+25%** |

**No gana.** Empata en los tamaños chicos, gana un 2% en uno, y pierde un 25%
en el más grande — donde LLVM claramente hace algo que el bucle a mano no:
prefetch, o un desenrollado distinto para datos que no entran en caché.

El kernel se deja en el árbol y en el banco, pero **fuera del despachador**:
no es un backend, es una medición que se puede repetir. Importa porque el
resultado **depende de la microarquitectura**, y hay un caso donde podría dar
vuelta: candle no usa **AVX-512** —`avx.rs` es de 256 bits— ni **SVE**
—`neon.rs` es de 128—, y esta máquina es un i5-10400, que no tiene AVX-512.
En una que lo tenga, duplicar el ancho del vector es algo que el compilador no
va a hacer solo mientras candle no se lo pida.

Lo que quedó claro, y vale más que el kernel: **contra un bucle que LLVM ya
vectoriza bien, escribirlo a mano no compra nada.** El sitio del ensamblador
es donde el compilador no tiene permiso para ir, no donde ya está.

### Cuándo entra el ensamblador, con un criterio comprobable

La pregunta vuelve, así que acá queda contestada con una medición y no con
"cuando el perfilado lo diga".

**El ensamblador gana donde el compilador se queda sin registros.** Es el
motivo por el que los microkernels de BLIS y OpenBLAS son ensamblador: cuando
hay que mantener doce o dieciséis acumuladores vivos a la vez, el asignador de
registros empieza a volcar a memoria, y volcar dentro del bucle caliente cuesta
más que cualquier cosa que el compilador gane reordenando.

Así que el criterio es inspeccionable, no opinable: **mirar el ensamblador
generado y contar volcados dentro del bucle**. Medido hoy sobre nuestros
kernels:

| kernel | líneas | registros `ymm` usados | volcados en el bucle |
|---|---:|---:|---:|
| `dot_i8_escalado_avx2` | 63 | 6 de 16 | **0** |
| `dot_i8_avx2` | 224 | 6 de 16 | **0** |
| `dot_avx2` | 84 | 8 de 16 | **0** |

Los cuatro `vmovaps [rsp...]` que aparecen en `dot_avx2` son el prólogo
guardando `xmm6` y `xmm7`, que en Windows x64 son no volátiles. No son
volcados de presión.

**Con 6 registros de 16 ocupados no hay nada que un humano pueda recuperar**, y
eso explica sin misterio por qué el `dot` en ensamblador salió bit a bit
idéntico y perdió un 25% en el tamaño grande.

#### Lo que sí abriría la puerta

Un **microkernel con bloqueo de registros**: calcular un bloque de salidas de
una vez —por ejemplo 4 columnas por 2 filas— manteniendo ocho acumuladores
vivos más los operandos. Eso sube la presión a propósito, y es además la
optimización que hace falta, porque reutiliza cada vector de pesos cargado para
varias filas en vez de releerlo.

El criterio queda así, y es falsable:

1. escribir el microkernel bloqueado con intrínsecos;
2. medirlo y **mirar si LLVM vuelca dentro del bucle**;
3. si vuelca, ahí va el ensamblador, y se compara contra la versión con
   intrínsecos — no contra código genérico.

Si no vuelca, no entra. Como hasta ahora.

### Se ejecutó el criterio, y la respuesta es que no

Se escribió `dot_q4k_i8_x4_avx2`: el kernel fusionado de Q4_K para **cuatro
columnas de una vez**, con los cuatro acumuladores vivos y la entrada cargada
una sola vez en vez de cuatro.

Los dos pasos del criterio, medidos:

| kernel | instrucciones | ymm usados | volcados a pila |
|---|---:|---:|---:|
| `dot_i8_escalado_avx2` | 151 | 6/16 | **0** |
| `dot_q3k_i8_avx2` | 277 | 8/16 | **0** |
| `dot_q4k_i8_avx2` | 568 | 8/16 | **0** |
| `dot_q4k_i8_x4_avx2` | 1031 | **12/16** | **0** |

LLVM subió a 12 registros y **no volcó nada**. Y lo segundo, que ni siquiera
hacía falta preguntar: el bloqueo es **más lento**, 855 ns contra 712 de
cuatro llamadas sueltas, un 20% peor.

**El motivo no tiene nada que ver con registros**, que es lo interesante.
Cuatro llamadas sueltas recorren **una fila contigua cada una**: acceso
secuencial perfecto, que es lo que el prefetcher hace bien. La bloqueada salta
entre cuatro filas separadas por el ancho de una fila entera, o sea cuatro
flujos entrelazados. Lo que se ahorra —una carga de la entrada, que está en
L1— cuesta menos que lo que se pierde.

Y el ensamblador no arregla eso: el patrón de acceso es del algoritmo, no de
la selección de instrucciones. Una persona escribiendo a mano choca con los
mismos cuatro flujos.

Por la regla 3 el microkernel se borró. Y con él se cierra la pregunta que
esta sección dejaba abierta desde hace versiones:

> **No hay ninguna apertura para el ensamblador en este código hoy.** Se midió
> la presión de registros de los cuatro kernels calientes —ninguno pasa de
> 12 de 16, ninguno vuelca— y se construyó el único algoritmo que se suponía
> que la crearía, que resultó más lento por una razón ajena a los registros.

Lo que **sí** apareció en el camino, y es lo contrario de lo que se buscaba:
el bloqueo que gana ya estaba puesto, y es el otro. `matmul_i8_fusionado`
recorre una columna de pesos y la usa para las `m` filas de entrada: un solo
flujo de pesos, contiguo, reutilizado. Bloquear por el lado de la salida
rompía justamente eso.

### Qué es exactamente lo que hay en `kernels/*.s`, dicho sin adornos

Conviene que esto esté escrito, porque mirando el repositorio la pregunta
aparece sola: los dos `.s` cubren `cosxamix_dot_bloques`, que es el producto
punto de `f32` —el **primer** kernel del proyecto— y **no** cubren los
fusionados de Q4_K y Q3_K, que son el camino caliente de hoy. El ensamblador
no siguió al resto.

Eso no es un olvido, y tampoco es que dé igual. Es el resultado de lo medido
arriba: ninguno de los cuatro kernels calientes pasa de 12 de 16 registros,
ninguno vuelca, el único algoritmo que iba a crear presión salió 20% más
lento, y adelantar el trabajo escalar fuera del bucle vectorial no movió la
aguja —188,9 ns contra 186,2, dentro del ruido—, o sea que LLVM ya lo estaba
programando bien. Tres hipótesis, tres mediciones, ninguna apertura.

Para qué siguen entonces: están **verificados idénticos a los intrínsecos**
por un test, y un job de CI comprueba que ensamblan en los tres formatos de
objeto —ELF, COFF y Mach-O—, que es donde se rompió un release una vez por
una directiva `.type` que solo existe en ELF. Ese valor es real y no depende
de que sean el camino rápido.

**Qué los haría crecer.** Una medición, no una intención. En concreto,
cualquiera de estas tres:

1. que un kernel caliente empiece a volcar registros a la pila;
2. que aparezca una instrucción que el compilador no emite y la CPU tenga
   —`vpdpbusd` de AVX-VNNI haría en una lo que hoy son dos, `vpmaddubsw` más
   `vpmaddwd`, pero este i5-10400 es Comet Lake y no la tiene, así que medirlo
   acá es imposible—;
3. que el perfil muestre al planificador de LLVM perdiendo ciclos en un bucle
   concreto.

### El ensamblador del kernel fusionado, y lo que midió

`kernels/q4k_avx2_win64.s` y `kernels/q4k_avx2_sysv.s` son el bucle vectorial
del fusionado de Q4_K, partiendo del que genera el compilador y limpiado a
mano: las constantes armadas con inmediatos en vez de leerse de una sección de
datos —que se declara distinto en COFF, ELF y Mach-O—, etiquetas numéricas
locales para no chocar con las que genera rustc dentro de `global_asm!`, y el
de System V sin `.type` y con el símbolo con y sin guion bajo, así sirve a ELF
y Mach-O a la vez. Solo `ymm0`–`ymm5`, sin prólogo. Un solo índice que avanza
de a 8 hace de tres punteros con las escalas del direccionamiento: `×4` son 32
bytes de `qs`, `×8` son 64 de `xq`, `×1` son dos `f32` de `esc`.

**Es correcto**: los tres tests pasan, contra la referencia escalar, contra los
intrínsecos y con filas de uno y de diecisiete bloques. Y mide así, en el
mismo proceso, dos pasadas, con las escalas y la reducción calculadas con el
**mismo código** en los dos caminos para que lo único distinto sea el bucle:

| camino | tiempo |
|---|---:|
| escalar | 690 ns |
| Zig | 296–301 ns |
| **ensamblador** | **107–110 ns** |
| intrínsecos | 92–93 ns |

6,4x sobre el escalar y 2,8x sobre Zig; 1,16x detrás de los intrínsecos. Y la
causa está medida, no supuesta: de las 82 instrucciones que ejecuta por
bloque, **14 son de armar y desarmar la llamada** —leer `acc`, reconstruir las
dos constantes, guardar `acc`, `vzeroupper`, volver—, o sea el 17%. Los
intrínsecos hacen eso una vez por fila; este kernel, una vez por bloque, ocho
por fila del 3B. **No es el bucle, es el contrato**: una llamada por bloque.

Para cerrar esa distancia el ensamblador tendría que recorrer la fila entera
en una llamada, con el acumulador y las constantes en registros de punta a
punta. Eso pide cambiar el contrato: el paso entre bloques —144 bytes, con
`qs` a 16 de cada comienzo— y las escalas de todos los bloques de la fila.

**El contrato por fila ya está del lado de Rust**: `cosxamix_q4k_fila`, con los
mismos cinco argumentos en las mismas posiciones —así la entrada de la ABI no
cambia— pero recorriendo bloques enteros. De los tres punteros, solo `qs` no es
contiguo a lo largo de la fila; `xq` y `esc` sí, y avanzan por par igual que en
el contrato por bloque. Rust calcula las escalas de a 64 bloques, 2 KB de pila,
y llama una vez por tanda: una sola llamada para cualquier fila del 3B.
`build.rs` detecta cada contrato por su **etiqueta** y no por el nombre a
secas, porque el nombre aparece en los comentarios de cabecera y eso daba por
presente una función ausente; puede haber uno, el otro o los dos en el mismo
archivo, y el bench los compara.

### El contrato por fila: el bucle gana, el kernel pierde

`cosxamix_q4k_fila` quedó escrito en los dos `.s` —un índice global para `xq` y
`esc` que no se reinicia entre bloques, un puntero propio para `qs` que salta
144 por bloque—, pasa los tres tests y da bit a bit lo mismo que el contrato por
bloque. Pero de punta a punta salió **más lento**, y hubo que desarmar el
porqué en capas.

**Una: el arreglo puesto en cero.** El envoltorio declaraba
`[0f32; TANDA * 8]`, 2 KB escritos en cada llamada aunque una fila de 8 bloques
usa 256 bytes, más una copia con comprobación de límites. La hipótesis vino del
lado del ensamblador, sin medirla, y era correcta: sin ponerlo en cero, de
139–141 a **120–121 ns**.

**Dos: el bucle aislado.** Midiendo los dos bucles con las escalas calculadas
fuera de lo cronometrado:

| | tiempo |
|---|---:|
| bucle por fila | **61–62 ns** |
| bucle por bloque | 71–73 ns |

El bucle por fila es **14% más rápido**, que es justo lo que se ahorra al no
armar la llamada en cada bloque. El ensamblador hizo exactamente lo que se le
pidió, y su bucle solo cuesta menos que el kernel entero de intrínsecos.

**Tres: lo que lo hunde es el envoltorio.** Por fila calcula todas las escalas
en una pasada aparte, las copia a un buffer y recién después llama. Llamarlo en
tandas chicas no lo arregla, y lo prueba un caso: con tanda 1 hace una llamada
por bloque, igual que el contrato por bloque, y aun así mide 130 ns contra 108.
La diferencia no está en el ensamblador sino en cómo se le entregan las escalas.

**Lo que haría ganar al ensamblador** es dejar de entregárselas: que calcule
las escalas él mismo, intercaladas con el bucle bloque por bloque, en una sola
pasada sin buffer y sin llamada por bloque. Ahí además puede usar `vcvtph2ps`,
que convierte los `f16` de `d` y `dmin` con una instrucción, donde el Rust de
hoy los convierte por software.

### El contrato completo: el primer ensamblador en el camino caliente

`cosxamix_q4k_completo` calcula todo en una sola pasada: `d` y `dmin` con
`vcvtph2ps`, las escalas con el desempaquetado sin ramas, el bucle con los 4
pares desenrollados, y el desplazamiento sobre `acc[8..16]`. En Windows no toca
ningún registro no volátil —así no necesita directivas de desenrollado— y
guarda las 8 escalas del bloque en el espacio de sombra; en System V, en la zona
roja. Pasa los tres tests, que lo recorren junto a los otros dos contratos.

Contra los intrínsecos de entonces ganaba **1,28x a 1,32x**. Pero no se le
atribuyó esa ganancia sin antes separarla, y fue lo correcto:

| | |
|---|---:|
| intrínsecos, `f16` por software | 96–99 ns |
| **intrínsecos con `vcvtph2ps`** | **78–79 ns** |
| ensamblador completo | 74–77 ns |

**La mayor parte era la instrucción.** Dándosela a los intrínsecos, sin tocar
nada más, pasaron de ~97 a ~79 ns: 1,24x. El Rust convertía los `f16` de cada
bloque por software desde siempre y nadie lo había notado, hasta que el
ensamblador usó la instrucción y ganó. Se pasó a `escalas_q4k`, y F16C entró a
la detección del camino AVX2 —todo procesador con AVX2 la tiene, pero una
máquina virtual puede esconder una extensión y dejar la otra—.

Con la misma instrucción de los dos lados, en tres pasadas:

| | 1 | 2 | 3 |
|---|---:|---:|---:|
| intrínsecos | 79,9 | 78,4 | 78,1 |
| **ensamblador completo** | **73,8** | **75,6** | **75,4** |
| ventaja | 1,08x | 1,04x | 1,04x |

Chica, pero en las tres pasadas en la misma dirección y con los intervalos sin
cruzarse. Por la regla 3, **el ensamblador entra primero en el despachador**
cuando `asm_q4k` está encendida: el primer ensamblador del proyecto en el camino
caliente, y entró midiendo, como cualquier otro kernel.

**De punta a punta.** En la matmul del 3B sobre Q4_K, el fusionado con una fila
—el caso del decode— queda **parejo con candle**, entre 0,81x y 1,09x en el mismo
proceso, donde antes perdía entre 0,55x y 0,85x: eso es `vcvtph2ps` llegando al
motor. En el 0,5B, generar con y sin el ensamblador da lo mismo, 28 tok/s, porque
ahí casi ningún tensor pasa por este kernel: el ancho, 896, no es múltiplo de
256, y solo `ffn_down` lo usa.

**Una lección de medición.** La primera comparación de punta a punta daba el
motor con ensamblador un 30% más lento. Era ruido, y con causa: con `cargo run`,
cambiar de features dispara una recompilación justo antes de medir, y la corrida
sale con el procesador caliente. Con los dos binarios compilados de antemano y
alternados, ronda por ronda, salen parejos. El motor completo se mide así.

**Una optimización que se probó antes de pedirla, y no servía.** El
ensamblador acumula el desplazamiento en memoria, `acc[8..16]`, con una lectura
y una escritura por bloque, porque no le queda un `ymm` libre sin tocar `xmm6`.
Tenerlo en un registro pediría salvar `xmm6`, y en Windows eso obliga a declarar
el prólogo con `.seh_savexmm` para el desenrollado. Antes de pedir ese cambio se
probó lo inverso en los intrínsecos, que ya lo tienen en registro: una variante
que lo lee y lo escribe en memoria en cada bloque, con accesos volátiles para que
LLVM no lo devolviera a un registro —verificado en el ensamblador generado, 11
movimientos con la pila contra 2—.

| | 1 | 2 | 3 |
|---|---:|---:|---:|
| en registro | 77,3 | 81,0 | 76,0 |
| en memoria | 78,7 | 78,7 | 77,5 |

La diferencia cambia de signo entre pasadas y no pasa del 3%. **El procesador
esconde la ida y vuelta**, porque la cadena del desplazamiento no depende de la
del acumulador principal. Así que la forma del ensamblador era la correcta: no
salvar `xmm6` le evita las directivas de desenrollado a cambio de nada que se
pueda medir. Es la segunda vez que probar en Rust antes de pedir un cambio en el
ensamblador evitó trabajo inútil; la primera fueron los acumuladores múltiples.

**Lo que no se ejecutó todavía**: el de System V. Ensambla para ELF y Mach-O,
pero se desarrolla en Windows. CI lo ejecuta ahora en `ubuntu-latest` con
`asm_q4k` en la matriz; el de Mach-O x86-64 sigue solo ensamblado, porque
`macos-latest` es ARM.

### ARM: donde el ensamblador no es una opción sino el único camino

En x86 el ensamblador le gana a los intrínsecos por un 4-8%, porque los dos
usan las mismas instrucciones. En ARM64 la pregunta es otra, y se comprobó
antes de escribir nada:

- **Rust estable no expone `sdot`.** `vdotq_s32` da E0658, detrás de
  `stdarch_neon_dotprod`. Lo que sí es estable es
  `is_aarch64_feature_detected!("dotprod")`: se puede preguntar si está, pero
  no usarla;
- **Zig no la genera.** `cosxamix_dot_q4k_i8_f32` compilado con
  `-mcpu=cortex_a76`, que la tiene, sale con 26 `smull`/`smlal` y ningún
  `sdot`: LLVM no reconoce el patrón de ensanchar y multiplicar.

`sdot` hace 16 productos de `i8` por instrucción, cuatro por carril de 32 bits,
que es la forma exacta del producto de Q4_K. Hoy el camino vectorizado de ARM es
el de Zig, que ensancha a `i32`. Así que el kernel NEON es la primera vez que un
lenguaje del proyecto entra **por habilitar algo que los otros no pueden**, que
es la segunda mitad de la regla 3.

El andamio está puesto con el mismo contrato `completo`, en AAPCS64: un solo
`kernels/q4k_neon.s` para Linux, Android y macOS, con `.arch_extension dotprod`
para que el ensamblador acepte la instrucción, sin tocar `v8`–`v15` ni `x18`.
`build.rs` lo detecta por la etiqueta, como a los de x86, y si falta avisa sin
romper nada.

**Lo que esta máquina no puede hacer es ejecutarlo**: es x86-64, sin WSL ni
QEMU. Se verificó lo que sí se puede —clippy para `aarch64-unknown-linux-gnu`
con el módulo, sus tests y el bench, con y sin Zig— y la ejecución queda en dos
lugares: el job `arm` de CI, en un runner ARM64 nativo con `dotprod`, que
prueba que **calcula bien**; y un teléfono con Termux, que es la plataforma
objetivo y la única medición que decide si **entra al despachador**.

### Las escalas sin ramas, y una regresión de 2x que salió de escribirlas mal

Las 8 escalas y los 8 mínimos de 6 bits se rearman ahora con máscaras sobre
tres palabras de 32 bits, sin ramas —la técnica de llama.cpp para estos mismos
bloques—, en lugar de 8 llamadas a `escala_y_minimo` con su `if j < 4`. Un test
lo compara **bit a bit** contra la versión legible sobre 5.000 bloques al azar.
Aislado calcula las escalas un 23% más rápido; en los tres kernels **no se
nota**, porque ese trabajo ya se superponía con el vectorial.

Pero la primera versión **duplicó** el tiempo del kernel de intrínsecos, de 90 a
178 ns, con un escalar de control que no se movió. El ensamblador generado
mostró por qué: leía los 12 bytes con `bloque[4 + i]` uno por uno, y las doce
comprobaciones de límites con sus caminos de pánico agrandaron la función lo
suficiente para que LLVM **dejara de ponerla en línea**. Con una llamada por
bloque y el resultado devuelto por memoria se perdió la superposición. Leyendo
los 12 bytes como un solo arreglo de tamaño fijo —una comprobación— vuelve a
quedar en línea y a 92 ns. La técnica estaba bien; la hundió cómo estaba
escrita, que es el mismo tipo de error que el de §15.

### Una hipótesis sobre el bucle que no se sostuvo

Antes de cambiar el contrato se probó otra explicación de la lentitud. En cada
par, los dos `vfmadd213ps` encadenan el acumulador: cada uno espera el
resultado del anterior, unos 4 ciclos, así que hay un piso de 8 ciclos por par.
Con varios acumuladores independientes las cadenas se solapan.

Se probó en los intrínsecos, donde se puede tocar sin costo, antes de pedir
tocarlo en el ensamblador:

| | pasada 1 | pasada 2 |
|---|---:|---:|
| 1 acumulador | 94,1 ns | 104,8 ns |
| 2 acumuladores | 90,5 ns | 104,5 ns |
| 4 acumuladores | 92,1 ns | 95,8 ns |

En cada pasada gana uno distinto, y las diferencias quedan dentro del ruido de
la máquina. **La cadena no es el cuello**: cada par tiene suficiente trabajo
independiente —cargas, `maddubs`, `madd`, conversiones— para que la CPU cubra
la espera del FMA. Se borró por la regla 3, y de haberse pedido en el
ensamblador sin probarlo, habría sido trabajo para nada.

Por la regla 3 el despachador sigue eligiendo los intrínsecos, igual que con
`dot_avx2_*.s`. La regla decide qué va primero en el camino caliente, no qué
lenguajes tiene el proyecto: el ensamblador queda compilado con
`asm_q4k`, con sus tests y en el bench.

Para el armado del lado de Rust hubo que corregir dos cosas del andamio. El
puente calculaba las escalas de a una, que es la forma lenta reemplazada por
el 1,90x, y medido así el ensamblador habría parecido peor por culpa del
envoltorio; ahora las calcula `x86::escalas_q4k`, compartida con los
intrínsecos. Y el test armaba los encabezados de los bloques con bytes al
azar, que con 17 bloques dieron un `f16` infinito: el kernel y la referencia
daban NaN los dos, y como NaN no es igual a NaN el test fallaba aunque
coincidieran.

#### Un detalle chico a favor del escrito a mano

LLVM usó `ymm6` y `ymm7` en `dot_avx2` y pagó por eso un prólogo y un epílogo
en Windows. El `.s` escrito a mano se limita a `ymm0`-`ymm5`, que son volátiles,
y no necesita ninguno de los dos. Es poco, pero es una decisión que el
compilador no tomó y un humano sí.

### Fortran

El argumento es real y no es nostalgia: **BLAS y LAPACK son Fortran**, y no por
inercia. Sus arreglos no admiten aliasing por defecto, y eso le deja al
compilador vectorizar sin las comprobaciones que un `*mut f32` de C obliga a
hacer. `gfortran` y `flang` generan código numérico excelente.

**Se intentó medirlo y está bloqueado por el toolchain, no por el argumento.**
El `gfortran` de esta máquina es MinGW de 32 bits —`-dumpmachine` dice
`mingw32` y el objeto que produce es COFF i386— mientras que Rust compila para
`x86_64-pc-windows-msvc`. Los objetos no enlazan, así que no hay comparación
posible acá.

Lo desbloquea un gfortran de 64 bits —el de MSYS2 `mingw-w64`— o `flang`.
Hasta entonces el argumento queda sin comprobar, que no es lo mismo que
descartado.

Y después queda la plataforma: haría falta un compilador de Fortran en cada
objetivo, y para `aarch64-linux-android` eso es exactamente el requisito que el
proyecto se quitó de encima al elegir candle sobre llama.cpp.

### Ada y SPARK

Lo que aporta no es velocidad: GNAT usa el mismo backend que los demás. Es
**verificación**. SPARK puede demostrar la ausencia de desbordamientos y de
accesos fuera de rango en un kernel, no comprobarla con tests.

Eso vale justo donde este proyecto es más frágil: los kernels corren con
`-OReleaseFast` y sin comprobaciones de desbordamiento, y lo único que hoy los
protege son los tests contra la referencia. Un kernel probado formalmente sería
más fuerte que uno probado con casos.

**No hay compilador instalado en esta máquina** —ni `gnat` ni `gnatmake`—, así
que tampoco se pudo probar. Es lo primero que haría falta para pasar de la idea
al dato.

### La regla sigue siendo la misma

Ninguno entra por ser mejor en abstracto. Entra el que **le gane a la
referencia midiendo**, o el que habilite una plataforma donde no había
aceleración — que es por lo que está Zig, que en x86-64 pierde un 3-8%.

---

## 6. Embeddings: 938 → 506 MB

La guía de teoría atribuía el exceso de RAM a que la carga convencional
"duplica el consumo". Medido, la causa era otra:

| | |
|---|---:|
| archivo en disco | 468,6 MB |
| tensores cuantizados | 463,0 MB |
| **tabla de embeddings pasada a f32** | **519,3 MB** |
| medido en el proceso | 938,1 MB |

`from_gguf` hacía `dequantize()` sobre la tabla entera, que pesa **más que el
modelo completo**. Un forward solo necesita las filas de los tokens que entran:
3,5 KB por token. Ahora queda cuantizada y se descuantizan solo esas filas.

**`mmap` habría ahorrado mucho menos.** La relación RAM/disco pasó de x2,0 a
**x1,03**.

### Un bug de candle encontrado acá

La primera versión generaba basura. Aislando: de siete filas pedidas, seis
salían exactas y solo fallaba **la primera del recorte** — el mismo índice 0
estaba bien si no iba primero.

```rust
fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {   // toma el Cow POR VALOR
    let ptr = data.as_ptr();
    unsafe { std::slice::from_raw_parts(ptr as *const T, ...) }
}                                                  // y aquí libera el Vec
```

Con `Cow::Owned` el `Vec` lo posee la variable local y se libera al salir; el
slice queda colgando y el `.to_vec()` de quien llama lee memoria liberada. Casi
toda sigue intacta —por eso seis de siete salían bien— menos los primeros
bytes, que el asignador pisa con su lista de libres.

Upstream no lo nota porque siempre pasa `Cow::Borrowed`. Haciendo lo mismo, la
diferencia contra descuantizar la tabla entera es **exactamente cero**.

---

## 7. Caché de atención por sesión

De fábrica hay **uno solo**, dentro del modelo. Con conversaciones intercaladas
—lo normal en WhatsApp— cada una invalidaba la de la otra:

| dos conversaciones, mismas 6 peticiones | tiempo | reaprovechados |
|---|---:|---:|
| secuencial | 5541 ms | 244 |
| intercalado | 9770 ms | **0** |

Las dos salidas obvias se midieron y se descartaron. **Un modelo por sesión**:
931 MB cada instancia, diez chats serían 9,1 GB. **La API del crate**:
`ModelWeights` no es `Clone` y su `kv_cache` es privado.

Así que `quantized_qwen2.rs` pasó al repositorio con tres métodos agregados
—`tomar_cache`, `poner_cache`, `limpiar_cache`—. Resultado: **intercalado y
secuencial empatan**, 244 tokens reaprovechados en los dos órdenes.

### Reutilizar el prefijo común, no "solo si extiende"

La primera versión exigía que el prompt nuevo empezara con *todo* lo cacheado.
Funciona en una conversación que crece, pero **no en el caso real de WinsiBot**,
que manda `system + pregunta` sin historial: el turno 2 comparte el system y
después diverge.

Medido contra el bot de verdad: **`cacheados=0` siempre**. El caché no servía.

Con prefijo común más largo y `truncar_cache`, pasa a **131** — el system
entero, con perfiles y catálogo de comandos, que antes se reprocesaba en cada
mensaje.

### Por qué el texto cambia y no es un bug

Con el caché puesto el mismo prompt da otro texto. Costó dos pruebas
demostrar que no está roto.

**La primera** fijó el prompt en 69 tokens y comparó tres caminos:

| | caché | prefill | salida |
|---|---|---|---|
| A | no | en bloque | *…controlar chats y mucho más* |
| B | no | de a un token | *…controlar tu conversación y mucho más* |
| C | **sí** | de a un token | *…y mucho más* |

**A y B difieren y ninguno usa caché.** Lo único que cambia es la forma de la
matriz: `gemm` acumula en otro orden según el bloque.

**La segunda** comparó los logits con un prompt real — con ids inventados la
distribución queda plana y el argmax lo decide el ruido:

```
rango de los logits  33,42
diferencia maxima     0,68   (2,0% del rango)
token mas probable   68012 sin cache | 68012 con cache  ->  IGUAL
```

---

## 8. Qwen2.5-3B

Tres tamaños, mismo cuantizado, 15 preguntas de respuesta verificable
—aritmética encadenada, seguir formato, conocimiento, trampas, negaciones— y
velocidad como mediana de 3 repeticiones:

| | 0,5B | 1,5B | 3B |
|---|---:|---:|---:|
| residente | 504 MB | 1108 MB | 2051 MB |
| decode | ~32 tok/s | 19,3 | 10,0 |
| respuesta de 40 tokens | ~2,1 s | ~2,9 s | ~5,6 s |
| aciertos | 7/15 | 8/15 | **10/15** |

**El 1,5B no es el punto dulce**, contra lo que se había recomendado: cuesta
1,7x de velocidad y gana una pregunta de quince, que es ruido. El salto está en
el 3B, que acierta el kilo de plomo contra el de plumas, los 206 huesos y
Canberra, y contesta "pingüino" donde el 0,5B dice "girafas" y el 1,5B
"urálico".

La primera versión de esta prueba tenía **6 preguntas triviales** y comparaba
por palabra clave; daba 5/6 para los dos y no distinguía nada. Una prueba sin
resolución es peor que ninguna, porque parece que decide.

---

## 9. Q3_K_M, y por qué no IQ3

**candle no soporta los formatos IQ.** Su enumeración llega a los k-quants
clásicos; IQ3_M e IQ2_XXS son de llama.cpp.

Q3K es lo más parecido disponible. Mismo 3B, mismas 15 preguntas:

| | disco | residente | decode | aciertos |
|---|---:|---:|---:|---:|
| Q2_K | 1313 MB | — | 13,7 tok/s | **6/15** |
| **Q3_K_M** | **1644 MB** | **1687 MB** | **11,9** | **10/15** |
| Q4_K_M | 2007 MB | 2051 MB | 10,8 | 10/15 |

Q3 acierta lo mismo que Q4 con 363 MB menos y 10% más rápido.

**Q2 se rompe**, y vale ver cómo: dice *"canossa"* por Canberra, *1972* por
1969, *"20 mandile"* donde va un número, y mete acentos franceses en medio del
español. No falla con un error — responde cosas que casi parecen bien.

---

## 10. Decodificación especulativa: implementada y borrada

La idea: el 0,5B propone K tokens y el 3B los verifica en una pasada, leyendo
sus pesos una vez en vez de K. Promete 2-3x.

Se implementó entera —borrador, verificación, recorte de caché— y salió
**1,91x más lenta** (46.045 ms contra 24.128). Se borró por la regla 3.

### Por qué no puede funcionar acá

La técnica supone que procesar K posiciones cuesta casi lo mismo que una.
Medido sobre el 3B:

| posiciones | ms | vs 1 |
|---:|---:|---:|
| 1 | 112 | 1,00x |
| 2 | 202 | 1,80x |
| 5 | 420 | 3,74x |
| 8 | 663 | 5,91x |

**Escala casi lineal.** La primera sospecha fue la capa de salida —151.936
tokens de vocabulario— pero resultó ser el **15%** del modelo, y el 15% no
explica 3,74x.

> **Revisado dos veces en la 0.3.0.** La primera revision concluyo que el motor
> esta limitado por memoria y que la culpa era del orden de los bucles de
> candle. El experimento la desmintio. Queda lo que la evidencia sostiene, y
> queda escrito el error porque costo medio dia.

Hay **dos regimenes distintos**, y confundirlos fue el error:

**Con una sola fila —el decode, que es el caso dominante— manda la memoria.**
Medido matriz por matriz a 6 hilos, el escalado depende de si la matriz entra
en los 12 MB de L3:

| matriz | pesos | 6 hilos |
|---|---:|---:|
| attn q · 2048→2048 | ~1,8 MB | **7,03x** |
| FFN gate · 2048→11008 | ~9,6 MB | 2,82x |
| FFN down · 11008→2048 | ~9,6 MB | 1,76x |
| salida · 2048→151936 | ~133 MB | 2,25x |

Eso explica el escalado de hilos del forward completo: ponderando cada matriz
por su participacion da **2,47x** contra los **2,60x** medidos.

**Con mas de una fila manda el calculo**, y es lo que hace cierta la conclusion
original. El bucle de filas de candle es el externo y relee la matriz entera
por fila, lo que parecia la causa:

```rust
for row_idx in 0..m {                    // en serie
    dst_row.into_par_iter().for_each(|col_idx, dst| { ... })
```

Se escribio el kernel con los bucles al reves —columnas afuera, filas adentro,
reusando la columna ya leida— y **no gano nada**: entre 0,91x y 1,17x segun la
corrida, dentro del ruido.

El motivo es que intercambiar los bucles ahorra releer **bytes**, pero
`vec_dot` rehace el desempaquetado del k-quant por cada par (fila, columna) de
todas formas. Para comprobarlo se escribió una segunda variante que descuantiza
cada columna una sola vez y después hace `m` productos punto en f32 — o sea que
elimina ese trabajo repetido por completo. Salió entre 5 y 25 veces más lenta:

| filas | candle | bucles al revés | descuantizar una vez |
|---:|---:|---:|---:|
| 1 | 0,292 ms | 1,05x | **0,04x** |
| 8 | 1,743 ms | 1,10x | 0,20x |

> **Corregido en la 0.3.0, y el error era mío.** Esa segunda variante reservaba
> el buffer de la columna **dentro** del bucle de columnas: once mil
> asignaciones por multiplicación. No medía el coste de descuantizar, medía el
> del asignador.
>
> Rehecha con un buffer por tramo, sobre los pesos reales del `ffn_gate` del
> 3B, **sí amortiza**:
>
> | filas | candle por fila | nuestra por fila |
> |---:|---:|---:|
> | 1 | 0,637 ms | 6,728 ms |
> | 2 | 0,636 | 3,010 |
> | 5 | 0,762 | 1,319 |
> | 8 | 0,773 | **1,072** |
>
> El coste por fila cae **6,3 veces** mientras el de candle se queda plano. La
> conclusión que se sostiene no es "separar el desempaquetado es inviable",
> sino otra: **separarlo amortiza, y lo que queda caro es el producto en f32**.
> A ocho filas la nuestra ya está cerca del techo aritmético de f32 en esta
> máquina, mientras candle multiplica en enteros sobre pesos de 3 bits.
>
> O sea que el camino no es volver a `vec_dot` por par, ni materializar f32:
> es **desempaquetar a registros una vez y hacer los `m` productos en enteros
> mientras el bloque sigue ahí**. Está en el [ROADMAP](ROADMAP.md) §9.

Los dos kernels se borraron por la regla 3.

**Conclusion que queda:** procesar K posiciones cuesta ~K de verdad, y la causa
es el trabajo aritmetico de descuantizar, que ningun orden de bucles evita. La
especulativa no puede ganar aca, y no por el hardware ni por un detalle de
candle: por lo que cuesta un k-quant.

**Vuelve a tener sentido** con un formato de pesos cuyo desempaquetado sea
barato o inexistente, o si el modelo pasara a f16 — ahi los pesos se leen una
vez y el lote si amortiza.

### Una advertencia sobre estas mediciones

Los milisegundos absolutos de esta seccion **no son estables entre corridas
distintas**. El FFN gate pesa 9,7 MB contra 12 MB de L3, asi que su tiempo
depende de que mas haya residente: la misma multiplicacion dio 1,10 ms en un
proceso con 150 MB de tensores cargados y 0,24 ms en uno con un solo tensor.
Verificado que no eran ni los pesos reales contra sinteticos ni la entrada en
cero contra datos: los dos pares dan lo mismo.

Lo comparable son **las relaciones dentro de una misma corrida**, no las cifras
entre secciones.

A eso se suma que el 0,5B coincide con el 3B en el **28-50%** de los tokens.

**Vuelve a tener sentido** si el motor se acerca al techo de memoria, o con un
borrador que coincida mucho más.

### Lo que quedó

Dos piezas que hicieron falta para intentarlo y que valen solas:

* **La máscara causal contempla el caché.** La original era cuadrada y solo
  valía con el caché vacío. Arreglada, el delta entra por lotes: el banco de
  intercalado pasó de 5422 a **4772 ms**.
* **`truncar_cache`**, que candle no expone — y sin la cual el caché por prefijo
  común del punto 7 no habría sido posible.

La regla 3 borra lo que no rinde, no lo que se aprendió haciéndolo.

---

## 11. El kernel fusionado propio: se escribió, y había que escribirlo

Se propuso escribir un `fused_dequant_gemv` para Q3_K, partiendo de que la ruta
convencional descuantiza a f32 y después multiplica.

**candle no hace eso.** Tiene `vec_dot_q3k_q8k` en `avx.rs`: un kernel
fusionado en intrínsecos AVX2 escritos a mano, derivado del de llama.cpp, que
desempaqueta los bloques directo a registros y acumula con FMA.

Competir contra eso tiene muy poca chance. Por la regla 3 se borraría igual,
solo que después de días de trabajo.

### Revertido, y por una medición

Eso era un pronóstico, no un número, y el número dice otra cosa. Se escribió:
`kernels::dot_q4k_i8`, con referencia escalar y versión AVX2, y
`ops::matmul_i8_fusionado`.

Sobre el `ffn_gate` del 3B en Q4_K, 2048 → 11008, todo en el mismo proceso:

| filas | candle | i8 directo | **fusionado** | fusionado vs candle |
|---:|---:|---:|---:|---:|
| 1 | 0,912 ms | 1,426 | **1,079** | 0,85x |
| 2 | 1,574 | 1,500 | **1,224** | **1,29x** |
| 5 | 4,037 | 2,734 | **2,179** | **1,85x** |
| 8 | 6,279 | 3,937 | **3,252** | **1,93x** |

En tres pasadas, el fusionado le gana al camino con buffer entre **1,16x y
1,44x** en todas las filas, y contra candle va de 0,82-0,92x con una fila a
1,93-2,39x con ocho. El caso de una fila, que estaba en 0,55x, queda en 0,85x:
candle sigue adelante pero por un 15%, no por el 80%.

Exactitud sin cambios: 0,26% de diferencia contra candle, a nuestro favor.

**Lo que el pronóstico no vio.** Dos cosas, y ninguna es "esforzarse más":

- **Q4_K no necesita el rodeo del signo.** Sus nibbles son enteros **sin
  signo**, 0 a 15, y la entrada cuantizada es con signo: eso es literalmente
  la firma de `vpmaddubsw`, `u8 × i8 → i16`. El kernel de `dot_i8_escalado` usa
  `vpsignb` para fabricar esa combinación a partir de dos operandos con signo,
  y acá no hay nada que fabricar. Se va también su único caso límite, el del
  −128 en los dos lados a la vez. Y no puede saturar: 15 × 127 = 1905, dos de
  esos son 3810, contra 32.767 que entran en un `i16` — 8,6 veces de margen.
- **Una fila en Q4_K son 1152 bytes.** Cabe de sobra en L1, así que las `m`
  filas de salida la reusan desde ahí. El camino con buffer tocaba `k` bytes
  de `i8` más sus escalas por cada columna.

Lo que el pronóstico **sí** acertó: competir con candle a una fila es duro, y
ahí sigue adelante. Pero perder por un 15% mientras se gana 1,9x en lote no es
lo mismo que no haberlo escrito.

### Q3_K también, y con el rodeo que le corresponde

Q3_K es el formato dominante del modelo por defecto, así que era el siguiente.
Sus valores van de **−4 a 3**, con signo, y `vpmaddubsw` quiere un operando sin
signo. El rodeo es sumarles 4 —con lo que quedan en 0 a 7— y descontarlo
después:

```text
    Σ x·q  =  Σ x·(q + 4)  −  4·Σx
```

y `Σx` por sub-bloque es exactamente `xs`, que el multiplicador ya calcula una
vez por fila. La corrección sale gratis. Su sub-bloque es de 16, que se
resuelve con el reparto de carriles ya usado en §15.

### Y acá apareció lo que el fusionado no puede: el lote

| filas | Q4_K con buffer | Q4_K fusionado | Q4_K `f32` | Q3_K con buffer | Q3_K fusionado |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,412 | **1,123** | 1,316 | 1,347 | **1,147** |
| 2 | 1,712 | **1,350** | 1,774 | **1,547** | 1,742 |
| 4 | 2,381 | **1,998** | 2,052 | **1,915** | 3,028 |
| 8 | 4,125 | 3,189 | **2,929** | **2,955** | 5,776 |
| 16 | 7,409 | 6,507 | **4,757** | **4,646** | 10,926 |
| 32 | 13,905 | 12,558 | **9,611** | **9,052** | 22,941 |

**El fusionado desempaqueta una vez por fila de salida; el camino con buffer,
una vez por columna.** Con una fila no hay nada que amortizar y el fusionado
gana; al crecer el lote paga `m` desempaquetados contra uno. Q3_K cruza antes
que Q4_K porque su desempaquetado cuesta mucho más —máscara de bits y
rearmado de 16 escalas de 6 bits por bloque, contra enmascarar nibbles—, y con
32 filas llega a ser **2,5x peor**.

Y aparece un tercero que no estaba en el plan: **con lote grande en Q4_K gana
el camino de `f32`**, porque la multiplicación deja de estar limitada por el
desempaquetado y los productos en `f32` tienen FMA de 8 carriles sin
recuantizar nada.

O sea que ninguno de los tres gana siempre. El punto del ROADMAP que decía
"hoy hay que elegir a mano, debería elegirlo el motor" queda cerrado con
`ops::matmul_auto`, que despacha por formato y por número de filas. Los
umbrales son **un valor por defecto medido en un tensor y esta máquina**, no
una ley; en otro hardware el cruce se mueve y la forma de saberlo es correr
`measure_own_matmul`.

### Leer el ensamblador que genera el compilador dio 1,90x

Esto llegó por el camino menos esperado. La pregunta era si los `.s` del
proyecto se estaban quedando atrás; la respuesta resultó ser que sí, pero lo
útil no fue escribir ensamblador sino **leerlo**.

Volcando lo que LLVM genera para `dot_q4k_i8_avx2`:

```
40 vmulss    16 vcvtsi2ss    8 vsubss        <- escalar
 8 vpmaddubsw  8 vpmaddwd  8 vfmadd213ps     <- vectorial
```

**Más instrucciones escalares que vectoriales** en el bucle caliente. Son las
escalas: ocho sub-bloques por bloque, cada uno con dos conversiones de entero
a flotante y cinco multiplicaciones, todo de a uno. Eso no se ve leyendo el
código, que parece razonable; se ve leyendo la salida del compilador.

Un intento previo las había **adelantado** fuera del bucle vectorial y no
cambió nada —188,9 ns contra 186,2, dentro del ruido—, y ahí se cerró mal la
pregunta: adelantarlas las mueve, no las quita. Lo que faltaba era
**calcularlas las ocho juntas**, y se puede porque `xe[bl*8 + j]` con `j` de 0
a 7 es contiguo, igual que `xs`:

| | tiempo |
|---|---:|
| escalar | 698 ns |
| AVX2, escalas de a una | 175,3 ns |
| **AVX2, escalas de a ocho** | **92,2 ns** |

**1,90x**, y sin tocar una sola instrucción del producto punto. `cvtepu8_epi32`
más `cvtepi32_ps` convierten ocho bytes en dos instrucciones donde había
dieciséis.

A Q3_K le tocó lo mismo y algo peor que arreglar: además de calcular las
escalas de a dos, armaba el vector de ocho con `_mm256_setr_ps` de valores
sueltos, que son ocho inserciones por grupo y ocho grupos por bloque. Ahora
sale de **una carga y una permutación** —`permutevar8x32_ps` con el patrón
`[0 0 0 0 1 1 1 1]`, que reparte dos escalas en las dos mitades de los
carriles, que es exactamente la forma que el reparto de AVX2 pide—. Queda en
156 ns contra 2.297 del escalar.

### Lo que eso le hizo a la tabla de despacho

La rehizo entera, y conviene decirlo porque es el riesgo de escribir umbrales
medidos:

| filas | Q4_K antes | Q4_K ahora | Q3_K antes | Q3_K ahora |
|---:|---|---|---|---|
| 1 a 2 | fusionado | **fusionado** | fusionado (solo 1) | **fusionado** |
| 4 | reñido | **fusionado** | con buffer | **fusionado** |
| 8 y más | `f32` | **fusionado** | con buffer | con buffer |

En Q4_K el fusionado pasó a ganar en **todo el rango medido**, de 1 a 32
filas, y el camino de `f32` dejó de tener un hueco donde ganara. Q3_K sigue
cruzando en 8 porque su desempaquetado cuesta bastante más.

Los tres umbrales anteriores eran correctos **para el kernel de entonces**.
Un umbral medido hay que volver a medirlo cuando cambia lo que mide.

Y en el forward entero, sobre el 0,5B en Q4_K: prefill **72 ms contra 149 de
candle, 2,07x**, y la generación voraz sale idéntica en los 24 tokens sobre el
modelo cuantizado, donde antes divergía en el 13. Eso último **no es una
mejora de precisión** y no hay que leerlo así: la diferencia de logits sigue
siendo del mismo orden, 1,67% contra 1,88%. Lo que cambió es de qué lado
cayeron los empates cercanos.

### Y en ARM lo hace Zig, porque si no no lo hace nadie

El kernel fusionado se escribió en intrínsecos de AVX2, que **no existen fuera
de x86-64**. En ARM y en Termux —objetivo declarado del proyecto— el
despachador caía a la referencia escalar: byte a byte, en el bucle más
caliente del motor. Era un agujero abierto por las dos tandas anteriores, y lo
tapa `cosxamix_dot_q4k_i8_f32` en `kernels/dot.zig`.

| camino | tiempo | contra escalar |
|---|---:|---:|
| escalar | 683 ns | 1,00x |
| **Zig** | **303 ns** | **2,25x** |
| AVX2 | 179 ns | 3,81x |

Zig pierde contra AVX2 y eso es esperado: no reproduce el truco de
`vpmaddubsw`, que suma pares de productos de 8 bits dentro de un registro de
16 y es lo que le da a AVX2 sus 32 carriles; Zig ensancha a `i32` y
multiplica, que con NEON de 128 bits son 8. Por eso el despachador sigue
eligiendo AVX2 primero en x86-64.

Pero **en ARM la elección no es Zig contra AVX2, es Zig contra el escalar**, y
ahí son 2,25x. Entra por el segundo brazo de la regla 3.

A cambio de la pérdida de ancho gana en otra cosa: `@Vector` deja que LLVM
elija la instrucción de cada arquitectura, así que en ARMv8.2 con `dotprod`
puede salir `sdot` —el producto punto de cuatro `i8` en un `i32`— y si el
objetivo no lo tiene sale la versión con `smull` y sigue andando. Verificado
que cruza: el objeto de `aarch64-unknown-linux-gnu` contiene el símbolo y
`clippy` para ese target pasa limpio.

Hay un test aparte para el camino de Zig, y hace falta: en x86-64 el
despachador elige AVX2 y nunca lo llamaría, así que sin ese test el único
camino que tiene ARM se quedaría sin probar en la máquina donde se desarrolla.

**Los cinco formatos que faltan** caen a `matmul_i8_directo`, que los cubre.
Cada uno resta algo distinto —32 en Q6_K, 16 en Q5_0, un mínimo por sub-bloque
en Q2_K y Q5_K— así que cada uno pide su propio rodeo, y cada uno tiene que
justificarse midiendo. Visto el cruce de arriba, varios puede que no lleguen a
justificarse nunca.

### Un detalle con consecuencias

```rust
#[cfg(target_feature = "avx2")]   // COMPILACIÓN, no ejecución
return super::avx::vec_dot_q3k_q8k(n, xs, ys);
```

Es un `cfg`, no detección en tiempo de ejecución. Si el target no declara AVX2,
el kernel vectorizado **no se compila** y cae al escalar, aunque la CPU lo
soporte, y sin avisar.

Eso explica el 4x del `target-cpu=x86-64-v3` de `.cargo/config.toml`:

```
genérico     4428 ms por respuesta    1,8 tok/s
x86-64-v3    1094 ms                  6,1 tok/s
native       1066 ms                  6,3 tok/s
```

Se eligió `x86-64-v3` y no `native` porque `native` ata el binario a la CPU que
compiló. El coste: no arranca en un x86 anterior a 2015.

---

## 12. Redes de tensores (MPS / Tensor-Train): aplazado

Es álgebra lineal clásica, real y aplicable — descomponer matrices grandes en
cadenas de rango bajo reduce parámetros de verdad.

Pero **reduce bytes que leer de la RAM**, y el motor está limitado por cálculo.
Es la misma trampa que mató a la especulativa: una optimización del lado de la
memoria sobre un motor que no está limitado por memoria.

No se descarta, se ordena: primero cerrar la brecha de cómputo.

**La excepción que sí valdría**: comprimir específicamente la **matriz de
salida**. En el 3B son 243 MB —el 15%— y se multiplica entera en cada token
para producir logits sobre 151.936 palabras, de las cuales se usa una. Ahí una
descomposición de rango bajo atacaría bytes leídos *y* aritmética a la vez.

Cuánto pesa el vocabulario, medido:

| | embeddings | salida | juntos |
|---|---:|---:|---:|
| 0,5B | 19,3% | 29,8% | **49,1%** |
| 3B | 7,8% | 14,9% | **22,6%** |

### Medido, y no se aplazó por lo que yo creía

El aplazamiento de arriba era un argumento, no un número, así que se midió.
`examples/measure_rank.rs` estima la fracción de energía de Frobenius que
captura la mejor aproximación de rango `k`, con un buscador de rango
aleatorizado e iteraciones de potencia —el método de Halko, Martinsson y
Tropp—, que da una cota **por debajo**: si dice que hay estructura, la hay.

La comparación es controlada: el mismo tensor y la misma forma en un modelo
entrenado (Qwen2.5-0.5B) y en uno recién inicializado (daiss-0.5b).

`output.weight`, 151.936 × 896, rango máximo 896:

| k | k/máx | Qwen entrenado | daiss aleatorio | ventaja |
|---:|---:|---:|---:|---:|
| 8 | 0,9% | **20,3%** | 0,9% | **22,1x** |
| 32 | 3,6% | 23,5% | 3,7% | 6,41x |
| 128 | 14,3% | 34,3% | 14,6% | 2,34x |
| 256 | 28,6% | 47,5% | 29,2% | 1,63x |

Y `blk.0.ffn_gate.weight`, 4864 × 896: 10,9% con k=8 contra 1,4%, o sea 7,58x,
bajando a 1,37x con k=256.

**Tres conclusiones, y una es contra mí.**

**Uno: sobre daiss no sirve, y ahora está probado.** La matriz aleatoria
captura exactamente `k / rango_máximo` —0,9% con k=8, 29,2% con k=256— que es
lo que la teoría predice para rango completo. Cero estructura. Toda
descomposición de rango bajo apuesta a que el entrenamiento concentró la
información en pocas direcciones, y en un peso recién inicializado no hay nada
concentrado. Que el estimador reproduzca la predicción teórica al décimo de
punto en todo el rango es además lo que valida el estimador.

**Dos: la excepción que §12 ya había aislado es la correcta, y ahora se sabe
por qué.** `output.weight` tiene la señal más fuerte con `k` chico, 22,1x
contra 7,58x del FFN: hay unas ocho direcciones que cargan el 20% de la
energía. Ese es el sitio donde una descomposición de rango bajo pagaría.

**Tres, y esto contradice a los papers que se citan a favor:** las cifras de
50-70% de reducción de parámetros con más del 98% de precisión **no se ven
acá**. Para llegar al 47,5% de la energía de `output.weight` hace falta el
28,6% del rango; recortar el 50-70% de los parámetros tiraría la mayor parte
de la matriz.

La salvedad, que es real y hay que decirla: **energía de Frobenius no es
precisión.** Puede que las direcciones descartadas importen funcionalmente
menos de lo que pesan, y los papers miden la precisión después de curar el
modelo con reentrenamiento. Así que esto no refuta sus cifras: muestra que
cualquier compresión de ese tamaño se apoya en una de esas dos cosas, y
ninguna de las dos es gratis. Que es exactamente lo que los 50-70% no dicen.

### El control que hubo que hacer antes de creerse nada

Los pesos de Qwen vienen cuantizados, y el error de cuantización es de rango
completo por naturaleza, así que infla el rango aparente. Si fuera el que
manda, la medición no diría nada sobre la matriz de verdad.

El mismo tensor del 3B a tres niveles de ruido, `blk.0.ffn_gate.weight`:

| k | Q2_K | Q3_K_M | Q4_K_M |
|---:|---:|---:|---:|
| 8 | 4,1% | 4,3% | 4,4% |
| 32 | 8,6% | 9,0% | 9,2% |
| 128 | 21,7% | 22,7% | 22,9% |

Entre Q2_K, que son unos 2,6 bits por valor, y Q4_K_M hay **entre 0,3 y 1,2
puntos**. El ruido no explica nada y lo medido es la matriz.

---

## 18. `mmap`: 1,25x en la carga, y el 1,22x era gratis

`Gguf::datos_de` abría el archivo **de nuevo por cada tensor** y copiaba sus
bytes a un `Vec`. En el daiss-7b son 339 aperturas y 7,5 GB al montículo.

Había dos arreglos y había que saber cuánto aporta cada uno antes de sumar una
dependencia por el segundo. `examples/measure_mmap.rs`, sobre el 3B en Q4_K,
1,95 GB en 435 tensores, con el archivo ya en la caché del sistema:

| camino | ms | GB/s | vs hoy |
|---|---:|---:|---:|
| reabriendo (hoy) | 2352 | 0,83 | 1,00x |
| un solo descriptor | 1924 | 1,02 | **1,22x** |
| mapeado, sin copias | 1820 | 1,07 | **1,29x** |

Estable entre corridas: 1,21x, 1,28x y 1,29x en tres.

**El 1,22x no cuesta nada**: es dejar de reabrir el archivo. Lo que compra
`memmap2` encima es un 4-5% de tiempo y **1,95 GB de montículo que no se
reservan** —7,5 GB en el daiss-7b—, y en una máquina donde el daiss-7b no
entra en RAM eso es el argumento, no el 5%.

### Y una afirmación mía que no sobrevivió

Al conectar el mapeo a los dos ejemplos del camino caliente, sus cifras
empeoraron —el desempaquetado de Q4_K de 2,36 a 3,25 ms— y lo atribuí a que
releer el mismo tensor miles de veces paga recorrer 3.072 páginas por vuelta,
cosa que un `Vec` residente no hace. Suena bien y es falso: estaba comparando
**entre procesos distintos**, que es justo lo que §15 prohíbe porque en esta
máquina el trabajo que recorre memoria varía hasta 1,8x entre corridas.

Medido en el mismo proceso, mapeado contra copiado sobre el mismo tensor de
12,1 MB con 21 pasadas: **1,17x, 0,93x y 1,04x** en tres corridas. O sea
ninguna diferencia. La copia que había agregado para arreglar el problema no
arreglaba nada porque no había problema, y se quitó.

La conclusión de diseño queda más simple de lo que yo había complicado: se
mapea y se usa la porción tal cual, en los dos regímenes.

Y habilita algo que el tiempo no muestra: `Pesos::de` devuelve una porción que
apunta al archivo, y `descuantizar`, `desempaquetar_i8` y las dos `matmul` ya
reciben `&[u8]`. El camino del disco al producto punto queda **sin una sola
copia**.

Lo que esto NO arregla: la primera lectura. El disco hace 41 MB/s (§17) y la
pagan los tres caminos igual.

### El test encontró el bug en el acto

`Pesos` es `unsafe`, y una porción fuera de rango sobre un mapeo no devuelve
`Err`: da una violación de acceso que mata el proceso. El test que prueba un
offset absurdo falló en la primera corrida — `inicio_datos + offset` se
desbordaba **antes** de llegar a la comprobación de límites, y en release eso
da un índice envuelto que apunta a cualquier parte del mapeo. Las tres cuentas
van comprobadas ahora.

---

## 13. QNLP / DisCoCat: fuera de alcance

El diagnóstico que acompaña a esta propuesta es correcto: los tokenizadores son
un parche heurístico, fragmentan sin entender morfología, penalizan al español
frente al inglés, y la capa de salida es un cuello real.

También es correcto desmontar el "token cuántico": las redes **ya** trabajan en
espacios continuos. El entero discreto solo existe en la entrada y la salida; en
cuanto cruza la tabla de embeddings es un vector en ℝᵈ. Confundir eso con
computación cuántica mezcla álgebra lineal sobre transistores con mecánica
cuántica sobre hardware criogénico.

**La afirmación que no se acepta sin medir**: "eficiencia por parámetro hasta
cinco órdenes de magnitud mayor que un Transformer". Cinco órdenes son
100.000x. Los resultados de DisCoCat y `lambeq` conocidos son de escala juguete
—decenas de frases, vocabularios de docenas de palabras— y comparar eficiencia
por parámetro en ese régimen contra un Transformer entrenado en billones de
tokens no es una comparación justa. Además, simular circuitos cuánticos en
hardware clásico cuesta exponencialmente.

Es exactamente el tipo de número que hay que medir antes de construir encima,
como pasó con la especulativa: prometía 2-3x y dio 0,52x.

**Lo que sí transfiere**: sea BPE, byte-patches, Mamba o lo que venga, abajo
sigue habiendo GEMM, RMSNorm y productos punto. El trabajo de kernels no se
tira.

---

## 14. El nombre: por qué Cosxamix y no el primero

El proyecto se llamó **Xoryn** las primeras semanas. Se cambió porque choca de
oído con **Zorin OS**, una distribución de Linux conocida: un nombre que se
confunde con otro producto obliga a explicarlo cada vez que se menciona, y eso
no se arregla después.

El error de fondo no fue elegir mal, fue **no verificar**. Así que el reemplazo
se buscó con un procedimiento, no con intuición: 131 candidatos inventados
contra nueve fuentes —crates.io, npm, PyPI, RubyGems, Docker Hub, usuario de
GitHub, repos de GitHub con ese nombre exacto, Wikipedia en español e inglés, y
búsqueda web— descartando cualquiera con un vecino reconocible.

Lo que enseñó el barrido, y vale para la próxima vez:

| | |
|---|---|
| nombres de **dos sílabas** | el espacio más disputado: de 44 probados, **2** pasaron limpios |
| nombres de **tres sílabas latinas** | pasan fácil, pero suenan a laboratorio: las terminaciones `-vex`, `-vrix` y `-velo` son el molde con que se inventan las marcas de medicamentos, y la búsqueda devuelve puro fármaco |
| derivar de **raíces griegas o latinas** | mala idea para la unicidad: cualquiera deriva lo mismo, y por eso `tachra` (de *tachys*) o `nebryx` (de *nebula*) ya tenían dueño |

**Cosxamix** dio cero coincidencias exactas en las nueve fuentes, con el usuario
de GitHub libre.

Se eligió sabiendo lo que cuesta, y queda escrito para que nadie lo
redescubra como si fuera un defecto nuevo:

* El grupo **`sx`** no existe en español. Quien oiga el nombre no sabe si
  escribir *Cosamix*, *Cossamix* o *Cosxamix*, y es un nombre que hay que
  tipear: `cargo add cosxamix`, la URL del repositorio, el comando. Si alguna
  vez conviene, el camino es publicar `cosamix` como alias en los registros,
  no renombrar otra vez.
* Tiene tres vecinos reconocibles, ninguno exacto y ninguno en software:
  **COSMAX** (cosmética, cotiza en bolsa en Corea), *Cosmix* (productor
  musical) y *Cosumix Plus* (medicamento veterinario).
* En español se descompone de oído como "cosa" + "mix".

La alternativa que se descartó fue **Duzanix**, igual de libre y más fácil de
escribir, pero con la terminación `-nix` —que en software se lee como "tipo
Unix": Unix, Linux, Minix, Xenix, NixOS, Nix el gestor de paquetes— apuntando a
un sistema o un empaquetador antes que a un motor de inferencia.

También se descartaron por colisión real, por si vuelven a proponerse:
**Tensil** (compilador de aceleradores de ML, 400 estrellas, el mismo terreno),
**Zarith** (aritmética de precisión arbitraria de OCaml, 276), **Tezro**
(estación de trabajo de Silicon Graphics), **Qorvo** y **Quevron** (eco a
Chevron), **Axiora** (≈ Axora), **Numetra** (≈ Numetric) y **Nuvrex**
(≈ Nuvex ApS, consultora de cloud e IA).

---

## 15. La comprobación de límites costaba más que todos los kernels juntos

Es la medición más incómoda del proyecto, porque dice que semanas de trabajo
sobre kernels vectoriales y ensamblador a mano estaban buscando donde no había
nada, mientras un factor de entre 4 y 12 esperaba en una línea de Rust
ordinario.

### Cómo apareció

La matmul propia le ganaba a candle en lote **sobre Q4_K**. Al probar el mismo
código sobre **Q3_K**, que es el formato dominante del 3B que el bot usa por
defecto, perdía en las cuatro filas medidas — y perdía sobre una matriz **más
chica**, 9,2 MB contra 12,1. Un código que va más lento sobre menos datos no
tiene un problema de ancho de banda.

Había dos sospechosos y adivinar cuál habría sido el error de siempre, así que
se escribió `examples/profile_i8.rs`, que mide el desempaquetado y el kernel
por separado sobre los dos formatos con el mismo número de valores:

| formato | sub-bloque | MB | desempaquetado | kernel |
|---|---:|---:|---:|---:|
| Q4_K | 32 | 12,1 | 10,733 ms | 1,517 ms |
| Q3_K | 16 | 9,2 | **31,616 ms** | 1,068 ms |

Las dos hipótesis eran que Q3_K desempaqueta bits más retorcidos, o que el
camino de sub-bloque 16 del kernel es más caro porque arma su vector de escalas
con un `setr_ps` de valores sueltos.

**La segunda era falsa, y por el lado contrario:** el kernel con sub-bloques de
16 resultó *más rápido* que con 32, 1,07 contra 1,52 ms. Queda anotado como
conjetura desmentida.

El desempaquetado era el **94% del trabajo de un hilo** — 31,6 ms contra 1,1.

### Lo que estaba pasando

Los bucles escribían por índice calculado:

```rust
for j in 0..32 {
    let b = qs[base + j];
    q[i * 256 + par * 64 + j] = (b & 0xF) as i8;
    q[i * 256 + par * 64 + 32 + j] = (b >> 4) as i8;
}
```

`q` es un `&mut [i8]` y el índice se calcula en tiempo de ejecución, así que
cada escritura lleva su comprobación de límites. Y una comprobación de límites
dentro del bucle **le prohíbe vectorizar**: LLVM no puede reordenar escrituras
que pueden entrar en pánico a mitad de camino.

Costaba 0,43 ns por valor en Q4_K, unos 1,3 ciclos a 3 GHz. Eso es un bucle
escalar, byte a byte, en el camino más caliente del motor.

El arreglo es pedir la ventana **una vez**, con su longitud fija, y recorrerla:

```rust
let desde = i * 256 + par * 64;
let (bajos, altos) = q[desde..desde + 64].split_at_mut(32);
for ((b, lo), hi) in qs[base..base + 32].iter().zip(bajos).zip(altos) {
    *lo = (b & 0xF) as i8;
    *hi = (b >> 4) as i8;
}
```

Un solo chequeo al cortar la ventana, y adentro un par de iteradores de
longitud conocida: nada que comprobar y nada que impida vectorizar.

Antes de eso hubo un segundo arreglo, más chico y del mismo tipo: en Q2_K y
Q3_K el bucle interno iba de 16 en 16 porque el formato tiene sub-bloques de
16, pero los dos sub-bloques seguidos comparten corrimiento y máscara y son
contiguos en el origen y en el destino. Partirlos dejaba el bucle a **medio
registro** de AVX2 sin ningún motivo. Fusionarlos a 32 dio la primera mejora.

### Lo medido

Desempaquetado a `i8`, un hilo, los 11008×2048 valores del `ffn_gate`:

| formato | antes | fusionando a 32 | sin comprobación de límites | total |
|---|---:|---:|---:|---:|
| Q4_K | 10,733 ms | 9,798 | **2,364** | **4,5x** |
| Q3_K | 31,616 | 17,054 | **2,623** | **12,1x** |
| Q2_K | 18,114 | 17,387 | **4,192** | **4,3x** |

Los mismos diez bucles existían en el camino de `f32`, que es el que usan
`matmul` y el rodeo, y se arreglaron igual. Sobre Q3_K con una fila, la matmul
en `f32` pasó de **5,827 a 1,126 ms**.

Y el total de `matmul_i8_directo` con una fila sobre Q3_K, con hilos, de
**5,989 a 0,880 ms**.

### Que siga siendo exacto es el punto

Un arreglo de rendimiento en un descuantizador no vale nada si cambia un bit.
`cargo run --features candle --example check_quant` compara valor por valor
contra candle sobre los modelos reales: **21 tensores, 8 formatos, cero
diferencias**, con el Q6_K de 311.164.928 valores de `output.weight` incluido.

### Lo que esto le hace al criterio del ensamblador

En §5 quedó escrito que el ensamblador entra cuando LLVM se queda sin
registros, y que medido no se quedaba: 6 de 16 registros ymm y cero volcados en
el bucle. Era cierto y sigue siéndolo, pero le faltaba un paso **anterior**:

> Antes de preguntar si una persona puede mejorar el código vectorizado que
> generó el compilador, hay que comprobar que el compilador vectorizó algo.

La cuenta de ns por valor lo dice en un renglón y no hace falta leer
ensamblador: si un bucle que procesa bytes anda a más de ~0,2 ns por byte, no
está usando vectores, y ahí no hay nada que afinar hasta que los use.

### Sobre medir en esta máquina

De paso se encontró un defecto en el banco mismo. `measure_own_matmul` fijaba
20 repeticiones por muestra, así que una llamada de 0,2 ms daba una muestra de
4 ms y el ruido del reloj y del planificador se la comía. Dos pasadas seguidas
del **mismo binario**, sin tocar una línea, dieron para candle con una fila
**0,205 y 0,361 ms** — un 76% de diferencia. Lo nuestro, que tardaba diez veces
más, variaba un 4%.

O sea que el banco estaba sesgado **a favor de quien fuera más rápido**, y una
medición que depende de a quién mide no decide nada. Ahora el número de
repeticiones se calibra para juntar 60 ms por muestra, con cinco muestras y
mediana: la celda de una fila pasó de ±76% a ±3%.

Queda una variación que no es del banco: en trabajo que recorre memoria, esta
máquina da hasta **1,8x de diferencia entre pasadas** del mismo binario — el
desempaquetado de Q2_K midió 4,19 y 7,54 ms. Por eso de acá en adelante solo se
comparan cifras tomadas **en el mismo proceso**, y las comparaciones entre
ejecuciones distintas no se usan para decidir.

---

## 16. daiss: qué es nuestro, qué no, y por qué dice `qwen2`

**daiss** es la familia de modelos propia: cinco tamaños de 0,08 a 14,77 mil
millones de parámetros, definidos en `src/daiss.rs`, inicializados y escritos
como GGUF por `src/gguf_write.rs`.

Conviene separar las tres cosas sin adornos, porque la diferencia importa y
confundirlas sería venderse humo a uno mismo.

**Nuestro**: los tamaños, la inicialización, el generador con semilla, el
cuantizador, el escritor de GGUF, y —cuando se entrene— los pesos, que son el
modelo.

**No nuestro**: la forma del bloque. daiss es un decoder con RMSNorm, atención
agrupada, SwiGLU y RoPE, que es la misma forma que usan Llama, Qwen y Mistral
desde 2023. Inventar una forma distinta sin tener con qué entrenarla sería
inventar un problema, no resolver uno.

**Prestado a propósito**: el vocabulario, 151.936 piezas de Qwen2.5. Un
tokenizador propio es el punto 3 del ROADMAP §8 y no está escrito; hasta que
lo esté, reusar uno que funciona permite evaluar el modelo en vez de
esperarlo.

### Por qué `general.architecture` dice `qwen2` y no `daiss`

Porque esa clave no nombra a un modelo sino a una **forma de bloque**, y es la
que daiss tiene. Poner `daiss` ahí haría que ningún runtime del mundo lo
cargara —ni el nuestro, que lee esa clave en `model/qwen2.rs`— a cambio de
nada. El nombre va en `general.name`, que es donde se mira.

### La verificación

Que nuestro lector lea lo que nuestro escritor escribe no prueba casi nada:
los dos podrían estar equivocados igual. La prueba es **candle**, que no
comparte una línea con el escritor. Sobre el daiss-7b recién creado:

- 339 tensores y 17 metadatos, los mismos que contamos nosotros;
- `output.weight` descuantizado, 544.538.624 valores, **cero diferencias**;
- los 339 tensores que el cargador pide, presentes y con su forma.

El modelo entero no se construye acá y no por culpa del archivo: cargarlo pide
7,5 GB y la máquina tiene 7,3 libres. Con el daiss-mini, que sí entra, candle
lo construye entero.

### Un hallazgo de la familia: el 0.5b no admite k-quants

`daiss-0.5b` tiene `dim` 896, que son 3,5 bloques de 256, y los k-quants piden
filas de 256 exactas. No es un descuido de la tabla: es exactamente la razón
por la que los GGUF de Qwen2.5-0.5B que circulan **mezclan formatos**, con la
atención en Q8_0 o Q5_0 y solo el FFN en k-quant. Hay un test que afirma que
el 0.5b es el único de la familia así, para que se entere si algún día cambia.

### Lo que este módulo no hace

No entrena. Un daiss recién escrito genera ruido. Lo que da es que el camino
completo exista y esté verificado, que es la condición previa de cualquier
entrenamiento.

---

## 17. El disco de esta máquina hace 41 MB/s, y eso cambia cuentas

Escribir el daiss-mini daba 24 MB/s. Se paralelizó la generación por tandas
acotadas y subió a 34. Se agruparon las filas para amortizar las reservas de
memoria y las escrituras —316.000 de cada una solo en el `token_embd`— y
subió a **35**.

Las dos hipótesis apuntaban al código. Las dos eran falsas, y la segunda no
movió la aguja un 3%. Lo que faltaba era medir el disco:

| unidad | escritura secuencial |
|---|---:|
| **F:**, donde vive el proyecto | **41 MB/s** |
| C:, el del sistema | 165 MB/s |

Estábamos al **85% del techo físico** de F:. El generador nunca fue el cuello
de botella. Escribiendo el mismo modelo en C: da **324 MB/s**, así que el
código puede dar al menos nueve veces lo que ese disco acepta.

La paralelización se queda igual, y por la regla 3: a 24 MB/s estaba por
debajo del techo del disco y ahora choca contra él, y en una unidad normal da
324. No se queda "por si acaso" — se queda porque se midió que gana.

Lo que esto cambia más allá de daiss: **cargar un modelo en frío desde F: es
disco y nada más.** El 3B en Q4_K son 12,1 GB, o sea unos 5 minutos de lectura
en frío si no está en la caché del sistema operativo. Cualquier medición de
arranque hecha sobre este disco está midiendo el disco.

---

## 19. El tokenizador propio: 3,4% más tokens por un modelo 9 veces más chico

Era el punto que bloqueaba los tamaños chicos de daiss, y la pregunta se podía
contestar con un número antes de decidir nada: **¿cuánto peor comprime un
vocabulario de 8.192 que uno de 151.936?**

`src/tokenizer.rs` es un BPE **a nivel de bytes**, con entrenamiento propio.
`examples/train_tokenizer.rs` lo entrena con los documentos del proyecto y los
comentarios de su código —382 KB de español técnico— y mide sobre el **20%
final, que el entrenamiento no vio**. Sin esa separación el número sería
mentira: un BPE medido sobre su propio corpus comprime de maravilla y no
generaliza nada.

| vocabulario | fusiones | tokens | bytes/token | entrenar |
|---:|---:|---:|---:|---:|
| 256 (solo bytes) | 0 | 78.256 | 1,00 | — |
| 1.024 | 768 | 32.190 | 2,43 | 0,9 s |
| 4.096 | 3.840 | 24.843 | 3,15 | 3,5 s |
| **8.192** | 7.185 | **23.433** | **3,34** | 5,6 s |
| **Qwen2.5, 151.936** | — | **22.659** | **3,45** | — |

**El nuestro de 8.192 necesita un 3,4% más de tokens que el de Qwen**, con un
vocabulario 18 veces más chico. Y lo que eso compra, en daiss-mini —256 de
ancho, 6 capas—:

| vocabulario | parámetros | % en tablas de vocabulario | Q8_0 |
|---:|---:|---:|---:|
| 1.024 | 5,2 M | 10% | 5 MB |
| **8.192** | **8,9 M** | **47%** | 9 MB |
| 32.768 | 21,5 M | 78% | 22 MB |
| 151.936 | 82,5 M | **94%** | 84 MB |

Con el vocabulario de Qwen, un daiss chico es **94% tabla de búsqueda**: el
transformer sería el 6% de los parámetros y el resto, memoria de palabras. Con
8.192 pasa a 47% y el modelo es **9,3 veces más chico**. El canje es a favor
por un margen que no deja lugar a discusión.

### Las dos salvedades, que son reales

**El entrenamiento se saturó en 7.185 fusiones.** Los objetivos de 16.384 y
32.768 dieron exactamente lo mismo que el de 8.192, porque en 306 KB no queda
ningún par que aparezca dos veces. Eso es lo correcto —llenar el vocabulario
con rarezas de un solo uso sería gastarlo— pero significa que el 3,4% está
medido con unos 7.441 tokens efectivos y que **no se sabe qué pasa con un
vocabulario grande bien lleno**. Para eso hace falta un corpus de gigabytes.

**El texto de prueba es del mismo dominio que el de entrenamiento.** Parte de
ese 3,4% es especialización y no eficiencia: nuestro tokenizador conoce
"descuantizar" y "sub-bloque" porque los vio, y el de Qwen los parte. Sobre
texto general la comparación le iría peor, y el límite ahí sería el **corpus**
y no el tamaño del vocabulario. La forma de la curva es creíble; el número
exacto no se extrapola.

### Decisiones de diseño, y por qué

**A nivel de bytes, con los 256 bytes como base.** Así no existe un token
desconocido y no hace falta un `<unk>`: cualquier secuencia se representa
siempre. Para un bot de WhatsApp eso no es teoría — un emoji nuevo o un byte
roto en un mensaje no puede tirar nada. Hay un test que lo afirma sobre bytes
al azar, incluidos los que no son UTF-8 válido.

**Codificar aplica siempre la fusión de menor rango primero.** No es un
detalle de implementación: es lo que hace que el resultado sea determinista y
coincida con lo que el entrenamiento aprendió. Fundiendo de izquierda a
derecha, el mismo texto daría tokens distintos según dónde empezara.

**El entrenamiento desempata por el par más chico.** `HashMap` no tiene orden,
así que entre dos pares igual de frecuentes el elegido dependería de la semilla
de la tabla y el tokenizador cambiaría en cada corrida. Hay un test que entrena
dos veces y compara.

**El formato guardado son solo las fusiones.** Los 256 bytes son implícitos y
la pieza de cada token se reconstruye desde su par, con el rango igual a su
posición: 8 bytes por fusión, y el de 8.192 pesa **57 KB**. Al cargar se
comprueba que cada fusión se apoye en tokens anteriores, que es lo que
convierte un archivo corrupto en un error con nombre en vez de un pánico por
índice.

### El círculo cerrado, y medido

Un tokenizador que no usa nadie es un adorno, así que daiss lo incrusta:
`daiss::escribir_con` acepta un `Bpe`, y cuando se le pasa uno **el
vocabulario del tokenizador manda sobre el de la configuración** — un modelo
cuya tabla de embeddings no coincide con su tokenizador está roto, y en
silencio, así que no se deja que las dos cifras puedan discrepar.

Medido de punta a punta, daiss-mini con nuestro vocabulario de 7.441 tokens:

| | vocabulario | archivo |
|---|---:|---:|
| prestado de Qwen2.5 | 151.936 | **84 MB** |
| propio | 7.441 | **9 MB** |

Nueve veces más chico, el tokenizador viaja dentro del archivo y el que sale
codifica igual que el que entró, y candle sigue construyendo el modelo entero
con cero diferencias sobre los pesos.

Las fusiones van bajo `cosxamix.tokenizer.*` y **no** bajo `tokenizer.ggml.*`,
que es la convención de ggml. Es a propósito: esa convención guarda las piezas
como cadenas y las nuestras son secuencias de bytes, que para representarse
como texto necesitan la tabla de byte-a-unicode de GPT-2. Esa tabla no está
escrita ni verificada, y poner las fusiones bajo la clave estándar sin ella
haría que otro runtime leyera un vocabulario equivocado **creyendo que lo
entiende**. Con nuestra clave, un runtime ajeno la ignora y lo dice.

**La pre-tokenización es más simple que la de GPT-2** —espacio pegado a la
palabra siguiente, corridas de letras, corridas de dígitos, y todo lo demás de
a un carácter— y lo que se pierde está dicho: en inglés, las contracciones con
apóstrofo dan tres piezas en vez de dos. En español no hay y la diferencia no
aparece.

------

## 20. Zig se compila para lo mismo que Rust

`.cargo/config.toml` fija `target-cpu=x86-64-v3` y lo dice claro: el binario no
arranca en un x86 anterior a 2015, y para esos hace falta compilar sin esa
línea. Esa compilación existía en el papel y no andaba con `zig`, por dos
motivos que se tapaban uno al otro:

1. `build.rs` compilaba Zig **siempre** para `x86_64_v3`. Rust sin AVX2, Zig con
   AVX2: primer kernel, instrucción ilegal.
2. Seis caminos de `kernels/mod.rs` preguntaban `matches!(elegido(),
   Backend::Avx2 | Backend::Zig)` antes de llamar a los kernels de Rust con
   AVX2. Con Zig elegido en una CPU sin AVX2, otra instrucción ilegal, aunque
   se arreglara lo primero.

**La regla ahora: Zig asume lo mismo que Rust, nunca más.** `build.rs` lee
`CARGO_CFG_TARGET_FEATURE`. Si Rust ya tiene AVX2 y FMA, Zig va en v3 y una
sola copia, porque una baseline no correría nunca. Si no, dos copias del mismo
fuente —baseline y `_v3`— y se elige al arrancar, como en WinsiBot 8.18.0,
donde se resolvió lo mismo para los kernels de imagen.

Se pensó primero en llevar siempre las dos copias. No: en la compilación de
siempre la baseline es peso muerto, y en un proyecto que mide cada MB no se
carga lo que no puede ejecutarse.

**Lo que se gana, medido** (Rust sin `target-cpu`, en este i5-10400 ejecutando
código SSE2):

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

Una primera medición, hecha en la compilación de siempre, daba números
parecidos y estaba mal: el "escalar" de Rust ya estaba compilado para v3 y el
compilador lo había vectorizado con AVX2 donde pudo. Se tiró y se midió de
nuevo con un binario de verdad sin `target-cpu`.

`dot_i8_escalado` en Zig se usaba solo fuera de x86-64. En un x86-64 sin AVX2
gana 1,5x a la referencia, así que ahora se usa también ahí. Y `silu` queda
como estaba, en Zig: el 6% está cerca del ruido, pero no pierde.

**Lo que esto no arregla:** candle sigue sin elegir en tiempo de ejecución, y
en esa compilación corre 4 veces más lento. Lo que sí cambia es que el motor
propio, que usa estos kernels, deja de pagar la referencia escalar.

---

## 21. Q3_K en tres lenguajes: el algoritmo dio 1,65x, el lenguaje nada

Q3_K es el formato del modelo por defecto y era el kernel fusionado más lento:
~19 ns por bloque contra ~10 de Q4_K, moviendo **menos** bytes (110 contra
144). O sea que no era la memoria: era trabajo de más.

Leyendo el kernel aparecieron cuatro cosas, y ninguna era de lenguaje:

| qué | antes | después |
|---|---|---|
| las 16 escalas de 6 bits | ~30 operaciones escalares (`escalas_q3k`) | 8 en vector: un `vpsrlvd` con `[0 2 4 6]` pone los dos bits altos de las cuatro palabras de una vez |
| dónde viven | 16 `f32` a la pila y 8 lecturas por bloque | dos registros y una permutación de patrón constante por grupo |
| el tercer bit | constante por grupo, `and`, `cmpeq`, `and` | corrimiento inmediato de 16 bits y `and`: lo que arrastra del byte vecino cae fuera del bit 2 |
| las FMA | una cadena | dos |

En Rust, **154 → 91 ns** por fila de 2048. Ese es el que usa el despachador.

**Después, el mismo algoritmo en C y en ensamblador**, para saber si el
lenguaje aporta algo encima. Mismas instrucciones, mismo orden; el de
ensamblador da **igual bit a bit** que los intrínsecos.

| | ns, tres tandas |
|---|---:|
| Rust | 90,1 – 91,5 |
| ensamblador | 90,1 – 92,2 |
| C (`zig cc`) | 94,4 – 95,4 |

**El ensamblador empata.** En Q4_K había ganado 4-8%, y se averiguó que la
mayor parte era una instrucción (`vcvtph2ps`) que los intrínsecos no usaban
(§5). Acá los intrínsecos ya la usan, y lo que le queda a la mano es el reparto
de registros: el compilador lo hace igual de bien. Un indicio de que el kernel
ya está contra el límite de los puertos 0 y 1, por donde pasan los
corrimientos, `vpmaddubsw`, `vpmaddwd`, la conversión y la FMA: probar **cuatro
acumuladores** en vez de dos lo hizo 3-4% más lento —el compilador se quedó sin
registros— en vez de más rápido.

**C pierde un 4%** con el mismo LLVM por debajo. No se buscó la causa: con
Rust a la par del ensamblador, un 4% de C en x86 no cambia ninguna decisión.

**Lo que C sí trae**, y por eso entra igual con su feature:

- las extensiones de CPU **por función**, `__attribute__((target))`, como
  `#[target_feature]` y a diferencia de Zig, que las fija por archivo con
  `-mcpu`. El objeto va en baseline y es seguro en cualquier x86-64;
- en ARM, **`vdotq_s32`** —la instrucción `sdot`— en `arm_neon.h`, estable. Rust
  estable no la expone (`stdarch_neon_dotprod`) y Zig no la genera (§5). Es el
  sitio donde C habilita una plataforma, el segundo brazo de la regla 3, y el
  siguiente paso: el kernel NEON de Q4_K y Q3_K en C, ejecutado en el runner
  ARM de CI.

**Por la regla 3, ni el de C ni el de ensamblador entran al despachador.**
Quedan detrás de `c` y `asm_q3k`, fuera del binario de siempre, con sus tests y
en el bench, como los contratos de Q4_K que perdieron: sirven de vara cuando
cambie el hardware. En una CPU con **AVX-VNNI** (`vpdpbusd` hace `vpmaddubsw` y
`vpmaddwd` en una instrucción, y libera justo los puertos que acá están
llenos), la comparación se repite.

---

## 22. `sdot` desde C: lo que ningún otro lenguaje del proyecto alcanzaba

En ARM el camino vectorizado de los fusionados era el de Zig, que ensancha
`i8` a `i32` y multiplica de a 8 carriles: 2,25x sobre el escalar pero 1,7x
por debajo de AVX2. `sdot` hace 16 productos de `i8` por instrucción, y estaba
fuera de alcance por tres lados (§5): Rust estable no expone `vdotq_s32`, Zig
no la genera, y el andamio en ensamblador seguía sin cuerpo.

En C es `arm_neon.h`. Comprobado antes de escribir el kernel: `zig cc -target
aarch64-linux-gnu` emite `sdot v0.4s, v1.16b, v2.16b` con
`__attribute__((target("dotprod")))` en la función, y el f16 se convierte con
un solo `fcvt`. Compila igual para Android y macOS, sin libc.

Lo que hace distinto del de x86:

- **Q3_K sin el rodeo del +4.** `vpmaddubsw` quiere un operando sin signo y
  obliga a subir los valores a 0-7 y descontar `4·Σx` después; `sdot`
  multiplica con signo los dos, así que se resta 4 en el byte y listo. El `xs`
  de Q3_K ni se lee.
- **Un `sdot` por sub-bloque de 16** en Q3_K y dos por cada uno de 32 en Q4_K,
  con la conversión a `f32` y una FMA por sub-bloque, en dos cadenas.

**Lo que no se sabe todavía es cuánto gana**, y por eso no está en el
despachador: esta máquina es x86-64 y no tiene cómo ejecutarlo, ni QEMU ni WSL.
Los tests —contra la referencia escalar en 1, 2, 8, 17 y 43 bloques, y contra
Zig— y el bench corren en el runner ARM64 de CI, un Neoverse N2 con `dotprod`.
La medición que decide es la de un teléfono con Termux, que es la plataforma
objetivo:

```bash
pkg install zig rust
cargo bench --features zig,c --bench kernels -- dot_q4k_i8
```

Ahí se ve `zig` contra `c con sdot` y `q3k escalar` contra `q3k c con sdot`.
Si gana, entra al despachador preguntando `dotprod`; en un A53 o un A72, que no
la tienen, sigue el de Zig.

**Ganó, y entró en la 0.5.1.** Medido en el runner ARM64 de CI, un Neoverse
N2, fila de 2048:

| | antes | con `sdot` | |
|---|---:|---:|---:|
| Q4_K | 382 ns (Zig) | 162 ns | 2,4x |
| Q3_K | 1998 ns (referencia) | 367 ns | 5,4x |

Q3_K es el número que importa: es el formato del modelo por defecto y en ARM no
tenía ningún camino vectorizado. Falta la cifra de un teléfono, que con menos
ancho de banda puede mover el reparto entre cómputo y memoria; con 2,4x y 5,4x
de margen, no la decisión.

---

## 23. El motor propio en el servidor, y por qué pasó a ser el de por defecto

El forward propio ya generaba texto, pero solo desde un ejemplo. Enchufarlo
al servidor fue escribir `MotorPropio` —sesiones, prefijo común, texto en
partes— detrás del mismo trait que candle, y hacer que el gestor de modelos
sirva cualquiera de los dos.

**La medición que decide es la del uso real, no la del ejemplo.**
`examples/generate` mide un prompt sin caché y 48 tokens: ahí el propio
procesa el prompt 1,8x más rápido y escribe un 8% más lento, y la suma no dice
mucho. WinsiBot manda siempre el mismo *system* y una pregunta distinta, con
el *system* reaprovechado del caché; cada mensaje es entonces procesar una
pregunta de ~15 tokens y escribir una respuesta de ~12. Medido así, con el 3B
Q3_K_M y el servidor de verdad:

| | candle | propio |
|---|---:|---:|
| carga del modelo | 2,7 – 3,6 s | **1,1 s** |
| primer mensaje (nada en caché, ~155 tokens) | 8,3 – 12,9 s | **6,2 – 9,4 s** |
| mensajes 2 a 8, media por mensaje | 2528 – 3727 ms | **1570 – 1940 ms** |
| velocidad contando todo | 3,3 – 4,9 tok/s | **5,7 – 7,1 tok/s** |
| tokens del *system* reaprovechados | 142 – 144 | 142 – 144 |

**1,6 a 1,9 veces más rápido por mensaje**, y la carga en un tercio.

**La calidad, con la batería y no con ocho preguntas.** La regla del
reemplazo (ROADMAP §9) dice que nada sale de candle hasta que la pieza propia
dé lo mismo sobre los modelos reales, y sobre un modelo cuantizado no da lo
mismo token por token: diverge a los 5 tokens en el 3B y a los ~28 en el 0,5B.
No es un error —sobre el modelo en F32 coinciden enteras, y la diferencia es
íntegramente cuantización de los dos lados—, pero cambia qué se le contesta a
la gente. En las ocho preguntas de la tabla cada motor erró un dato distinto,
y ocho no alcanzan. Decidió la batería de 15 preguntas verificables con la
que se eligió el 3B (§8), en modo voraz, sobre el mismo archivo:

| | candle | propio |
|---|---:|---:|
| aciertos | 10/15 | **11/15** |
| decode, mediana de 3 | 10,5 tok/s | **11,3 tok/s** |
| prefill | 1782 ms | **1279 ms** |
| carga, desde el disco | 43,7 s | **39,2 s** |

candle saca el mismo 10/15 de cuando se eligió el 3B: la batería reproduce.
En 13 de las 15 responden lo mismo. El propio acierta además la de dos pasos
("100 menos 37, dividido por 7": 9, y candle dijo 25). Su respuesta del barco
es confusa y cuenta solo porque nombra el agua, así que con rigor es 10 contra
10 u 11 contra 10.

**Igual o mejor en calidad, y 1,6-1,9x más rápido en el uso real: desde la
0.6.0 es el de por defecto.** Candle no sale del proyecto: queda detrás de
`COSXAMIX_MOTOR=candle` y como vara de los tests, que es lo que la regla del
reemplazo pide que siga siendo.

---

## 24. La matmul por bloques, y lo que el freno resultó ser

`examples/measure_response` midió una respuesta como la de WinsiBot: en el
primer mensaje de un chat, **el 72% del tiempo era leer el prompt** (333
tokens, 11,2 s) y el decode ya usaba el 67% de la RAM. El aire estaba en el
prefill, que es una multiplicación de matrices de verdad y no de un vector.

**Primera hipótesis: la caché.** `matmul_i8_directo` hacía un producto punto
por llamada y, por cada columna, recorría la entrada entera: 680 KB con 333
filas, que no entran en L2. `matmul_i8_bloques` desempaqueta el tramo de
columnas una vez, recorre la entrada en franjas y calcula teselas de 4 filas
× 2 columnas en registros. Ganó 1,53x. Pasar los pesos a sin signo y las
escalas a una por carril, que quitaban instrucciones, casi no movió nada.

**Segunda hipótesis: las cuentas.** Las escalas enteras de llama.cpp dejan 3
instrucciones por trozo en vez de 5. Ganaron solo un 8 a 12%, y con una
escala cada 32 valores, nada. **Y partir `k` en tajadas que entraran en L2
tampoco ganó.** Si ni menos instrucciones ni mejor caché mueven el número,
el freno estaba en otro lado.

**Lo que era: dos etapas en un solo hilo.** Medido por etapas dentro de una
llamada: cuantizar la entrada y transponer la salida —las dos en un hilo,
con los otros once esperando— eran entre el 24% (`ffn_gate`) y el 64%
(`attn_k`) de cada multiplicación. En paralelo, la mejora pasó de 1,65x a
**2x**, y de punta a punta el prompt de 333 tokens bajó a **6,6 s (1,70x)**.

La lección, para la próxima: **medir por etapas antes de optimizar un
kernel.** Dos de los tres intentos atacaron algo que no era el freno.

**Las escalas enteras se sacaron, aunque ya con el freno resuelto ganaban un
15%.** `examples/measure_precision` compara la distribución entera del
próximo token contra el camino que no cuantiza la entrada: el que quedó
pierde KL 0,0011 —lo mismo que el camino de antes, 0,0010—, y las enteras
entre 0,0027 y 0,0071 según cada cuánto lleva escala la entrada. En la
batería, 10 de 15 contra 11. Un 15% de prefill no paga 3 a 7 veces más
error, y el que lo ve es el usuario del bot.

## 25. El decode: medir dentro del forward

Con el prefill resuelto, el decode quedó como lo que más pesa en una
respuesta larga, al 66% de lo que da la RAM. `examples/measure_decode`
recorrió todas las matrices en orden y señaló la de salida —Q6_K, sin kernel
fusionado— como el 25% del token a 12,9 GB/s. Se le escribió el kernel, y la
matmul suelta bajó a la mitad... y el forward no se movió.

**La medición aislada mentía.** Medida dentro del forward —con un cronómetro
temporal por etapa—, la salida por el camino con buffer costaba 11,3 ms y no
19,7, y el kernel nuevo la dejó en 9,7: un 2%, no un 12%. Lo que sí mostró la
medición en contexto fue otra cosa: **q, k y v iban a 12 GB/s** contra 21 de
las ffn. Tres llamadas paralelas seguidas, dos de ellas a matrices de 0,2 MB
donde el costo fijo de repartir los hilos (~10 µs) era casi todo. Juntarlas
en una pasada, y gate con up, bajó el token un 3 a 5%.

Lo que queda: las ffn van a 21-22 GB/s de 28. Con los pesos en caché el
kernel de Q3_K hace 33 GB/s, solo 1,2 veces lo que da la RAM, así que ahí el
límite es casi tanto de cálculo como de memoria, y las escalas enteras que lo
aliviarían ya se descartaron por precisión (§24).

## 26. El prefijo entre conversaciones

El caché de atención se guarda por sesión, y WinsiBot usa una sesión por
chat. Desde que su *system* dejó de cambiar con cada mensaje (WinsiBot 9.1.0),
dentro de un chat solo se lee lo nuevo. Quedaba el primer mensaje de cada
chat, y el de cada grupo cada media hora, cuando se renueva la foto de su
perfil: ahí se leía el *system* entero, aunque su comienzo —quién es el bot,
su personalidad, las reglas— fuera el mismo en todos los grupos.

**Copiar es exacto.** El caché de un token depende solo de los tokens
anteriores, así que si dos prompts empiezan igual, el tramo común es el mismo
en los dos. Se busca entre las sesiones guardadas —16 como mucho, comparar
prefijos no cuesta nada— la que comparte más, y se copia ese tramo. Copiar
130 tokens del 3B son ~10 MB; leerlos, unos 2,5 s.

**La otra mitad del arreglo estaba en WinsiBot.** Sus dos instrucciones fijas
del final ("responde como un miembro del grupo", "no recicles los ejemplos")
iban después del perfil del grupo, fuera del tramo común. Pasarlas arriba
llevó el tramo de 68 a 130 tokens, y el primer mensaje de 4,9 a 3,5 s (6,0
sin préstamo).

**No es bit a bit idéntico a leerlo**: el tramo copiado puede haberse
calculado en un lote de otro tamaño —y desde 24 filas va por la matmul por
bloques—, así que puede diferir en el redondeo. En las pruebas, la respuesta
sale igual; y `check_sessions` confirma que la conversación de la que se
copia no cambia.

## Cifras que hubo que corregir

Se listan porque una cifra mal medida dirige el trabajo hacia el lado
equivocado, y todas estas lo hicieron un rato.

| se dijo | era | por qué |
|---|---|---|
| 6,1 tok/s de generación | ~25 tok/s | se dividían tokens por el tiempo **total**, con respuestas de 1-2 tokens donde el prefill domina |
| al 30% del techo de memoria | el 3B está al **51%** | el 30% era del 0,5B |
| el 1,5B es el punto dulce | no lo es | la prueba de 6 preguntas no tenía resolución |
| la capa de salida explica la especulativa lenta | es el 15% | la causa es el coste de descuantizar cada k-quant, que se paga por fila |
| el motor está limitado por cálculo | **sigue siendo cierto**, y se confirmó midiendo | la "corrección" que lo negaba duró medio día: dos kernels la desmintieron (§10) |
| el escalado de hilos no tiene explicación | sí la tiene, para el decode | las matrices que no entran en L3 no escalan, y son el 89% de los pesos (§10) |
| candle comparte los pesos entre instancias | **931 MB cada una** | el parseo de `tasklist` estaba roto y daba 0,0 MB |
| la matmul propia le gana 2,10x a candle a 8 filas | **1,19x a 1,30x** sobre Q4_K, 1,05x a 1,12x sobre Q3_K | la cifra salió de una pasada cuya columna de candle marcó 7,717 ms donde el mismo código sin tocar marca hoy 2,5 a 4,3; no es reproducible y la causa no quedó establecida (§15) |
| el camino de enteros le gana siempre al de `f32` | **depende del formato**: en Q3_K gana el de enteros, en Q4_K gana `f32` desde 2 filas | quitar la comprobación de límites le aprovechó más al de `f32` (§15) |
| el sub-bloque de 16 encarece el kernel | lo **abarata**: 1,07 contra 1,52 ms | era una conjetura sobre el coste de `setr_ps`, nunca medida (§15) |
| escribir modelos es lento por el generador | es **el disco**: F: hace 41 MB/s y escribíamos a 35 | dos hipótesis seguidas sobre el código, ninguna medida; el disco no se midió hasta el final (§17) |
| el mapeo pierde 1,4x al releer el mismo tensor | **no hay diferencia**: 1,17x, 0,93x, 1,04x | se comparó entre procesos distintos, que es lo que §15 prohíbe (§18) |
| escribir el kernel fusionado no vale la pena porque candle ya lo tiene | **1,16x a 1,44x** sobre nuestro camino, y el caso de una fila de 0,55x a 0,85x contra candle | era un pronóstico y no una medición; Q4_K no necesita el rodeo del signo, que es lo que el pronóstico no vio (§11) |
| el microkernel con bloqueo de registros es la primera oportunidad del ensamblador | **es 20% más lento**, y por el patrón de acceso, no por los registros | se escribió y se midió: 12 de 16 registros, cero volcados, y cuatro flujos entrelazados donde antes había uno contiguo (§5) |
| adelantar el cálculo de las escalas fuera del bucle no sirve, LLVM ya lo hace | cierto, pero la conclusión era otra: **vectorizarlo da 1,90x** | adelantar las mueve, no las quita; se midió lo primero y se dio por cerrado lo segundo sin probarlo (§11) |
| los umbrales de `matmul_auto` | **los tres cambiaron** al duplicarse la velocidad del kernel | eran correctos para el kernel de entonces; un umbral medido hay que volver a medirlo cuando cambia lo que mide (§11) |

---

## Lo que no se pudo explicar

**El escalado de hilos.** De 1 a 6 núcleos se obtienen 2,6x, no 6x:

| hilos | decode | núcleos efectivos | CPU consumida |
|---:|---:|---:|---:|
| 1 | 9,86 tok/s | 0,98 | 32,9 s |
| 6 | 25,58 | 4,06 | 53,2 s |
| 12 | 26,08 | 6,23 | 81,8 s |

Se encontró un factor —`with_min_len(128)` en el reparto de candle impide
partir las proyecciones K y V, de 256 columnas, en más de 2 trozos— pero K+V
son ~1,4% de los parámetros y eso explica ~3%, no la brecha.

Diagnosticarlo pide un perfilador de verdad. Con el 3B al 51% del techo, el
margen restante es menos de 2x.

De 6 a 12 hilos se gana **2%** y se gasta **54% más de CPU**: los dos hilos de
un núcleo compiten por la misma unidad vectorial. Por eso el pool se fija a los
núcleos físicos.

---

## Apéndice: las investigaciones recibidas, una por una

Cuatro análisis entraron al proyecto desde fuera. Esto es qué se tomó de cada
uno y qué no, para que no haya que releerlos para saberlo.

### A. Políglota: Zig, D, C/C++, ASM, código máquina

**Lo que proponía**: un dueño por responsabilidad, en el lenguaje que mejor la
sirva, con una frontera única y estable entre ellos.

**Lo que se tomó**: la arquitectura entera. `src/kernels/mod.rs` es esa
frontera: una firma, tres implementaciones, elección en tiempo de ejecución y
`COSXAMIX_KERNEL` para forzar una. Rust manda y es el dueño de la memoria; nadie
más asigna ni libera.

**Lo que no**: D (punto 4) y ASM por ahora (punto 5). La afirmación de que Zig
ganaría en velocidad **no se cumplió** — pierde 3-8% en x86 — pero se queda por
cobertura, que es un argumento mejor que el original.

**Lo que corrigió una cifra del análisis**: decía que cruzar la FFI cuesta
"menos de 1,5 ns" sin respaldo. Medido con funciones vacías: ~0,8 ns. Acertaba.

### B. Arquitectura de un runtime de IA

**Lo que proponía**: tener el modelo cargado y caliente en vez de arrancarlo por
petición; caché de atención reutilizable entre turnos; procesar peticiones por
lotes; un gestor que decida qué modelo vive en RAM.

| pieza | estado |
|---|---|
| modelo caliente entre peticiones | **hecho** — `MotorCandle` lo mantiene cargado |
| caché de atención por conversación | **hecho** — punto 7, con LRU y prefijo común |
| procesar el delta por lotes | **hecho** — vino de arreglar la máscara causal, 5422 → 4772 ms |
| gestor de varios modelos en RAM | **pendiente** — hoy carga uno ([ROADMAP](ROADMAP.md) §4) |
| lotes entre peticiones distintas | **pendiente** — las sesiones se intercalan, no se agrupan |

El diagnóstico de fondo era correcto y es el que más rindió: el caché de sesión
hizo que intercalar conversaciones pase de costar +76% a costar nada.

### C. Modelos pequeños, cuantizado agresivo, especulativa

**Lo que proponía**: bajar a un modelo pequeño, cuantizar fuerte, y recuperar
velocidad con decodificación especulativa.

**Lo que se tomó**: el cuantizado agresivo, sí — Q3_K_M acierta lo mismo que Q4
con 363 MB menos (punto 9).

**Lo que se dio vuelta**: la dirección del tamaño. La propuesta empujaba hacia
abajo; medido, el 3B acierta 10/15 contra 7/15 del 0,5B, y el proyecto se queda
con el grande. Lo que hay que hacer chico es el **runtime**, no el modelo.

**Lo que se midió y se borró**: la especulativa, 1,91x más lenta (punto 10). Y
el kernel fusionado propio, que no se escribió porque candle ya lo tiene (punto
11). Las dos promesas más grandes del análisis fueron las dos que no sobrevivieron.

### D. QNLP, DisCoCat y redes de tensores

**Lo que se tomó**: el diagnóstico de los tokenizadores y el desmontaje del
"token cuántico", que son correctos y están resumidos en el punto 13. Y la idea
de comprimir por rango bajo, acotada al único sitio donde hoy rinde: la matriz
de salida ([ROADMAP](ROADMAP.md) §2).

**Lo que no**: la ruta cuántica, por fuera de alcance, y la cifra de cinco
órdenes de magnitud, que no está medida a escala real. Las redes de tensores
quedan aplazadas por una razón concreta y no por desconfianza: atacan la memoria,
y el motor está limitado por cálculo.
