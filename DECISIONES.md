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

## 11. El kernel fusionado propio: no se escribió porque ya existe

Se propuso escribir un `fused_dequant_gemv` para Q3_K, partiendo de que la ruta
convencional descuantiza a f32 y después multiplica.

**candle no hace eso.** Tiene `vec_dot_q3k_q8k` en `avx.rs`: un kernel
fusionado en intrínsecos AVX2 escritos a mano, derivado del de llama.cpp, que
desempaqueta los bloques directo a registros y acumula con FMA.

Competir contra eso tiene muy poca chance. Por la regla 3 se borraría igual,
solo que después de días de trabajo.

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
