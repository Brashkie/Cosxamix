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

Hasta la 0.2.0 este punto decía "el método es bueno, el momento no" y se
apoyaba en un razonamiento: que el bucle que importa ya está escrito a mano en
candle. Razonar no es medir. Así que se escribió.

El kernel vive en **dos archivos de ensamblador sueltos**,
`kernels/dot_avx2_win64.s` y `kernels/dot_avx2_sysv.s`, no en `asm!` metido
entre Rust. Son dos y no uno porque el ensamblador es **el único lenguaje de
esta frontera que tiene que saber la convención de llamada**: Rust, Zig y
Fortran reciben sus argumentos donde el compilador decida, acá hay que escribir
en qué registro llega cada uno, y Windows usa `rcx rdx r8 r9` donde System V
usa `rdi rsi rdx rcx`.

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

> **Revisado dos veces en la 0.2.0.** La primera revision concluyo que el motor
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
todas formas. Para comprobarlo se escribio una segunda variante que descuantiza
cada columna una sola vez y despues hace `m` productos punto en f32 — o sea que
elimina ese trabajo repetido por completo. Salio **entre 5 y 25 veces mas
lenta**:

| filas | candle | bucles al reves | descuantizar una vez |
|---:|---:|---:|---:|
| 1 | 0,292 ms | 1,05x | **0,04x** |
| 2 | 0,445 ms | 0,91x | 0,06x |
| 5 | 1,287 ms | 1,17x | 0,16x |
| 8 | 1,743 ms | 1,10x | 0,20x |

Porque `vec_dot_q3k_q8k` desempaqueta **dentro de los registros**, fundido con
el producto, sin materializar nunca los f32. Separarlo cuesta 8 KB de escritura
y lectura por columna y pierde el AVX2 escrito a mano. Es lo mismo que ya decia
el punto 11 de este documento, aprendido otra vez.

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

## Cifras que hubo que corregir

Se listan porque una cifra mal medida dirige el trabajo hacia el lado
equivocado, y las cuatro lo hicieron un rato.

| se dijo | era | por qué |
|---|---|---|
| 6,1 tok/s de generación | ~25 tok/s | se dividían tokens por el tiempo **total**, con respuestas de 1-2 tokens donde el prefill domina |
| al 30% del techo de memoria | el 3B está al **51%** | el 30% era del 0,5B |
| el 1,5B es el punto dulce | no lo es | la prueba de 6 preguntas no tenía resolución |
| la capa de salida explica la especulativa lenta | es el 15% | la causa es el coste de descuantizar cada k-quant, que se paga por fila |
| el motor está limitado por cálculo | **sigue siendo cierto**, y se confirmó midiendo | la "corrección" que lo negaba duró medio día: dos kernels la desmintieron (§10) |
| el escalado de hilos no tiene explicación | sí la tiene, para el decode | las matrices que no entran en L3 no escalan, y son el 89% de los pesos (§10) |
| candle comparte los pesos entre instancias | **931 MB cada una** | el parseo de `tasklist` estaba roto y daba 0,0 MB |

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
