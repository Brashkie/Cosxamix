# Roadmap

El objetivo de Cosxamix es correr rápido en hardware débil —de un PC de oficina a
un teléfono— sin depender de NVIDIA. Todo lo de acá se ordena por eso, y el
orden sale de mediciones, no de preferencias.

**Regla 3**: un kernel o una técnica se queda si le gana a la referencia en
velocidad **o** habilita una plataforma donde no había aceleración. Si no, se
borra. Ya borró la decodificación especulativa.

---

## Dónde estamos, medido

Máquina de referencia: i5-10400 (6 núcleos / 12 hilos, AVX2), DDR4-2133 doble
canal: 34,1 GB/s teóricos y **~30 GB/s medidos** (`examples/measure_ceiling.rs`,
DECISIONES §33). Sin GPU en el camino.

| | |
|---|---|
| modelo | Qwen2.5-3B-Instruct Q3_K_M, 1644 MB |
| residente | 1687 MB (x1,03 del archivo) |
| generación | ~14,7 tok/s (con la precarga de la 0.9.0) |
| lectura del prompt | ~50 tok/s (333 tokens en 6,6 s, matmul por bloques) |
| **del techo de memoria** | **~81%** (`30 ÷ 1,644 = 18,2 tok/s`) |
| calidad | 10/15 preguntas verificables |
| sesiones concurrentes | 16, caché LRU, intercalar no cuesta nada |
| plataformas con kernel vectorizado | x86-64 (AVX2) y ARM/Android (Zig) |

El techo es `tokens/s ≈ ancho de banda / tamaño del modelo`. El 51% que decía
esta tabla se calculaba contra 37,5 GB/s, que esta RAM no puede dar; con el techo
medido, y la precarga, el decode está al **~81-87%** según el formato
([DECISIONES](DECISIONES.md) §33). Lo que queda en esta máquina es poco y está en
los kernels de Q4_K y Q6_K. **La mejora más grande ya no es de software**: con la
RAM a 2666 (XMP), el techo sube un 25%.

---

## Corto plazo

### 1. El escalado de hilos: cerrado, y sin nada que hacer

Era el bloqueante declarado. Ya está explicado, y la respuesta es que **no se
arregla con código**.

Medido matriz por matriz a 6 hilos, lo que entra en los 12 MB de L3 escala
**7,03x** —superlineal— y lo que no entra se queda entre 1,76x y 2,82x. El FFN
y la capa de salida son el **89% de los pesos**, así que el forward hereda su
techo: ponderando por participación da **2,47x** contra los **2,60x** medidos.

Es el muro de memoria, no sincronización. Más hilos no lo mueven; lo único que
lo mueve es **leer menos bytes**, y eso es el punto 2.

**Lo que se probó y se borró por la regla 3**, para que no se reintente:

| intento | resultado |
|---|---|
| bucles de la multiplicación al revés, reusando la columna de pesos | 0,91x–1,17x — dentro del ruido |
| descuantizar cada columna una vez y hacer `m` productos en f32 | **0,04x–0,20x**, de 5 a 25 veces peor |

El segundo explica al primero: `vec_dot_q3k_q8k` desempaqueta el k-quant dentro
de los registros, fundido con el producto, sin materializar nunca los f32.
Ningún orden de bucles evita ese trabajo, y separarlo es mucho peor. El detalle
está en [DECISIONES](DECISIONES.md) §10.

De paso quedó una advertencia de método: los milisegundos absolutos de estas
mediciones **no son estables entre corridas**, porque el FFN gate pesa 9,7 MB
contra 12 MB de L3 y su tiempo depende de qué más haya residente. Solo son
comparables las relaciones dentro de una misma corrida.

### 2. Comprimir la matriz de salida

La única optimización del lado de la memoria que vale la pena **ahora**, porque
ataca bytes leídos *y* aritmética a la vez.

| | embeddings | salida | juntos |
|---|---:|---:|---:|
| 0,5B | 19,3% | 29,8% | **49,1%** |
| 3B | 7,8% | 14,9% | **22,6%** |

En el 3B son 243 MB que se multiplican enteros en cada token para producir
logits sobre 151.936 palabras, de las que se usa **una**. Dos caminos:

- **Descomposición de rango bajo** de `output.weight`. Hay que medir cuánta
  calidad se pierde contra las 15 preguntas antes de aceptarla.
- **Vocabulario candidato**: proyectar solo sobre los k tokens plausibles. Más
  barato pero cambia la distribución, así que necesita la prueba de logits de
  `compare_logits` para demostrar que el argmax no se mueve.

### 3. Cerrar el camino de ARM y Android

Lo que ya está: `dot.zig` cruza a `aarch64-linux-android`, toda la pila compila
para ese target —el motor ya no usa `tokenizers`, que era lo que más costaba
cruzar—, y el cruce a `aarch64-unknown-linux-gnu` lo **comprueba CI en cada push** — leyendo
la cabecera ELF de la librería de Zig, no solo que compile.

Lo que falta es **medirlo en un teléfono de verdad**. Las cifras de ARM hoy son
una expectativa, no un dato, y este proyecto no acepta expectativas:

- `dot` y `rmsnorm` en ARM NEON vía Zig, contra la referencia escalar.
- Si candle no tiene kernels fusionados para NEON —hay que confirmarlo— la ruta
  de la multiplicación cuantizada cae a escalar en ARM, y ahí sí habría un
  kernel propio que escribir con razón: sería *habilitar una plataforma*, el
  segundo brazo de la regla 3.
- Los fusionados propios de Q4_K y Q3_K con `sdot` ya están, en C
  (`kernels/neon_dotprod.c`), y desde la 0.5.1 en el despachador: 2,4x y 5,4x
  en el runner ARM de CI. `cargo bench --features zig,c --bench kernels --
  dot_q4k_i8` en Termux da la cifra del teléfono.
- Comprobar cuál es el techo de memoria en un teléfono. Con ancho de banda
  mucho menor, el 3B puede quedar fuera y el 0,5B volver a tener sentido ahí.

### 4. Afinar el servicio

Trabajo sin misterio, pendiente:

- ~~Respuestas en *streaming*~~ **hecho en la 0.3.0**: SSE en el dialecto de
  OpenAI, primer trozo a los 2,14 s contra 5,82 de la respuesta completa, y la
  generación se corta si el cliente cierra.
- ~~Carga de más de un modelo a la vez~~ **hecho en la 0.3.0**: registra todos
  los `.gguf` del directorio, carga bajo demanda y desaloja el menos usado
  cuando no entra en `COSXAMIX_RAM_MB`.
- ~~Reaprovechar el prefijo entre conversaciones distintas~~ **hecho en la
  0.7.0**: un chat nuevo copia de otro el comienzo que comparten, y el primer
  mensaje de un grupo de WinsiBot baja de 6,0 a 3,5 s ([DECISIONES](DECISIONES.md)
  §26).
- ~~Persistir el caché de sesión en disco~~ **medido y descartado**. Lo que
  ahorra es el prefill del *system prompt* en el primer mensaje de cada
  conversación tras reiniciar: medido contra WinsiBot, **~1 s** (4905 ms a
  3891 ms). Lo que cuesta un reinicio es **cargar el modelo: 45-48 s**. Es
  optimizar un segundo de cada cincuenta.

  Y el precio no es cero: 16 sesiones de 1000 tokens del 3B son **1,1 GB** que
  escribir al apagar, más un formato en disco que mantener y un riesgo que no
  avisa — restaurar un caché contra otros pesos no da error, da basura, así que
  haría falta una huella del modelo y la disciplina de comprobarla.

  **Vuelve a tener sentido** si alguna vez reiniciar deja de costar 45 s —por
  ejemplo con carga por `mmap`—, porque ahí el segundo del prefill pasaría a
  ser una parte grande del total. Antes, no.

  *Revisado en la 0.7.0:* el motor propio ya carga por `mmap` en ~0,5 s, así
  que la condición se cumplió. Pero sigue sin convenir por otro lado:
  WinsiBot renueva el perfil de cada grupo en su *system* cada media hora,
  así que un caché guardado vencería enseguida, y el préstamo de prefijo ya
  cubre el tramo fijo, que es lo único que sobrevive.

---

## Medio plazo

### 5. Revisar la especulativa cuando cambie la condición

Está borrada, no descartada. Vuelve a tener sentido si se cumple **una** de
estas dos, y las dos son verificables:

- el punto 1 cierra la brecha de cómputo y el motor pasa a estar limitado por
  memoria — ahí procesar K posiciones sí costaría ~1x;
- aparece un borrador que coincida con el 3B mucho más que el 28-50% medido con
  el 0,5B.

El código se puede recuperar del historial; `truncar_cache` y la máscara con
caché, que eran lo difícil, ya están en el árbol.

**Revisado en la 0.9.0: la condición 1 se cumplió y la cuenta todavía no da.** El
decode está al ~81-87% del techo, pero eso no hace barato verificar `K`
posiciones: el prefill, que ya va por bloques, cuesta ~26 ms por token en el 3B
(256 en 6,6 s) contra ~68 ms un token de decode. Verificar 4 cuesta entonces al
menos ~1,5 tokens de decode. Con el 0,5B aceptado el 40% —en el medio del 28-50%
medido—, cada verificación deja `(1 − 0,4⁵) / (1 − 0,4) ≈ 1,65` tokens, y los 4
del borrador cuestan ~1 token más del 3B. **Es una estimación, no una medición**:
~2,5-2,7 tokens de costo por 1,65 de avance, más lento. Lo que la cambiaría, y hay que
medir antes de escribir código:

- cuánto cuesta de verdad el forward con `K` = 2, 4 y 5 posiciones en el motor
  propio, que no es el de candle con el que se midió la tabla del README;
- un borrador que acierte mucho más: un daiss entrenado para imitar al 3B, o
  *n*-gramas del mismo prompt, que no cuestan casi nada.

### 6. Redes de tensores (MPS / Tensor-Train)

**Medido, y el resultado reordena el punto.** `examples/measure_rank.rs` dice
que un peso aleatorio no tiene nada que descomponer —captura exactamente
`k / rango`— y que uno entrenado sí, pero mucho menos de lo que prometen los
papers: `output.weight` del 0,5B necesita el 28,6% del rango para el 47,5% de
la energía.

Dónde sí vale, con número: `output.weight` con `k` chico, **22,1x** sobre el
control aleatorio con k=8 contra 7,58x del FFN. Hay unas ocho direcciones que
cargan el 20% de la energía.

Así que el punto queda dividido:

- **el FFN y la atención salen de la lista**: 7,58x cayendo a 1,37x no paga la
  complejidad de una Tensor-Train;
- **la proyección de salida se queda**, y pasa a depender de medir el efecto en
  los logits y no en la energía, porque energía de Frobenius no es precisión;
- **nada de esto se aplica a daiss hasta que esté entrenado**, y eso ya no es
  una opinión.

Después, en la 0.8.0 (§32): **cortar después pierde**, también en las tablas de
vocabulario (lo que TensorGPT promete sin reentrenar), y **entrenar desde el
principio con las tablas en Tensor-Train gana**. Guardar los núcleos en vez de
las tablas densas es **QCML**, el contenedor propio que va a ser un proyecto
aparte (§11).

El detalle y el control de ruido de cuantización, en
[DECISIONES](DECISIONES.md) §12.

### 7. Entrenamiento y ajuste fino en hardware débil

Es la mitad del objetivo del proyecto y hoy no existe nada: Cosxamix solo
infiere. Ya está medido que **es viable**, y con qué límites.

| | |
|---|---:|
| paso de LoRA r=8, 0,5B, 256 tokens | ~3,3 s |
| 1000 pasos | ~55 min |
| parámetros entrenables | 540.672, el 0,1% del modelo |

El límite es la memoria, y define el alcance: la base hay que tenerla en f32
porque **`QMatMul` no implementa `bwd`** y el gradiente no atraviesa una
multiplicación cuantizada. El 0,5B en f32 son ~2 GB y entra; el 3B serían
~12 GB y no.

~~1. **El entrenador**~~ y ~~2. **Guardar y cargar adaptadores**~~ están
hechos en la 0.3.0: `src/training/`, feature `training`. Entrena de verdad —la
pérdida baja— y el adaptador sale de 2,1 MB. El paso real son **~5,2 s**, no
los 3,3 s proyectados: aquella cifra era un piso declarado y se quedó en el
63% de la realidad, porque dejaba fuera la atención, RoPE, las normalizaciones
y la capa de salida sobre 151.936 palabras.

Lo que falta:

3. **Quitar el gradiente que sobra.** `backprop.rs` calcula `dL/dW` de las
   capas congeladas aunque nadie lo use: 23,1 ms de los 135 por capa, un
   **17%** del paso. Se arregla con un `CustomOp1` que guarde la matriz dentro
   de la operación en vez de como nodo del grafo, de modo que su `bwd` solo
   devuelva `dL/dx`.
4. **El backward cuantizado**, que es lo que haría entrenable el 3B sin pagar
   los 12 GB. Es el kernel que hoy no existe: `dL/dy · W` sobre una matriz
   guardada como `(n, k)` cuantizada. Va último porque es lo más caro y porque
   los puntos 1 a 3 sirven igual sin él.

### 8. daiss: crear modelos, no solo servirlos

**La familia existe y se escribe.** `src/daiss.rs` define cinco tamaños,
`src/gguf_write.rs` los guarda como GGUF, y los dos están verificados:

| modelo | params | capas | dim | cab/kv | Q8_0 | F16 | caché/token | k-quants |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| daiss-mini | 0,08 B | 6 | 256 | 4/2 | 0,08 GB | 0,15 GB | 6,0 KB | sí |
| daiss-0.5b | 0,63 B | 24 | 896 | 14/2 | 0,62 GB | 1,17 GB | 24,0 KB | **no** |
| daiss-3b | 3,40 B | 36 | 2048 | 16/2 | 3,36 GB | 6,33 GB | 72,0 KB | sí |
| daiss-7b | 7,61 B | 28 | 3584 | 28/4 | 7,54 GB | 14,18 GB | 112,0 KB | sí |
| daiss-14b | 14,77 B | 48 | 5120 | 40/8 | 14,62 GB | 27,51 GB | 384,0 KB | sí |

El daiss-7b está escrito y comprobado: 339 tensores, 7,54 GB, y **candle lo
lee con cero diferencias** sobre los 544.538.624 valores de `output.weight`,
con los 339 tensores que el cargador pide presentes y con su forma. Que una
implementación que no comparte una línea con la nuestra lea el archivo es lo
que separa escribir GGUF de escribir un formato parecido.

Se genera en flujo, fila por fila y en paralelo: el pico de memoria son 16 MB
sea cual sea el modelo, así que el 14b no pide 14 GB. Y con semilla, porque un
modelo que no se puede regenerar bit por bit es un archivo, no un modelo — hay
un test que escribe el mini dos veces y compara los bytes.

**Lo que NO es**: un modelo entrenado. Los pesos son ruido con la desviación
correcta y genera texto sin sentido. Lo que sí es: el camino entero —definir,
inicializar, cuantizar, escribir, leer, cargar— existiendo y verificado, que
es lo que hay que tener antes de poder entrenar nada. El razonamiento sobre
qué parte de daiss es nuestra y cuál no está en [DECISIONES](DECISIONES.md)
§16.

Lo que sigue bloqueado es **preentrenar** uno grande desde cero.

Son miles de horas de GPU y cientos de GB de texto, y eso es aritmética, no
pesimismo. Pero preentrenar desde cero no es la única forma de crear un modelo
grande, ni la más usada:

| camino | qué cuesta | ¿acá? |
|---|---|---|
| **Fusionar** dos o más modelos (SLERP, TIES, DARE) | cero entrenamiento: leer, combinar pesos, escribir | **sí, hoy** |
| **Crecer en profundidad**: apilar capas de uno existente y seguir entrenando | entrenamiento, pero partiendo de pesos que ya saben | sí para el 0,5B |
| **Convertirlo en MoE**: clonar los FFN y entrenar el enrutador | cirugía más entrenamiento del enrutador | probablemente |
| **Preentrenamiento continuado** sobre un corpus propio | entrenamiento completo sobre un modelo grande | no para el 3B en f32 |
| **Destilar** de un modelo grande a uno chico | hace falta correr el grande muchas veces | lento, pero posible |
| **Preentrenar desde cero** un modelo grande | miles de horas de GPU | no |

Fusionar es el que se puede hacer **ya**: no necesita gradientes, solo leer
GGUF, combinar y escribir. Y produce un modelo que no existía.

Para los que sí entrenan, el límite medido es la memoria: la base tiene que
estar en f32 porque `QMatMul` no implementa `bwd`, y el 3B en f32 son ~12 GB
contra 7,3 libres. Eso lo levanta el backward cuantizado del §7, no más RAM.

Lo que decide el orden del trabajo es esta cuenta, que es aritmética y no
opinión. Parámetros de un modelo según su vocabulario:

| forma | vocab 151.936 | vocab propio 8.192 |
|---|---:|---:|
| h=256, 6 capas | 44,6 M — **87% embedding** | 7,8 M — 27% embedding |
| h=512, 8 capas | 108,2 M — 72% embedding | 34,6 M — 12% embedding |
| h=768, 12 capas | — | 108,9 M — 6% embedding |

Con el vocabulario de Qwen, un modelo chico es **casi todo tabla de
búsqueda**: el transformer sería el 13% de los parámetros. Así que el orden es:

1. ~~**Inicialización aleatoria**~~ **hecho**: `daiss::escribir`, con
   `normal(0, 0.02)` y las dos proyecciones que escriben en el residual
   divididas por `sqrt(2 * capas)`, que es lo que evita que la varianza crezca
   con la profundidad.
2. ~~**Exportar a GGUF**~~ **hecho**: `gguf_write`, en flujo y verificado
   contra candle. Lo que se cree acá lo sirve el mismo motor y corre en un
   teléfono.
3. ~~**Un tokenizador propio**~~ **hecho**: `src/tokenizer.rs`, un BPE a nivel
   de bytes con entrenamiento propio, y `src/tokenizer_json.rs`, que lee el de
   Qwen y es lo que usa el motor; el crate `tokenizers` queda solo para
   comparar ([DECISIONES](DECISIONES.md) §27). Medido sobre texto que no vio, el nuestro de 8.192
   necesita **3,4% más tokens** que el de Qwen de 151.936, y eso lleva a
   daiss-mini de 94% tabla de búsqueda y 82,5 M de parámetros a **47% y
   8,9 M** — nueve veces más chico. Con dos salvedades medidas: el
   entrenamiento se satura en 7.185 fusiones porque el corpus son 382 KB, y el
   texto de prueba es del mismo dominio, así que parte del 3,4% es
   especialización ([DECISIONES](DECISIONES.md) §19). **Lo que falta ahora es
   un corpus**, no código.
4. ~~**Medir un paso**~~ **hecho**: `src/training/full.rs`. En daiss-mini con lote
   16 × 128, 1,3 a 2 s por paso, y en 10 minutos de CPU baja de `xz`
   ([DECISIONES](DECISIONES.md) §32).

**Chavín** existe desde la versión sin publicar: `chavin-30b`, 29,5B, 9,02 GB en
Q2_K, escrito, leído por candle y corriendo a 2,38 tok/s, el 71% del techo de
memoria medido (§33), con pesos al azar ([DECISIONES](DECISIONES.md) §29 y §30). Lo que sigue,
en orden:

1. ~~**Bytes contra BPE, con más tiempo**~~ **hecho a 30 minutos**: BPE 8.192 da 1,77
   bits por byte y bytes 1,95 ([DECISIONES](DECISIONES.md) §32). Falta repetir la de
   contexto parejo con la CPU libre, y probar más tiempo.
2. ~~**Embeddings en Tensor-Train**~~ **hecho en la 0.8.0** (§32): cortado después
   pierde; entrenado desde el principio, con rango 32, le gana al modelo sin
   tokenizador con los mismos parámetros. Guardar los núcleos en el archivo, para
   que también se achique, es **QCML**, un proyecto aparte (§11).
3. ~~**Servir un daiss entrenado**~~ **hecho en la 0.8.0**: el motor lee
   `cosxamix.tokenizer.merges` y, sin plantilla de chat, continúa el texto.
4. **Un kernel de entrenamiento propio**: repartir el lote da 2,5× en 6 núcleos.
5. **Medir el kernel de Q2_K en Zig en un ARM sin `sdot`**, que es donde se usa.
6. **Kernels para el cuantizador**: escribir Chavín son 14,5 minutos y casi todo
   es la búsqueda de Q2_K, 115 ms por millón de valores contra 20 de Q3_K.
7. **Capas que comparten pesos**: un núcleo común a todas las capas y un
   adaptador chico por capa, como proponen SCADA-Tucker y Minima. Cortarlo después
   en un modelo entrenado casi seguro pierde, como el Tensor-Train; lo que se
   puede probar es **entrenar** un daiss-mini así y comparar bits por byte con los
   mismos parámetros.
8. **Imágenes propias.** `/api/image` corre SD-Turbo, un modelo de terceros
   (Stability AI Community License): es el "Qwen de las imágenes", lo que sirve
   hoy mientras se construye lo propio. Entrenar desde cero un modelo de
   difusión de la calidad de SD en una CPU no da: son meses de GPU. Lo que sí se
   puede probar: uno chico, de 64×64 o 128×128 (íconos, stickers), entrenado
   sobre los latentes de TAESD, que es MIT; y las convoluciones con kernels
   propios en vez de las de candle, que además tienen el error de §`image_gen`
   (canales = alto = ancho). Hoy, por imagen: ~12,6 s el UNet y 3 s TAESD.

### 9. El motor propio, reemplazando candle pieza por pieza

Es el cambio de rumbo de la 0.3.0: dejar de apoyarse en candle, por lo medido
—los cinco techos que están en el README— y porque la superficie que de verdad
se usa es estrecha.

| fase | qué | estado |
|---|---|---|
| **1** | lector de GGUF propio | **hecho**: coincide con candle en los 5 modelos, incluidos bytes crudos y offsets |
| 1b | descuantización de los k-quants y de Q5_0/Q8_0 | **hecho**: los 8 formatos, **valor por valor idénticos** a candle en 21 tensores de los 5 modelos. El desempaquetado directo a `i8` cubre los 7 cuantizados, con sub-bloque de 16 y de 32 |
| **2** | las operaciones: matmul cuantizada, RoPE, softmax | **hecha**: la matmul le gana a candle en lote, y con una fila —el decode— quedó **pareja** desde que los `f16` se convierten con `vcvtph2ps`; RoPE y softmax coinciden con candle **valor por valor** |
| **3** | el forward de Qwen2 sobre la pila propia | **anda y genera texto**: idéntico a candle token por token, prefill 2,8x más rápido, decode a la par y 1,26x de punta a punta. **Enchufado al servidor y el de por defecto desde la 0.6.0**: 1,6-1,9x más rápido que candle por mensaje usado como lo usa WinsiBot, y 11/15 contra 10/15 en la batería ([DECISIONES](DECISIONES.md) §23). `COSXAMIX_MOTOR=candle` vuelve al de antes |
| 4 | el backward cuantizado | es lo que candle no tiene y lo que desbloquea entrenar el 3B |

**La matmul propia ya le gana a candle en lote.** Medida sobre el `ffn_gate`
del 3B, dos pasadas por formato y todo dentro del mismo proceso — las cifras de
procesos distintos no se comparan, por lo medido en
[DECISIONES](DECISIONES.md) §15:

| filas | candle Q4_K | nuestra | | candle Q3_K | nuestra | |
|---:|---:|---:|---|---:|---:|---|
| 1 | 0,494 ms | 0,895 ms | 0,55x | 0,319 ms | 1,023 ms | 0,31x |
| 2 | 1,080 | 1,205 | 0,90x | 0,574 | 1,046 | 0,55x |
| 5 | 2,128 | 2,462 | 0,86x | 1,723 | 1,633 | **1,06x** |
| 8 | 3,776 | **3,170** | **1,19x** | 2,551 | **2,433** | **1,05x** |

El coste por fila cae de 0,90 a 0,40 ms en Q4_K y de 1,02 a 0,30 en Q3_K
mientras el de candle se queda plano, y eso es lo que necesita el prefill.

Dos cosas lo consiguieron, y ninguna fue afinar un kernel:

- **quitar un rodeo**: la primera versión descuantizaba la columna a `f32` y la
  volvía a cuantizar a `i8`. Ahora se desempaqueta directo del formato, porque
  los pesos **ya eran enteros con una escala**. De 10,3 a 2,65 ms con una fila,
  4x sin tocar una instrucción vectorial;
- **quitar la comprobación de límites** de los diez bucles de desempaquetado,
  que les impedía vectorizar: entre 4,3x y 12,1x según el formato, y el 94% del
  trabajo de un hilo estaba ahí. El detalle, en §15.

Y es además más precisa que candle: 0,25% de diferencia, a nuestro favor,
porque candle usa una escala cada 256 valores y nosotros una cada 32.

**Lo que falta:**

1. ~~**El caso de una fila**~~ **hecho para Q4_K**: el kernel fusionado
   existe, `kernels::dot_q4k_i8` y `ops::matmul_i8_fusionado`, y pasó de 0,55x
   a **0,85x** contra candle con una fila mientras gana **1,93-2,39x** con
   ocho. Candle sigue adelante en el decode, pero por un 15%. Lo que falta son
   **los otros seis formatos**, y cada uno pide su propio truco porque todos
   restan algo —32 en Q6_K, 16 en Q5_0, hasta 4 en Q3_K según una máscara— y
   restar rompe el rango sin signo que `vpmaddubsw` necesita. Q3_K es el
   siguiente, porque es el formato del modelo por defecto
   ([DECISIONES](DECISIONES.md) §11). **Su kernel fusionado pasó de 154 a 91
   ns por fila** con cuatro cambios de algoritmo, y el mismo algoritmo en C y
   en ensamblador no le gana ([DECISIONES](DECISIONES.md) §21). Lo que sigue
   en lenguajes es ARM: **hecho en la 0.5.1** —`sdot` desde C, 2,4x sobre
   Zig en Q4_K y 5,4x sobre la referencia en Q3_K, en el despachador
   ([DECISIONES](DECISIONES.md) §22)—. Falta la cifra de un teléfono.
2. ~~**Elegir el camino según el formato**~~ **hecho**: `ops::matmul_auto`
   despacha por formato y por número de filas, con los umbrales medidos — y
   resultó que hay **tres** caminos en juego y ninguno gana siempre. Lo que
   falta es medirlo en los cinco modelos y no en un tensor, y en otro hardware,
   porque el cruce se mueve ([DECISIONES](DECISIONES.md) §11).
3. El microkernel con bloqueo de registros, que es donde el ensamblador tendría
   su primera oportunidad real — ver [DECISIONES](DECISIONES.md) §5, con el
   paso previo que le faltaba al criterio en §15.

**RoPE y softmax están**, y los dos coinciden con candle valor por valor. De
RoPE lo que importaba no era la velocidad sino acertar la convención: Qwen2 usa
la de **mitades partidas** y no la intercalada, y elegir mal **no da ningún
error** — el modelo carga, genera y dice cosas sin sentido. Por eso el test que
vale es el que compara contra candle y no el de propiedades, aunque los dos
estén. Las tablas de ángulos se calculan en `f64` y se guardan en `f32`: con
32.768 de contexto el ángulo llega a 32.768 radianes, donde un `f32` tiene
0,002 de resolución.

**La atención también está**, con máscara causal y cabezas agrupadas, y
coincide con candle valor por valor.

**Y la fase 3 anda.** `src/forward.rs` encadena las piezas propias —lector de
GGUF mapeado, descuantización, `matmul_auto`, RoPE, atención, softmax— hasta
los logits, sin candle en el camino. Comparado contra candle sobre el 0,5B en
Q4_K, con `examples/compare_forward.rs`:

| régimen | tokens | nuestro | candle | peor diferencia | argmax |
|---|---:|---:|---:|---:|---|
| prefill | 6 | **68 ms** | 130 ms | 1,88% del rango | el mismo |
| decode | 1 | 36 ms | 30 ms | 1,35% | el mismo |

Y la prueba que de verdad decide, porque coincidir en un argmax puede ser
suerte: generando con el mismo criterio voraz, las dos producen **los mismos
13 tokens** y divergen en el catorceavo, donde dos candidatos estaban casi
empatados.

**De dónde sale esa diferencia, medido y no supuesto.** La hipótesis obvia era
que la causa fuera nuestra cuantización de la entrada, así que se agregó
`COSXAMIX_MATMUL=f32` para forzar el camino que no cuantiza nada. El resultado
la descarta: con `f32` la diferencia de logits **sube** a 2,19% y la coincidencia
**sube** a 16 tokens. O sea que no es nuestra cuantización. Nuestro camino de
`f32` es en principio el más preciso de los dos y aun así se aparta un 2%, lo
que apunta a la cuantización de activaciones de candle a Q8_K —una escala cada
256 valores contra una cada 32, ya medida en 0,25% por matmul a nuestro favor—
compuesta a lo largo de 24 capas.

**Y se zanjó, con el modelo sin cuantizar.** `examples/make_f32_model.rs`
convierte un GGUF cuantizado a `F32` con el escritor propio —tensor por
tensor, así que un modelo más grande que la RAM se convierte igual— y la
conversión es exacta: cero diferencias sobre los 136.134.656 valores de
`token_embd`. Con esos pesos ninguno de los dos cuantiza nada, y entonces
tienen que coincidir. Coinciden:

| régimen | nuestro | candle | peor diferencia | argmax |
|---|---:|---:|---:|---|
| prefill | 136 ms | 146 ms | **0,0000** | el mismo |
| decode | 92 ms | 119 ms | **0,0000** | el mismo |

Y la generación voraz sale **idéntica en los 24 tokens**, no 13.

Eso prueba las dos cosas que antes solo se podían inferir: el forward propio
es **correcto**, y el 1,88-2,19% del modelo cuantizado era **íntegramente
cuantización**, de los dos lados.

### Y genera

`src/sampler.rs` es el muestreador propio —temperatura y núcleo de top-p, la
misma semántica que el `LogitsProcessor` de candle— y `Modelo::generar` hace
el prefill de una vez y después va de a un token, parando en el `eos` del
archivo o cuando el que recibe los tokens dice que basta. Sobre el 0,5B, con
el mismo prompt de chat que arma el bot y los dos motores calentados:

| | prefill | decode | total |
|---|---:|---:|---:|
| propio | **303–315 ms** | 23,0–23,5 tok/s | **2,06–2,09 s** |
| candle | 864–891 ms | 23,4–23,5 tok/s | 2,61–2,64 s |

Las dos secuencias salen **idénticas en los 42 tokens**, en las dos corridas.
El prefill gana por lo mismo que gana la matmul en lote, y el decode queda a la
par por lo mismo que la matmul de una fila.

La primera medición dio un prefill **9x peor** que candle, y era el disco: el
mapeo trae los pesos en el primer toque, así que la primera llamada leía 0,45
GB a los 41 MB/s de F:, mientras candle los lee en `from_gguf`, fuera de su
cronómetro, y encima con el archivo ya en caché gracias a nosotros. Los dos lo
pagan; en sitios distintos. Ahora los dos calientan antes de medir.

De paso, el experimento encontró un bug que nadie habría visto de otra forma:
`matmul_auto` mandaba los formatos sin cuantizar al camino de enteros, que no
tiene nada que desempaquetar, y fallaba con *"F32: sin desempaquetado directo
a i8"*. Nunca había aparecido porque todos los modelos vienen cuantizados. Ya
está arreglado y el test del despacho cubre `F32` y `F16`.

**La regla del reemplazo**: nada sale de candle hasta que la pieza propia dé lo
mismo sobre los modelos reales. Candle se queda como dependencia de **test**
aunque salga del binario — tener motor propio no es razón para tirar la forma
más barata de saber si está bien.

Y es donde el ensamblador deja de competir contra un bucle que LLVM ya
vectoriza: en la matmul cuantizada propia el bucle caliente es **nuestro**, y
ahí sí se puede probar AVX-512, que candle no usa.

### 10. Esparcidad de las activaciones: medir antes de escribir

La entrada de `ffn_down` es `silu(gate) · up`. Los papers dicen que el 50-70% es
cero, pero **SiLU no da ceros exactos**: eso es de ReLU. Lo que se puede hacer es
tratar como cero lo que está bajo un umbral (TEAL), saltar esas columnas y no
leer sus pesos, y eso cambia el resultado. Antes de escribir un kernel:

1. medir, en el Qwen 3B, qué fracción de esa entrada queda bajo cada umbral, capa
   por capa;
2. medir cuánto cuesta en precisión con la infraestructura de KL de
   `measure_precision`, saltando de verdad esas columnas;
3. si el costo es chico, resolver el formato: los pesos van por filas en bloques
   de 256, así que saltar columnas pide guardar `ffn_down` traspuesta o en
   bloques por columna.

Si el 1 o el 2 no dan, se anota y se cierra, como la especulativa.

### 11. QCML: el formato propio que quiere superar a GGUF, en un proyecto aparte

**QCML** (`.qcml`) es un formato de archivo binario propio para guardar modelos
de lenguaje. Se va a crear como **un proyecto aparte**, sin fecha todavía;
Cosxamix sigue optimizando y evolucionando por su lado y lo va a leer cuando
exista. **El objetivo es superar a GGUF en todos los aspectos**, no solo en uno:
tamaño, carga, memoria, velocidad, precisión, robustez y capacidad de crecer. Y
como todo acá, "superar" quiere decir **medido contra GGUF sobre los mismos
modelos**, aspecto por aspecto, no declarado.

#### Qué significa el nombre

**QCML = *Quantum Compression for Machine Learning***, compresión cuántica para
aprendizaje automático:

- ***Quantum*** (cuántica): por las **redes de tensores**, que nacieron en la
  física cuántica. El estado de un sistema de muchas partículas tiene un tamaño
  que crece exponencialmente, y la física lo guarda como una cadena de tensores
  chicos, un **MPS** (*matrix product state*), donde el *rango de enlace* decide
  cuánto se guarda. En matemática aplicada la misma estructura se llama
  **Tensor-Train** (Oseledets, 2011). Una matriz de pesos se puede tratar igual:
  como un tensor de varios índices, guardado como una cadena de núcleos. **No hay
  cómputo cuántico**: son productos de matrices en la CPU. El nombre dice de dónde
  viene la matemática, no que haga falta una computadora cuántica.
- ***Compression*** (compresión): guardar el mismo modelo en menos bytes,
  combinando dos cosas que se complementan: **factorizar** (redes de tensores, que
  quitan redundancia entre filas y columnas) y **cuantizar** (menos bits por
  valor, como los K-quants que ya hace Cosxamix).
- ***for Machine Learning*** (para aprendizaje automático): hecho para pesos de
  modelos —matrices, tablas de vocabulario, normas— y para leerlos mapeados y
  multiplicar sin copiar, no como un formato de compresión general.

#### Por qué hace falta, medido

En la 0.8.0, un daiss-mini con las tablas de vocabulario entrenadas en
Tensor-Train de rango 32 tiene esas tablas **30,6× más chicas** y le gana al
modelo sin tokenizador con los mismos parámetros (§8 y
[DECISIONES](DECISIONES.md) §32). Pero GGUF solo sabe guardar la matriz:
`daiss-mini-8192-tt32-30m.gguf` pesa 35,7 MB, lo mismo que el modelo sin
Tensor-Train. **El ahorro existe en el modelo y se pierde en el archivo.**
Guardando los núcleos, la estimación —**a medir**— es de 35,7 a ~19,5 MB (−45%)
con los mismos bits por byte.

#### Qué es cierto de GGUF y qué no

Para que la comparación sea justa: GGUF **sí** guarda tensores de hasta 4
dimensiones, **sí** se mapea en memoria sin copiar (`gguf::Pesos` lo hace) y **sí** permite cambiar la alineación con `general.alignment`. QCML no
gana por eso. Gana, o tiene que ganar, en lo de la tabla de abajo.

#### Aspecto por aspecto: dónde GGUF se queda corto y cómo se mide

| aspecto | GGUF hoy | QCML | cómo se mide |
|---|---|---|---|
| **redes de tensores** | no tiene cómo describirlas: solo matrices | Tensor-Train / MPS nativo, después Tucker y Tensor-Ring, con factores y rangos en el índice | tamaño y bits por byte, mismo modelo |
| **llegar a un tensor** | el índice de tensores va **después** de los metadatos: en un Qwen hay que recorrer **5,96 MB** de cabecera, 5,93 de ellos el tokenizador en texto, para leer un índice de 24,8 KB (medido) | cabecera fija de 64 bytes que dice dónde empieza el índice; índice ordenado para buscar por nombre sin recorrer todo | tiempo de abrir y de encontrar un tensor |
| **tokenizador** | 151.936 tokens, sus 151.936 tipos y 151.387 *merges* como valores sueltos de los metadatos | en una sección propia y compacta, que se lee solo si se usa | bytes y tiempo de carga |
| **alineación** | 32 por defecto, configurable | fija a 64 bytes en todo bloque: una línea de caché y un registro de AVX-512 | — (por construcción) |
| **integridad** | sin sumas de comprobación: un archivo corrupto se detecta, si se detecta, generando basura | una suma por tensor (por ejemplo xxHash o CRC32C), verificable bajo pedido | detectar un byte cambiado a propósito |
| **validación** | formas y tipos se comprueban afuera, en cada lector | esquema estricto: el índice dice tamaños, y el lector rechaza lo que no cuadra antes de mapear | tests con archivos rotos a propósito |
| **disposición para los kernels** | bloques en el orden del formato; llama.cpp los reordena al cargar para algunas CPU | puede guardar los bloques ya intercalados para el kernel que los va a leer, sin reordenar al cargar | tiempo de carga y tok/s |
| **dimensiones** | hasta 4 | sin ese tope para los núcleos | — |
| **ecosistema** | atado a cambios de llama.cpp | propio, versionado, con su especificación | — |

Y lo que QCML **no puede empeorar**: con un modelo denso, cuantizado igual, tiene
que dar el mismo tamaño, la misma velocidad y los mismos bytes de pesos que el
GGUF. Si en algo queda peor, se anota y se corrige antes de decir que lo supera.

#### La estructura propuesta

En *little endian*:

1. la marca `QCML` (`0x51 0x43 0x4D 0x4C`);
2. una cabecera fija de 64 bytes: versión, alineación, cuántos metadatos y
   tensores hay, y dónde empiezan el índice, los datos y el tokenizador;
3. la tabla de metadatos, con tipos;
4. el índice de tensores, alineado a 64: por tensor, su nombre, sus dimensiones,
   su tipo, si va denso —cuantizado como en GGUF— o como red de tensores, y en ese
   caso la factorización, los factores, los rangos y dónde está cada núcleo; más
   su tamaño y su suma de comprobación;
5. el tokenizador, en su sección;
6. los datos contiguos, cada bloque alineado a 64, para mapearlos y leerlos sin
   copiar.

#### Lo que promete y lo que no

Las redes de tensores sirven para modelos **entrenados** con esas capas así,
donde la cadena es el modelo. **No sirven para comprimir un modelo que ya
existe**: cortar las tablas de un daiss entrenado después lo empeora 0,6-1,25
bits por byte, contra +0,004 de Q2_K (§32), y en el Qwen `output.weight` necesita
el 28,6% del rango para el 47,5% de la energía (§6). El "50-70% menos sin perder
precisión" de los papers no se cumple con pesos ya entrenados, así que el
convertidor desde GGUF guarda en red de tensores solo lo que se corta sin error,
y el resto lo guarda denso. Con un modelo como el Qwen, QCML gana en todo lo de
la tabla salvo en el tamaño de los pesos, donde empata.

#### Los pasos

1. **El formato**: la especificación, el lector y el escritor, y el convertidor
   desde GGUF. Medir contra GGUF, sobre los mismos modelos: tamaño, tiempo de
   abrir, RAM y bits por byte.
2. **Reconstruir al cargar** las matrices de las redes de tensores, y medir.
3. **Multiplicar directamente desde los núcleos**, sin la matriz densa: una fila
   de embeddings contrayendo la cadena, y los logits multiplicando por ella.
   Medir velocidad y RAM contra la tabla densa en Q8_0 y Q2_K.
4. **Los kernels de contracción** en Rust, Zig, C y ensamblador, como todos
   (regla 3).

#### Ideas para después, sin medir

- **El plan de contracción en la cabecera**: el orden en que se multiplican los
  núcleos decide los FLOPs y la memoria temporal. En una cadena de 4 núcleos es
  obvio, pero en Tucker o Tensor-Ring vale guardarlo calculado.
- **Una factorización por tipo de capa**: Tensor-Train en el vocabulario, Tucker
  en la atención, Tensor-Ring en el FFN, y las capas sensibles sin comprimir. Por
  lo de §32, solo vale entrenando así.
- **Un núcleo compartido entre capas**, con un adaptador por capa: es el punto 7
  de §8.

## Decidido y cerrado

No se reabren sin un dato nuevo. El razonamiento completo está en
[DECISIONES.md](DECISIONES.md).

| | estado |
|---|---|
| llama.cpp como motor | **no** — pide toolchain de C++ en el teléfono |
| D en el camino crítico | **no** — mismo backend LLVM, y su runtime no cruza la FFI |
| ASM en el bucle interno | **no todavía** — el bucle ya es AVX2 a mano; la pérdida no está ahí |
| kernel fusionado propio de Q3_K | **no** — candle ya tiene `vec_dot_q3k_q8k` en AVX2 |
| formatos IQ (IQ3_M, IQ2_XXS) | **imposible** — candle no los soporta; Q3_K_M es el equivalente |
| un modelo por sesión | **no** — 931 MB por instancia; diez chats serían 9,1 GB |
| `mmap` para los pesos | **innecesario** — el exceso era la tabla de embeddings, ya resuelto |
| QNLP / DisCoCat | **fuera de alcance** — la promesa de 5 órdenes de magnitud no está medida a escala real |
| `target-cpu=native` | **no** — `x86-64-v3`, que no ata el binario a la CPU que compiló |

---

## Integración con WinsiBot

Cosxamix ya responde al bot de verdad: el id de sesión (`group_jid`, o `sender_jid`
en privado) viaja desde `ai_chat.rs` y selecciona el caché. Tres turnos reales
medidos: 25.256 ms el primero —con las páginas frías—, después 4905 y 3891 ms,
con `cacheados=131` a partir del segundo.

Falta en el lado de WinsiBot:

```
OLLAMA_URL=http://127.0.0.1:11435
OLLAMA_MODEL=qwen2.5-3b-instruct-q3_k_m
OLLAMA_TIMEOUT=60000
```

El *timeout* de 40 s no alcanza para el primer turno en frío. Y queda por
decidir si Cosxamix se arranca solo desde `index.ts`, como la Session API, o si se
deja aparte — con el sesgo puesto en que instalar siga siendo `npm install` y
nada más.
