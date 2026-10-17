# Roadmap

El objetivo de Cosxamix es correr rápido en hardware débil —de un PC de oficina a
un teléfono— sin depender de NVIDIA. Todo lo de acá se ordena por eso, y el
orden sale de mediciones, no de preferencias.

**Regla 3**: un kernel o una técnica se queda si le gana a la referencia en
velocidad **o** habilita una plataforma donde no había aceleración. Si no, se
borra. Ya borró la decodificación especulativa.

---

## Dónde estamos, medido

Máquina de referencia: i5-10400 (6 núcleos / 12 hilos, AVX2), DDR4-2666 doble
canal, **~37,5 GB/s** reales. Sin GPU en el camino.

| | |
|---|---|
| modelo | Qwen2.5-3B-Instruct Q3_K_M, 1644 MB |
| residente | 1687 MB (x1,03 del archivo) |
| generación | ~11,9 tok/s |
| **del techo de memoria** | **51%** |
| calidad | 10/15 preguntas verificables |
| sesiones concurrentes | 16, caché LRU, intercalar no cuesta nada |
| plataformas con kernel vectorizado | x86-64 (AVX2) y ARM/Android (Zig) |

El techo es `tokens/s ≈ ancho de banda / tamaño del modelo`. Al 51%, lo que
queda por ganar en esta máquina **es menos de 2x**, y hace falta cerrar la brecha
de cómputo para cobrarlo.

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
para ese target gracias al `default-features = false` de `tokenizers`, y el
cruce a `aarch64-unknown-linux-gnu` lo **comprueba CI en cada push** — leyendo
la cabecera ELF de la librería de Zig, no solo que compile.

Lo que falta es **medirlo en un teléfono de verdad**. Las cifras de ARM hoy son
una expectativa, no un dato, y este proyecto no acepta expectativas:

- `dot` y `rmsnorm` en ARM NEON vía Zig, contra la referencia escalar.
- Si candle no tiene kernels fusionados para NEON —hay que confirmarlo— la ruta
  de la multiplicación cuantizada cae a escalar en ARM, y ahí sí habría un
  kernel propio que escribir con razón: sería *habilitar una plataforma*, el
  segundo brazo de la regla 3.
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

### 6. Redes de tensores (MPS / Tensor-Train)

Álgebra real y aplicable, pero reduce **bytes que leer**, y el motor está
limitado por cálculo. Es la misma trampa que mató a la especulativa.

Se mide después del punto 1, no antes. El punto 2 es un caso particular de esto
que sí vale ya, porque la matriz de salida es el único sitio donde comprimir
también recorta aritmética.

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

### 8. Crear modelos, no solo servirlos

Hoy Cosxamix sirve modelos ajenos y ajusta uno con LoRA. Crear modelos propios
es un objetivo declarado, **chicos y grandes**.

Lo que no sale de una máquina sin GPU es **preentrenar** uno grande desde cero:
son miles de horas de GPU y cientos de GB de texto, y eso es aritmética, no
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

1. **Un tokenizador propio.** Es lo primero, no un accesorio: sin él no hay
   modelo chico que valga la pena. El crate `tokenizers` ya está en el árbol y
   sabe entrenar un BPE.
2. **Inicialización aleatoria** en vez de cargar pesos. Es un delta pequeño
   sobre `src/training/model.rs`, que hoy construye la misma estructura leyendo
   un GGUF — hay que hacer `Var` todos los parámetros en vez de solo los
   adaptadores.
3. **Medir un paso** con todos los parámetros entrenables, que es otro régimen:
   el LoRA entrena el 0,1% y acá se entrena el 100%, así que el optimizador
   deja de ser gratis y la memoria del backward crece mucho.
4. **Exportar a GGUF**, para que lo creado lo pueda servir el mismo motor y
   correr en un teléfono. Sin esto el modelo entrenado no sale del entrenador.

Va después del punto 7 porque reusa todo lo de ahí, y lo que falta medir —el
punto 3— no se puede estimar desde el LoRA.

### 9. El motor propio, reemplazando candle pieza por pieza

Es el cambio de rumbo de la 0.3.0: dejar de apoyarse en candle, por lo medido
—los cinco techos que están en el README— y porque la superficie que de verdad
se usa es estrecha.

| fase | qué | estado |
|---|---|---|
| **1** | lector de GGUF propio | **hecho**: coincide con candle en los 5 modelos, incluidos bytes crudos y offsets |
| 1b | descuantización de los k-quants y de Q5_0/Q8_0 | **hecho**: los 8 formatos, **valor por valor idénticos** a candle en 21 tensores de los 5 modelos. El desempaquetado directo a `i8` cubre los 7 cuantizados, con sub-bloque de 16 y de 32 |
| **2** | las operaciones: matmul cuantizada, RoPE, softmax | **la matmul le gana a candle en lote**: 1,19x a 8 filas sobre Q4_K y 1,05x sobre Q3_K. Con una fila candle sigue adelante, 1,8x y 2,7x |
| 3 | el forward de Qwen2 sobre la pila propia | |
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

1. **El caso de una fila**, donde candle sigue adelante — 1,8x sobre Q4_K y
   2,7x sobre Q3_K. Ya no es un misterio: candle a una fila mueve los 9,2 MB en
   0,32 ms, que son 28 GB/s y lo pone contra el muro de memoria de esta
   máquina, mientras lo nuestro hace 2,4 ms de desempaquetado más 1,0 de kernel
   en **dos pasadas separadas**, con la ida y vuelta a L1 entre una y otra.
   El arreglo es fusionarlas: desempaquetar un bloque de 256 y multiplicarlo
   sin que salga de los registros, que es lo que hace `vec_dot_q3k_q8k` de
   candle ([DECISIONES](DECISIONES.md) §11). Pide AVX2 a mano por formato, así
   que es trabajo de días y no de horas — y es la primera tarea donde el
   ensamblador y Zig tienen algo real que hacer.
2. **Elegir el camino según el formato.** En Q3_K gana el de enteros y en Q4_K
   gana el de `f32` desde 2 filas, porque el arreglo de los límites le
   aprovechó más. Hoy hay que elegir a mano; debería elegirlo el motor, y para
   eso hace falta medirlo en los cinco modelos y no en uno.
3. El microkernel con bloqueo de registros, que es donde el ensamblador tendría
   su primera oportunidad real — ver [DECISIONES](DECISIONES.md) §5, con el
   paso previo que le faltaba al criterio en §15.

**La regla del reemplazo**: nada sale de candle hasta que la pieza propia dé lo
mismo sobre los modelos reales. Candle se queda como dependencia de **test**
aunque salga del binario — tener motor propio no es razón para tirar la forma
más barata de saber si está bien.

Y es donde el ensamblador deja de competir contra un bucle que LLVM ya
vectoriza: en la matmul cuantizada propia el bucle caliente es **nuestro**, y
ahí sí se puede probar AVX-512, que candle no usa.

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
