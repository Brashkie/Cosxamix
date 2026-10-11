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

### 1. Explicar el escalado de hilos

El bloqueante real. De 1 a 6 núcleos se obtienen **2,6x**, no 6x:

| hilos | decode | núcleos efectivos | CPU consumida |
|---:|---:|---:|---:|
| 1 | 9,86 tok/s | 0,98 | 32,9 s |
| 6 | 25,58 | 4,06 | 53,2 s |
| 12 | 26,08 | 6,23 | 81,8 s |

Se encontró un factor y **no alcanza**: el `with_min_len(128)` del reparto de
candle impide partir las proyecciones K y V, de 256 columnas, en más de 2 trozos
— pero K+V son ~1,4% de los parámetros, o sea ~3% del tiempo.

Lo que falta es un perfilador de verdad (`perf`, VTune o `cargo flamegraph`) para
saber si la pérdida está en sincronización por capa, en fallos de caché L2/L3, o
en que las barreras de rayon se pagan 36 veces por token. **No tengo la
explicación todavía**, y adivinar acá sale caro.

Es lo primero porque si la brecha se cierra, el motor pasa a estar limitado por
memoria — y eso reabre de golpe los puntos 4 y 5.

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

- Carga de más de un modelo a la vez, con expulsión por LRU igual que las
  sesiones. Hoy es uno.
- Respuestas en *streaming* (`stream: true`), que el dialecto de Ollama define y
  Cosxamix todavía no implementa.
- Persistir el caché de sesión en disco, para que reiniciar no obligue a
  reprocesar el *system prompt* de cada conversación.

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

Es la mitad del objetivo del proyecto y hoy **no hay nada**: Cosxamix solo infiere.
El orden razonable es LoRA / QLoRA sobre el modelo cuantizado antes que
cualquier entrenamiento desde cero, y candle tiene autograd, así que la pieza
base existe.

No se empieza antes que el punto 1 porque entrenar con el reparto entre hilos
rindiendo al 43% sería arrastrar el mismo problema a un trabajo diez veces más
caro.

---

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
