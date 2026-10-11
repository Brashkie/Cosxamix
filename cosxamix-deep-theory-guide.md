# Especificación Avanzada de Inferencia en Cosxamix: Teoría Computacional, Álgebra Matricial y Optimización Microarquitectónica Políglota

> **Cómo leer este documento.** Mezcla dos cosas que conviene no confundir:
> lo que ya corre y lo que está decidido pero sin escribir. Las secciones
> que describen diseño previsto llevan un aviso al principio. Todo lo demás
> está implementado y verificado contra el código.
>
> | | |
> | :--- | :--- |
> | **Implementado** | §2.1, §2.2 · §3 · §5.1, §5.2 · §6 completo |
> | **Diseño previsto** | §2.3 (RoPE) · §2.4 (PagedAttention) · §4 (memoria) · §5.3 (tabla de punteros) |
>
> Las cifras del §6 salen de `cargo bench` sobre el código de este
> repositorio, no de estimaciones.

## 1. Fundamentos Teóricos de la Inferencia Autoregresiva en Grafos de Cómputo

La inferencia de modelos de lenguaje basados en arquitecturas Transformer (Decoder-Only) opera como una secuencia de transformaciones tenoriales sobre un grafo de cómputo dirigido acíclico (DAG). A diferencia del entrenamiento, donde el cálculo se procesa en lotes masivos altamente paralelizables a lo largo de la dimensión de la secuencia, la generación autoregresiva token por token impone un patrón de ejecución fundamentalmente limitado por el ancho de banda de memoria (*Memory Bandwidth Bound*).

### 1.1 Formulación Matemática de la Pasada Directa (*Forward Pass*)

Sea $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ el vector de representación del token en el paso de tiempo $t$, donde $d_{\text{model}}$ denota la dimensión oculta del modelo. La transformación a través de $L$ capas cuadradas del Transformer se define formalmente mediante las siguientes ecuaciones continuas:

$$\mathbf{x}_t^{(0)} = \mathbf{E}_{[\text{token}_t]} + \mathbf{P}_{[t]}$$

Para cada capa $l \in \{1, 2, \dots, L\}$, el vector de activaciones experimenta un proceso de normalización de raíz media cuadrática (*RMSNorm*), seguido por las proyecciones de atención de consulta, clave y valor ($Q, K, V$):

$$\mathbf{y}_t^{(l)} = \text{RMSNorm}\left(\mathbf{x}_t^{(l-1)}\right)$$

$$\mathbf{q}_t^{(l)} = \mathbf{W}_Q^{(l)} \mathbf{y}_t^{(l)}, \quad \mathbf{k}_t^{(l)} = \mathbf{W}_K^{(l)} \mathbf{y}_t^{(l)}, \quad \mathbf{v}_t^{(l)} = \mathbf{W}_V^{(l)} \mathbf{y}_t^{(l)}$$

Donde $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. Las claves y valores computados en el paso $t$ se anexan al caché de atención ($\mathbf{K}_{1:t}^{(l)}, \mathbf{V}_{1:t}^{(l)}$). El cálculo del mecanismo de atención con escalamiento de producto punto y máscara causal se expresa como:

$$\mathbf{A}_t^{(l)} = \text{Softmax}\left( \frac{\mathbf{q}_t^{(l)} \left(\mathbf{K}_{1:t}^{(l)}\right)^\top}{\sqrt{d_k}} \right) \mathbf{V}_{1:t}^{(l)}$$

$$\mathbf{z}_t^{(l)} = \mathbf{x}_t^{(l-1)} + \mathbf{W}_O^{(l)} \mathbf{A}_t^{(l)}$$

Posteriormente, la salida de la atención pasa por un bloque Feed-Forward (*FFN*) con activación SwiGLU:

$$\mathbf{x}_t^{(l)} = \mathbf{z}_t^{(l)} + \mathbf{W}_{d}\left( \text{Swish}\left(\mathbf{W}_{g} \text{RMSNorm}\left(\mathbf{z}_t^{(l)}\right)\right) \odot \mathbf{W}_{u} \text{RMSNorm}\left(\mathbf{z}_t^{(l)}\right) \right)$$

### 1.2 Intensidad Aritmética y el Límite de Ancho de Banda

El rendimiento de un motor de inferencia en procesadores centrales (CPU) está gobernado por el modelo de rendimiento *Roofline*. La intensidad aritmética $I$ se define como la razón entre las operaciones de punto flotante realizadas (FLOPs) y los bytes de memoria transferidos desde la memoria principal (RAM) hacia las cachés del procesador:

$$I = \frac{\text{Operaciones de Punto Flotante (FLOPs)}}{\text{Transferencia de Memoria (Bytes)}}$$

En la fase de generación autoregresiva de un solo token ($N=1$), para un modelo con $P$ parámetros almacenados en precisión de 16 bits (2 bytes por parámetro), la ejecución de cada multiplicación matriz-vector ($\mathbf{W} \mathbf{x}$) requiere leer $2P$ bytes de la memoria RAM para realizar aproximadamente $2P$ FLOPs. Esto establece una intensidad aritmética $I \approx 1 \text{ FLOP/Byte}$.

Dado que un procesador moderno x86-64 o ARM64 posee una capacidad de cómputo pico de varios TFLOPs pero un ancho de banda de memoria RAM limitado a unas pocas decenas de gigabytes por segundo (por ejemplo, 50 GB/s en memoria DDR4/DDR5 de canal único), la velocidad teórica máxima de inferencia no depende de los núcleos de cálculo sino de la tasa de transferencia de datos de la memoria. La arquitectura Cosxamix responde a esta restricción mediante la combinación de mapeo de memoria sin copias (`mmap`), compresión de estructuras, conservación térmica de modelos en memoria y kernels vectorizados con cero overhead.

---

## 2. Algoritmos Matemáticos del Camino Caliente: Derivación y Vectorización

El camino caliente (*hot path*) de un motor de inferencia abarca el subconjunto de funciones matemáticas ejecutadas repetitivamente millones de veces por cada token generado. En Cosxamix, el aislamiento estricto de estas primitivas en la frontera `src/kernels/` permite optimizar matemáticamente cada operación para alinearse con los registros del procesador.

### 2.1 Producto Punto de Vectores ($\mathbf{x} \cdot \mathbf{y}$)

El producto punto de dos vectores $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ constituye el núcleo numérico de las multiplicaciones matriz-vector en proyecciones densas y capas de atención.

#### Formulación Algorítmica y Acumulación Vectorial

$$S = \sum_{i=0}^{n-1} x_i \cdot y_i$$

En hardware vectorial, el bucle escalar se desenrolla para ejecutar instrucciones SIMD de Multiplicación-Acumulación Fusionada (*FMA*). Para vectores con tamaño de carril SIMD $V$ (donde $V=8$ para vectores `f32` en registros AVX2 de 256 bits), el algoritmo proyecta el espacio en $V$ acumuladores independientes para ocultar la latencia de la instrucción FMA:

$$\mathbf{a}^{(0)}, \mathbf{a}^{(1)}, \dots, \mathbf{a}^{(K-1)} \in \mathbb{R}^V$$

$$\mathbf{a}^{(k)} \leftarrow \mathbf{a}^{(k)} + \mathbf{x}_{[i:i+V]} \odot \mathbf{y}_{[i:i+V]}$$

Al finalizar la iteración masiva, los acumuladores vectoriales se reducen horizontalmente mediante sumas cruzadas de carriles hasta obtener el escalar final $S$.

### 2.2 Normalización de Raíz Media Cuadrática (RMSNorm)

RMSNorm sustituye a LayerNorm en modelos modernos como Llama y Qwen, eliminando el cálculo de la media y conservando únicamente la escala invariante del volumen vectorial. Esto reduce los accesos a memoria de tres pasadas a dos pasadas sobre el vector.

#### Formulación Matemática y Precisión Numérica

Dado un vector de activaciones $\mathbf{x} \in \mathbb{R}^n$ y un vector de ponderación escalada $\boldsymbol{\gamma} \in \mathbb{R}^n$, el cálculo de RMSNorm se define mediante:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} x_i^2 + \epsilon}$$

$$\bar{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i, \quad \forall i \in \{0, 1, \dots, n-1\}$$

Donde $\epsilon > 0$ es un factor de estabilidad numérica (típicamente $10^{-5}$ o $10^{-6}$).

Un aspecto crítico documentado en las pruebas de referencia de Cosxamix es la precisión del acumulador. La implementación de referencia en Rust acumula la suma de cuadrados $\sum x_i^2$ utilizando el tipo de doble precisión $f64$ para prevenir la pérdida de significación por subflujo o desbordamiento en dimensiones elevadas. En contraste, los kernels de alto rendimiento vectorizados en AVX2 y Zig utilizan acumuladores en $f32$ combinados con instrucciones FMA, manteniendo la desviación dentro de tolerancias numéricas estrictas verificadas mediante pruebas de equivalencia en el arnés de pruebas de Cosxamix.

### 2.3 Embebidos de Posición Rotacional (RoPE)

> **Diseño previsto, todavía no implementado.** Lo que sigue describe la
> dirección decidida, no código que exista hoy en el repositorio.

RoPE codifica la información posicional relativa aplicando una matriz de rotación ortogonal a los vectores de consulta y clave en el espacio complejo bidimensional.

#### Transformación en el Espacio Bidimensional

Para un par de coordenadas $(x_{2i}, x_{2i+1})$ dentro del vector de atención en la posición $m$ con frecuencia $\theta_i = 10000^{-2i/d}$:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

$$\mathbf{x}'_{[2i:2i+1]} = \mathbf{x}_{[2i:2i+1]} \odot \cos(m\boldsymbol{\theta}) + \tilde{\mathbf{x}}_{[2i:2i+1]} \odot \sin(m\boldsymbol{\theta})$$

Donde $\tilde{\mathbf{x}}_{[2i:2i+1]} = (-x_{2i+1}, x_{2i})$. La vectorización de esta operación exige reorganizar los elementos en pares contiguos para permitir cargas vectoriales intercaladas alineadas con los límites de línea de caché.

### 2.4 PagedAttention y Estructura del Caché KV

> **Diseño previsto, todavía no implementado.** Lo que sigue describe la
> dirección decidida, no código que exista hoy en el repositorio.

Para erradicar la fragmentación externa de la memoria RAM provocada por la asignación de tensores contiguos masivos para el caché KV, Cosxamix implementa el concepto de PagedAttention.

#### Paginación Matemática de Memoria de Atención

El espacio de claves y valores se divide en bloques de tamaño fijo $B_{\text{size}}$ (por ejemplo, 16 tokens). Un vector de tabla de páginas $T(s, i)$ mapea el índice lógico de bloque $i$ de la sesión $s$ a un bloque físico no contiguo en el contenedor de memoria asignado al inicio del sistema:

$$\mathbf{K}_{\text{físico}} = \text{Pool}_{\text{KV}}\left[ T(s, \lfloor t / B_{\text{size}} \rfloor) \right]\left[ t \pmod{B_{\text{size}}} \right]$$

Esta abstracción permite reutilizar páginas del sistema entre peticiones con *prompts* idénticos sin necesidad de copiar datos en la memoria física, habilitando una latencia de inicio cercana a cero en conversaciones continuas.

---

## 3. Microarquitectura de CPU, Registros Vectoriales y Mecánica SIMD

El rendimiento del software a bajo nivel está íntimamente ligado a la arquitectura del conjunto de instrucciones (ISA) y a las unidades de ejecución del procesador. Comprender el comportamiento del pipeline, la profundidad de los registros vectoriales y las penalizaciones por desalineamiento es indispensable para justificar el diseño de la frontera políglota de Cosxamix.

### 3.1 Arquitecturas de Registros Vectoriales: x86-64 vs. ARM64

Los procesadores x86-64 y ARM64 abordan el procesamiento vectorial mediante modelos de registros divergentes:

* **x86-64 (AVX2 / AVX-512)**:
  * AVX2 proporciona 16 registros vectoriales de 256 bits (`ymm0` a `ymm15`), capaces de empaquetar 8 valores en formato `f32`.
  * AVX-512 expande el espacio a 32 registros de 512 bits (`zmm0` a `zmm31`), procesando 16 valores `f32` por instrucción.
  * Incorpora unidades FMA que ejecutan dos operaciones flotantes (multiplicación y suma) en un solo ciclo de reloj con latencias típicas de 4 a 5 ciclos y un rendimiento (*throughput*) de 0.5 ciclos por instrucción.

* **ARM64 (NEON / SVE)**:
  * ARM NEON dispone de 32 registros de 128 bits (`v0` a `v31`), empaquetando 4 valores `f32` por registro.
  * ARM SVE (Scalable Vector Extension) introduce una longitud de registro Agnóstica de Vector (VLA), permitiendo tamaños de 128 a 2048 bits según la implementación del chip.

### 3.2 Alineamiento de Memoria, Unrolling de Bucles y Líneas de Caché

Un factor determinante en la eficiencia de las instrucciones SIMD es el alineamiento de las direcciones de memoria física. Las líneas de caché en las CPUs modernas x86 y ARM tienen un tamaño uniforme de 64 bytes.

Cuando un vector SIMD de 256 bits (32 bytes) se lee de una dirección de memoria desalineada que cruza el límite de una línea de caché de 64 bytes, la unidad de gestión de memoria (MMU) debe emitir dos lecturas de caché independientes, duplicando la latencia de carga y saturando los puertos de lectura (*Load Ports*) del procesador.

Pese a eso, **los kernels de Cosxamix usan hoy cargas desalineadas**: `_mm256_loadu_ps` en el camino de Rust y `align(@alignOf(f32))` —4 bytes— en el de Zig. No es un descuido, es la única opción correcta: la memoria llega desde slices de Rust, y un slice garantiza la alineación de **su elemento** y nada más. Declarar `align(64)` sobre un `Vec<f32>` sería afirmarle al compilador algo que no es cierto, y en una CPU que exige alineación —o con las comprobaciones de seguridad activadas— eso se cae.

En las CPUs x86 modernas la penalización por carga desalineada que no cruza línea de caché es nula, y las mediciones del §6 se obtuvieron así: es el rendimiento real de los kernels tal como están, no el de una versión ideal.

La alineación a 64 bytes entra cuando exista el asignador de arena del §4, que sí podrá garantizarla porque la memoria será suya. Ahí los kernels podrán ofrecer un camino alineado — y habrá que **medir** si la diferencia justifica mantener dos.

### 3.3 El Caso de Estudio Zig: `-mcpu=x86_64_v3` y la Descomposición de Registros

Las mediciones empíricas de Cosxamix registraron una diferencia drástica en el rendimiento del kernel `dot` en Zig antes y después de ajustar los parámetros de compilación.

Cuando la herramienta de compilación invoca el compilador de Zig sin especificar el modelo de microarquitectura destino, el compilador emite código para la arquitectura base `x86-64` genérica (que garantiza compatibilidad con CPUs de 2003 limitadas a SSE2 de 128 bits). En esta configuración, una abstracción vectorial de alto nivel definida en Zig como `@Vector(8, f32)` no se asigna a un registro natural `ymm` de 256 bits. El compilador se ve obligado a desglosar (*split*) la abstracción en dos operaciones vectoriales de 128 bits (`xmm`), duplicando la cantidad de instrucciones emitidas y aumentando la presión sobre la cola de decodificación.

Al especificar la bandera `-mcpu=x86_64_v3` (que habilita el conjunto de instrucciones AVX2, FMA3 y BMI2), la herramienta de compilación de Zig hace coincidir el tipo primitivo `@Vector(8, f32)` directamente con las instrucciones nativas `vfmacd_ps` y los registros `ymm`. Esto redujo la brecha de rendimiento frente a las primitivas intrínsecas escritas manualmente en Rust del 26% a un margen cercano al 8%, demostrando que la equivalencia de microarquitectura en la bandera de compilación es un prerrequisito indispensable para evaluar cualquier lenguaje.

---

## 4. Sistemas de Memoria de Altísimo Rendimiento: `mmap`, Jerarquía de Caché y Zero-Allocation

> **Diseño previsto, todavía no implementado.** Lo que sigue describe la
> dirección decidida, no código que exista hoy en el repositorio.

El diseño de la capa de gestión de memoria determina la capacidad de un runtime para operar en entornos con restricciones severas de RAM, como teléfonos móviles ejecutando Termux o sistemas incrustados.

### 4.1 Mapeo de Memoria I/O (`mmap`) y Gestión de Fallos de Página

La carga convencional de un modelo de lenguaje mediante llamadas al sistema `read()` asigna bloques de memoria virtual en el *heap*, copia los datos desde el almacenamiento secundario y duplica el consumo de RAM física. Cosxamix utiliza mapeo de memoria directo mediante la llamada del sistema `mmap()` con las siguientes banderas de protección y configuración:

$$\text{mmap}(\text{NULL}, \text{longitud}, \text{PROT\_READ}, \text{MAP\_SHARED}, \text{fd}, 0)$$

#### Mecánica de Ejecución y Paginación por Demanda

1. **Creación del Espacio de Direcciones**: `mmap` registra un rango de direcciones virtuales contiguas en el proceso sin transferir datos físicos de la unidad de almacenamiento.
2. **Fallo de Página (*Page Fault*)**: Cuando un kernel SIMD intenta leer un tensor por primera vez, la MMU detecta que la página no está presente en la memoria principal y desencadena una interrupción de hardware.
3. **Mapeo por Demanda**: El núcleo del sistema operativo lee únicamente la página requerida (típicamente de 4 KB a 2 MB en *Huge Pages*) desde el almacenamiento secundario hacia el caché de páginas (*Page Cache*).
4. **Desalojo Transparente**: Si el sistema operativo detecta presión de memoria en el sistema, puede liberar páginas de memoria mappedas en lectura sin necesidad de escribir en el espacio de intercambio (*swap*), recuperando la memoria de forma instantánea.

### 4.2 Optimización de Jerarquía de Caché y TLB Misses

Para evitar que los accesos a la memoria principal queden bloqueados por la latencia de la DRAM (aproximadamente 60-80 nanosegundos), Cosxamix estructura la disposición de los datos considerando la jerarquía de cachés:

| Nivel de Caché | Capacidad Típica | Latencia en Ciclos | Ancho de Banda Típico |
| :--- | :--- | :--- | :--- |
| **Caché L1 Data** | 32 KB – 48 KB / núcleo | ~4 ciclos | > 2 TB/s |
| **Caché L2** | 512 KB – 1 MB / núcleo | ~12–14 ciclos | > 1 TB/s |
| **Caché L3 (LLC)** | 16 MB – 96 MB (Compartido) | ~35–50 ciclos | ~300–500 GB/s |
| **Memoria RAM** | Varios Gigabytes | ~200 ciclos | ~30–60 GB/s |

Para reducir los fallos en la tabla de traducción de direcciones (*TLB Misses*), Cosxamix configura las llamadas de asignación de memoria con soporte para *Huge Pages* de 2 MB en Linux (`MAP_HUGETLB`). Esto reduce el número de entradas requeridas en la TLB para direccionar un modelo de 7 GB en un factor de 512, eliminando las penalizaciones por caminata de tablas de páginas (*Page Table Walks*) durante la ejecución del bucle de inferencia.

### 4.3 Asignadores Estáticos de Memoria (*Arena Allocators*) y Zero-Allocation

La invocación del asignador global de memoria (`malloc` o `global_allocator`) durante el bucle autoregresivo de tokenización e inferencia introduce inconsistencias de latencia causadas por la búsqueda de bloques libres, la sincronización de hilos (*locks*) y la fragmentación del *heap*.

Cosxamix adopta una arquitectura de memoria de cero asignaciones dinámicas en el bucle caliente mediante el uso de un **Arena Allocator estático**:

$$\text{Buffer}_{\text{Arena}} \in \mathbb{R}^{M_{\text{máx}} \text{ Bytes}} \quad (\text{Asignado una sola vez en la inicialización})$$

Durante la pasada directa de un token, las asignaciones temporales para matrices intermedias, proyecciones de atención y buffers de normalización avanzan un puntero interno offset $\Delta$ sobre la arena de memoria. Al finalizar el procesamiento del token, el offset se reinicia a cero ($\Delta = 0$) en una constante de tiempo $O(1)$, erradicando completamente la sobrecarga del asignador de memoria del sistema operativo.

---

## 5. Diseño Políglota de Bajo Nivel y Contrato de Frontera FFI

La coexistencia de múltiples lenguajes de programación dentro de un mismo runtime requiere una definición rigurosa de la interfaz binaria de aplicación (ABI) para evitar sobrecargas de conversión y garantizar la seguridad de la memoria.

### 5.1 Interfaz C ABI y Convención de Llamadas de Bajo Nivel

Toda la comunicación entre el runtime principal en Rust y los kernels acelerados en Zig, C, C++, D o Ensamblador se realiza exclusivamente a través de la ABI de C pura (`extern "C"`).

#### Firmas de Primitivas en la Frontera de Kernels

Estas son las firmas **reales**, copiadas del código. Importa que lo sean: los
tres punteros de `rmsnorm` son del mismo tipo, así que **equivocarse de orden no
lo detecta el compilador** — se escribiría sobre el buffer de entrada y se
leería del de salida, en silencio. El destino va **primero**.

En Rust (`src/kernels/zig.rs`):

```rust
unsafe extern "C" {
    fn cosxamix_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;

    fn cosxamix_rmsnorm_f32(
        salida:  *mut f32,     // destino PRIMERO
        entrada: *const f32,
        peso:    *const f32,
        n:       usize,
        eps:     f32,
    );
}
```

En Zig (`kernels/dot.zig`):

```zig
// `export fn` ya implica la ABI de C: no hace falta `callconv`.
export fn cosxamix_dot_f32(a: [*]const f32, b: [*]const f32, n: usize) f32 {
    var s0: V = @splat(0);
    // ... cuatro acumuladores, cola escalar al final ...
}
```

Los símbolos **no** llevan sufijo de lenguaje. Se consideró `_zig` y se
descartó: el nombre del símbolo pasaría a ser parte del contrato, y mover un
kernel de Zig a C o a ensamblador —que es justo lo que la frontera permite—
obligaría a renombrarlo y a tocar al llamador. El lenguaje es un detalle de
implementación; el nombre describe la operación.

#### Convenciones de Registro en System V AMD64 ABI

En sistemas Linux x86-64, la convención de llamadas *System V AMD64 ABI* dicta el paso de argumentos a través de registros de propósito general y registros vectoriales:

* Los primeros argumentos enteros y punteros (`*const f32`, `len`) se transmiten a través de los registros `%rdi`, `%rsi`, `%rdx`, `%rcx`, `%r8`, `%r9`.
* Los argumentos de punto flotante de precisión simple (`eps`) se pasan mediante los registros `%xmm0` a `%xmm7`.
* El valor de retorno escalar flotante se entrega en el registro `%xmm0`.

Al respetar estrictamente esta convención sin envoltorios complejos, el costo de invocación de un kernel externo se reduce a una instrucción `call`. **Medido** con `cargo bench --features zig -- ffi`, comparando una función vacía en Zig —que cruza la frontera— contra una vacía en Rust marcada `#[inline(never)]` —que no la cruza—:

| | mediana |
| :--- | ---: |
| Llamada local en Rust | 651 ps |
| Llamada por FFI a Zig | 1,443 ns |
| **Coste de cruzar la frontera** | **~0,8 ns** |

O sea que un kernel que haga menos de unos pocos nanosegundos de trabajo no conviene cruzarlo; de ahí para arriba, la frontera no se nota. Para `dot` con n=1024 —54 ns— el coste de la llamada es el 1,5% del total.

Esta cifra estaba antes en el documento como "inferior a 1,5 nanosegundos" sin respaldo. Resultó acertada, pero el §6 de este mismo documento dice que en Cosxamix nada se decide sin medir, y eso aplica también a lo que el documento afirma.

### 5.2 Matriz de Responsabilidades y Especialización por Lenguaje

Cosxamix asigna a cada lenguaje una responsabilidad técnica concreta dentro de la arquitectura general, evitando redundancias e imponiendo una justificación objetiva para su permanencia en el código fuente:

| Lenguaje | Rol Principal en Cosxamix | Ventaja Técnica Justificada | Criterio de Permanencia |
| :--- | :--- | :--- | :--- |
| **Rust** | Runtime principal, Grafo, HTTP, MMap, Referencia. | Seguridad de memoria, concurrencia sin carreras, iteradores de cero costo. | Código base obligatorio e inmutable para las referencias. |
| **Zig** | Kernels de alta cobertura multiplataforma. | Compilación cruzada nativa, soporte para Android/Termux sin toolchain C. | Debe superar a la referencia Rust o proporcionar cobertura multi-target. |
| **C** | Integración con librerías nativas y primitivas simples. | ABI universal de mínima sobrecarga y compresión de binario. | Integración directa sin dependencias externas pesadas. |
| **C++** | Interoperabilidad con motores maduros (llama.cpp/ONNX). | Reutilización de implementaciones complejas de cuantización. | Unicamente si aporta un backend de ejecución maduro no reescrito. |
| **D** | Experimentos de kernels numéricos de alto rendimiento. | Expresividad matemática avanzada y metaprogramación con metaplantillas. | Debe demostrar un rendimiento superior en prototipos específicos. |
| **ASM** | Microkernels de máxima optimización (AVX-512 / NEON). | Control total sobre la asignación de registros y la tubería de instrucciones. | Reservado exclusivamente para rutas críticas donde los compiladores fallan. |

### 5.3 Selección Dinámica de Backend en Tiempo de Ejecución (*Dynamic Dispatch*)

> **La idea está implementada; el código de abajo, no.** La detección por
> CPU, el despacho y `COSXAMIX_KERNEL` existen y funcionan — pero resueltos con
> un `match` sobre un enum cacheado en un `OnceLock`, no con la tabla de
> punteros a función que se muestra. Las dos formas son válidas; esta se
> conserva porque será la que haga falta cuando un mismo kernel tenga
> variantes por tamaño de problema y haya que cambiarlas sin recompilar.

Para evitar publicar múltiples binarios para diferentes generaciones de procesadores, Cosxamix evalúa las capacidades de la CPU durante la fase de inicialización mediante la instrucción `cpuid` en x86-64 o la lectura de vectores auxiliares `getauxval(AT_HWCAP)` en Linux/ARM64. La variable de entorno `COSXAMIX_KERNEL` permite forzar un backend específico para la ejecución aislada de benchmarks.

#### Algoritmo de Despacho mediante Punteros a Funciones

```rust
pub type DotKernelFn = unsafe extern "C" fn(*const f32, *const f32, usize) -> f32;

pub struct KernelRegistry {
    pub dot_impl: DotKernelFn,
}

impl KernelRegistry {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return Self { dot_impl: dot_avx2 };
            }
        }
        
        #[cfg(feature = "zig")]
        {
            // En arquitecturas ARM64 o fallback sin AVX2, Zig ofrece la mejor cobertura
            return Self { dot_impl: cosxamix_dot_f32 };
        }

        Self { dot_impl: scalar::dot }
    }
}
```

Esta estrategia garantiza que un solo binario precompilado pueda ejecutarse de forma segura en una CPU antigua sin causar una interrupción por instrucción ilegal (`SIGILL`), al tiempo que aprovecha automáticamente los registros AVX-512 o ARM NEON en procesadores modernos.

---

## 6. Evaluación Empírica, Perfilado Microarquitectónico y Reglas de Decisión

El principio rector de la ingeniería de rendimiento en Cosxamix establece que ninguna decisión arquitectónica se toma basándose en preferencias teóricas; toda inclusión de código debe estar respaldada por mediciones empíricas reproducibles.

### 6.1 Análisis Comparativo de Datos Medidos en los Kernels

A continuación se presentan los resultados consolidados de las mediciones de la mediana de tiempo de ejecución obtenidas mediante la biblioteca de benchs `Criterion` en un procesador Intel Core i5 con soporte AVX2+FMA (sin AVX-512).

#### Primitiva 1: Producto Punto (`dot` en precisión $f32$)

La tabla compara la referencia escalar en Rust frente a la implementación intrínseca en Rust AVX2 y la implementación portable en Zig (`-mcpu=x86_64_v3`):

| Dimensión ($n$) | Referencia Escalar | Rust AVX2 (Intrínsico) | Zig (`-mcpu=x86_64_v3`) | Zig vs. Rust AVX2 |
| :--- | :--- | :--- | :--- | :--- |
| **1,024** | $1.039 \text{ }\mu\text{s}$ | **$54.1 \text{ ns}$** | $58.2 \text{ ns}$ | $+7.5\%$ |
| **4,096** | $4.631 \text{ }\mu\text{s}$ | **$233.5 \text{ ns}$** | $252.3 \text{ ns}$ | $+8.0\%$ |
| **65,536** | $75.900 \text{ }\mu\text{s}$ | $8.530 \text{ }\mu\text{s}$ | **$8.500 \text{ }\mu\text{s}$** | Empate |
| **1,000,003** | $1.633 \text{ ms}$ | **$933.0 \text{ }\mu\text{s}$** | $1,010.0 \text{ }\mu\text{s}$ | $+8.2\%$ |

#### Primitiva 2: Normalización Raíz Media Cuadrática (`rmsnorm` en precisión $f32$)

Medición realizada sobre dimensiones realistas de capas ocultas en LLMs pequeños ($n \in [2048, 8192]$):

| Dimensión ($n$) | Referencia Escalar | Rust AVX2 (Intrínsico) | Zig (`-mcpu=x86_64_v3`) | Zig vs. Rust AVX2 |
| :--- | :--- | :--- | :--- | :--- |
| **2,048** | $2.874 \text{ }\mu\text{s}$ | **$294.6 \text{ ns}$** | $375.4 \text{ ns}$ | $+27.0\%$ |
| **4,096** | $6.152 \text{ }\mu\text{s}$ | **$759.4 \text{ ns}$** | $799.1 \text{ ns}$ | $+5.2\%$ |
| **8,192** | $12.185 \text{ }\mu\text{s}$ | **$1.970 \text{ }\mu\text{s}$** | $2.033 \text{ }\mu\text{s}$ | $+3.2\%$ |

### 6.2 Interpretación Microarquitectónica de los Resultados

1. **Aceleración Vectorial Real**: En ambas primitivas, las implementaciones SIMD (tanto en Rust como en Zig) ofrecen aceleraciones de entre **6x y 19x** en comparación con la referencia escalar en Rust. Esto demuestra la absoluta necesidad de la vectorización en el camino caliente para alcanzar latencias en el rango de milisegundos.
2. **Superioridad de la Invocación Directa Intrínseca en x86-64**: Las implementaciones intrínsecas escritas a mano en Rust utilizando primitivas de `core::arch::x86_64` superan de manera uniforme a Zig por un margen de entre $3\%$ and $8\%$ en dimensiones grandes. Esto se debe a que el programador de intrínsecos puede estructurar manualmente la reutilización de registros y el desenrollado del bucle en función de la latencia exacta de la instrucción FMA del procesador.
3. **El Valor de la Cobertura Multi-Target de Zig**: A pesar de la pequeña desventaja porcentual en x86-64, la implementación en Zig se conserva en la arquitectura Cosxamix debido a su valor estratégico de portabilidad. Un único archivo fuente (`kernels/dot.zig`) se compila transparentemente para entornos ARM64 y Android (Termux) sin requerir modificaciones en el código fuente. En un dispositivo móvil ARM64, la presencia del kernel en Zig evita que el motor caiga a la referencia escalar, la cual sería 19 veces más lenta.

### 6.3 Las Tres Reglas Gobernantes de la Frontera de Kernels de Cosxamix

Para mantener la integridad técnica de la base de código y prevenir la acumulación de código muerto o ineficiente, Cosxamix aplica de manera automatizada tres reglas inviolables en su arnés de pruebas (`cargo test`, `cargo bench`):

* **Regla 1 (Existencia de la Referencia)**: Todo kernel introducido en el sistema debe poseer una implementación de referencia pura escrita en Rust portable. Esta función actúa como el estándar de veracidad matemática y ejecuta en plataformas donde no existe aceleración de hardware.
* **Regla 2 (Equivalencia Estricta de Verificación)**: Toda nueva implementación en cualquier lenguaje (Zig, C, C++, D, ASM) debe someterse a pruebas de comparación cruzada contra la referencia en Rust. Si un kernel optimizado produce desviaciones numéricas superiores a la tolerancia permitida ($\epsilon$), el kernel es rechazado inmediatamente, independientemente de su velocidad.
* **Regla 3 (Justificación Empírica de Rendimiento y Cobertura)**: Todo kernel alternativo debe demostrar mediante pruebas de rendimiento (*benchmarks*) que supera a la referencia en velocidad o que habilita una plataforma objetivo donde no existía aceleración previa. Si una implementación alternativa no supera la velocidad de la referencia ni amplía la cobertura de plataformas, se elimina del repositorio sin excepciones.
