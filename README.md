# Cosxamix

***English** · [Español](README.es.md)*

An inference runtime of my own. It serves language models over HTTP, speaking
the same dialect as Ollama so it can replace it without touching whatever calls
it.

> **Status: real inference running, and WinsiBot already answers through it.**
> It serves a GGUF model over HTTP in Ollama's dialect, pre-warmed at startup,
> with a per-conversation attention cache that reuses the common prefix,
> *streaming* responses, several models at once with use-based eviction, and
> LoRA fine-tuning.

## The documents

| | |
|---|---|
| this file | how it works, how to use it, and the engine's measurements |
| [README.es.md](README.es.md) | the same in Spanish |
| [DECISIONES.md](DECISIONES.md) | why each thing is the way it is, with the measurement that backs it — including the negative results and the figures that had to be corrected |
| [CHANGELOG.md](CHANGELOG.md) | what changed in each version |
| [ROADMAP.md](ROADMAP.md) | what's left, in what order, and what is settled |
| [cosxamix-deep-theory-guide.md](cosxamix-deep-theory-guide.md) | the underlying theory; each section states whether it is implemented or planned |
| [THIRD-PARTY.md](THIRD-PARTY.md) | the licenses of the models it uses and of the dependencies; since 0.11.0 no code is copied from another project |

The four documents above are **written in Spanish**, and deliberately so: they
carry every measurement with its context, and keeping two copies of a number is
the reliable way to end up with two different numbers. This README is the
translated entry point.

---

## Why it exists, and what it is NOT

There's a misunderstanding worth clearing up before writing a single line:
**Ollama does not implement inference.** Ollama is a service layer — model
registry, load and unload, an HTTP API — sitting on top of `llama.cpp`, which
does the numerical work.

That defines what makes sense to build here and what would be throwing years
away:

| layer | who does it | ours? |
|---|---|---|
| HTTP API, model management, cache, scheduler | **Cosxamix** | **yes** |
| Graph execution, attention, quantization, images, training | **Cosxamix** — no `candle` since 0.11.0 | **yes** |
| Individual hot-path kernels | Cosxamix, **only where profiling justifies it** | maybe |

**That second row said "no" until 0.3.0, and what changed it was
measurements, not ambition.** Leaning on candle was cheap and turned out to be
a ceiling in five places at once, all verified in this repository:

| | |
|---|---|
| `QMatMul` doesn't implement `bwd` | gradients can't flow through a quantized matmul, so training the 3B needs 12 GB in f32 against 7.3 free |
| its `matmul` re-reads the whole matrix per batch row | 8 rows cost 8.55x one, and that's what killed speculative decoding |
| no AVX-512, no SVE | `avx.rs` is 256-bit and `neon.rs` is 128-bit |
| no IQ quantization formats | its enumeration stops at the classic k-quants |
| its backprop computes `dL/dW` for frozen layers | 17% of the training step, wasted |

None of the five is fixable from the outside. And the surface actually in use
is narrow — `Tensor`, `QMatMul`, the GGUF reader and a dozen loose functions —
so replacing it is a bounded job, not a rewrite of `llama.cpp`.

It was done **piece by piece, with candle as the oracle**: each part of ours
was compared against theirs on the real models before replacing anything. In
0.11.0 it left entirely, tests included: what candle computed for them was saved
(`tests/fixtures/ops_candle.json`), and training is checked against finite
differences, which need nobody.

The service layer is still where the decisions that move perceived latency the
most live: that the model isn't unloaded, that the prompt isn't recomputed in
full, that one request doesn't wait behind another.

**Creating models is in scope; creating a large one is not.** A
billion-parameter model is thousands of GPU-hours and hundreds of GB of text,
and that doesn't come out of a machine without a GPU. A **small, specialized**
one does, and it fits the rest of the project: if the goal is running on a
phone, an 8-million-parameter model trained for one task beats a 3-billion one
that doesn't fit.

One detail decides the order of that work, and it's arithmetic, not opinion.
With Qwen's vocabulary — 151,936 entries — a model 256 wide and 6 layers deep
would be **87% embedding table**: 38.9 M out of 44.6 M. It wouldn't be a model,
it would be a lookup table with a transformer attached. With a custom 8,192
vocabulary, the same shape is 7.8 M with the embedding at 27%.

So **creating a model starts with creating a tokenizer**. It's in the
[ROADMAP](ROADMAP.md) §8 with the figures.

---

## The contract, which is the first milestone

Cosxamix was born so WinsiBot would stop depending on an external service.
WinsiBot talks to Ollama through **exactly two endpoints**, verified by reading
its `ai_chat.rs`:

```
GET  /api/tags              → {"models":[{"name":"..."}]}
POST /v1/chat/completions   → OpenAI format
```

So that's the minimum goal: implement those two and have WinsiBot work
**without changing a line**, just pointing its `OLLAMA_URL` at Cosxamix.

Imitating someone else's dialect instead of inventing our own was a deliberate
choice: it lets us swap the engine underneath — or go back to Ollama — without
touching the consumer. A format of our own would force changing every client
and maintaining it forever, in exchange for nothing.

```bash
# In WinsiBot, without touching code:
OLLAMA_URL=http://127.0.0.1:11435
```

---

## Usage

```bash
cargo run --release
```

```bash
curl http://127.0.0.1:11435/api/tags

curl -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"cosxamix-demo","messages":[{"role":"user","content":"hello"}]}'
```

With `"stream": true` the response comes over SSE in OpenAI's dialect, one
`data:` per chunk and a `data: [DONE]` at the end. Measured on the 3B: the
first chunk at **2.14 s** and the last at 5.82 — **3.68 s of text appearing**
instead of a blank screen. And if the client hangs up, generation stops:
verified by closing at 1.87 s on a 120-token request, where the server stopped
at 5.

```bash
curl -N -X POST http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hello"}]}'
```

Environment variables:

| | default | |
|---|---|---|
| `COSXAMIX_PORT` | `11435` | One off from Ollama's 11434, so both can coexist |
| `COSXAMIX_MODELOS` | `./modelos` | Where to look for model files |
| `COSXAMIX_GGUF` | `qwen2.5-3b-instruct-q3_k_m.gguf` | Which one gets pre-warmed at startup |
| `COSXAMIX_RAM_MB` | `4096` | Memory budget for loaded models |
| `RAYON_NUM_THREADS` | physical cores | Threads for the matrix math |
| `COSXAMIX_SIN_BLOQUES`, `COSXAMIX_SIN_Q6K_FUSIONADO`, `COSXAMIX_SIN_VARIAS`, `COSXAMIX_SIN_PRESTAMO` | unset | Each one turns off a 0.7.0 optimization —the blocked prefill matmul, the Q6_K kernel, q-k-v in one pass, the prefix shared across conversations— to measure with and without it ([DECISIONES §24-26](DECISIONES.md)) |
| `COSXAMIX_PRECARGA` | `2` | Rows ahead that the decode asks the cache for while it computes one; `0` turns it off ([DECISIONES §33](DECISIONES.md)) |
| `COSXAMIX_SIN_Q5K_FUSIONADO` | unset | Turns off the Q5_K fused kernel, to measure with and without it ([DECISIONES §34](DECISIONES.md)) |
| `COSXAMIX_IMAGEN_LIBERAR_MIN` | `10` | Minutes without image requests before the model (2.6 GB peak) is released |
| `COSXAMIX_TESELA` | unset | `vieja` goes back to the pre-0.12.0 prefill tile and `c` uses the C one, to compare ([DECISIONES §38](DECISIONES.md)) |
| `COSXAMIX_SGEMM_MICRO` | `rust` | `fortran` uses the Fortran microkernel in the f32 GEMM, with the `fortran` feature ([DECISIONES §37](DECISIONES.md)) |
| `RUST_LOG` | `info` | The log level: `error`, `warn`, `info`, `debug` or `trace` |

And for building and measuring:

| | |
|---|---|
| `COSXAMIX_ESPERAR_CPU` | Seconds a measuring example waits for the CPU to drop below 15% before measuring; without it, it only warns ([DECISIONES §37](DECISIONES.md)) |
| `COSXAMIX_GFORTRAN` | The gfortran for the `fortran` feature; otherwise MSYS2's on Windows, or the one in PATH |
| `COSXAMIX_GNAT_GCC` | GNAT's gcc for the `ada` feature; otherwise it is looked up in the Alire cache and in PATH |

**Every `.gguf` in the directory is registered**, not only the one in
`COSXAMIX_GGUF`: that variable says which one is pre-warmed, not which one
exists. Asking for another by `model` loads it, evicting the least recently
used one if it doesn't fit the budget.

The budget is in **bytes, not in number of models**, because they don't weigh
the same: in a normal directory they run from 468 to 2007 MB, and a cap of
"three models" would let in 1.4 GB or 6 GB depending on which. It is measured
by file size, which is a good approximation now that embeddings are no longer
dequantized: the RAM/disk ratio came down to x1.02-x1.08.

There is no `COSXAMIX_TOKENIZER` any more: one variable cannot serve five
models. Each GGUF is paired automatically, looking first for
`<model>.tokenizer.json`, then `tokenizer-<size>.json`, and finally
`tokenizer.json`. The pairing **is logged at startup**, because a wrong
tokenizer doesn't raise an error: it produces nonsense.

---

## Building, testing and publishing

```bash
cargo run --release --features zig,c                # text and images, with the Zig and C kernels: 72 crates
```

The features: `zig` and `c` (kernels, they need Zig), `training` (training),
`asm_q4k` and `asm_q3k` (the assembly kernels), `fortran` (needs gfortran) and
`ada` (needs GNAT; with Alire installed it is found on its own). None is on by
default, and without any Cosxamix builds and runs on Rust and AVX2.

What each thing checks, and it all runs the same locally as in CI:

```bash
cargo fmt --all -- --check
cargo clippy --features zig,c,training --all-targets -- -D warnings
cargo test --features zig,c,training                 # none needs a model
cargo test --features fortran,ada                    # if gfortran and GNAT are there
python scripts/check_docs.py                         # tables, links and old names
python scripts/check_kernel_arch.py "" x86_64        # that the kernel matches the target
cargo run --release --example check_models           # that every GGUF in modelos/ passes the bounds rule
```

And the SPARK proof, from `ada/`:

```bash
alr -n build
alr -n exec -- gnatprove -P cosxamix_checks.gpr --level=4 -j0 --checks-as-errors=on
```

**`RUSTFLAGS` is used nowhere here, on purpose.** That environment variable
*replaces* the `rustflags` under `[target.*]` in `.cargo/config.toml` instead of
adding to them: a `RUSTFLAGS=-D warnings` would wipe out `target-cpu=x86-64-v3`,
which is a [measured 4x](#the-biggest-find-target-cpu), and nothing would say
so — it compiles, it passes the tests, and it runs four times slower. Warnings
are denied by passing the flag to clippy as its own argument.

### The two workflows

`ci.yml` runs on every push: the Linux, Windows and macOS matrix across the
feature combinations, the Zig and C kernels with their equivalence tests (and
the Fortran one, with gfortran), **the cross to `aarch64`**, a run on a native
ARM64 runner —which also measures the f32 GEMM without and with Fortran—, the
SPARK proof with GNATprove and the tests with the `ada` feature, and the
documentation.

The cross job isn't there for completeness. Zig is in this project because it's
the only vectorized path on ARM and Android; if cross-compiling broke, the ARM
binaries would ship with scalar kernels — 19 times slower — and nothing would
warn. It already caught a real failure: `build.rs` was passing `-target=X` as
one argument and Zig wants it as two.

`release.yml` publishes 6 targets on tag:

| platform | kernels | why |
|---|---|---|
| Linux x86-64 · Windows x86-64 · macOS Intel | Rust AVX2 | Zig loses 3-8% there, [measured](#what-the-first-kernel-measured) |
| Linux ARM64 · Android ARM64 · macOS Apple Silicon | NEON via Zig, and `sdot` via C (`zig,c`) | without them the quantized matrix product falls back to scalar |

One file per platform, no DLL or `.so` alongside, each with its `.sha256`.
macOS Intel is cross-compiled from the ARM runner instead of asking for an
Intel runner, which is the one GitHub has been retiring. Android is separate
from `aarch64-unknown-linux-gnu` because it uses Bionic instead of glibc, and a
binary of the latter downloads fine and fails to run.

---

## What's next

It's in **[ROADMAP.md](ROADMAP.md)**, with the measured state up top and the
order set by how much it moves latency, not by how interesting it is to write.

In short: **reading the prompt** runs at ~57 tok/s on the 3B since 0.12.0
(~47 before, and 1.7x that of 0.6.0), and the start of the *system* prompt is
read once for every chat (0.7.0).
**Generation** runs at **~81-87% of the measured memory ceiling** (~30 GB/s,
DDR4-2133), with the row prefetch of 0.9.0 ([DECISIONES §33](DECISIONES.md));
the 51% quoted before was computed against a ceiling this RAM cannot deliver.
What's left is small and lives in the kernels. Why 6 cores yield 2.6x instead
of 6x **is now explained**: a single thread already reads 19 GB/s and 4 saturate
the RAM. Speculative decoding stays out: with the engine near the ceiling, the
numbers still don't add up (ROADMAP §5).

What can move the ceiling most now is **weights as formulas** (ROADMAP §12):
quantizing 8 weights at a time on the E8 lattice gave, on the whole 0.5B, the
score of Q3_K at 2.06 bits per weight instead of 3.44
([DECISIONES §40](DECISIONES.md)). If it holds on the 3B, the model would weigh
half, and the decode, which runs at the pace of RAM, would go close to twice as
fast.

Later on, as a separate project, **QCML** will be born (`.qcml`, *Quantum
Compression for Machine Learning*): an in-house format that aims to **beat GGUF
in every respect**, measured one aspect at a time. It stores tensor networks
—the mathematics of quantum states, with no quantum computing involved— as
their cores rather than as dense matrices, so the 0.8.0 savings —tables 30.6x
smaller— reach the file too (ROADMAP §11).

---

## daiss: the project's own models

Cosxamix doesn't only serve other people's models. **daiss** is its own
family: five sizes the project defines, initialises and writes as GGUF.

| model | parameters | Q8_0 | cache/token |
|---|---:|---:|---:|
| daiss-mini | 0.08 B | 0.08 GB | 3.0 KB |
| daiss-0.5b | 0.63 B | 0.62 GB | 12.0 KB |
| daiss-3b | 3.40 B | 3.36 GB | 36.0 KB |
| daiss-7b | 7.61 B | 7.54 GB | 56.0 KB |
| daiss-14b | 14.77 B | 14.62 GB | 192.0 KB |

The attention cache is f16 since 0.12.0; in f32 it was twice as much.

```
cargo run --release --example create_daiss
COSXAMIX_DAISS=daiss-7b cargo run --release --example create_daiss
```

They are generated streaming and in parallel, with peak memory capped at 16 MB
whatever the size, and seeded: the same seed yields the same file byte for
byte. daiss-7b has been created, and candle read it with zero differences
while it was still in the project; today the own engine reads it.

And with the project's own tokenizer — `src/tokenizer.rs`, a byte-level BPE it
trains itself — the vocabulary drops from 151,936 pieces to 8,192 for a
measured **3.4% more tokens** on held-out text, taking daiss-mini from
**84 MB to 9 MB**:

```
cargo run --release --example train_tokenizer
COSXAMIX_BPE=modelos/daiss-8192.bpe cargo run --release --example create_daiss
```

The tokenizer travels inside the model file.

**Chavín** is the big model: `chavin-30b`, 29.5 B parameters, sized for a 16 GB
machine. Models can now be written in k-quants, and it goes in Q2_K, 9.02 GB:

```
COSXAMIX_DAISS=chavin-30b cargo run --release --example create_daiss
COSXAMIX_TIPO=q3k COSXAMIX_DAISS=daiss-3b cargo run --release --example create_daiss
```

It runs here at 2.38 tok/s, 57% of what memory allows, thanks to a fused Q2_K
kernel written in Rust, assembly, C and Zig.

And they can now be trained from scratch, on this CPU: `examples/train_daiss.rs`
trains every parameter, and the exported GGUF is served by the API with its own
vocabulary, no JSON needed. With no chat template, the messages arrive one after the
other and the model continues them. A daiss-mini reaches 2.03 bits per byte on unseen
text after 10 minutes, below `xz` (2.15), and 1.77 after 30:

```
COSXAMIX_VOCAB=8192 cargo run --release --features training --example train_daiss -- corpus.txt
```

It writes Spanish-looking sentences that don't mean anything yet.

**A freshly written daiss or Chavín is not trained**: its weights are noise with the
right standard deviation, so it generates nonsense. What exists and is
verified is the whole path — define, initialise, quantise, write, read, load —
which is the precondition for any training. What's missing is in
[ROADMAP.md](ROADMAP.md) §8.

---

## The engine: our own, and where it comes from

Cosxamix was born on **candle** (from Hugging Face), picked by measuring: pure
Rust, without the C++ toolchain llama.cpp asks for, which on Termux is exactly
the requirement we wanted gone. From 0.3.0 on it was replaced piece by piece
—the GGUF reader, quantization, matrix multiplications, attention, the
forward— always comparing against candle until they matched. In 0.6.0 our
engine became the default: 11 of 15 on the question battery against 10, and
1.6-1.9x faster per message ([DECISIONES §23](DECISIONES.md)). **In 0.11.0 the
candle left entirely**: text, images (CLIP, the SD UNet and TAESD on our own
GEMM, 2.3x faster and 1.6 GB lighter) and training (a hand-written backward,
1.9x faster for LoRA) are ours. The binary weighs 2.56 MB instead of 7.3, images included, with
72 crates instead of 139 (3.25 MB and 80 in 0.11.0; tokio and logging with only what is used,
DECISIONES §38).

The measurements of that era —the attention cache, the embeddings table, the
speculative decoding that was deleted— are in [DECISIONES.md](DECISIONES.md)
and in the [CHANGELOG](CHANGELOG.md), from 0.1.0 to 0.10.0. What still holds:

### The biggest find: `target-cpu`

The first measurement gave **1.8 tokens/s** with a 0.5B model. **Rust compiles
for the basic x86-64 of 2003 by default**, without AVX2. Same code, same
model, only the flag changes:

| | per answer | tokens/s |
|---|---:|---:|
| generic (default) | 4428 ms | 1.8 |
| **`x86-64-v3`** (AVX2+FMA) | **1094 ms** | **6.1** |
| `native` | 1066 ms | 6.3 |

`x86-64-v3` is pinned in `.cargo/config.toml`, not `native`, which ties the
binary to the CPU that built it. **The cost**: it does not start on an x86
older than 2015. For those:

```bash
# For a CPU without AVX2 (Celeron, Pentium, Atom...). RUSTFLAGS replaces the config.
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release --features zig
```

With `zig`, that build carries the Zig kernels twice —SSE2 and AVX2— and picks
at startup: on an old CPU it uses the SSE2 one, **2.4x to 12x** faster than the
scalar reference ([DECISIONES §20](DECISIONES.md)).

### Which model and which quantization

The 3B in Q3_K_M, measured with the same 15 questions with checkable answers
(`examples/compare_models.rs`):

| | disk | correct |
|---|---:|---:|
| 0.5B Q4_K_M | 504 MB | 7/15 |
| 1.5B Q4_K_M | 1108 MB | 8/15 |
| 3B Q2_K | 1313 MB | 6/15, it breaks |
| **3B Q3_K_M** | **1644 MB** | **10/15** (11/15 with the own engine) |
| 3B Q4_K_M | 2007 MB | 10/15 |
| **Qwen3-4B-Instruct-2507 Q3_K_M** | **1979 MB** | **14/15** |

Q3_K_M gets as many right as Q4_K_M with 363 MB less, and fewer bytes per
weight are fewer bytes to read from RAM on every token. The 3B is under the
**Qwen Research License** (non-commercial); the 0.5B and 1.5B, Apache-2.0
([THIRD-PARTY.md](THIRD-PARTY.md)).

**Qwen3** is read since 0.12.0 (no biases on q, k and v, a per-head RMSNorm on
q and k, the head width taken from the file). The 4B-Instruct-2507 in Q3_K_M
gets **14/15** right, against 11/15 for the 3B with the own engine, on the same
battery and the same machine: decode **11.0 tok/s against 14.1**, a 40-token
answer in ~4.8 s against ~3.8, and 335 MB more. It is Apache-2.0, so it can
also be used commercially. `compare_models -- --qwen3` repeats it.

### Threads, cache and memory

- **Threads: physical cores, not logical ones.** Going from 6 to 12 threads on
  an i5-10400 the decode gains 2% for 54% more CPU: both threads of a core
  fight for the same AVX2 unit. `RAYON_NUM_THREADS` changes it.
- **Attention cache per session**: every conversation keeps its own, and the
  delta of a new message goes in without recomputing what came before.
- **Memory ≈ the file**: weights are mapped and the embeddings table is read
  row by row; a model takes 1.02-1.08x its disk size in RAM. What the engine
  asks for on top —activations, buffers, cache— peaks at ~150 MB on the 3B's
  first message, and 192 MB on Qwen3-4B ([DECISIONES §39](DECISIONES.md)).
- **The attention cache is f16**, half of f32, and attention is split by token
  and head: in the decode it ran on a single thread. Since 0.12.0, +9% decode
  on the 3B and +27% on Qwen3-4B.
- **Every tensor of a GGUF is validated when it is opened**: how many values,
  how many bytes, and that they fall inside the file, with no overflow. With
  the `ada` feature a rule proven with SPARK decides it
  ([DECISIONES §41](DECISIONES.md)).
- **Speculative decoding** was implemented, measured and deleted in the candle
  era; when it makes sense again is in the [ROADMAP](ROADMAP.md) §5.

## Several languages, one single boundary

Cosxamix isn't written in Rust alone. Rust is the main one, and the boundary is
open to whichever language wins the measurement:

| | what it brings | what's missing |
|---|---|---|
| **Zig** | one source covering x86-64, ARM64 and Android | nothing: it's already in |
| **Assembly** | the fused Q4_K kernel on x86-64, first in the dispatcher with `asm_q4k` (1.04x-1.08x over intrinsics). The Q3_K one (`asm_q3k`) ties with intrinsics, bit for bit | the NEON body, `kernels/q4k_neon.s`, and measuring it on ARM |
| **Fortran** | in with the `fortran` feature (`kernels/*.f90`, gfortran, `bind(C)` and no Fortran runtime): the microkernel of the f32 GEMM used by images and training. Four lines of loop from which gfortran emits the same 12 FMAs as 40 lines of intrinsics: on x86-64 it **ties** with Rust, bit for bit ([DECISIONES §37](DECISIONES.md)). On ARM64, where Rust only has the scalar one, it is the default, with the NEON gfortran generates | the ARM figure: the CI ARM64 job measures it without and with the feature; and a gfortran for Android |
| **Ada / SPARK** | in with the `ada` feature (`ada/src/`, GNAT's gcc): the bounds of every tensor of a GGUF, with GNATprove proving that no computation overflows and that it says "valid" exactly when the rule holds (110 proof obligations, with gnatprove 14 and 16). Without the feature the same rule in Rust applies, and a test compares both ([DECISIONES §41](DECISIONES.md)) | a proven compute kernel; GNAT doesn't build for Android, so there the Rust rule always applies |
| **C** | in with the `c` feature, built with `zig cc` and no libc. Per-function CPU extensions (`__attribute__((target))`), and on ARM a stable `vdotq_s32` — `sdot` — which Rust doesn't expose and Zig doesn't emit. On x86 its Q3_K loses 4% to Rust ([DECISIONES §21](DECISIONES.md)) | a phone measurement: on the CI ARM runner the `sdot` fused kernels win 2.4x (Q4_K) and 5.4x (Q3_K), and since 0.5.1 they are in the dispatcher |

The risk in that isn't the languages: it's having **N different places** where
each one gets in. That's why there is a single boundary, in `src/kernels/`, and
three rules governing it.

1. **Every kernel has a portable reference implementation in Rust.** It defines
   which result is correct and it runs when the CPU has nothing better. It is
   never deleted.
2. **Every implementation is compared against the reference**, in the tests. A
   kernel that runs faster and gives a different number is not faster: it's
   wrong.
3. **Every implementation must justify itself by measurement**: beat the
   reference on speed, **or** enable a platform where there was no acceleration.
   If it does neither, it gets deleted — a backend isn't kept just because it's
   already written.

The backend is chosen **at runtime**, by looking at what the CPU supports. A
binary compiled where AVX-512 exists and run where it doesn't dies with an
illegal instruction; detecting at startup is what makes it possible to ship **a
single binary** that runs everywhere and uses the best of each machine.
`COSXAMIX_KERNEL` forces one, so two can be measured against each other on the
same machine.

```bash
cargo test                      # reference and AVX2
cargo test  --features zig      # and Zig as well
cargo bench --features zig      # rule 3, executable
cargo test  --features asm_q4k  # the hand-written assembly kernels
cargo test  --features asm_q3k  # the Q3_K kernel in assembly
cargo test  --features c        # the C kernels (needs Zig, for `zig cc`)
cargo test  --features fortran  # the Fortran one (needs gfortran)
cargo test  --features ada      # the bounds rule proven with SPARK (needs GNAT)
```

### What the first kernel measured

`dot` (`f32` dot product), an i5 with AVX2+FMA and no AVX-512, Criterion median:

| n | scalar | AVX2 (Rust) | Zig | Zig vs AVX2 |
|---|---:|---:|---:|---:|
| 1,024 | 1.039 µs | **54.1 ns** | 58.2 ns | +7.5% |
| 4,096 | 4.631 µs | **233.5 ns** | 252.3 ns | +8.0% |
| 65,536 | 75.9 µs | 8.53 µs | **8.50 µs** | tie |
| 1,000,003 | 1.633 ms | **933 µs** | 1.010 ms | +8.2% |

Two conclusions, and the second is the one that matters:

**On x86-64, Zig does not win.** It stays 8% behind hand-written AVX2
intrinsics, and ties at exactly one size. If the criterion were only speed on
this machine, rule 3 would delete it. But the rule covers coverage, and that's
why it stays:

**The Zig kernel is ONE source covering x86-64, ARM64 and Android.** Verified:
the same `kernels/dot.zig`, without touching a line, compiles for
`aarch64-linux-gnu` and for `aarch64-linux-android` — Termux's target. The Rust
path is `core::arch::x86_64`: **x86 only**. On a phone, today, it falls back to
the scalar reference, which is 19 times slower.

And since 0.11.0, with candle gone, that also holds for the quantized matrix
multiply, where 99% of the work is: on ARM it is done by the C kernel with
`sdot` (feature `c`) and the Zig one (feature `zig`). Up to 0.10.0 candle's
`neon.rs` covered it; today an ARM build without `zig` or `c` falls back to the
scalar reference for everything. That's why the ARM binaries ship with
`zig,c`.

So what Zig contributes here isn't speed: it's **coverage**. A portable kernel
that performs nearly as well as the specific one beats a specific kernel that
doesn't exist on half the fleet. That's the reason it stays — and it's written
down so that if someone writes the NEON path in Rust, it gets measured again and
decided again.

### Second kernel: `rmsnorm`

It went in to answer a question the first one couldn't: **does the boundary
generalize?** One that works for a single case may not serve another.

`rmsnorm` has a different shape on purpose — two passes over the data, a square
root, three vectors instead of two, and it writes into an output buffer instead
of returning a scalar. **Nothing in the selection mechanism had to change**:
same enumeration, same dispatch, same pointer-and-length FFI. That's what we
wanted to find out.

Realistic sizes: a small LLM's dimension runs from 2048 to 8192. Measuring it
with million-element vectors would say nothing, because a transformer never
calls it that way.

| n | scalar | AVX2 (Rust) | Zig | Zig vs AVX2 |
|---|---:|---:|---:|---:|
| 2,048 | 2.874 µs | **294.6 ns** | 375.4 ns | +27% |
| 4,096 | 6.152 µs | **759.4 ns** | 799.1 ns | +5.2% |
| 8,192 | 12.185 µs | **1.970 µs** | 2.033 µs | +3.2% |

Same pattern as `dot`: AVX2 ahead, Zig 3-27% behind depending on size, and both
far above the reference.

**A warning about that last column**, so nothing more is read into it than it
says: the reference accumulates the sum of squares in `f64`, on purpose, because
its job is to be correct and `f32` loses precision on long vectors. So the 6-10x
against it **is not vectorization alone**: part of it is the precision
difference. The clean language-to-language comparison is the last two columns,
which do accumulate the same way.

### Third kernel: `silu`, and the first time Zig wins on speed

`silu` is the MLP activation — `x * sigmoid(x)` — and the MLP is 79% of the
weights, so it runs over 11008 elements per layer per token.

It has **two** implementations, not three, and that's the interesting part:

| n | scalar | Zig | |
|---:|---:|---:|---|
| 4,864 | 21.23 µs | **16.88 µs** | 1.26x |
| 11,008 | 47.96 µs | **38.76 µs** | 1.24x |

There is no AVX2 path because **Rust cannot vectorize an exponential without
hand-writing it**: `f32::exp` is scalar, and applying it to an `__m256` forces
a polynomial approximation with range reduction — forty delicate lines where a
wrong coefficient breaks nothing, it just gives slightly worse answers. Zig's
`@exp` takes a vector and lowers to the LLVM intrinsic.

It's the first time Zig's advantage here isn't "it runs on ARM" but **the
language brings the tool**.

That said, 1.25x is not 8x. If `@exp` truly vectorized across all eight lanes
we'd see far more, so most likely LLVM expands it into eight scalar calls and
only saves the loop. It's a kernel worth re-measuring when the Zig version
changes.

### A trap that cost a measurement

The first comparison had Zig losing between 3% and **26%**. The reason wasn't
the language: `build.rs` was invoking `zig build-lib` without `-mcpu`, so it
compiled for the basic x86-64 of 2003 and its `@Vector(8, f32)` was split into
128-bit operations. With `-mcpu=x86_64_v3` — which is exactly AVX2+FMA, the same
thing Rust's `#[target_feature]` enables — the gap dropped to 8%. Comparing
without matching the target is comparing two targets, not two languages.

---

## Training without a GPU

Cosxamix no longer only runs inference. With the `training` feature it
fine-tunes a model using **LoRA**: it trains 0.1% of the parameters instead of
100%.

```bash
cargo run --release --features training --example train_lora -- corpus.txt 1000
```

The output is an `adaptador.safetensors` of **2.1 MB**. The base model is never
touched, so the adapter ships on its own.

Measured on an i5-10400 with no GPU, Qwen2.5-0.5B, 256-token sequences:

| | |
|---|---:|
| trainable parameters | 540,672 out of 494 million |
| full step | ~2.8 s (with candle, ~5.2) |
| 1000 steps | ~47 min |
| resulting adapter | 2.1 MB |

The forward and backward are ours (`src/training/transformer.rs`), every
gradient checked against finite differences. With the base frozen, neither
`dL/dW` nor the vocabulary tables' gradient is computed, which is a good part
of why the step went from 5.2 s to 2.8.

**The base is held in f32, not quantized**, and that bounds the scope: the
backward needs `dL/dy · W` — multiplying by `W` untransposed — and the quantized
kernels only do `x · Wt`. The 0.5B in f32 is ~2 GB and fits; the 3B would be
~12 GB and does not. Training the 3B needs that kernel, and it's in the
[ROADMAP](ROADMAP.md) §7.

The training model is an f32 reimplementation of the inference one, and that
they are the same **is checked**, because a mistake there — RoPE order, how KV
heads are repeated, the attention scale — would still make the loss go down,
optimizing against a model that isn't the one later served:

```bash
cargo run --release --features training --example check_training_model
```

It compares the logits of both paths on a real prompt. It doesn't demand
equality — one dequantizes to f32 and the other multiplies in integers — but
that they **pick the same token**. Measured: 2.6% of the range as the largest
difference, same token. Against candle's Qwen2 in f32, before removing it, the
difference was 4e-5: the 2.6% is what the engine quantizes, not a mistake in
the training model.

---

## License

**GPL-3.0-or-later** — see [LICENSE](LICENSE): it can be used, studied,
modified and redistributed, and whatever is distributed modified stays GPL,
with its source. Up to 0.10.0 Cosxamix was released under MIT, and those
versions stay MIT for whoever has them; from 0.11.0 on, GPL.

The models it uses that don't ship in the repository, and their licenses, are
in [THIRD-PARTY.md](THIRD-PARTY.md).
