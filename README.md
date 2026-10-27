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
| [THIRD-PARTY.md](THIRD-PARTY.md) | the one file copied from another project, where it comes from and why |

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
| Graph execution, attention, quantization | **Cosxamix** — in progress, replacing `candle` piece by piece | **yes** |
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

It happens **piece by piece, with candle as the oracle**: each part of ours is
compared against theirs on the real models before it replaces anything, and
candle stays as a dev-dependency even after it leaves the binary. Owning the
engine is no reason to throw away the cheapest way to know whether it's right.

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
cargo run --release                                  # HTTP layer only, 6 crates
cargo run --release --features candle                # the real engine, 155 crates
cargo build --release --features zig,candle          # with the Zig kernels
```

What each thing checks, and it all runs the same locally as in CI:

```bash
cargo fmt --all -- --check
cargo clippy --features zig,candle --all-targets -- -D warnings
cargo test --features zig,candle                     # 22 tests, none needs a model
python scripts/check_docs.py                         # tables, links and old names
python scripts/check_kernel_arch.py "" x86_64        # that the kernel matches the target
```

**`RUSTFLAGS` is used nowhere here, on purpose.** That environment variable
*replaces* the `rustflags` under `[target.*]` in `.cargo/config.toml` instead of
adding to them: a `RUSTFLAGS=-D warnings` would wipe out `target-cpu=x86-64-v3`,
which is a [measured 4x](#the-biggest-find-target-cpu), and nothing would say
so — it compiles, it passes the tests, and it runs four times slower. Warnings
are denied by passing the flag to clippy as its own argument.

### The two workflows

`ci.yml` runs on every push: the Linux, Windows and macOS matrix across both
feature combinations, the Zig kernels with their equivalence tests, **the cross
to `aarch64`**, and the documentation.

The cross job isn't there for completeness. Zig is in this project because it's
the only vectorized path on ARM and Android; if cross-compiling broke, the ARM
binaries would ship with scalar kernels — 19 times slower — and nothing would
warn. It already caught a real failure: `build.rs` was passing `-target=X` as
one argument and Zig wants it as two.

`release.yml` publishes 6 targets on tag:

| platform | kernels | why |
|---|---|---|
| Linux x86-64 · Windows x86-64 · macOS Intel | Rust AVX2 | Zig loses 3-8% there, [measured](#what-the-first-kernel-measured) |
| Linux ARM64 · Android ARM64 · macOS Apple Silicon | NEON via Zig | it's the only vectorized path that exists |

One file per platform, no DLL or `.so` alongside, each with its `.sha256`.
macOS Intel is cross-compiled from the ARM runner instead of asking for an
Intel runner, which is the one GitHub has been retiring. Android is separate
from `aarch64-unknown-linux-gnu` because it uses Bionic instead of glibc, and a
binary of the latter downloads fine and fails to run.

---

## What's next

It's in **[ROADMAP.md](ROADMAP.md)**, with the measured state up top and the
order set by how much it moves latency, not by how interesting it is to write.

In short: the engine is at **51% of this machine's memory-bandwidth ceiling**,
meaning there's less than 2x left to win here. Why 6 cores yield 2.6x instead
of 6x **is now explained**: the matrices that don't fit in the 12 MB of L3
don't scale, and they are 89% of the weights. It's the memory wall, and more
threads won't fix it — which is exactly what killed speculative decoding.

---

## daiss: the project's own models

Cosxamix doesn't only serve other people's models. **daiss** is its own
family: five sizes the project defines, initialises and writes as GGUF.

| model | parameters | Q8_0 | cache/token |
|---|---:|---:|---:|
| daiss-mini | 0.08 B | 0.08 GB | 6.0 KB |
| daiss-0.5b | 0.63 B | 0.62 GB | 24.0 KB |
| daiss-3b | 3.40 B | 3.36 GB | 72.0 KB |
| daiss-7b | 7.61 B | 7.54 GB | 112.0 KB |
| daiss-14b | 14.77 B | 14.62 GB | 384.0 KB |

```
cargo run --release --example create_daiss
COSXAMIX_DAISS=daiss-7b cargo run --release --example create_daiss
```

They are generated streaming and in parallel, with peak memory capped at 16 MB
whatever the size, and seeded: the same seed yields the same file byte for
byte. daiss-7b has been created and candle reads it with zero differences.

And with the project's own tokenizer — `src/tokenizer.rs`, a byte-level BPE it
trains itself — the vocabulary drops from 151,936 pieces to 8,192 for a
measured **3.4% more tokens** on held-out text, taking daiss-mini from
**84 MB to 9 MB**:

```
cargo run --release --example train_tokenizer
COSXAMIX_BPE=modelos/daiss-8192.bpe cargo run --release --example create_daiss
```

The tokenizer travels inside the model file.

**A freshly written daiss is not trained**: its weights are noise with the
right standard deviation, so it generates nonsense. What exists and is
verified is the whole path — define, initialise, quantise, write, read, load —
which is the precondition for any training. What's missing is in
[ROADMAP.md](ROADMAP.md) §8.

---

## The engine: candle, and why

The decision was made by measuring, not by reading comparisons.

| | candle | llama.cpp |
|---|---|---|
| dependencies | 155 crates, **pure Rust** | + a C++ toolchain per platform |
| release build | 84 s | minutes, and you need a compiler |
| crosses to `aarch64-linux-android` | **yes** | with the NDK and a fight |

The last row is what decides it. Termux is a target of this project, and an
engine that needs a C++ compiler on the phone reintroduces the very requirement
we were removing — moved elsewhere, but the same one.

**A trap you only see by trying**: the `tokenizers` crate pulls in the `onig`
feature by default, which is Oniguruma — a C library — and it breaks the cross
to Android with `failed to run custom build command for onig_sys`. It's turned
off with `default-features = false`, and the alternative path uses
`fancy-regex`, pure Rust. It's annotated in `Cargo.toml` because it's the kind
of dependency that creeps back in without anyone noticing.

### The biggest find: `target-cpu`

With the engine running, the first measurement gave **1.8 tokens/s** — unusable
for a 0.5B model. The reason wasn't candle: **Rust compiles by default for the
basic x86-64 of 2003**, so the matrix multiplications weren't using AVX2.

It's exactly the same trap we'd hit with Zig and `-mcpu`, now in the other
language. Same code, same model, only the flag changes:

| | per response | tokens/s |
|---|---:|---:|
| generic (default) | 4428 ms | 1.8 |
| **`x86-64-v3`** (AVX2+FMA) | **1094 ms** | **6.1** |
| `native` | 1066 ms | 6.3 |

`x86-64-v3` was chosen over `native`, and it's pinned in `.cargo/config.toml`.
`native` looks at the CPU doing the compiling and ties the binary to it: one
built on a server with AVX-512 dies with an illegal instruction on a machine
without it. `x86-64-v3` is AVX2+FMA — any CPU from 2015 onward — and delivers
97% of the performance.

**The cost, stated plainly**: the binary does not start on an x86 older than
2015. Those need a build without that line, and it will run 4 times slower. The
kernels in `src/kernels/` do choose at runtime, but candle's internals do not.

```bash
# For a CPU without AVX2 (Celeron, Pentium, Atom...). RUSTFLAGS replaces the config.
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release --features candle,zig
```

With `zig`, that build carries the Zig kernels twice — SSE2 and AVX2 — and picks
one at startup: on an old CPU the in-house engine uses the SSE2 copy, **2.4x to
12x** faster than the scalar reference ([DECISIONES §20](DECISIONES.md)).

### Measured end to end

Qwen2.5-0.5B-Instruct Q4_K_M (491 MB), CPU, no GPU:

```
model load           ~1.9 s   (once, at startup)
first response        3.0 s   cold
typical response      1.1 s   warm
speed                 6.1 tokens/s
```

Cosxamix **pre-warms at startup**: the user doesn't pay the 2 s load on their
first question. That's the reason this project exists — Ollama unloads the
model after 5 minutes of inactivity, and a bot with spaced-out messages pays
that reload almost every time.

With no model in `modelos/`, or built without `--features candle`, it starts
with the demo engine and the HTTP contract still holds.

```bash
cargo run --release --features candle
```
### Reusing the attention cache

Prefill dominates latency: generating **one** token cost 857 ms, of which ~817
were processing the prompt. And in a conversation that work repeats almost
entirely on every turn, because turn N's prompt starts exactly like turn N-1's.

Cosxamix reuses the **longest common prefix** between the new prompt and what's
in the cache, truncates the cache to that part, and processes only the rest.

The first version required the new prompt to start with *everything* cached.
That works in a conversation that grows, but not in WinsiBot's real case, which
sends `system + question` with no history: turn 2 shares the system with turn 1
and then diverges. Measured against the real bot, it gave **`cacheados=0`
always**. With the common prefix it goes to **131**, which is the whole system
prompt — style profiles and command catalog that used to be reprocessed on
every message.

That was only possible because `truncar_cache` already existed, from the
speculative-decoding attempt.

| turn 2 of a conversation | |
|---|---:|
| no cache | 2454 ms |
| **with cache** (53 of 69 tokens reused) | **1562 ms** |
| saved | 892 ms · **1.6x** |

The advantage grows with the conversation: the longer the history, the larger
the part that doesn't need reprocessing.

#### A candle limitation, and why the delta goes one token at a time

`ModelWeights::forward` builds the causal mask as **seq_len × seq_len**, which
is only correct with an empty cache: with a cache, attention is
seq_len × (cached + seq_len) and the shapes don't line up — the attempt gives
`cannot broadcast [16, 16] to [1, 14, 16, 69]`. Candle supports either a full
prefill or decoding one token at a time, not a block with a cache in play.

One at a time there is no mask (`seq_len == 1` skips it) and it works. It costs
twice per token what block prefill does — ~40 ms against ~21 — but it skips the
entire cached prefix. The arithmetic: reusing C tokens and feeding D one at a
time wins while **D < 1.1·C**, which holds comfortably from the second turn on.

#### Why the answer changes, and why that is NOT a bug

With the cache in place, the same prompt produces different text. The first
reading is that the cache is broken. It isn't, and telling the difference took
isolating the variable.

Comparing text is useless: with argmax, a one-bit difference in the logits
changes a token and everything diverges from there, so two different outputs
don't distinguish a broken cache from floating-point noise. Two experiments
were needed.

**The first** fixes the prompt at 69 tokens and compares three paths:

| | cache | prefill | output |
|---|---|---|---|
| A | no | in a block | *…enviar mensajes, **controlar chats** y mucho más* |
| B | no | one token at a time | *…enviar mensajes, **controlar tu conversación** y mucho más* |
| C | **yes** | one token at a time | *…enviar mensajes, **y mucho más*** |

**A and B differ, and neither uses the cache.** The only thing that changes
between them is the shape of the matrix, because `gemm` accumulates in a
different order depending on the block. That rules the cache out as the cause.
The `COSXAMIX_PREFILL_INCREMENTAL=1` switch exists so B can be run; it is not
for production, only for isolating this variable.

**The second** compares the logits, which is the quantity the model actually
produces, using a real prompt — not made-up ids, which give a flat distribution
where the most likely token is decided by noise:

```
rango de los logits  33,42
diferencia maxima     0,68   (2,0% del rango)
token mas probable   68012 sin cache | 68012 con cache  ->  IGUAL
```

A cache attending to the wrong tokens wouldn't produce similar logits: it would
produce different ones, and the maximum would move. It picks the same token.

```bash
cargo run --release --features candle --example compare_logits   # correctness
cargo run --release --features candle --example check_cache      # performance
```
#### The real limit: there is only one cache

The cache lives inside the model, so there is **one**. While a single
conversation moves forward it works very well; when several arrive interleaved
— which is normal on WhatsApp — each one invalidates the other's.

Measured with two three-turn conversations, the same six requests, with only
the order changing:

| | time | tokens reused |
|---|---:|---:|
| sequential (all of A, then B) | 5541 ms | 244 |
| interleaved (A1 B1 A2 B2 A3 B3) | 9770 ms | **0** |
| | **+76%** | |

In the good case you can watch the advantage grow: turn 3 reused **93 of 107**
tokens and took 871 ms, *less* than turn 2 despite a longer prompt. In the
interleaved case it never hits once.

**The cache is fragile to a mismatched history.** It only reuses if the new
tokens extend the cached ones exactly, and the cached ones include what the
model generated. If the caller returns text different from what was generated —
a trim, a replacement, a summary — the prefix doesn't fit and nothing is reused.
The first version of this measurement built the history from invented text and
reported zero reuse **in the sequential case too**; it looked like the cache was
useless, and what was useless was the measurement.

#### Which model: measured, not estimated

Three models from the same family, same quantization (Q4_K_M), same machine.
Speed as the median of 3 runs — a single run once gave 21.6 and 32.3 tok/s for
the same model — and quality over 15 questions with verifiable answers: chained
arithmetic, following a requested format, knowledge, trick questions and
negations.

| | 0.5B | 1.5B | 3B |
|---|---:|---:|---:|
| on disk | 468 MB | 1066 MB | 2007 MB |
| resident | 504 MB | 1108 MB | 2051 MB |
| RAM/disk ratio | x1.08 | x1.04 | x1.02 |
| decode | ~32 tok/s | 19.3 | 10.0 |
| prefill | 846 ms | 814 ms | 1561 ms |
| 40-token response | ~2.1 s | ~2.9 s | ~5.6 s |
| correct | 7/15 | 8/15 | **10/15** |

**The 1.5B is not the sweet spot.** It costs 1.7x in speed and wins one
question out of fifteen, which is noise. The real jump is the 3B: it gets right
the ones the other two miss — a kilo of lead versus a kilo of feathers, the 206
bones, Canberra — and answers "penguin" where the 0.5B says "giraffes" and the
1.5B says "Uralic".

The choice is between **0.5B and 3B**, depending on whether latency or accuracy
weighs more. Neither is good: 10 out of 15 on basic questions is still weak.

The RAM/disk ratio dropped to **x1.02-x1.08** after removing the embedding
dequantization. It used to be x2, and on the 3B that would have been 4 GB
instead of 2.

```bash
cargo run --release --features candle --example compare_models
cargo run --release --features candle --example measure_memory -- <gguf> <tokenizer>
```

#### Quantization: Q3_K_M, not IQ3

**candle does not support the IQ formats.** Its type enumeration stops at the
classic k-quants — `Q2K Q3K Q4K Q5K Q6K Q8K` — and the IQ ones (IQ3_M, IQ2_XXS)
belong to llama.cpp. They're better per bit, but they're not available here.

Q3K is the closest thing that is. Same 3B model, same 15 questions:

| | on disk | resident | decode | 40 tokens | correct |
|---|---:|---:|---:|---:|---:|
| Q2_K | 1313 MB | — | 13.7 tok/s | ~4.0 s | **6/15** |
| **Q3_K_M** | **1644 MB** | **1687 MB** | **11.9 tok/s** | **~4.7 s** | **10/15** |
| Q4_K_M | 2007 MB | 2051 MB | 10.8 tok/s | ~5.2 s | 10/15 |

**Q3 gets as many right as Q4 with 363 MB less and 10% faster.** Fewer bytes
per weight means fewer bytes to read from RAM on every token, which is exactly
the bottleneck of generation on CPU.

Q2 no longer holds: the damage is visible at a glance. It says *"canossa"* for
Canberra, *1972* for 1969, *"20 mandile"* where a number belongs, and drops
French accents — *"debido à su arquitectura"* — in the middle of Spanish. That's
how an over-quantized model breaks: it doesn't fail with an error, it answers
things that almost look right.

Hence the default is **Qwen2.5-3B Q3_K_M**: 1687 MB resident and ~4.7 s per
40-token response.

```bash
cargo run --release --features candle --example compare_models -- --cuant
```

#### Parallelism: how many cores it really uses

The theoretical ceiling on CPU is `RAM bandwidth / model size`. With an
i5-10400 and dual-channel DDR4 that's ~40 GB/s over 0.47 GB: **~85 tokens/s**.

Measured separating prefill from decode — two runs with the same prompt and a
different token count, so prefill cancels out in the subtraction:

| threads | prefill | decode | effective cores | CPU consumed |
|---:|---:|---:|---:|---:|
| 1 | 3173 ms | 9.86 tok/s | 0.98 | 32.9 s |
| 2 | 1639 ms | 16.94 | 1.76 | 33.8 s |
| 4 | 1060 ms | 23.65 | 2.95 | 41.8 s |
| **6** | 933 ms | **25.58** | 4.06 | **53.2 s** |
| 12 | 854 ms | 26.08 | 6.23 | 81.8 s |

**From 6 to 12 threads you gain 2% and spend 54% more CPU.** The two threads of
one core compete for the same vector unit, and in work that is almost entirely
AVX2 the second one adds nothing. On a phone or a laptop that's heat and
battery in exchange for nothing.

That's why Cosxamix pins the pool to **physical** cores at startup. `num_cpus`
is needed because `available_parallelism()` returns 12 on a machine that has 6.
`RAYON_NUM_THREADS` still wins if someone sets it.

##### A figure that was wrong and needs correcting

**6.1 tokens/s** was reported earlier, and on that basis it was concluded that
the engine was at 7% of the ceiling and that speculative decoding wasn't worth
considering.

That figure was an artifact: it came from dividing generated tokens by the
**total** time, on one- and two-token responses where prefill takes almost
everything. It wasn't measuring generation speed. The real number is **~25
tok/s**, which is **30% of the ceiling**, not 7%.

It changes the conclusion: about 3x of headroom remains before hitting memory,
so speculative decoding — which trades weight reads for arithmetic — does have
room to pay off. Less than it would at 95% of the ceiling, but room.

#### 938 MB for a 469 MB model: the embedding table

§4.1 of the theory guide blames conventional loading for "doubling physical RAM
use" and proposes `mmap`. It was measured before being accepted, and the cause
was something else:

| | |
|---|---:|
| file on disk | 468.6 MB |
| quantized tensors | 463.0 MB |
| **embedding converted to f32** | **519.3 MB** |
| estimated | 982.3 MB |
| measured | 938.1 MB |

`from_gguf` was calling `tok_embeddings.dequantize()` on the whole table:
151936 x 896 in f32 weighs **more than the entire model**. And a forward pass
only needs the rows of the tokens coming in — 3.5 KB per token.

Now the table stays quantized and only those rows are dequantized. The table is
stored row-major and each row occupies a whole number of blocks (896 elements
in blocks of 32 is 28), so the bytes can be sliced without taking anything
apart.

**938 MB → 506 MB.** Resident memory ends up close to the file, which is what's
correct. `mmap` would have saved far less; the problem was never the copy.

##### A candle bug found along the way

The first version generated garbage. Isolating it: of seven rows requested, six
came out **exact** and only the first one of the slice failed — the very same
index 0 was fine when it wasn't first. It wasn't the index: it was the position.

The cause is in `candle-core`:

```rust
fn as_t_slice<T>(data: Cow<'_, [u8]>) -> &[T] {   // takes the Cow BY VALUE
    let ptr = data.as_ptr();
    unsafe { std::slice::from_raw_parts(ptr as *const T, ...) }
}                                                  // and frees the Vec here
```

With `Cow::Owned` the `Vec` is owned by the local variable and is freed on
return: the returned slice dangles, and the caller's `.to_vec()` reads freed
memory. Most of it is still intact — which is why six rows out of seven came
out fine — except the first bytes, which the allocator overwrites with its free
list.

Upstream doesn't hit it because it always passes `Cow::Borrowed`. The same is
done here, and the difference against dequantizing the whole table became
**exactly zero**.

```bash
cargo run --release --features candle --example measure_weights   # the diagnosis
cargo run --release --features candle --example check_embeddings  # row by row
```

#### Speculative decoding: implemented, measured and deleted

The idea: the 0.5B proposes K tokens and the 3B verifies them **in a single
pass**, reading its weights once instead of K times. It promises 2-3x.

It was implemented in full — draft, verification and cache truncation — and came
out **1.91x slower**. Rule 3 says what doesn't win gets deleted, so it was
deleted. This is written down so nobody retries it blind.

##### Why it cannot work here

The technique assumes that processing K positions costs about the same as one:
the weights are the same and get read once. Measured on the 3B:

| positions | ms | vs 1 |
|---:|---:|---:|
| 1 | 112 | 1.00x |
| 2 | 202 | 1.80x |
| 4 | 368 | 3.28x |
| 5 | 420 | 3.74x |
| 8 | 663 | 5.91x |

**It scales almost linearly.** The first suspect was the output layer — Qwen's
vocabulary is 151936 tokens — but it turned out to be only **15%** of the model,
and 15% does not explain 3.74x.

What does explain it: if the pass were limited by reading weights from RAM, 5
positions would cost ~1x. That they cost 3.74x means it is limited by
**compute**. And speculative decoding only pays off when the bottleneck is
memory — it converts weight reads into arithmetic, and here arithmetic is
already the bottleneck.

It fits the bandwidth ceiling measured separately: the engine runs at **30%** of
that ceiling. Until it gets near 100%, batching buys nothing.

On top of that, the 0.5B only agrees with the 3B on **28-50%** of tokens, so of
every 4 proposals 1.4 survive.

**It becomes worth revisiting** if the engine ever gets close to the memory
ceiling — with better matrix-multiplication kernels, for instance — or with a
draft model that agrees far more often. Not before.

##### What did survive from that work

Two pieces that were needed to attempt it and are worth having on their own:

* **The causal mask accounts for the cache.** The original was square and only
  valid with an empty cache. Fixed, a conversation's delta goes in as a batch
  instead of token by token: the interleaving benchmark went from 5422 to
  **4772 ms**, 12%, with individual turns halving.
* **`truncar_cache`.** Candle offers no way to truncate the attention cache. Now
  there is one, and it's what would enable any scheme that processes tentative
  tokens.

```bash
cargo run --release --features candle --example measure_projection
```

#### Per-session cache

The cache lives inside the model, so out of the box there is **one**. With
several interleaved conversations — normal on WhatsApp — each invalidated the
other's. Measured before fixing it, with two three-turn conversations and the
same six requests, changing only the order:

| | time | tokens reused |
|---|---:|---:|
| sequential | 5541 ms | 244 |
| interleaved | 9770 ms | **0** |
| | **+76%** | |

The two obvious ways out were measured and both discarded. **Loading the model
per session**: 931 MB per instance, twice the GGUF — ten chats would be 9.1 GB.
**Using the crate's API**: `ModelWeights` is not `Clone` and its `kv_cache` is
private inside `LayerWeights`; `quantized_qwen2` only exposes `from_gguf` and
`forward`.

So `quantized_qwen2.rs` is **in this repository**, under `src/model/`, with
three added methods — `tomar_cache`, `poner_cache`, `limpiar_cache` — and the
rest untouched. The model has one active cache; the other conversations' caches
are kept aside and swapped in on entry, moving pointers without copying data.

| after the change | time | tokens reused |
|---|---:|---:|
| sequential | 5422 ms | 244 |
| interleaved | 5357 ms | **244** |
| | no difference | |

Isolation is verified separately, because it's the dangerous failure: restoring
the wrong cache would make the model answer conditioned on another chat, with
no error and with an answer that sounds fine. `check_sessions` runs conversation
A alone and then interleaved with B, and compares: **all three turns come out
identical**.

The session travels in the request body as `cosxamix_sesion`, outside the
OpenAI dialect — which has no such concept, since every call there is
independent. A client that doesn't send it works the same, it just reprocesses
the whole prompt.

Capped at 16 sessions with least-recently-used eviction, and 4096 tokens per
conversation. What it costs **depends on the model**, and measuring it
corrected a figure this document had wrong:

| | KB per token | 16 sessions of 1000 tokens |
|---|---:|---:|
| 0.5B | 24.0 | 0.4 GB |
| **3B Q3** (the default) | **72.0** | **1.1 GB** |

36 layers against 24, and 128 head dimensions against 64: three times more per
token. This README said "~24 KB per token, 16 sessions are ~380 MB" without
saying which model, and with the default one that underestimates memory by
**three times**. Measured with `examples/measure_cache_disk.rs`.

Even so the trade still holds next to the 931 MB that **each** model instance
costs, which is the alternative that was ruled out.

#### The cost of keeping the model in this repository

It's 338 lines of someone else's code under our maintenance, which don't receive
upstream's fixes. When candle is updated, it has to be diffed against the
crate's. That's why the file starts by saying where it comes from, under what
license — MIT OR Apache-2.0, from huggingface/candle — and every change of ours
is marked `// COSXAMIX:`: without that marker, in six months nobody can tell our
change from a fix we're missing.

```bash
cargo run --release --features candle --example measure_interleaved  # performance
cargo run --release --features candle --example check_sessions       # isolation
cargo run --release --features candle --example measure_memory       # the 931 MB
```
---

## Several languages, one single boundary

Cosxamix isn't written in Rust alone. Rust is the main one, and the boundary is
open to whichever language wins the measurement:

| | what it brings | what's missing |
|---|---|---|
| **Zig** | one source covering x86-64, ARM64 and Android | nothing: it's already in |
| **Assembly** | the fused Q4_K kernel on x86-64, first in the dispatcher with `asm_q4k` (1.04x-1.08x over intrinsics). The Q3_K one (`asm_q3k`) ties with intrinsics, bit for bit | the NEON body, `kernels/q4k_neon.s`, and measuring it on ARM |
| **Fortran** | arrays without aliasing, which is why BLAS and LAPACK are still Fortran | a compiler on every platform, and Android is exactly the problem |
| **Ada / SPARK** | proving a kernel cannot overflow, instead of testing that it doesn't | a kernel critical enough to justify the toolchain |
| **C** | in with the `c` feature, built with `zig cc` and no libc. Per-function CPU extensions (`__attribute__((target))`), and on ARM a stable `vdotq_s32` — `sdot` — which Rust doesn't expose and Zig doesn't emit. On x86 its Q3_K loses 4% to Rust ([DECISIONES §21](DECISIONES.md)) | measuring on ARM the fused kernels with `sdot` (`kernels/neon_dotprod.c`), already written |

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

That shouldn't be read as more than it says: what falls back to scalar on ARM
are **these** kernels, `dot` and `rmsnorm`. The quantized matrix multiply,
where 99% of the work is, is covered by candle's own hand-written `neon.rs`.
The engine on a phone is not scalar; these two standalone kernels would be,
without Zig.

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
| full step | ~5.2 s |
| 1000 steps | ~1.4 h |
| resulting adapter | 2.1 MB |

**The base is held in f32, not quantized**, and that bounds the scope. candle's
`QMatMul` does not implement `bwd`: gradients cannot flow through a quantized
matmul, because the backward needs `dL/dy · W` — multiplying by `W`
untransposed — and candle only knows how to do `x · Wt`. The 0.5B in f32 is
~2 GB and fits; the 3B would be ~12 GB and does not. Training the 3B needs a
kernel that doesn't exist yet, and it's in the [ROADMAP](ROADMAP.md) §7.

The training model is an f32 reimplementation of the inference one, and that
they are the same **is checked**, because a mistake there — RoPE order, how KV
heads are repeated, the attention scale — would still make the loss go down,
optimizing against a model that isn't the one later served:

```bash
cargo run --release --features training --example check_training_model
```

It compares the logits of both paths on a real prompt. It doesn't demand
equality — one dequantizes to f32 and the other multiplies in integers — but
that they **pick the same token**. Measured: 3.2% of the range as the largest
difference, same token.

---

## License

**MIT** — see [LICENSE](LICENSE).

One file in this repository comes from another project: `src/model/qwen2.rs`,
copied from [candle](https://github.com/huggingface/candle) under MIT OR
Apache-2.0, with every change marked `// COSXAMIX:`. The details, and why it's
copied instead of using the crate's, are in [THIRD-PARTY.md](THIRD-PARTY.md).
