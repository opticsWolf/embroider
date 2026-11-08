# embroider — Architecture

How embroider works, for contributors and consumers debugging the space.
The user manual lives in [quickref.md](quickref.md); this doc records
the design decisions and measurements behind it.

## Module layout

| Module | Role |
|---|---|
| `providers` | provider-name matrix (`cuda`/`rocm`/`directml`/`openvino`/`coreml` + implicit `cpu`) + clone-and-fallback application (`apply_providers` = arena on; `apply_providers_with_arena` carries the flag) |
| `probe` | corrected CUDA availability check (`OnceLock`-cached) |
| `policy` | `DeviceReq` (`auto`/`cpu`/`cuda`) + `Precision` (`auto`/`fp32`/`fp16`/`int8`/`q8`) + explicit `SessionPolicy` (`text_embed()`, `rerank_pair()`, `decide_pair()`, `laya()`, `decider2b()`, `ort_defaults()`) |
| `acquire` | validated `owner/name` parsing, HF client, tokenizer-only fetch, FP16-mirror selection for the default id, offline cache inspection (`cache_info`, generic `cache_info_files`), embedding registry (`ModelSpec`) + task registry (`TaskSpec`: rerank/decide), registration overlay |
| `error` | anyhow-based error plumbing (`ort` errors stringified at boundaries) |
| `diag` | runtime inventory — `ORT_DYLIB_PATH` value, venv installs, CUDA DLLs, usability probe (`runtime_info`, `prepare_cuda_dirs`, `save_runtime_info`, `configure_runtime`) |
| `jina` | `JinaV5` + `TokenizerHandle` — the frozen embedding contract |
| `vision` | `JinaV5Vision` — dynamic-grid vision into text-nano space |
| `gemma` | `GemmaVision` — vision tower + text-tower fusion into Gemma space |
| `rerank` | `Reranker` — cross-encoder scoring (pair seam + `score_long` windowing) |
| `decide` | `Decider` — typed decisions (`Choice`/`Score`, truncate-state-side-first, both label layouts) |
| `laya` | `LayaDecider` — marker-layout contract, fitted temperatures, `predict_long`, binning overlay |
| `decider2b` | `Decider2B` — slot-layout contract (Qwen3.5-2B decoder, parallel executor on CUDA) |

The default (pure-Rust) build is Python-free — no `pyo3` in downstream
trees; the `extension-module` Cargo feature gates the PyO3 bindings and
is enabled only for wheel builds (maturin), the same pattern bobine uses.

## Using embroider without Jina

The policy layer is model-agnostic — bobine uses it for converter models
(RapidOCR, layout, tables, formulas) without ever touching `JinaV5`.
The recipe, in full (reference implementation: bobine's
`engine::session_builder`):

```rust
let builder = ort::session::Session::builder()?;
let tuned = embroider::SessionPolicy::ort_defaults().apply(builder)?;
let session = embroider::apply_providers(tuned, providers)?.commit_from_file(path)?;
```

plus `embroider::cuda_available()` for the auto-GPU decision and
`embroider::report()` for the which-binary-serves-what diagnostic.

The boundary is deliberate:

| embroider owns | Consumer owns |
|---|---|
| Session tuning (`SessionPolicy`), provider fallback (`apply_providers`), CUDA probe, dylib report, `owner/name` parsing | Model fetch (own HF client, own repo constants), preprocessing, IO wiring, decoding |

Generic fetch + session-build helpers (a model-agnostic "N files from
repo R" to complement the Jina-shaped acquisition) are a natural
future addition — additive, no contract risk. Inference itself stays
with the consumer by design.

## Runtime discovery

`ort` loads dynamically (`load-dynamic`, same pin as bobine:
`2.0.0-rc.13`). Resolution order: `ORT_DYLIB_PATH` first (user override
always wins), else the pip-installed `onnxruntime`/`onnxruntime-gpu`
build when unset. okfgraph's `resolve_ort_dylib()` runs before the
native module is imported, so bobine and embroider share **one** ORT
binary — no version/CUDA drift between ingest and import.

Since 0.14.0 the loader path is self-servicing: before the first CUDA
session, `embroider` prepends the venv's NVIDIA wheel directories, the
`onnxruntime` capi dirs, and the system CUDA/cuDNN roots to `PATH`
when missing, and delivers the venv's best dylib via `ORT_DYLIB_PATH`
only when unset — a fresh process no longer depends on ambient `PATH`
for `cudnn64_9.dll`, and an explicit pin still wins. Exposed as
`embroider.prepare_cuda_runtime()` (idempotent, never throws) and
`embroider::prepare_cuda_dirs()` on the Rust side; the same dirs stay
listed in `runtime_info()["load_dirs"]`.

`ORT_DYLIB_PATH` is first-touch: the dylib binds on first `open()`
(the probe counts), so set it before anything opens a session — even
`runtime_info()`'s probe counts as a first touch.

**Pitfall: stale `onnxruntime.dll` on Windows.** Windows boxes can carry
a stale `C:\Windows\System32\onnxruntime.dll` (v1.17.1 in the wild). With
`ORT_DYLIB_PATH` unset, `ort` may load it and die with `BadVersion`, plus
an abort at shutdown from ort's exit handler. Point `ORT_DYLIB_PATH` at
a modern build — e.g. the venv's `onnxruntime/capi/onnxruntime.dll`.
Same pitfall bobine documents in its `docs/benchmarks.md`.

## Session and threading policy (measured)

Tuning is `Level3`, intra = physical-cores/2, inter = 1 — kept because
it measured fastest, not because it was inherited. Reference box:
Windows, 32 logical cores, CPU-only ORT 1.29, warm model cache, best-of-5
reps on 4 fixed docs (short → ~400 tokens):

| Config | Session cold open | `encode_batch` (4 docs) | Notes |
|---|---|---|---|
| Level3, intra=16, inter=1 (**current**) | 4.7 s | **375 ms** | kept |
| Level1, intra=16, inter=1 | 5.5 s | 433 ms (+15%) | slower *and* bit-different vectors |
| Level3, intra=32, inter=1 | 4.5 s | 411 ms (+10%) | full-logical loses to phys/2 (SMT contention) |
| 4× `encode_one` vs 1× `encode_batch` | — | 389 vs 375 ms | one boundary crossing saves ~3%; sequential stays |
| Tokenizer-only cold open | 0.5 s | — | 9× cheaper than session open; budgeted reads stay cold |

Consequences:

- **Do not mix tuning in one index.** Level1 vs Level3 fuse the graph
  differently — bits differ (hashes diverged at 1e-8 formatting). Same
  model + same build + same tuning, or re-embed.
- **Sequential batching stays.** Padded batching would waste attention
  on variable-length docs to save ~14 ms of boundary overhead — not
  worth the numerics risk.

`SessionPolicy::ort_defaults()` exists for consumers (bobine's
converter-model sessions) that never tuned — policy is data, never a
forced default. Re-measure on new hardware/ORT before changing policy.

The one exception is Decider-2B on CUDA: the sequential planner aliases
a shape buffer on that graph (ORT #4829 class — a chunk-scan Reshape
reads a zeroed dim and fails on the first batch), the C-level
`enable_mem_reuse=false` is unreachable from the C API, and an explicit
`inter_op_num_threads=1` re-triggers it — all probed. So
`SessionPolicy::decider2b` runs the parallel executor with default
thread pools, and the CUDA e2e leg gates that switch.

## The frozen contract

One pipeline per model: task prefix → tokenize → ONNX forward →
pooling → L2 → Matryoshka truncate → re-normalise.

- **Pooling and prompts are per-model.** Jina ids use last-token
  pooling with `Query:`/`Document:` prefixes; `google/embeddinggemma-2`
  uses mean pooling with Gemma prompts (`task: search result | query:`
  / `title: none | text:`). Swapped prefixes silently fork the space.
- **Session IO is discovered at load.** `input_ids` + `attention_mask`
  required; `token_type_ids` fed only if declared (v5's export doesn't
  declare it — where generic runners fail). Embedding output prefers
  `last_hidden_state`; rerank reads `logits`.
- **Pooling takes the last attended token** (`mask_sum - 1`, clamped
  ≥ 0); `input_ids`/`attention_mask` feed as int64.
- **`truncate_dim`** validated 32–1024 with a warning off the
  Matryoshka ladder (nano tops at 768). **`max_length`** validated
  1..=32768 (`None` = 8192 compat default); same (text, `max_length`)
  → same vector, short inputs bit-identical at any limit. A 32K-token
  forward is O(n²) memory — size the limit to the machine.
- **Batch encoding is sequential by design** (padded batches waste
  attention compute on variable-length docs). The GIL is released
  during encode. A failed `open()` is cached and re-raised — config
  errors fail fast once.
- **Vision slots** run one image per call. Jina: dynamic-grid graph,
  six inputs, host-computed grid tensors bit-identical to transformers
  (fixture-pinned), every encode re-checks `seq == image_tokens + 15`.
  Gemma: vision tower (`pixel_values` + `pixel_position_ids` in,
  `image_features` out) fused through the text tower with framing ids
  and the native `sentence_embedding` head; host math ports the
  transformers processor exactly (multiples of 48, /255, patch 16,
  `[col,row]` positions, `[-1,-1]` pad). Explicit fp16-on-CPU is an
  error in both slots — the Jina graph stalls, the Gemma tower NaNs.
- **Decide slots** truncate state-side-first in both label layouts
  (text-first small-v1.0, labels-first v4-edge); options are the
  question and are never cut. Laya collates marker columns to
  `max(kmax, 2)` (the exported head takes `topk(2)`); decode is
  `logits/t → softmax → unpermute → per-type shape` in float64, cast
  at the boundary; fitted per-bucket temperatures clamp to [0.5, 5.0]
  with queryable rejections.

## Registry and precision design

Builtins are frozen `&'static` specs (embedding `ModelSpec`, task
`TaskSpec`); callers register new models under known kinds without a
source change (`register_task` / `register_model`, or
`load_registry_file`). Rules: ids must be `owner/name`; builtin and
duplicate ids are refused loudly (mirrors take their own ids); unlisted
precision tiers fall back to fp32; `Auto` never resolves to a
caller-supplied non-fp32 tier — serving unmeasured weights takes two
explicit actions (register the tier, request it by name), and `open()`
warns with the entry's `registered_from` attribution. Every builtin
artifact pins its revision (commit SHA, verified by unit test); the
caller revision always wins. New *input contracts* stay code — a new
kind needs a Rust loader (see
[model-registry-record.md](model-registry-record.md) layer 2).

Precision policy: `auto` follows the landed device (CUDA→FP16,
CPU→FP32). Explicit fp16-on-CPU warns for text, errors for vision.
`q8` (weight-only int8, activations fp32) is an explicit deployment
choice wherever listed — decider-2b, v4-edge, all three Laya, both
rerankers; GLiClass-small failed the 0-flip gate and stays fp32-only.
Embeddings never take int8/int4 (quant shifts the vector space — needs
retrieval re-eval, not parity); fp16 mirrors ship for CUDA serving only,
parity-gated.

## Conformance and testing philosophy

Goldens are the contract proof: `fixtures/golden_jina_v5_text_small.json`
pins the vector space (4 texts × Query/Document × dims 64/512 +
token counts, `abs=1e-6` — catches wrong model, pooling, prefix, or
truncation, immune to cross-CPU noise); vision host math is pinned
bitwise under `fixtures/vision/`; every engine ships fixture + ignored
env-gated e2e (`tests/*_e2e.rs`, no weight downloads in CI). Regenerate
only on an intentional contract change — which is a new minor version
plus a re-index-everything notice. Releases are immutable: the tag must
equal both version files, and `embroider` publishes before consumers
move their pins.

Rust unit tests (111, pure — no network, no dylib, no tokenizer file)
cover parsing, resolution, registry validation (including the revision
completeness test), the truncation/normalisation math, contract
constants, validation-before-IO, and offline cache inspection. Python
parity lives with okfgraph (`tests/test_parity.py`, slow): Rust output
against a numpy/transformers replication at ≤ 1e-5.
