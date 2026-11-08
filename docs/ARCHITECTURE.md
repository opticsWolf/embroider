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
`runtime_info()`'s probe counts as a first touch. The venv scan follows
`VIRTUAL_ENV` with a `sys.prefix` fallback (unset = installs unknown, the
probe fields still report); duplicate spellings of one venv (`Lib/` vs
`lib/`) list once. Report keys are stable (`ort_dylib`,
`ort_dylib_exists`, `sibling_provider_dlls`, `cuda_usable`,
`ort_installs`, `cuda_runtime_dlls`, `load_dirs`, `virtual_env`).

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
  (int64) required; `token_type_ids` fed only if declared (v5's export
  doesn't declare it — where generic runners fail). Embedding output
  prefers `last_hidden_state`; rerank reads `logits` (`[B,1]`).
- **Pooling takes the last attended token** (`mask_sum - 1`, clamped
  ≥ 0).
- **`truncate_dim`** validated 32–1024 with a warning off the
  Matryoshka ladder (nano tops at 768). **`max_length`** validated
  1..=32768 (`None` = 8192 compat default; `MAX_LENGTH` /
  `MODEL_MAX_TOKENS` are exposed for the caller's window guard).
  Same (text, `max_length`) → same vector; short inputs are bit-identical
  at any limit, while longer inputs change once the old truncation lifts.
  A 32K-token forward is O(n²) memory — size the limit to the machine.
- **Batch encoding is sequential by design** (padded batches waste
  attention compute on variable-length docs). The GIL is released
  during encode. A failed `open()` is cached and re-raised — config
  errors fail fast once. `count_tokens()` reports true length through a
  truncation-free handle (shared helper with JinaV5); internal tokenize
  matches `tokenizer.encode(t, add_special_tokens=False)` — no
  transformers anywhere in the runtime path.
- **Vision slots** run one image per call (`seq = patches/4 + 15`,
  ≤1295 tokens). Jina: dynamic-grid graph, six inputs
  (`input_ids`/`attention_mask` int64, `pixel_values` float32,
  `vision_pos_ids`/`interp_indices` int64, `interp_weights` float32),
  host-computed grid tensors bit-identical to transformers
  (fixture-pinned); output `sentence_embedding` float32 `[1,768]`, then
  the same L2 → truncate → re-normalise as text. Every encode re-checks
  `seq == image_tokens + 15`, so a wrong tokenizer fails loudly instead
  of shifting the space.
  Gemma: vision tower (`pixel_values` + `pixel_position_ids` in,
  `image_features` out) fused through the text tower with framing ids
  and the native `sentence_embedding` head; host math ports the
  transformers processor exactly (multiples of 48, /255, patch 16,
  `[col,row]` positions, `[-1,-1]` pad). Explicit fp16-on-CPU is an
  error in both slots — the Jina graph stalls, the Gemma tower NaNs.
- **Decide slots** truncate state-side-first in both label layouts
  (text-first small-v1.0, labels-first v4-edge); options are the
  question and are never cut. Concatenation is manual id stitching, not
  the pair-tokenizer `OnlySecond` pattern — which would cut the wrong
  side. Laya collates marker columns to `max(kmax, 2)` (the exported
  head takes `topk(2)`); decode is `logits/t → softmax → unpermute →
  per-type shape` in float64, cast at the boundary; fitted per-bucket
  temperatures clamp to [0.5, 5.0] with queryable rejections.
  `option_order` must be an exact permutation — a silent misindex would
  attach probabilities to the wrong options.

## Engine serving notes

- **Rerank.** `score_long` slides windows (default: session cap, 50%-
  overlap steps) with MaxP aggregation; chunk planning is pure and
  weight-free unit-tested, and single-window inputs score exactly like
  `score_one` (e2e-pinned ≤1e-6). CUDA is opportunistic (~1.5x at batch,
  worse at b1 — short forwards are overhead-bound).
- **Decide.** `Choice` is single-label softmax; `Score` is multi-label
  sigmoid, native and unthresholded. `Noul` is declared but
  loud-unsupported on GLiClass graphs. Temperature is identity — the
  calibration lives in the v4 graph, not the port.
- **Laya.** `predict_long` scans states past the cap: token windows
  (default `max(64, max_len - head_max_len - 8)` — the checkpoint's own
  state budget — clamped to the room the questions leave, 50%-overlap
  steps), one shared session run for all windows, per-type aggregation
  (noul: strongest P(true); choice/score: most-confident window). It
  returns the deciding window's decision plus its token span — that
  window's numbers, not calibrated document numbers. A `BinningMap`
  overlay remaps `answer_confidence` for fitted buckets; no checkpoint
  ships one.
- **Decider-2B.** Lettered choices (`(A) …`, one label token each past
  10 options) with a trailing `Answer: (` slot scored over the 255
  label ids; score questions become one yes/no row per level (isolated
  fit). Noul carries BOTH criteria (`yes:` / `no:`). CPU is slow-tier
  (~1.5–2 s/row); CUDA runs the parallel executor (see above).

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

Goldens are the contract proof. The fixture inventory:

- `fixtures/golden_jina_v5_text_small.json` — 12 vectors (4 texts ×
  Query/Document × dims 64/512) + exact token counts, generated with
  embroider 0.1.3 / onnxruntime 1.29.0 CPU, byte-identical green on
  1.31.1. Unchanged by 0.1.4 (`max_length`) and 0.1.5 (`precision`,
  `cpu_arena`) for short inputs at defaults. Assert live vectors at
  `abs=1e-6`.
- `fixtures/vision/` — `target_sizes.json`, `host_tensors.json`,
  `pixel_cases.json` (from the spike's verified `grid.py`); the Rust
  port asserts bitwise equality on all 31 grids and the pixel pipeline
  on 3 cases. `fixtures/vision/e2e/` (5 real figures + torch goldens)
  backs the ignored end-to-end parity test (cos ≥ 0.9999 fp32 /
  ≥ 0.999 fp16).
- `fixtures/golden_rerank_{minilm,mmarco}.json` — torch parity
  2.9e-6/7.7e-6 at pinned revisions; re-recorded from 1.29.0 at 0.13.0
  (max drift 2.9e-6). The ignored e2e asserts goldens plus
  per-language relevant-beats-irrelevant, single-window equivalence,
  MaxP sanity, and validation order.
- `fixtures/golden_decide_gliclass_{small,v4edge}.json` — small abs
  1e-6, v4 abs 0.03 (the embedded calibrator accumulates kernel-order
  drift); exact-match labels; B0 torch probes exact (small) / Δ 0.028
  (v4). The ignored e2e asserts goldens plus the layout proof per
  recipe.
- `fixtures/golden_laya_td.json` — agent-built rows + raw float64
  reference decode, 7 cases incl. permutation, solo, and relevance
  probes; ignored e2e asserts exact input-id + marker equality and 1e-6
  on every number.
- `fixtures/golden_decider2b.json` — 8 cases incl. the 12-option wide
  path and a mixed multi batch; exact input ids and 5e-4 answers on CPU
  *and* CUDA (the CUDA leg gates the executor switch).
- `fixtures/gemmavision/` — sentence-transformers references for 3
  figures; ignored e2e asserts cos = 1.0 fp32 CPU/CUDA and fp16 CUDA
  plus a cross-modal rank pin.

Regenerate only on an intentional contract change — which is a new minor version
plus a re-index-everything notice. ## Consumers

- **okfgraph** holds a lazy proxy: construction validates the wheel
  import and device string eagerly, but the session opens on the first
  real encode — PPR search, budgeted reads, diff, and doctor stay cold.
  It vendors the goldens and asserts live vectors (`test_parity.py`),
  reports cache state through `model_info()` (over `cache_info`), and
  routes local files through `OKFRouter(model_path=..., tokenizer_path=...)`
  (both or neither; missing files raise `FileNotFoundError`).
- **bobine** builds its own converter-model sessions from
  `SessionPolicy::ort_defaults()` + `apply_providers` + `cuda_available()`
  + `report()` — never touching the loaders — and reports cache state
  through `cache_info_files` (its `model_status()`). Sharing runtime
  policy never means sharing one inference session.

Releases are immutable: the tag must
equal both version files, and `embroider` publishes before consumers
move their pins.

Rust unit tests (111, pure — no network, no dylib, no tokenizer file)
cover device/precision parsing, precision-follows-device resolution,
FP16-mirror selection, model-id parsing, the provider matrix, task-prefix
idempotence, the L2 → truncate → re-normalise math, contract constants,
`open()`/`open_files()` validation firing before I/O (dims, `max_length`),
offline cache inspection (registry artifact resolution, fake-cache
hit/partial/full miss, `auto`/bad-id refusal; the Gemma entry incl.
contract, fp32/fp16 artifacts, mirror revision, prompt/mean-pool
selection; generic `cache_info_files` required/optional semantics, pinned
revision, delegation parity), the vision registry entry, the resize
contract (incl. banker's rounding and aspect rejection), host tensors
bit-identical on all 31 grids, the pixel pipeline bit-identical on 3
cases, and the vision fp16-on-CPU refusal. Python
parity lives with okfgraph (`tests/test_parity.py`, slow): Rust output
against a numpy/transformers replication at ≤ 1e-5.
