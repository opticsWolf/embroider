# embroider
### Jina v5 embeddings + cross-encoder rerank over ONNX Runtime (Rust core, PyO3)

[![CI](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml/badge.svg)](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/embroider)](https://crates.io/crates/embroider)
[![docs.rs](https://img.shields.io/docsrs/embroider)](https://docs.rs/embroider)
[![PyPI](https://img.shields.io/pypi/v/embroider)](https://pypi.org/project/embroider/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/crates/l/embroider)](LICENSE-MIT)
[![Website](https://img.shields.io/badge/website-embroider-blue)](https://opticswolf.github.io/embroider/)

## Contents

- [What it is](#what-it-is)
- [Models](#models) — registry, precisions, custom repos, licences
- [Install](#install)
- [Module layout](#module-layout)
- [Using embroider without Jina](#using-embroider-without-jina) — session-policy recipe for other models
- [Runtime: ONNX Runtime discovery](#runtime-onnx-runtime-discovery)
- [Text embeddings](#text-embeddings)
- [Vision embeddings](#vision-embeddings)
- [Rerank](#rerank) — cross-encoder pair scoring (`Reranker`, 0.4.x)
- [Decide](#decide) — typed decisions (`Decider`, 0.5.x) · [Laya](#laya) — marker layout (`LayaDecider`, 0.7.x) · [Decider-2B](#decider-2b) — slot layout (`Decider2B`, 0.10.x)
- [Running without downloads](#running-without-downloads) — tokenizer counts, offline cache checks, local files
- [Session/threading policy](#sessionthreading-policy-measured)
- [Failure policy](#failure-policy) · [Contract notes](#contract-notes) · [Conformance](#conformance) · [Testing](#testing)
- [Quick reference](docs/quickref.md) — copy-paste Python/Rust surface

## What it is

`embroider` turns text — and images — into vectors: Jina v5 embeddings
served through ONNX Runtime from a Rust core, with optional PyO3 bindings
for Python. Since 0.4.x it also scores them: cross-encoder rerankers
(`Reranker`) reorder retrieval candidates with pair-encoded logits from
the same runtime — no second engine, no torch anywhere in the chain.

### One frozen contract

Task prefix → tokenize → ONNX forward → last-token pooling → L2 →
Matryoshka truncate → re-normalise. Every consumer lands in the same
vector space, or gets a loud error instead of a silent fork.

### Two consumers

**okfgraph** uses the full Jina v5 text + vision contract.
**bobine** (PDF/Office → Markdown) reuses the ONNX plumbing — session
policy, provider fallback, CUDA probe — while owning its own
converter-model sessions (RapidOCR, layout, tables, formulas).

### One backend

No torch / transformers / optimum in the hot path, so every vector in an
index stays directly comparable. If something can't be embedded exactly
to contract, embroider errors loudly rather than quietly mixing vector
spaces.

## Models

Three blessed embedding entries (`src/acquire.rs`), all non-commercial
weights (licences tabled in [docs/compat.md](docs/compat.md)). Download
bytes, per-model latencies and CPU-vs-CUDA guidance live in the
[Model sizes & serving](docs/compat.md#model-sizes--serving) table —
read it before choosing what a consumer downloads:

| Registry id | Artifact repo | Precisions | Used for |
|---|---|---|---|
| `jinaai/jina-embeddings-v5-text-small-retrieval` (default) | itself (fp32) + `opticsWolf/…-onnx-fp16` mirror (fp16) | fp32, fp16 | General text graphs |
| `jinaai/jina-embeddings-v5-text-nano-retrieval` | itself (in-repo fp32 + fp16 + int8) | fp32, fp16, int8 | Compact graphs; **required text partner** for vision |
| `jina-v5-omni-nano-retrieval-vision` | `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` | fp32, fp16 | Image bytes → text-nano space (`text_partner` pinned) |

Plus the task registry (`TaskSpec`, same file) — rerankers, which have
no vector space and take no `truncate_dim`. The first entries are
Apache-2.0, the registry's first commercially-usable weights:

| Registry id | Artifact repo | Precisions | Used for |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | itself (`onnx/model.onnx`) + `opticsWolf/ms-marco-MiniLM-L-6-v2-q8-onnx` | fp32, q8 | Rerank, English |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | itself (`onnx/model.onnx`) + `opticsWolf/mmarco-mMiniLMv2-L12-H384-v1-q8-onnx` | fp32, q8 | Rerank, multilingual (100+ langs) |
| `knowledgator/gliclass-small-v1.0` | itself (`onnx/model.onnx`) | fp32 | Typed decisions (Choice/Score), 144M |
| `knowledgator/gliclass-v4-edge` | itself (`onnx/model.onnx`) + `opticsWolf/gliclass-v4-edge-q8-onnx` | fp32, q8 | Typed decisions + instruction prompts, labels-first, 440M |
| `convaiinnovations/laya-typed-decisions` | `opticsWolf/laya-typed-decisions-fp32-onnx` | fp32, q8 | Typed decisions, marker layout, fitted temps |

### Custom models

Unknown ids fall through to the legacy path: the id itself is treated as
an HF repo holding `onnx/model.onnx` (`lookup_model` → `None`,
`artifact_for` default layout). Any downstream consumer can point
embroider at a compatible repo with no registry change. Fully local
weights skip acquisition entirely — see
[Running without downloads](#running-without-downloads).

### Registering models

New models under known kinds need no source change (`register_task` /
`register_model`, or `load_registry_file("models/extra.json")` for
the file form — `{"registry_version": 1, "tasks": [...],
"models": [...]}`). Kinds are the existing ones (`rerank`,
`decide`, `decide-laya`, `decide-2b`, `embed`); a new input contract
still needs a Rust loader (see `docs/model-registry-draft.md`
layer 2). Rules: ids must be `owner/name`, builtin ids and
duplicates are refused loudly (mirrors take their own ids), unlisted
precision tiers fall back to fp32, and caller-supplied non-fp32 tiers
run flagged-unmeasured (`available_models` shows `source` /
`measured`; `open()` at such a tier warns with the entry's
`registered_from` attribution — the drift is the caller's gate).
`export_registry()` dumps the overlay for round-tripping.
`probe=True` runs an offline `cache_info` after registering and
reports what `open()` would need.

## Install

PyPI wheels (Linux / Windows / macOS-arm64, Python 3.11–3.13) — `okfgraph`
pulls it in automatically; standalone:

```bash
pip install embroider
```

From source (Rust toolchain + maturin; `maturin develop` needs pip, which
uv venvs lack — build the wheel and install it instead):

```bash
maturin build --release
uv pip install --python <venv> target/wheels/embroider-*.whl --reinstall
```

## Module layout

| Module | Role |
|---|---|
| `providers` | provider-name matrix (`cuda`/`rocm`/`directml`/`openvino`/`coreml` + implicit `cpu`) + clone-and-fallback application (`apply_providers` = arena on; `apply_providers_with_arena` carries the flag) |
| `probe` | corrected CUDA availability check (`OnceLock`-cached) |
| `policy` | `DeviceReq` (`auto`/`cpu`/`cuda`) + `Precision` (`auto`/`fp32`/`fp16`/`int8`) + explicit `SessionPolicy` (`text_embed()`, `rerank_pair()`, `decide_pair()`, `laya()`, `decider2b()`, `ort_defaults()`) |
| `acquire` | validated `owner/name` parsing, HF client, tokenizer-only fetch, FP16-mirror selection for the default id, offline cache inspection (`cache_info`, generic `cache_info_files`), embedding registry (`ModelSpec`) + task registry (`TaskSpec`: rerank/decide) |
| `error` | anyhow-based error plumbing (`ort` errors stringified at boundaries) |
| `diag` | `OrtReport` — `ORT_DYLIB_PATH` value + CUDA usability for logs |
| `jina` | `JinaV5` + `TokenizerHandle` — the frozen embedding contract |
| `rerank` | `Reranker` — the cross-encoder scoring contract (pair seam + `score_long` windowing) |
| `decide` | `Decider` — the typed-decision contract (`Choice`/`Score` questions, truncate-state-side-first) |
| `decider2b` | `Decider2B` — the slot-layout contract (Qwen3.5-2B decoder, `input_ids` + `slot_pos`, 32k ctx, parallel executor on CUDA) |

The default (pure-Rust) build is Python-free — no `pyo3` in downstream
trees; the `extension-module` Cargo feature gates the PyO3 bindings and is
enabled only for wheel builds (maturin), the same pattern bobine uses.

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

## Runtime: ONNX Runtime discovery

`ort` loads dynamically (`load-dynamic`, same pin as bobine:
`2.0.0-rc.13`). Resolution order: `ORT_DYLIB_PATH` first (user override
always wins), else the pip-installed `onnxruntime`/`onnxruntime-gpu`
build when unset. okfgraph's `resolve_ort_dylib()` runs before the native
module is imported, so bobine and embroider share **one** ORT binary — no
version/CUDA drift between ingest and import.

## Text embeddings

### Lazy sessions

`JinaV5.open` (model download + ONNX session build) is the single
expensive step. OKFgraph therefore holds a lazy proxy: construction
validates the wheel import and device string eagerly, but the session
opens on the first real encode — PPR search, budgeted reads, diff, and
doctor stay cold.

`JinaV5.open(model_id, revision=None, cache_dir=None, truncate_dim=512,
device="auto", max_length=None, precision=None, cpu_arena=False)`:
`max_length` caps encodes at 1..=32768 (`None` = 8192 compat default;
`.max_length` reports the effective limit), `precision` selects weights
(`auto`/`fp32`/`fp16`, `None` = `auto`; `.precision` reports the
effective choice), `cpu_arena=False` disables the CPU arena allocator
(8x lower peak RSS for ~1.4x encode time, measured on the FP32 text
model).

A failed session open is cached and re-raised — configuration errors fail
fast once, not once per encode.

### Weight precision (`auto` by default)

`auto` follows the *resolved* device (CUDA → FP16, CPU → FP32), so
CUDA-requested-but-missing degrades to FP32 weights instead of stranding
FP16 on CPU (FP16-on-CPU runs >40x slower than FP32-CPU — measured).
`fp16` maps the default model id to the published FP16 mirror repo
(`opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16`);
explicit model ids (omni tower, mirrors) always win untouched. An explicit
`fp16`-on-CPU warns loudly but is honoured. Do not mix precisions in one
index — FP32 vs FP16 weights shift vectors, same as mixing tuning levels.
`q8` selects the weight-only int8 tier where the registry lists one
(decider-2b, v4-edge, all three Laya, both rerankers — GLiClass-small
failed the 0-flip gate and stays fp32-only). Unlike fp16, q8 runs on any
device (activations stay fp32); `auto` never resolves to q8 — it is an
explicit deployment choice, same rule as int8.

### Memory: CPU arena off by default

`cpu_arena=False` registers the CPU execution provider explicitly with
its arena allocator disabled — measured 8x lower peak RSS (15.3 → 1.9 GB
on FP32) for ~1.4x encode time. Pass `True` only when peak throughput
beats memory pressure. Vision slots likewise run with the CPU arena off.

## Vision embeddings

Image bytes into the text-nano space (text-nano graphs only — the
registry's `text_partner` pin refuses anything else).

### Opening and encoding

`JinaV5Vision.open(model_id="jina-v5-omni-nano-retrieval-vision",
truncate_dim=512, device="auto", precision=None, gpu_mem_limit=None)`
embeds one already-resized RGB image (`encode_image(rgb_bytes, h, w)`
where `(h, w)` is its own `vision_target_size(h, w)` — Pillow bicubic
resize on the caller side; anything else is rejected rather than
embedded under the wrong resolution contract). Image vectors compare
against text-nano text vectors only (`text_partner` in the registry).
The dynamic-grid graph takes six inputs; the three grid tensors
(`vision_pos_ids`, `interp_indices`, `interp_weights`) are computed on
the host, bit-identical to transformers (fixture-pinned). `auto`
precision → fp16 on CUDA / fp32 on CPU; explicit fp16-on-CPU is an
error (the graph stalls on CPU — it doesn't merely run slow, so
fail-fast beats a hung batch). Sessions use the vision-slot policy
(ORT defaults, never the text tuning), CUDA
`arena_extend_strategy=kSameAsRequested`, optional `gpu_mem_limit`
bytes, CPU arena off. Every encode re-checks `seq == image_tokens +
15` so a wrong tokenizer fails loudly instead of shifting the space.

## Rerank

Cross-encoder pair scoring: one `(query, document)` forward → one
relevance logit (higher = more relevant). Same ORT runtime, same session
policy machinery, no vector space — scores reorder candidates, they never
enter an index.

`Reranker.open(model_id, revision=None, cache_dir=None, device="auto",
max_length=None, precision=None, cpu_arena=False)`: `max_length` caps
pairs at 1..=512 (`None` = the registered ceiling; `.max_len` reports
the effective limit), `precision` selects weights (`auto`/`fp32`/`q8`; q8 tiers are listed where measured — the quant-spike rule, so `auto`
lands fp32 everywhere for rerank today). `score_one(query, doc)` scores a pair;
`score_many([(q, d), …])` scores a batch in one padded session run
(order-preserving); `count_tokens(text)` reports true length without
specials through a truncation-free handle (shared helper with JinaV5).

Truncation is truncate-doc-side-first: the query is never cut to fit the
document, and a query that alone fills the cap raises instead of scoring
garbage. The loader feeds `token_type_ids` only when the graph declares
them (MiniLM 3-input vs mMARCO 2-input, discovered at load) and reads
the `logits` output. Documents past the pair cap go through
`score_long(query, doc, window=None, stride=None)` — sliding windows
with 50%-overlap steps by default and MaxP aggregation (a document is
relevant if any part answers); chunk planning is weight-free and
unit-tested, and single-window inputs score exactly like `score_one`
(e2e-pinned).

Goldens live in `fixtures/golden_rerank_*.json` (ORT 1.29.0 CPU, abs
1e-6; torch parity 2.9e-6/7.7e-6 at pinned revisions) with the ignored
`tests/rerank_e2e.rs` asserting them plus per-language relevant-beats-
irrelevant ordering. Precision notes from the spike: fp16 rejected
(slower + drift), int8 gated on a wider harness, CUDA opportunistic
(~1.5x at batch, worse at b1) — see
docs/[rerank-typed-decision-plan.md](docs/rerank-typed-decision-plan.md).

## Decide

Typed questions over a state: one calibrated distribution per question,
single forward pass. `Decider.open(model_id, revision=None,
cache_dir=None, device="auto", max_length=None, precision=None,
cpu_arena=False)`: `max_length` caps sequences at 1..=512 (`None` = the
registered ceiling; `.max_len` reports the effective limit). `decide(state,
questions)` answers a batch in one padded session run, order-preserving;
`decide_many([(state, question), …])` batches across states.

`Question::Choice` is single-label (stable softmax — exactly one option
wins); `Question::Score` is multi-label (sigmoid, native and
unthresholded — the consumer fits thresholds, because the B0 probe showed
raw multi-label sigmoids systematically over-confident: `quantum physics`
at 1.0 on a football text). `Question::Noul` is declared but
loud-unsupported — no registered model serves none-of-the-above, so it
errors instead of degrading into a threshold nobody fitted.

Input construction is contract, pinned like the rerank pair seam.
Small-v1.0 is text-first: `state + (<<LABEL>>option)* + <<SEP>> +
instructions` (`prompt_first=false`); v4-edge is labels-first:
`(<<LABEL>>option)* + <<SEP>> + instructions + state`
(`prompt_first=true`, recorded per model in the registry — `open()`
resolves it, `open_files` takes it caller-side). Truncate-state-side-first
with manual id concatenation in both layouts (the pair-tokenizer
`OnlySecond` pattern would cut the wrong side). Temperature is identity
— neither model ships fitted temps (v4's calibration lives in its
graph). Goldens live in `fixtures/golden_decide_gliclass_{small,v4edge}.json`
(ORT 1.29.0 CPU; small abs 1e-6, v4 abs 0.03 — the embedded calibrator
accumulates kernel-order drift; exact-match labels, torch exact / Δ
0.028 on the B0 probes) with the ignored `tests/decide_e2e.rs`
asserting them plus the layout proof per recipe. GLiClass-small is
the seam-proving spike; v4-edge is the better model (calibrated,
instruction prompts); Laya (native typed questions, descriptions,
level-scores, noul) is served by `LayaDecider` — see `## Laya` below.

## Laya

Marker-layout typed decisions: the state goes in as plain tokens and
the questions arrive as **marker positions** — 5 inputs (`input_ids`,
`attention_mask`, `marker_pos`, `marker_mask`, `qtype`), 2 outputs
(`logits`, `act_logits`). `LayaDecider.open(model_id, ...)` serves the
three registered `convaiinnovations/laya-*` checkpoints from the
`opticsWolf/laya-*-fp32-onnx` self-exports (graph + sidecar +
tokenizer + `rl_agent_config.json`, all from the artifact repo);
`open_files(onnx_path, tokenizer_path, config_path, ...)` takes
explicit files — the config is required, never optional: it carries
the checkpoint truth (ceilings, temperatures, layout check) and
caller-supplied temps would silently fork the calibration.

`LayaQuestion::{Choice, Score, Noul}` mirrors laya's qdefs, not
`Question`: Choice carries labels with optional descriptions
(single-label softmax), Score carries ordered LEVELS (one distribution
over levels — expected value in `.value`, top level in `.label`), Noul
carries one statement (P(true) — the rerank primitive). The four
structural tokens (`[CLS]`/`[SEP]`/`[MASK]`/`[PAD]`) resolve from the
sibling `tokenizer_config.json` exactly the way HF resolves
`cls_token_id` — the multilingual checkpoint maps them onto
`<bos>`/`<eos>`/`<mask>`/`<pad>`, and hardcoding the BERT names
refused to open it. Score here is
*not* GLiClass's multi-label sigmoid; the two must never share a
question type, so `Decider::open` refuses Laya ids (and `LayaDecider`
refuses non-Laya ids) instead of failing late on the wrong graph.
`option_order` permutes option slots (reported back in caller order);
it must be an exact permutation — a silent misindex would attach
probabilities to the wrong options.

Temperature is fitted, not identity: per-bucket scales from the config
(`choice:2`, `choice:3-5`, …, `noul:2`) clamped to [0.5, 5.0] with the
rejections queryable via `temperature_rejections()` (the shipped
`choice:11+` 0.1006 sharpens ~10x — the clamp refuses it). Decode is
`logits/t → softmax → unpermute → per-type shape` in float64 (the
agent's numpy arithmetic), cast to f32 at the boundary; act head reads
`softmax(act_logits)[0]` (`.act_probability`, feeds escalation).
Collation pads marker columns to `max(kmax, 2)` — the exported head
takes `topk(2)`, so one-option questions still collate two columns.
`predict_long(state, questions, window, stride)` scans states past the
cap: token windows (default `max(64, max_len - head_max_len - 8)` — the
checkpoint's own state budget — clamped to the room the questions
leave, 50%-overlap steps), one shared session run for all windows,
per-type aggregation (noul: strongest P(true); choice/score:
most-confident window). Returns the deciding window's decision plus
its token span — that window's numbers, not calibrated document
numbers. A `BinningMap` overlay (`load_binning_map`, bare map or laya
calibration payload) remaps `answer_confidence` for fitted buckets
and passes the rest through; no checkpoint ships one. Goldens live in `fixtures/golden_laya_td.json` (agent-built rows + raw
float64 reference decode, 7 cases incl. permutation + solo + relevance
probes) with the ignored `tests/laya_e2e.rs` asserting exact input-id
+ marker equality and 1e-6 on every number. Deliberately not ported:
`predict_long` scans (caller windows + aggregates), `binning_map`
(no checkpoint ships one), `lang_temperatures`, hooks/serve.

## Decider-2B

Slot-layout typed decisions: `Decider2B.open(model_id, ...)` serves
the `Mapika/decider-2b` checkpoint (Qwen3.5-2B decoder + slot head)
from the `opticsWolf/decider-2b-fp32-onnx` self-export — 7.6MB
weightless graph + 3.76GB vendored safetensors + tokenizer +
`decision.json` + `calibration.json`, all fetched from the artifact
repo (`open_files(onnx, tokenizer, decision, calibration, ...)` takes
explicit files; both configs are required, never optional). One row
per question: `Context:` + state, then options with lettered choices
(`(A) …`, one label token each past 10 options) and a trailing
`Answer: (` whose slot is scored over the 255 label ids; score
questions become one yes/no row per level (isolated fit). Two inputs
(`input_ids`, `slot_pos`, no attention mask — causal, right padding
trails the slot), one output (`label_logits`). Temperatures are the
checkpoint's own `(choice, score, noul)` triple from
`calibration.json` clamped to its `[0.05, 5.0]` range — the fitted
noul temp IS 0.05, so a Laya-style 0.5 floor would corrupt it.
Questions mirror `LayaQuestion` except noul, which carries BOTH
criteria (`("noul", ins, true_criterion[, false_criterion])` —
rendered `yes: …` / `no: …`; the reference layout has no
`option_order`, so neither does this port). CUDA runs the parallel
executor with default thread pools: the sequential planner aliases a
shape buffer on this graph (ORT #4829 class — a chunk-scan Reshape
reads a zeroed dim and fails on the first batch), the C-level
`enable_mem_reuse=false` is unreachable from the C API, and an
explicit `inter_op_num_threads=1` re-triggers it — all probed.
`fixtures/golden_decider2b.json` (8 cases incl. the 12-option wide
path + a mixed multi batch) pins exact input ids and 5e-4 answers on
CPU *and* CUDA (`tests/decider2b_e2e.rs` — the CUDA leg gates the
executor switch). 32k context, fp32 + q8 (q8: 1.59 GB int8 linears + the vendored safetensors alongside, worst 0.0020, 0 flips), CPU slow-tier (~1.5–2 s/row).

## Running without downloads

### Tokenizer-only counts

`JinaTokenizer.open` fetches only `tokenizer.json` for exact token counts
without the session (~0.5 s cold, 9× cheaper than a session open). It
never truncates, so counts report true length — `JinaV5.count_tokens()`
instead reflects the session's `max_length`.

### Cache inspection (offline)

`embroider.cache_info(model_id, revision=None, cache_dir=None,
precision=None)` answers "is this model already cached, and where?"
without opening a session or touching the network — the pre-flight check
for UIs and config validation (okfgraph's `model_info()`):

```python
from embroider import cache_info

info = cache_info("jinaai/jina-embeddings-v5-text-small-retrieval", precision="fp16")
info["cached"], info["repo"], info["snapshot_path"], info["disk_usage_bytes"]
info["files"]   # per-file paths: model, sidecar (if any), tokenizer.json
```

It resolves the artifact exactly like `open()` (`lookup_model` →
`artifact_for`): the default id at `precision="fp16"` reports the FP16
mirror repo, multi-file repos list their `.onnx_data` sidecar.
`cached=True` means an offline `open()` would succeed — model file +
`tokenizer.json` present (a missing sidecar only warns at open). No
device probe: `precision=None` reads fp32; `precision="auto"` and bare
legacy ids (`no-slash`) raise `ValueError` before any I/O. Rust
consumers get the same inspection as `embroider::cache_info` →
`CacheReport`.

The generic `embroider.cache_info_files(repo, files, revision=None,
cache_dir=None)` covers non-registry layouts (converter models, custom
repos): `files` takes bare names (required) or `(name, required)`
pairs, `cached` is true when every required file is present, `model_id`
echoes the repo, and `precision` is `None`. Same offline contract, same
dict keys — the surface bobine's `model_status()` builds on.

### Runtime inventory (which dylib, which CUDA)

`embroider.runtime_info()` inventories the ONNX Runtime situation in
one dict: the loaded dylib (`ORT_DYLIB_PATH`, with existence check),
its sibling provider DLLs, every `onnxruntime*` install in the active
venv (CPU and GPU builds side by side), the CUDA/cuDNN runtime DLLs
the `nvidia*` dists ship (grouped: `nvidia/cu13`, `nvidia/cudnn`),
and the CUDA-usability probe. `embroider.save_runtime_info(path)`
writes the same report as pretty JSON (creating parent dirs) — a
bench harness saves it next to its numbers so a later reader knows
exactly which dylib, providers and CUDA files produced them:

```python
from embroider import runtime_info, save_runtime_info, configure_runtime
info = runtime_info()
info["ort_dylib"], info["sibling_provider_dlls"], info["cuda_usable"]
save_runtime_info("bench/run-001/runtime.json")
```

`configure_runtime(dylib_path=None, dll_dirs=None)` is the write
half: `dylib_path` becomes `ORT_DYLIB_PATH` (must exist; effective
only before the first `open()` in the process — ort pins the first
library it initializes, and even `runtime_info()`'s probe counts as
a first touch, so set the dylib before anything else), `dll_dirs`
are prepended to `PATH` (effective immediately — pass
`info["load_dirs"]` to reproduce a saved layout, which is what
makes `cudnn64_9.dll` resolve out of site-packages for cuDNN-backed
ops like GLiClass's `ReduceMax`). Returns the post-call report. The
venv scan follows `VIRTUAL_ENV` with a `sys.prefix` fallback, so a
non-activated interpreter still sees its own installs.

The venv scan follows `VIRTUAL_ENV` (unset = installs unknown, the
probe fields still report); duplicate spellings of one venv
(`Lib/` vs `lib/`) list once. Keys are stable
(`ort_dylib`, `ort_dylib_exists`, `sibling_provider_dlls`,
`cuda_usable`, `ort_installs`, `cuda_runtime_dlls`,
`virtual_env`) — JSON today, trivially TOML-convertible.

### Explicit local files (air-gapped)

`JinaV5.open_files(onnx_path, tokenizer_path, truncate_dim=512,
device="auto", max_length=None, cpu_arena=False)` and
`JinaTokenizer.open_files(tokenizer_path)` skip every download. The
sidecar (`model.onnx_data`-style) must sit next to the ONNX file — ORT
resolves it relative to the model path, same as the HF cache layout.
Precision selection does not apply here — the files are what they are
(reported precision reads `fp32`; pass FP16 files explicitly for FP16
weights). OKFgraph's `OKFRouter(model_path=..., tokenizer_path=...)`
uses them (both or neither; missing files raise `FileNotFoundError` at
construction). Same bytes in → same vectors out (test-pinned against HF
acquisition).

## Session/threading policy (measured)

Tuning is `Level3`, intra = physical-cores/2, inter = 1 — kept because it
measured fastest, not because it was inherited. Reference box: Windows,
32 logical cores, CPU-only ORT 1.29, warm model cache, best-of-5 reps on
4 fixed docs (short → ~400 tokens):

| Config | Session cold open | `encode_batch` (4 docs) | Notes |
|---|---|---|---|
| Level3, intra=16, inter=1 (**current**) | 4.7 s | **375 ms** | kept |
| Level1, intra=16, inter=1 | 5.5 s | 433 ms (+15%) | slower *and* bit-different vectors |
| Level3, intra=32, inter=1 | 4.5 s | 411 ms (+10%) | full-logical loses to phys/2 (SMT contention) |
| 4× `encode_one` vs 1× `encode_batch` | — | 389 vs 375 ms | one boundary crossing saves ~3%; sequential stays |
| Tokenizer-only cold open | 0.5 s | — | 9× cheaper than session open; budgeted reads stay cold |

Two consequences:

- **Do not mix tuning in one index.** Level1 vs Level3 fuse the graph
differently, so bits differ (hashes diverged at 1e-8 formatting). Same
model + same build + same tuning, or re-embed.
- **Sequential batching stays.** Padded batching would waste attention on
variable-length docs to save ~14 ms of boundary overhead — not worth the
numerics risk.

`SessionPolicy::ort_defaults()` exists for consumers (bobine's converter-model
sessions — RapidOCR, layout, tables, formulas) that never tuned — policy is
data, never a forced default.
Re-measure on new hardware/ORT before changing the policy.

## Pitfall: stale `onnxruntime.dll` on Windows

Windows boxes can carry a stale `C:\Windows\System32\onnxruntime.dll`
(v1.17.1 in the wild). With `ORT_DYLIB_PATH` unset, `ort` may load it and
die with `BadVersion { version_str: "1.17.1" }`, followed by an abort at
shutdown (fallout from ort's exit handler, not the root cause). Point
`ORT_DYLIB_PATH` at a modern build — e.g. the venv's
`onnxruntime/capi/onnxruntime.dll`. Same pitfall bobine documents in its
`docs/benchmarks.md`.

## Failure policy

| Level | Behaviour |
|---|---|
| Install | The wheel is a core dependency of the consumer; if it is missing or fails to import, the consumer raises a clear `RuntimeError` with the install hint — never an `ImportError` from deep inside, never a silent fallback. |
| Device | Accelerators are opportunistic: `auto`/`cuda` use CUDA when the loaded ORT registers the EP, else warn (stderr) + CPU. `used_cuda` reports the outcome. Never fatal. Unknown provider names warn and are skipped; registration failure degrades to CPU. `precision='auto'` follows the landed device (CUDA→FP16, CPU→FP32); explicit `fp16`-on-CPU warns but is honoured. |
| Encode | **Fail fast.** No fallback at encode time — vectors must stay bit-comparable within one index. |
| Tokenizer | No transformers in the runtime path, anywhere: internal tokenize + `count_tokens()` (== `tokenizer.encode(t, add_special_tokens=False)`) feed the context-window guard. |

## Contract notes

- Session IO is discovered at load (`input_ids` + `attention_mask` required,
  `token_type_ids` fed only if declared — v5's export doesn't declare it,
  which is where generic runners fail). Output prefers `last_hidden_state`.
- `truncate_dim` validated (32–1024, warning off the Matryoshka ladder).
  `max_length` validated (1..=32768, `None` = 8192 compat default);
  `MAX_LENGTH` / `MODEL_MAX_TOKENS` are exposed for the window guard.
  Same (text, `max_length`) → same vector; short inputs are bit-identical
  at any limit, while longer inputs change once the old truncation lifts.
  A 32K-token forward is O(n²) memory — size the limit to the machine.
- Batch encoding is sequential by design (padded batches waste attention
  compute on variable-length docs). GIL is released during encode.
- Vision: one image per call (`seq = patches/4 + 15`, ≤1295 tokens).
  Inputs `input_ids`/`attention_mask` int64, `pixel_values` float32,
  `vision_pos_ids`/`interp_indices` int64, `interp_weights` float32;
  output `sentence_embedding` float32 `[1,768]`, then the same
  L2 → truncate → re-normalise as text (same Matryoshka ladder).
- `input_ids`/`attention_mask` feed as int64; pooling takes the last
  attended token (`mask_sum - 1`, clamped ≥ 0).

## Conformance

`fixtures/golden_jina_v5_text_small.json` pins the frozen vector space:
4 canonical texts × Query/Document prefixes × dims 64/512 (12 vectors) +
exact token counts, generated with embroider 0.1.3 / onnxruntime 1.29.0
CPU / text-embed policy. Unchanged by 0.1.4 (`max_length`) and 0.1.5
(`precision`, `cpu_arena`) for short inputs at defaults. Consumers vendor
this file and assert live vectors against it (`abs=1e-6` — catches wrong
model, pooling, prefix, or truncation; immune to cross-CPU noise). Regenerate only on an
intentional contract change, which is a new minor version plus a
re-index-everything notice. See [docs/compat.md](docs/compat.md) for the release matrix.
Vision host math is pinned the same way: `fixtures/vision/`
(`target_sizes.json`, `host_tensors.json`, `pixel_cases.json`) generated
from the spike's verified `grid.py`; the Rust port asserts bitwise
equality, and `fixtures/vision/e2e/` (5 real figures + torch goldens)
backs the ignored end-to-end parity test.

## Testing

- **Rust unit tests** (49, pure — no network, no dylib, no tokenizer
  file): device/precision parsing, precision-follows-device resolution,
  FP16 repo selection, model-id parsing, provider-matrix mapping,
  task-prefix idempotence, the L2 → truncate → re-normalise math,
  contract constants, `open()`/`open_files()` validation (dims,
  `max_length`) firing before I/O, offline cache inspection
  (`cache_info`: registry artifact resolution, fake-cache hit/partial/
  full miss, `auto`/bad-id refusal; generic `cache_info_files`:
  required/optional semantics, pinned revision, delegation parity), the vision registry entry, the
  vision resize contract (incl. banker's rounding and aspect rejection),
  host tensors bit-identical to `fixtures/vision/` on all 31 grids, the
  pixel pipeline bit-identical on 3 cases, and the vision fp16-on-CPU
  refusal.

  ```bash
  cargo test --locked
  ```

- **Vision end-to-end** (`tests/vision_e2e.rs`, ignored — needs ORT
  1.29 dylib, the vision weight files, and ideally CUDA): 5 real
  figures vs torch-native goldens, cos ≥ 0.9999 fp32 / ≥ 0.999 fp16.

  ```bash
  ORT_DYLIB_PATH=.../onnxruntime.dll VISION_E2E_FP32=.../model.onnx \
    VISION_E2E_FP16=.../model.onnx VISION_E2E_TOK=.../tokenizer.json \
    VISION_E2E_CUDA=1 cargo test --offline --test vision_e2e -- --ignored
  ```

- **Python parity** lives with the consumers: OKFgraph's
  `tests/test_parity.py` (marked `slow`) pins Rust output against a
  numpy/transformers replication across dims × tasks × texts at ≤ 1e-5;
  `tests/test_rust_backend.py` / `tests/test_rust_e2e.py` cover the
  wheel import, the count-tokens contract, and real-model encodes.
