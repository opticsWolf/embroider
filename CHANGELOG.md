# Changelog

All notable changes to `embroider` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com).

## [0.6.0] — 2026-10-29

### Added
- Labels-first decide layout + second registry entry
  (`knowledgator/gliclass-v4-edge`, 440M, Apache-2.0, fp32-only):
  `TaskSpec.prompt_first` records the input order per model
  (`open()` resolves it, `open_files` takes it caller-side with no
  silent default) and `decide_many` assembles
  `[CLS] + labels + [SEP] + prompt? + state + [SEP]` when set
  (truncate-state-side-first unchanged — the state keeps its head in
  both layouts). v4 brings instruction prompts, 2048-token ceiling,
  and an embedded held-out-fitted calibrator (multi-label is
  decision-ready: the small-v1.0 over-confidence finding does not
  reproduce). Golden `fixtures/golden_decide_gliclass_v4edge.json`
  (ORT 1.29.0 CPU, 6 cases incl. prompt + single-option relevance
  probes, abs 0.03, exact-match labels — torch Δ 0.028 on the
  calibrated branch at pinned revision `885af1af`) + v4 gates in
  `tests/decide_e2e.rs` (layout proof for the labels-first recipe).
  The in-repo `model_q8.onnx` was measured and rejected (Δ 5.5 on
  the logit scale — same rule as every other quant).

## [0.5.0] — 2026-10-29

### Added
- Typed-decision contract (`src/decide.rs`, Python `embroider.Decider`):
  `Decider::open` / `open_files` / `decide` / `decide_many`, mirroring
  the `Reranker` seams (device→precision ordering, registry ceilings,
  warn-and-honour fp16-on-CPU, `count_tokens` via the shared helper).
  `Question::Choice` (single-label stable softmax) vs `Question::Score`
  (multi-label native sigmoid, unthresholded — thresholds stay consumer
  policy); `Question::Noul` declared but loud-unsupported (no registered
  model serves none-of-the-above). Input construction is contract:
  `state + (<<LABEL>>option)* + <<SEP>> + instructions`
  (`prompt_first=false`), truncate-state-side-first with manual id
  concat; temperature is identity (the spike model ships no fitted
  temps). Batched rows with different option counts share one padded
  run (first-Cᵢ readout — B0-verified row-prefix-exact, tail slots
  unread). Registry `Decide` entry: `knowledgator/gliclass-small-v1.0`
  (144M, Apache-2.0, fp32 tier only until measured), `available_models()`
  kind `decide`. Golden `fixtures/golden_decide_gliclass_small.json`
  (ORT 1.29.0 CPU, abs 1e-6, exact-match labels, torch exact on 8 B0
  probes at pinned revision `21edefaf`) + ignored
  `tests/decide_e2e.rs` (incl. the layout proof and validation order).
  B0 finding, recorded: single-label 3/3 correct; multi-label raw
  sigmoids systematically over-confident — not decision-ready without
  calibration/thresholds (see `docs/typed-decision-plan.md` §8).

## [0.4.1] — 2026-10-29

### Added
- Long-document reranking (`Reranker::score_long`, Python
  `r.score_long(query, doc, window=None, stride=None)`): sliding windows
  over the document with max-pool (MaxP) aggregation. Window caps the
  whole pair (default: session `max_len`); stride steps the windows
  (default: 50% overlap); larger-than-budget strides and
  query-fills-window fail loudly. Chunk planning is a pure,
  weight-free unit-tested function; template overhead is derived per
  architecture, never hardcoded.
- `Reranker::count_tokens` (Python `r.count_tokens`): true token counts
  without specials via a truncation-free handle — the same helper JinaV5
  uses, reused. Golden fixtures pin per-pair counts; the ignored e2e
  asserts single-window equivalence with `score_one`, multi-window
  MaxP sanity, and validation order.

## [0.4.0] — 2026-10-28

### Added
- Cross-encoder rerank contract (`src/rerank.rs`, Python
  `embroider.Reranker`): `open` / `open_files` / `score_one` /
  `score_batch`, mirroring the `JinaV5` seams (device→precision ordering,
  registry ceilings, warn-and-honour fp16-on-CPU). Pair encoding with
  truncate-doc-side-first and per-batch longest padding; `token_type_ids`
  fed only when the graph declares it. Session tuning via
  `SessionPolicy::rerank_pair()` (starts from the text-embedding tuning).
- Vendored goldens `fixtures/golden_rerank_{minilm,mmarco}.json` (MiniLM-L-6
  English + mMARCO-L12 multilingual, ORT 1.29.0 CPU, abs 1e-6) with torch
  parity measured at 2.9e-6 / 7.7e-6; ignored `tests/rerank_e2e.rs` asserts
  both goldens plus per-language relevant-beats-irrelevant ordering.
- Shared session builder now prefers the `logits` output when a graph
  declares it (embeddings still read `last_hidden_state`; no contract
  impact — each graph declares exactly one of the two).

## [0.3.4] — 2026-10-28

### Added
- Task registry (`acquire::{TaskSpec, TaskKind, builtin_tasks,
  lookup_task, artifact_for_task}`): non-embedding sibling of the
  `ModelSpec` registry for rerankers and typed-decision models (no
  `native_dim`/`ladder`). `cache_info` resolves task ids through the
  same offline delegation; `available_models()` gains a `kind` key
  (`embed`/`rerank`/`decide`) with all existing keys unchanged. No
  existing contract moves — purely additive, no consumer impact.
- First task entries: the rerank twin-spike (`cross-encoder` MiniLM-L-6
  English + mMARCO MiniLMv2-L12 multilingual, Apache-2.0, fp32 tier
  only until measured). Licence rows in `docs/compat.md` — these are
  the registry's first commercially-usable weights.

## [0.3.3] — 2026-10-06

### Added
- Generic offline cache inspection: `acquire::cache_info_files(repo,
  files, revision, cache_dir)` looks an arbitrary (repo, files) pair up
  in the hub cache — same offline-only mechanics and `CacheReport`
  contract as `cache_info`, for non-registry layouts (bobine's converter
  models). `files` pairs each repo-relative filename with
  required/optional; `cached` is true when all required files are
  present; `model_id` echoes the repo and `precision` is `None` (no tier
  here). Python surface: `embroider.cache_info_files(repo, files,
  revision=None, cache_dir=None)` (bare names read required,
  `(name, required)` pairs opt out); bad repos, empty lists, and
  malformed entries raise `ValueError`/`TypeError` before any I/O.
- `acquire::cache_info` now delegates to `cache_info_files` (one lookup
  implementation underneath; registry resolution + id/tier patching
  stays in `cache_info`) — behavior unchanged, pinned by a
  delegation-parity test.

### Changed
- `CacheReport.precision` is now `Option<Precision>` (`Some` from
  `cache_info`, `None` from `cache_info_files`). No consumer impact:
  the struct is only constructed inside `acquire`, and the Python dict
  keeps identical keys (`cache_info` still reports the tier string).

## [0.3.2] — 2026-10-05

### Added
- Offline cache inspection: `embroider.cache_info(model_id, revision=None,
  cache_dir=None, precision=None)` (Python), backed by
  `acquire::{cache_info, CacheReport}` (Rust). Resolves the artifact
  exactly like `open()` (`lookup_model` → `artifact_for`) and looks each
  required file up in the hub cache with `local_files_only` — never the
  network, never a session, no device probe (`precision=None` reads
  fp32; 'auto' and bare legacy ids raise `ValueError` before any I/O).
  Returns repo (the fp16 mirror for the default id at fp16), resolved
  precision, per-file paths, `cached` (sidecar optional, matching
  `open()`), snapshot dir, disk usage. Replaces okfgraph's
  `huggingface_hub`-based `model_info` workaround (missing dependency,
  and the wrong repo for fp16/multi-file layouts).

### Changed
- Homepage metadata now points at the project website
  (<https://opticswolf.github.io/embroider/>) in Cargo.toml and
  pyproject.toml ([project.urls]), and the webpage reflects 0.3.2
  (version stamps, test count, the new `cache_info` in the workflow and
  capability sections).

## [0.3.1] — 2026-10-03

### Changed
- Docs only: README restructure (TOC, Models section with registry +
  custom-model loading, consumer recipe for non-Jina sessions), new
  `docs/quickref.md`, `COMPAT.md` → `docs/compat.md`. No code change.

## [0.3.0] — 2026-10-03

### Added
- Vision contract (`vision::{JinaV5Vision, vision_target_size}`, Phase 6):
  dynamic-grid omni-nano image embeddings sharing the text-nano vector
  space. Exact Rust port of the spike's `grid.py` (pinned bit-identical by
  `fixtures/vision/`: 24 resize targets, 31 host-tensor grids, 3 pixel
  pipelines). Registry id `jina-v5-omni-nano-retrieval-vision`
  (`text_partner = jina-v5-text-nano-retrieval`, fp32/fp16 artifacts from
  `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx`, explicit
  sidecar layout, CC BY-NC 4.0). `Precision::Auto` → fp16 on CUDA / fp32
  on CPU; explicit fp16-on-CPU fails fast (the graph stalls on CPU, it
  doesn't just run slow). Vision-slot sessions: ORT defaults + CUDA
  `arena_extend_strategy=kSameAsRequested` + optional `gpu_mem_limit`;
  CPU arena off. Reverses the "no vision models" non-goal — minor bump
  with this note. End-to-end parity vs torch native (ignored test,
  `tests/vision_e2e.rs`): cos 1.00000/0.99998 fp32, ≥0.99991 fp16 on 5
  real figures.

## [0.2.1] — 2026-10-01

### Changed
- Compatibility refresh only (no vector change): `docs/compat.md` current
  (okfgraph 0.7.x / bobine 0.5.x+ / embroider 0.2.x / onnxruntime 1.29.0),
  floor pin `>=0.2,<0.3`, no-torch-chain note. Bobine moves to
  `embroider = "0.2"` on this release.

## [0.2.0] — 2026-09-15

### Added
- Model registry (`acquire::{ModelSpec, Artifact, builtin_models,
  lookup_model, artifact_for}`): the loader now resolves (model,
  precision) pairs against a static contract — id, native dim, token
  ceiling, Matryoshka ladder, per-precision artifacts — instead of one
  hardcoded repo. Builtins: text-small (fp32 + FP16 mirror) and
  text-nano (official in-repo fp32/fp16/dynamic-int8, measured in the
  nano probe: fp32/fp16 @0.99994, int8 rank-kept @0.99980, q4 killed).
  `JinaV5::open(model_id, ...)` needs no signature change: registered
  ids validate against their own ladder/ceiling (nano: 768 dim max,
  8192 ctx) and fetch their own file layout (sidecar derived per
  artifact stem); unknown ids keep the frozen legacy path. Default id
  unchanged — adding models never moves existing graphs.
- `Precision::Int8` (explicit opt-in only; `auto` never selects it).
  Unlisted (model, precision) pairs fall back to fp32, never to
  unvalidated weights.
- Python `embroider.available_models()` + `NANO_TEXT_MODEL` constant
  for UIs and config validation.

## [0.1.5] — 2026-09-14

### Added
- Weight precision selection: `JinaV5::open` (and the Python `precision`
  kwarg, default `None` = `auto`) accepts `fp32` / `fp16`. `auto` follows
  the *resolved* device (CUDA → FP16, CPU → FP32), so CUDA-requested-
  but-missing degrades to FP32 weights instead of stranding FP16 on CPU
  (FP16-on-CPU runs >40x slower than FP32-CPU — measured). `fp16` maps
  the default model id to the published FP16 mirror repo
  (`opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16`);
  explicit model ids (omni tower, mirrors) always win untouched, and
  explicit files bypass selection (they report `fp32`). An explicit
  `fp16`-on-CPU warns loudly but is honoured. The session exposes its
  effective precision (`.precision` / `precision()`).
- CPU-arena control: `open` / `open_files` (and the Python `cpu_arena`
  kwarg, default `False`) disable the CPU arena allocator — measured 8x
  lower peak RSS (15.3 → 1.9 GB on FP32) for ~1.4x encode time.
  `apply_providers` keeps its exact historical behaviour (arena on);
  the new `apply_providers_with_arena` carries the flag, retrying
  CPU-only on accelerator failure so the flag survives fallback.

## [0.1.4] — 2026-09-13

### Added
- Configurable token limit: `JinaV5::open` / `open_files` (and the Python
  `max_length=None` kwarg) accept 1..=32768 (`MODEL_MAX_TOKENS`, the Qwen3
  position ceiling from the v5 config); `None` keeps the historical 8192
  default (`MAX_LENGTH`) so existing graphs keep bit-identical vectors.
  Out-of-range values fail before any I/O. The session exposes its
  effective limit (`max_len` / `.max_length`); the counting-only
  `TokenizerHandle` no longer truncates, so counts report true length.

### Notes
- Raising the limit changes vectors for inputs longer than the old
  truncation only; short inputs are bit-identical at any limit.
  A 32K-token forward is O(n^2) memory — size the limit to the machine.

## [0.1.3] — 2026-09-13

### Fixed
- `cuda_available()` is panic-free when no ORT dylib is resolvable: ort
  load-dynamic panics on first API use (`.expect`ed init); the probe now
  contains it and reports "no CUDA". Bobine's suite surfaced this — its
  table-slot test runs without a dylib.

## [0.1.2] — 2026-09-13

### Changed
- `apply_providers` is now infallible (returns the builder directly): every
  path already yielded a usable builder, so the `Result` forced needless
  error-mapping on consumers (bobine's seven slot call sites).
- `SessionPolicy::apply(builder)` added — policies are now publicly
  appliable to a builder; `build_session` uses it internally. Bobine adopts
  `ort_defaults().apply(...)` in Phase 2 of the spin-off.

## [0.1.1] — 2026-09-13

### Fixed
- PyPI wheel interpreter matrix: 0.1.0 shipped only the runners' default
  interpreters (cp312/cp314). The release workflow now builds the proven
  3-platform × Python 3.11/3.12/3.13 matrix (`--interpreter` per OKFgraph's
  okf-embed release job), and `requires-python` is pinned to `>=3.11` to
  match what wheels are actually built for.

## [0.1.0] — 2026-09-13

### Added
- Initial release — clean move from OKFgraph's `rust/okf-embed`
  (github.com/opticsWolf/OKFgraph). Jina v5 text-embedding contract
  (`JinaV5`: prefix → tokenize@8192 → last-token pooling → L2 → Matryoshka
  truncate → re-normalise; `JinaTokenizer` for session-free exact counts),
  lazy session lifecycle, air-gapped `open_files`, cross-provider ONNX
  Runtime plumbing (`providers`/`probe`/`policy`/`acquire`/`diag`),
  opt-in PyO3 `extension-module` for wheels, pure-Rust rlib for crates.io.
  21 pure unit tests.
