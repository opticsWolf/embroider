# Changelog

All notable changes to `embroider` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

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
