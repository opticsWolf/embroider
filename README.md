# embroider — Jina v5 text embeddings (Rust core, PyO3)

[![CI](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml/badge.svg)](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/embroider)](https://crates.io/crates/embroider)
[![docs.rs](https://img.shields.io/docsrs/embroider)](https://docs.rs/embroider)
[![PyPI](https://img.shields.io/pypi/v/embroider)](https://pypi.org/project/embroider/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/crates/l/embroider)](LICENSE-MIT)

`embroider` turns text into vectors — Jina v5 embeddings served through
ONNX Runtime from a Rust core, with optional PyO3 bindings for Python.
One frozen contract covers the whole path: task prefix → tokenize → ONNX
forward → last-token pooling → L2 → Matryoshka truncate → re-normalise,
so every consumer lands in the same vector space.

One engine, two consumers: **bobine** (PDF/Office → Markdown) reuses the
ONNX plumbing, **okfgraph** uses the full Jina v5 text-embedding contract.
It began as a clean move out of OKFgraph's `rust/okf-embed` — an exact
port of `EmbeddingEngine._encode` — and stays pinned against a
numpy/transformers replication by OKFgraph's parity harness
(`tests/test_parity.py`, max abs diff ≤ 1e-5).

Single backend, kept comparable: one lean runtime with no torch /
transformers / optimum in the hot path, so every vector in an index stays
directly comparable. If something can't be embedded exactly to contract,
embroider errors loudly rather than quietly mixing vector spaces.

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
| `policy` | `DeviceReq` (`auto`/`cpu`/`cuda`) + `Precision` (`auto`/`fp32`/`fp16`) + explicit `SessionPolicy` (`text_embed()` vs `ort_defaults()`) |
| `acquire` | validated `owner/name` parsing, HF client, tokenizer-only fetch, FP16-mirror selection for the default id |
| `error` | anyhow-based error plumbing (`ort` errors stringified at boundaries) |
| `diag` | `OrtReport` — `ORT_DYLIB_PATH` value + CUDA usability for logs |
| `jina` | `JinaV5` + `TokenizerHandle` — the frozen embedding contract |

The default (pure-Rust) build is Python-free — no `pyo3` in downstream
trees; the `extension-module` Cargo feature gates the PyO3 bindings and is
enabled only for wheel builds (maturin), the same pattern bobine uses.

## Runtime: ONNX Runtime discovery

`ort` loads dynamically (`load-dynamic`, same pin as bobine:
`2.0.0-rc.13`). Resolution order: `ORT_DYLIB_PATH` first (user override
always wins), else the pip-installed `onnxruntime`/`onnxruntime-gpu`
build when unset. okfgraph's `resolve_ort_dylib()` runs before the native
module is imported, so bobine and embroider share **one** ORT binary — no
version/CUDA drift between ingest and import.

## Lifecycle: lazy session, cheap tokenizer

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

`JinaTokenizer.open` fetches only `tokenizer.json` for exact token counts
without the session. It never truncates, so counts report true length —
`JinaV5.count_tokens()` instead reflects the session's `max_length`.
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

### Memory: CPU arena off by default

`cpu_arena=False` registers the CPU execution provider explicitly with
its arena allocator disabled — measured 8x lower peak RSS (15.3 → 1.9 GB
on FP32) for ~1.4x encode time. Pass `True` only when peak throughput
beats memory pressure.

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

`SessionPolicy::ort_defaults()` exists for consumers (bobine's vision
sessions) that never tuned — policy is data, never a forced default.
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
re-index-everything notice. See `COMPAT.md` for the release matrix.

## Testing

- **Rust unit tests** (28, pure — no network, no dylib, no tokenizer
  file): device/precision parsing, precision-follows-device resolution,
  FP16 repo selection, model-id parsing, provider-matrix mapping,
  task-prefix idempotence, the L2 → truncate → re-normalise math,
  contract constants, and `open()`/`open_files()` validation (dims,
  `max_length`) firing before I/O.

  ```bash
  cargo test --locked
  ```

- **Python parity** lives with the consumers: OKFgraph's
  `tests/test_parity.py` (marked `slow`) pins Rust output against a
  numpy/transformers replication across dims × tasks × texts at ≤ 1e-5;
  `tests/test_rust_backend.py` / `tests/test_rust_e2e.py` cover the
  wheel import, the count-tokens contract, and real-model encodes.
