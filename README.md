# embroider
### Frozen embedding, rerank, and decision vectors over ONNX Runtime (Rust core, PyO3)

[![CI](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml/badge.svg)](https://github.com/opticsWolf/embroider/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/embroider)](https://crates.io/crates/embroider)
[![docs.rs](https://img.shields.io/docsrs/embroider)](https://docs.rs/embroider)
[![PyPI](https://img.shields.io/pypi/v/embroider)](https://pypi.org/project/embroider/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/crates/l/embroider)](LICENSE-MIT)
[![Website](https://img.shields.io/badge/website-embroider-blue)](https://opticswolf.github.io/embroider/)

## What it is

`embroider` turns text — and images — into vectors, scores, and typed
decisions: embeddings, cross-encoder rerank, and Choice/Score/Noul
gates served through ONNX Runtime from a Rust core, with optional PyO3
bindings. Two embedding families live in the registry — Jina v5
(non-commercial) and EmbeddingGemma 2 (Apache-2.0) — alongside
rerankers, GLiClass deciders, Laya marker-layout gates, and the
Decider-2B slot engine. No torch, transformers, or optimum in the hot
path, so every vector in an index stays directly comparable — or the
encode fails loudly instead of silently forking the space.

Two consumers share it differently: **okfgraph** uses the text +
vision contracts for its graph; **bobine** (PDF/Office → Markdown)
reuses only the ONNX plumbing — session policy, provider fallback,
CUDA probe — while owning its converter-model sessions.

## Engines

| Capability | Loader (Python / Rust) | Registry ids |
|---|---|---|
| Text embeddings (Jina v5, Gemma 2) | `JinaV5` / `jina` | `jinaai/jina-embeddings-v5-*`, `google/embeddinggemma-2` |
| Vision → text space (Jina, Gemma) | `JinaV5Vision`, `GemmaVision` / `vision`, `gemma` | `jina-v5-omni-nano-retrieval-vision`, `google/embeddinggemma-2-vision` |
| Cross-encoder rerank | `Reranker` / `rerank` | `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Typed decisions (GLiClass) | `Decider` / `decide` | `knowledgator/gliclass-small-v1.0`, `knowledgator/gliclass-v4-edge` |
| Typed decisions (Laya markers) | `LayaDecider` / `laya` | `convaiinnovations/laya-{typed-decisions,english,multilingual}` |
| Typed decisions (decider-2B slots) | `Decider2B` / `decider2b` | `Mapika/decider-2b` |
| Custom models under known kinds | `register_task` / `register_model`, `load_registry_file` | caller-supplied `owner/name` ids |

Full artifact repos, precisions, and licences per entry live in
[docs/compat.md](docs/compat.md); the registry design (builtin +
overlay, flagged-unmeasured tiers, revision pins) in
[docs/model-registry-record.md](docs/model-registry-record.md).

## Install

```bash
pip install embroider                       # Linux / Windows / macOS-arm64, Python 3.11–3.13
pip install onnxruntime==1.31.1             # or -gpu[cuda,cudnn]==1.31.1 — exactly one, never both
```

From source: `maturin build --release`, then install the wheel
(`maturin develop` needs pip, which uv venvs lack).

```python
from embroider import JinaV5
m = JinaV5.open("jinaai/jina-embeddings-v5-text-small-retrieval")
v = m.encode("what do badgers eat?", task="query")   # 512-d unit vector, golden-pinned
```

## Docs

- [docs/quickref.md](docs/quickref.md) — the complete user manual:
  every loader, every parameter, copy-paste Python and Rust.
- [docs/architecture.md](docs/architecture.md) — how it works: module
  layout, session policy and measurements, runtime discovery, the
  frozen-contract rules, registry and precision design, conformance.
- [docs/compat.md](docs/compat.md) — release matrix, model sizes and
  serving guidance, per-entry licences.
- [docs/rerank-typed-decision-record.md](docs/rerank-typed-decision-record.md) —
  how the rerank and decide engines were chosen, measured, and gated.
- [docs/model-registry-record.md](docs/model-registry-record.md) — the
  registry design (builtin + overlay, revision pins).
- [docs/model-conversion.md](docs/model-conversion.md) — the conversion
  playbook: weights to parity-gated ONNX.

## Failure policy

| Level | Behaviour |
|---|---|
| Install | Missing wheel → clear `RuntimeError` with the install hint; never a silent fallback. |
| Device | Accelerators are opportunistic: `auto`/`cuda` use CUDA when the loaded ORT registers the EP, else warn + CPU. `used_cuda` reports the outcome. |
| Encode | **Fail fast.** No fallback — vectors must stay bit-comparable within one index. |
| Precision | `auto` follows the landed device (CUDA→FP16, CPU→FP32); explicit fp16-on-CPU warns for text, errors for vision. |

Same (model, precision, tuning, `max_length`) or re-embed — never mix
precisions or tuning levels in one index.
