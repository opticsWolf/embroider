# Compatibility matrix

One pinned stack in practice. A new `embroider` minor without an okfgraph
update is a **supported** state (the floor pin allows it); the vendored
golden vectors (`fixtures/golden_jina_v5_text_small.json`) are the contract
proof — okfgraph's suite asserts live vectors against them.

| okfgraph | bobine | embroider | onnxruntime (pip) | embroider wheels |
|---|---|---|---|---|
| 0.7.x | 0.5.x+ | 0.2.x | 1.29.0 | py3.11–3.13 × linux / win / macOS-arm64 |
| 0.8.x | 0.5.x+ | 0.3.x | 1.29.0 | py3.11–3.13 × linux / win / macOS-arm64 |

Rules:

- **Single pinned ORT**: one `onnxruntime` binary shared by every Rust
  consumer (`ORT_DYLIB_PATH`-overridable). Never float it per project.
- **Floor pin**: okfgraph 0.7.x requires `embroider>=0.2,<0.3`, 0.8.x
  requires `embroider>=0.3,<0.4` — the Jina contract moves only with
  okfgraph releases (0.3.0 adds the vision contract; the text contract
  is untouched, same goldens).
- **No torch anywhere in the chain**: text embeddings (embroider/Jina) and
  bobine vision models are all ONNX Runtime; okfgraph image embeddings are
  caption text vectors (`mode=text`) or ONNX vision vectors (`optional`/`omni`
  via `JinaV5Vision`) since okfgraph 0.8.0.
- **Frozen vector space**: prefixes, last-token pooling, truncation order
  never change inside a minor. A contract change is a new minor plus a
  re-index-everything notice, and the golden fixture is regenerated.
- **Truncation limit is a parameter, not the contract** (0.1.4): same
  (text, `max_length`) always yields the same vector; the default stays
  8192 so existing graphs are unaffected. Raising the limit changes
  vectors for inputs longer than the old truncation only — reimport
  long docs after changing it, don't mix limits in one graph.
- **Wheel matrix**: embroider ships cp311/cp312/cp313 wheels per platform;
  okfgraph requires Python ≥ 3.11, matching exactly.
- **Cache inspection (0.3.2)**: `embroider.cache_info(model_id, revision,
  cache_dir, precision)` reports what is already downloaded — offline
  only (no download, no session, no device probe; `precision=None` reads
  fp32). Consumers use it instead of `huggingface_hub`-based checks,
  which crash without the dependency and resolve the wrong repo for
  fp16/multi-file layouts.
- **Reranker contract (0.4.0)**: `Reranker::open` / `open_files` /
  `score_one` / `score_many` (`src/rerank.rs`, Python `embroider.Reranker`).
  Pair encoding `[query, doc]` with truncate-doc-side-first (`OnlySecond`)
  and per-batch longest padding; `token_type_ids` fed only when the graph
  declares them (MiniLM 3-input vs mMARCO 2-input, discovered at load).
  Goldens `fixtures/golden_rerank_{minilm,mmarco}.json` (ORT 1.29.0 CPU,
  abs 1e-6) + ignored `tests/rerank_e2e.rs`, same gates as vision.
- **Decider contract (0.5.0)**: `Decider::open` / `open_files` /
  `decide` / `decide_many` (`src/decide.rs`, Python `embroider.Decider`).
  `Question::{Choice, Score}` over a state (single-label softmax vs
  multi-label native sigmoid — thresholds stay consumer-side);
  `Noul` declared but loud-unsupported (no registered model serves it).
  Input construction is contract:
  `state + (<<LABEL>>option)* + <<SEP>> + instructions`
  (`prompt_first=false`), truncate-state-side-first with manual id
  concat (the pair-tokenizer `OnlySecond` pattern would cut the wrong
  side); temperature is identity (no fitted temps ship with the model).
  Golden `fixtures/golden_decide_gliclass_small.json` (ORT 1.29.0 CPU,
  abs 1e-6 on probs, exact-match labels) + ignored
  `tests/decide_e2e.rs`, same gates as rerank (incl. the layout proof:
  Rust-built ids equal a single recipe encode).
- **Decider labels-first layout (0.6.0)**: `TaskSpec.prompt_first` +
  `Decider::prompt_first` — `open()` resolves it from the registry,
  `open_files` takes it caller-side (explicit files carry no registry
  metadata, so a wrong layout would score garbage, never an error).
  Second registry entry: `knowledgator/gliclass-v4-edge` (440M,
  Apache-2.0, labels-first recipe `labels + [SEP] + prompt? + state`,
  2048 ceiling, fp32 only — the in-repo q8 drifts 5.5 and is
  rejected). Its graph embeds the held-out-fitted calibrator, so its
  golden (`fixtures/golden_decide_gliclass_v4edge.json`, ORT 1.29.0
  CPU) pins ORT values at a 0.03 tolerance, not 1e-6.
- **Task registry (0.3.4)**: `acquire::{TaskSpec, TaskKind, builtin_tasks,
  lookup_task, artifact_for_task}` — the non-embedding sibling of
  `ModelSpec` (no `native_dim`/`ladder`; scores and decisions have no
  vector space). `cache_info` resolves task ids through the same
  delegation + id/tier patching; `available_models()` gains a `kind`
  key (`embed`/`rerank`/`decide`, existing keys unchanged). First
  entries: the Apache-2.0 rerank spikes (MiniLM-L-6 English +
  mMARCO-L12 multilingual, fp32 tier only until measured).
- **Generic cache inspection (0.3.3)**: `embroider.cache_info_files(repo,
  files, revision, cache_dir)` answers the same question for arbitrary
  (repo, files) pairs — `files` as `(name, required)` pairs, `cached`
  true when all required files are present. `cache_info` delegates to it
  (one lookup implementation); `CacheReport.precision` is now
  `Option<Precision>` (`None` for the generic surface — no consumer
  impact, dict keys unchanged).

## Model licences

embroider's code is MIT OR Apache-2.0. The weights its registry
(`src/acquire.rs`) downloads are licensed separately:

| Registry model | Artifact repo | Licence |
|---|---|---|
| `jina-embeddings-v5-text-small-retrieval` fp32 | `jinaai/jina-embeddings-v5-text-small-retrieval` | CC BY-NC 4.0 |
| `jina-embeddings-v5-text-small-retrieval` fp16 | `opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16` | CC BY-NC 4.0 (ONNX conversion of the above) |
| `jina-embeddings-v5-text-nano-retrieval` fp32 / fp16 / int8 | `jinaai/jina-embeddings-v5-text-nano-retrieval` | CC BY-NC 4.0 |
| `jina-v5-omni-nano-retrieval-vision` fp32 / fp16 | `opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx` | CC BY-NC 4.0 (dynamic-grid ONNX export of `jinaai/jina-embeddings-v5-omni-nano-retrieval`; card carries base_model, attribution, changes) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` fp32 | `cross-encoder/ms-marco-MiniLM-L-6-v2` (`onnx/model.onnx`) | Apache-2.0 (rerank, English; upstream repo referenced, never mirrored; CPU parity ≤1e-6 vs goldens, torch Δ 2.9e-6) |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` fp32 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (`onnx/model.onnx`, not the O1–O4 optimum variants) | Apache-2.0 (rerank, multilingual; upstream repo referenced, never mirrored; CPU parity ≤1e-6 vs goldens, torch Δ 7.7e-6) |
| `knowledgator/gliclass-small-v1.0` fp32 | `knowledgator/gliclass-small-v1.0` (`onnx/model.onnx`) | Apache-2.0 (decide, single/multi-label; upstream repo referenced, never mirrored; CPU parity ≤1e-6 vs goldens, torch exact on 8 B0 probes) |
| `knowledgator/gliclass-v4-edge` fp32 | `knowledgator/gliclass-v4-edge` (`onnx/model.onnx`) | Apache-2.0 (decide, single/multi-label + instruction prompts, labels-first layout; upstream repo referenced, never mirrored; CPU parity ≤0.03 vs goldens — the graph embeds the calibrator, torch Δ 0.028 on the calibrated branch; in-repo q8 rejected at Δ 5.5) |

- **Every built-in model is non-commercial.** Any consumer (okfgraph,
  bobine's callers) that runs them commercially needs a commercial licence
  from Jina AI.
- **Mirrors and conversions** we publish (e.g. the fp16 mirror) keep the
  upstream licence. Their model card must carry `license`, `base_model`,
  attribution to the upstream authors and a list of the changes made.
- **New registry entries** record the artifact's licence in this table in
  the same change. A model whose licence forbids redistributing adapted
  weights is referenced by upstream repo only, never mirrored.
- bobine's PDF models (layout, OCR, tables, formulas) are fetched by bobine,
  not by this registry. They are listed with their licences in okfgraph's
  README, "Model licences" section.
- Licences are as declared on each Hugging Face model card when checked
  (2026-09-27). The cards are authoritative; this is not legal advice.
