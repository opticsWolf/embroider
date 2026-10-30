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
  2048 ceiling, fp32 + q8 (weight-only; the dynamic-int8 route drifts 5.5 and stays rejected) and is
  rejected). Its graph embeds the held-out-fitted calibrator, so its
  golden (`fixtures/golden_decide_gliclass_v4edge.json`, ORT 1.29.0
  CPU) pins ORT values at a 0.03 tolerance, not 1e-6.
- **Laya marker layout (0.7.0)**: `LayaDecider` (`src/laya.rs`, Python
  `embroider.LayaDecider`) serving the three registered
  `convaiinnovations/laya-*` checkpoints from the
  `opticsWolf/laya-*-fp32-onnx` self-exports. 5-input collation
  (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`),
  `k_cols >= 2` (the exported head's `topk(2)`), options truncated to
  48 tokens at the tokenizer with the head-budget cut
  (`max(4, (head_max_len-16)/n)` under 16 tokens of room, instruction
  floor 8). Own question enum (`Choice` with descriptions +
  `option_order`, level-`Score` with expected value, statement-`Noul`)
  — Score is a distribution over levels, never GLiClass's multi-label
  sigmoid, so `Decider` refuses Laya ids and `LayaDecider` refuses
  non-Laya ids. Fitted per-bucket temperatures clamped to [0.5, 5.0]
  (rejections queryable, `choice:11+` refused); decode in float64,
  `encode_row` exposes the construction for inspection. `predict_long`
  (checkpoint-sized windows, per-type aggregation) + `load_binning_map`
  (fitted overlay, none shipped) round out the long-document and
  calibration stories. Golden
  `fixtures/golden_laya_td.json` (agent-built rows + raw reference
  decode, 7 cases) + ignored `tests/laya_e2e.rs` (exact ids + markers,
  abs 1e-6 on every number). `TaskKind::DecideLaya` (`decide-laya`)
  keeps the routing unambiguous; `cache_info` requires
  `rl_agent_config.json` for it.
- **Decider-2B slot layout (0.10.0)**: `Decider2B` (`src/decider2b.rs`,
  Python `embroider.Decider2B`) serving `Mapika/decider-2b` (Qwen3.5-2B
  decoder + slot head) from the `opticsWolf/decider-2b-fp32-onnx`
  self-export (7.6MB graph + 3.76GB vendored safetensors + tokenizer +
  `decision.json` + `calibration.json`). One row per question
  (`Context:` + state, lettered/wide options, the `Answer: (` slot
  scored over 255 label ids; isolated yes/no rows for score), 2
  inputs (`input_ids`, `slot_pos`), no attention mask, ort_rows-style
  batching. Own question enum mirroring Laya's except noul, which
  carries BOTH criteria (no `option_order` — the reference layout has
  none). Checkpoint temperatures clamped to its own [0.05, 5.0]
  (fitted noul temp is 0.05). CUDA runs the parallel executor with
  default thread pools (sequential planner aliases a shape buffer —
  ORT #4829 class; `enable_mem_reuse=false` unreachable from C;
  `inter_op=1` re-triggers it — all probed, gossamer-sourced).
  Goldens `fixtures/golden_decider2b.json` (8 cases incl. the wide
  path + mixed batch) + ignored `tests/decider2b_e2e.rs` (exact ids,
  5e-4 CPU *and* CUDA). `TaskKind::Decide2B` (`decide-2b`);
  `cache_info` requires both configs for it.
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
| `knowledgator/gliclass-v4-edge` fp32 | `knowledgator/gliclass-v4-edge` (`onnx/model.onnx`) + `opticsWolf/gliclass-v4-edge-q8-onnx` | Apache-2.0 (decide, single/multi-label + instruction prompts, labels-first layout; upstream fp32 referenced, q8 mirrored; CPU parity ≤0.03 vs goldens — the graph embeds the calibrator, torch Δ 0.028 on the calibrated branch; dynamic-int8 q8 rejected at Δ 5.5, weight-only q8 ships at worst 0.090, 0 flips) |
| `convaiinnovations/laya-typed-decisions` fp32 | `opticsWolf/laya-typed-decisions-fp32-onnx` (`laya-td-fp32.onnx` + sidecar) | Apache-2.0 (decide-laya, marker layout, fitted temps; self-export of `1a793eb`, torch Δ 2.0e-6; agent-row goldens at 1e-6, billing vector unit-pinned) |
| `convaiinnovations/laya` fp32 | `opticsWolf/laya-english-fp32-onnx` (`laya-base-fp32.onnx` + sidecar) | Apache-2.0 (decide-laya, base English 512/192; near-chance zero-shot per Ollaya — faithful export, not classifier duty) |
| `convaiinnovations/laya-multilingual` fp32 | `opticsWolf/laya-multilingual-fp32-onnx` (`laya-multi-fp32.onnx` + sidecar) | Apache-2.0 (decide-laya, multilingual mmBERT-base 1024/256; fastest CPU of the three) |
| `Mapika/decider-2b` fp32 | `opticsWolf/decider-2b-fp32-onnx` (`model.onnx` + `model.safetensors` + `decision.json` + `calibration.json`) | Apache-2.0 (decide-2b, slot layout, 32k ctx; self-export of `7614a6c7`, torch Δ 1e-4; reference-row goldens at 5e-4 CPU+CUDA) |

- **Every built-in model is non-commercial.** Any consumer (okfgraph,
  bobine's callers) that runs them commercially needs a commercial licence
  from Jina AI.
- **Mirrors and conversions** we publish (e.g. the fp16 mirror) keep the
  upstream licence. Their model card must carry `license`, `base_model`,
  attribution to the upstream authors and a list of the changes made.

## Model sizes & serving

Download bytes per registry entry (graph + sidecar + tokenizer, as
`open()` fetches them) and where each one runs. Latencies measured on
the reference rig (RTX 3090 + 5950X, ORT 1.29, single question unless
noted) — re-measure on yours before signing an SLO; the ordering, not
the digits, is the contract. Weights-bytes ≈ VRAM floor (fp32 = 4
B/param); fp16 halves download + VRAM with **zero CPU relief**
(emulated half-precision) — it is a CUDA-serving variant everywhere
it appears, never a CPU upgrade.

| Registry model | Download | CPU | CUDA |
|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` | 91 MB (**q8: 52 MB**) | 1–3 ms/pair, CPU-native | same |
| `mmarco-mMiniLMv2-L12-H384-v1` | 471 MB (**q8: 379 MB**) | 1–3 ms/pair, CPU-native | same (q8 nDCG within 0.016 on all 4 Nano benches) |
| `gliclass-small-v1.0` | 575 MB | **10.5 ms, the CPU sweet spot** | faster, unneeded |
| `gliclass-v4-edge` | 1.77 GB (**q8: 860 MB**) | **38 ms, fine on CPU** | **2 ms** |
| `laya-typed-decisions` | 1.70 GB (fp16: 846 MB, **q8: 409 MB**) | 65 ms / 225 ms per 4-row batch — **slow-tier** | **~15 ms/batch** (fp16: 10–23 ms) |
| `laya` (base English) | 1.70 GB (fp16: 846 MB, **q8: 409 MB**) | ~220 ms — not classifier duty (near-chance zero-shot per Ollaya) | ~15 ms |
| `laya-multilingual` | 1.32 GB (fp16: 678 MB, **q8: 827 MB**) | **~97 ms, fastest Laya on CPU** | **~10 ms** |
| `decider-2b` | 3.76 GB (graph 7.6 MB; **q8: 1.59 GB + sidecar**) | 1.5–2 s/row — **slow-tier, CUDA duty** | **~75–90 ms/row** (parallel executor) |
| `jina-embeddings-v5-text-small` | 2.40 GB (fp16: 1.20 GB) | ms-scale, CPU-native | — |
| `jina-embeddings-v5-text-nano` | 866 MB (fp16: 425 MB, int8: 247 MB) | ms-scale, CPU-native | — |
| `jina-v5-omni-nano` (vision) | 1.25 GB (fp16: 623 MB) | slow | fp16 CUDA |

Consumer shapes that follow from the table:

- **CPU-only floor** (~1 GB): one reranker (MiniLM for English at 91
  MB, mMARCO at 471 MB for multilingual) + GLiClass-small (575 MB, 10
  ms) + nano-embed int8 (247 MB). This is the okfgraph floor-pin shape.
- **CPU with headroom** (+~3 GB): v4-edge (calibrated multi-label +
  prompts, 38 ms) and Laya-multilingual (~100 ms) for descriptions +
  P(true) relevance. Laya-typed-decisions and base stay off CPU duty.
- **CUDA**: everything collapses to single-digit ms (v4-edge 2 ms,
  Laya-td 15 ms/batch, multi 10 ms) — take the fp16 Laya variants to
  halve download and VRAM. GLiClass stays fp32 until its own parity
  gate is wired.
- **Not served**: dynamic/static int8 on any decision model (rejected — the −1e4 mask
  sentinel is unquantizable, drift 2.8–9997); GLiClass-small q8 (failed the 0-flip gate);
  fp16 decider-2b (no precision story). Weight-only q8 ships where it gates clean (see table).
- **New registry entries** record the artifact's licence in this table in
  the same change. A model whose licence forbids redistributing adapted
  weights is referenced by upstream repo only, never mirrored.
- bobine's PDF models (layout, OCR, tables, formulas) are fetched by bobine,
  not by this registry. They are listed with their licences in okfgraph's
  README, "Model licences" section.
- Licences are as declared on each Hugging Face model card when checked
  (2026-09-27). The cards are authoritative; this is not legal advice.
