# Rerank + typed-decision integration plan

Goal: serve **cross-encoder rerankers** and **typed-decision models** (Jev-shaped
Choice/Score/Noul) out of embroider's ORT runtime, reusing the `acquire` /
`policy` / `providers` / `cache_info` seams — same posture as the Jina and
vision contracts. Research background (model shortlists, ONNX verification,
licences) is in the session notes; this doc is the build order.

Non-goals: generative/decoder judges (Qwen causal-LM ONNX, small-LLM +
constrained decoding), hosted APIs (Jev, Clef, OpenAI Decisions, Perplexity
Decider), training/fine-tuning code, embedding-space changes. The frozen
Jina/vision contracts are untouched — new modules only.

## 0. Conventions this plan inherits

- **Pinned stack**: ORT 1.29 (single binary, `ORT_DYLIB_PATH`-overridable),
  `tokenizers` 0.21, Rust core + opt-in PyO3 (`extension-module`).
- **Measured artifacts only**: a (model, precision) pair ships iff parity was
  measured. Precedent: int8 killed for text-small @0.93 drift, q4 killed for
  nano/vision — never shipped.
- **Golden fixtures are the contract proof**
  (`fixtures/golden_*.json`, vendored; okfgraph asserts live outputs against
  them). New contract = new fixture + new minor + compat.md entry.
- **Env-gated parity tests** (`tests/*_e2e.rs`, `#[ignore]`, env vars for
  weights — the `vision_e2e.rs` pattern). No weight downloads in CI.
- **Licence table**: every registry entry records its weight licence in
  `docs/compat.md` in the same change. Note: all current entries are
  CC BY-NC 4.0 — the Apache-2.0 candidates below would be the first
  commercially-usable weights in the registry. Flag that, don't bury it.
- **Thresholds are consumer policy**: embroider returns native scores /
  probabilities; allow/review/block cutoffs live in okfgraph and are re-fit
  per model on held-out labels (same-API ≠ same-quality).

## 1. Registry generalisation (do first, unblocks both tracks)

`ModelSpec` is embedding-shaped (`native_dim`, `ladder`) — rerankers and
deciders have neither. Do **not** stretch it; add siblings:

```rust
pub struct TaskSpec {
    pub id: &'static str,        // canonical id (names the contract)
    pub max_tokens: usize,       // positional ceiling
    pub fp32: Artifact<'static>, // reuse Artifact (repo + file + sidecar)
    pub fp16: Option<Artifact<'static>>,
    pub int8: Option<Artifact<'static>>,
    pub kind: TaskKind,          // Rerank | Decide
}
```

- `builtin_tasks() -> &'static [TaskSpec]`, `lookup_task(id)`,
  `artifact_for_task(id, precision)` mirroring the existing functions
  (same fp32-fallback rule for unlisted pairs; same legacy-layout fallback
  for unknown ids).
- `available_models()` gains the task entries (add a `kind` key; embedding
  keys unchanged — no consumer impact).
- `cache_info` covers task ids through the same delegation it already uses
  (`cache_info_files` covers arbitrary layouts meanwhile — Laya's
  per-checkpoint tokenizer subfolders included).
- Licence column added to the compat.md table per entry, same change.

## 2. Track A — rerank (`src/rerank.rs`)

> **Status 2026-10-28: A0 + A1 landed (0.4.0).** Both spikes verified on
> the ORT 1.29 pin: opset 14, `logits [B,1]` outputs, standard ops only
> (MiniLM 785 nodes / 3-input incl. type_ids, WordPiece; mMARCO 1532 nodes /
> 2-input, Unigram). Torch↔ONNX max|Δ| 2.86e-6 (L-6) / 7.63e-06 (mMARCO)
> at pinned revisions; ORT 1.29.0 CPU ~2 ms/pair (L-6) / ~4–5 ms/pair
> (mMARCO). Goldens + `tests/rerank_e2e.rs` green on the 1.29 dylib.
> Remaining: production bakeoff (v2-m3 export vs Qwen3-seq-cls vs
> jina-v2-multilingual) on okfgraph's language mix.
>
> **Qwen3-0.6B-seq-cls proposal 2026-10-29: DO NOT ADOPT now.** Artifact
> verified (`shawnw3i/...-seq-cls-ONNX` @ e5d273d: single 1.1GB model.onnx,
> opset 17, 2-input, logits [B,1], Apache-2.0; README Mars/Venus example
> reproduced modulo sigmoid). Integration fits the seam (left padding +
> formatter helper + 32K ceiling — small change). KILLER: ~120–150 ms/pair
> CPU (~100x the spikes; an early 1417 ms single-sample was cold-start
> noise, superseded by multi-rep runs) with NO demonstrated CUDA relief
> on the 1.29 pin (EP load flaky on this box; GQA/RoPE-heavy graph →
> expected fallback). mMARCO already covers 100+ langs at 1/50th the cost.
> Adopt IFF a bakeoff shows a quality gap that matters AND serving closes
> the cost gap. bge-v2-m3 (export) and jina-v2-multilingual still
> unexamined — the production slot stays open.
>
> **Precision sweep 2026-10-28 (RTX 3090 + Ryzen 9 5950X, ORT 1.29.0).**
> Latency ms/pair — CPU fp32: L-6 1.3–1.6, mMARCO 2.1–3.2 (flat vs batch).
> CUDA fp32 short pairs (~20 tok), ms/pair, corrected 2nd run: L-6
> b1 3.6 → b2 2.2 → b4 2.0 → b8 1.6 → b16 1.2 → b32 1.0; mMARCO b1 5.3 →
> b16 2.1 → b32 1.7. So: CUDA ≈ CPU or worse at b1 (launch overhead),
> ~1.3–1.9x better at batch, saturating ~b16 — short forwards are
> overhead-bound on both runtimes. (An earlier 14x/8x reading failed to
> reproduce and is discarded as a measurement artifact.) Long pairs
> (~500 tok): CUDA b1 ≈ CPU b1 (L-6 29.5 vs 21.1, mMARCO 49.2 vs 49.9);
> long-batch CUDA degraded anomalously on this desktop rig — no claim
> until a controlled rerun. CUDA↔CPU logit drift ~1e-3 (kernel-order
> noise; goldens stay CPU-only). CUDA path confirmed working; e2e stays
> CPU-gated like vision. Practical upshot: batch candidate lists through
> one `score_batch` call for a free ~1.5x on GPU; CPU stays respectable.
> fp16 (optimum `--dtype fp16` export, keep-io): drift 3.3e-3 (L-6) /
> 5.3e-3 (mMARCO), order kept — but SLOWER than fp32 on both CPU (~1.5–2x)
> and CUDA (launch-bound tiny graphs). Not shipped; warn-and-honour stays.
> int8 (vendor qint8_avx512 / vnni / quint8_avx2, 4x smaller: 23MB / 116MB):
> ~1.4x / ~1.7x faster CPU, rank intact 6/6 on a 12-pair probe — but drift
> up to 0.55 (L-6) / 1.10 (mMARCO) logit units, and the probe is too thin
> for a ship decision (plus int8 needs its own golden set, not the 1e-6
> one). NOT listed in the registry; gate on the production-bakeoff
> question set with a wider rank-parity harness.
> bf16 (optimum `--dtype bf16` export): KILLED at load — ORT 1.29 rejects
> the graph on every provider (`Pow` in LayerNorm fed bf16 is a type
> error; the cast is naive, no op-level keep-list). Harder failure than
> fp16, and fixing it means custom mixed-precision export surgery for a
> dtype that couldn't beat fp32 on launch-bound graphs anyway. Not listed,
> not pursued.

### A0. Artifact verification (per candidate, before any Rust)

Twin-spike: two small models proving the seam across arch *and*
tokenizer variance (BERT WordPiece vs XLM-R SentencePiece), then one
production bakeoff on okfgraph's real language mix.

| Candidate | Role | Licence | Verify |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M) | **spike, English** (proves the pair-encoding seam) | Apache-2.0 | in-repo ONNX + `tokenizer.json`; opset vs ORT 1.29 pin |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M) | **spike, multilingual** (same seam, XLM-R arch + SP tokenizer) | Apache-2.0 | in-repo `onnx/model.onnx` (plain graph, not the O1–O4 optimum variants) + `tokenizer.json`; opset vs pin; ~5× L-6 params — record CPU latency, still spike-cheap |
| `BAAI/bge-reranker-v2-m3` (568M) | production multilingual (self-export, no in-repo ONNX) | Apache-2.0 | export recipe + parity vs torch |
| `shawnw3i/Qwen3-Reranker-0.6B-seq-cls-ONNX` (596M) | production alt (100+ langs, 32K ctx, seq-cls graph) | Apache-2.0 | single `model.onnx`; prove `score = w·h[CLS]` parity with Route-B torch |
| `jinaai/jina-reranker-v2-base-multilingual` (278M) | production alt, stronger mid-size (in-repo ONNX incl. fp16/int8) | CC BY-NC-4.0 | input format (torch side is `custom_code`); licence = same non-commercial status quo as today's Jina weights |
| `cross-encoder/ettin-reranker-{17,32,68,150}m-v1` | reserve tiny siblings | (verify) | scope/licence/ONNX unverified — check before adopting |
| `cross-encoder/msmarco-MiniLM-L12-en-de-v1` | niche | (verify) | EN–DE bilingual only |

Per candidate, record: file layout + sidecar, opset, tokenizer path,
parity numbers vs torch/`sentence-transformers` on a fixed query/doc set,
CPU latency @ batch 1/8. Kill criteria: custom ops, opset above the pin,
logit drift beyond tolerance, licence mismatch.

### A1. `Reranker` (mirror the `JinaV5` seams)

- `open(model_id, revision, cache_dir, device, max_length, precision, cpu_arena)`
  and `open_files(onnx_path, tokenizer_path, …)` — same signatures minus
  `truncate_dim` (no Matryoshka ladder; scores are scalars).
- `score_one(query, doc) -> f32`, `score_many(pairs) -> Vec<f32>` (batched
  pair encoding, one session run per chunk — the `ONNXAgent.predict_batch`
  shape).
- Pair encoding: `[CLS] query [SEP] doc [SEP]`; truncation policy =
  truncate-doc-side-first, `max_length` caps the pair (parameter, not
  contract — same rule as embeddings §0.1.4 note in compat.md).
- `SessionPolicy::rerank_pair()` — start from `text_embed()` (Level3,
  intra=phys/2, inter=1), re-measure; short pairs may want a different
  batching sweet spot.
- Golden fixtures, one per spike model (`fixtures/golden_rerank_minilm.json`,
  `fixtures/golden_rerank_mmarco.json`): fixed pairs, expected logits, abs
  tolerance **1e-6** (logits, not cosine — rerank scores have no
  normalisation to hide behind). Both tokenizers load through the same
  `open_files(onnx_path, tokenizer_path)` seam — no per-arch special cases
  in the loader.
- PyO3 `PyReranker`: `open` / `open_files` / `score` / `score_batch` +
  `model_id`, `used_cuda`, `max_length`, `precision` getters.
- Env-gated `tests/rerank_e2e.rs` (`RERANK_E2E_ONNX/TOK` + 
  `RERANK_MM_E2E_ONNX/TOK` for the multilingual twin), ignored by default.

### A2. Acceptance

`cargo test --locked` green; golden fixture asserted in CI (CPU, fp32);
spike model end-to-end from `cache_info` → `open` → `score_batch`;
compat.md row (licence, precisions measured, truncation policy); CHANGELOG.

## 3. Track B — typed decision (`src/decide.rs`)

### B0. Artifact bakeoff (before any Rust)

Contenders: **Laya English** (Jev-shaped Choice/Score/Noul natively, first-party
`ONNXAgent` = proven ORT graph, community exports e.g.
`inferenceprince/laya-onnx` as cross-check) vs **GLiClass-small-v1.0**
(in-repo ONNX + int8, general classifier, labels-spliced input format to
port). Record per contender: exact ONNX file + tokenizer path (Laya's live
in per-checkpoint subfolders), opset, input-construction recipe (special
tokens / question schema), parity vs torch runtime on a fixed state×question
set (probabilities, abs tolerance), CPU latency, licence (both Apache-2.0).

Decision rule: Laya if its ONNX graph loads on the 1.29 pin at parity —
native typed questions are worth more than GLiClass's smaller size;
GLiClass-small stays the fallback (144M, single-pass, proven in-repo ONNX).

### B1. `Decider` (typed questions in, typed answers + probabilities out)

```rust
pub enum Question { Choice { instructions: String, criteria: Vec<(String, String)> },
                    Score  { instructions: String, /* … */ },
                    Noul   { instructions: String, /* … */ } }
pub struct Decision { pub label: String, pub probabilities: Vec<(String, f32)>, /* … */ }
```

- `open` / `open_files` mirror the `Reranker` shape. `decide(state, &[Question])`
  single-pass batches all questions for one state (the GLiClass/Laya
  single-forward property — the whole point vs per-label NLI forwards);
  `decide_batch(states, …)` for throughput.
- Input construction (special tokens, label splicing) is **part of the
  contract**: golden fixture pins token streams, not just outputs, so a
  regression in prompt formatting fails loudly.
- Probabilities returned native and unthresholded; abstention lives in the
  consumer (thresholds re-fit per model — §0).
- Golden fixture `fixtures/golden_decide_<model>.json`: fixed
  state×question sets, expected labels + probability vectors, abs tolerance
  1e-6 on probs, exact-match on labels.
- PyO3 `PyDecider` + `cache_info` coverage via the task registry; env-gated
  `tests/decide_e2e.rs`.

### B2. Acceptance

Same as A2, plus: per-question-type parity (Choice/Score/Noul each pinned);
multilingual checkpoint (if adopted) gets per-language golden cases, not one
aggregate — the vs-laya review flags weak slices hiding in averages.

## 4. Sequencing and releases

1. **Registry generalisation** (§1) — no model, no weights; pure refactor +
   `available_models`/`cache_info` extension. Ships in a patch.
2. **Twin MiniLM spike** (A0-spikes + A1 + A2) — proves pair-encoding →
   logits → goldens → `cache_info` across both arch/tokenizer families.
   Release **0.4.0** (new contract, two fixtures, compat.md rows, okfgraph
   floor-pin coordination).
3. **Production reranker decision** (A0 bakeoff v2-m3-export vs Qwen3-seq-cls
   ONNX on okfgraph's language mix) → second registry entry, same 0.4.x
   mechanics. If the winner needs a new input contract (32K context
   handling), it takes its own minor.
4. **Decider bakeoff → `decide.rs`** (B0 + B1 + B2). Release **0.5.0**.
5. okfgraph-side (out of scope here, listed so it isn't dropped):
   confidence-gated routing (cheap local decide → rerank → LLM/human
   escalation), threshold re-fit harness on held-out labels, regime curves
   (labels × traffic × review coverage) before promoting any candidate.

## 5. Risks

- Custom architectures (`LayaTypedDecisions`, `GLiClassModel`) — torch side
  needs `trust_remote_code`; the ONNX graph is the contract, pin bytes +
  fixture it. If a graph needs custom ops on the pin, kill it.
- Tokenizer layouts that aren't repo-root `tokenizer.json` (Laya
  subfolders, DeBERTa SentencePiece) — `open_files` takes explicit paths;
  registry entries record the exact file.
- Publisher numbers (Laya 33 ms, multilingual coverage, Clef benchmarks)
  are vendor results until reproduced on our hardware — the e2e tests are
  the reproduction.
- Licence drift: re-check card licences at adoption (Jina v3 family showed
  survey-vs-API disagreement); Apache-2.0 entries change the registry's
  commercial story — update okfgraph's README model-licence section too.
