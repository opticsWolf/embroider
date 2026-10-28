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
