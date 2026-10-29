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

**Moved to `docs/reranker-plan.md`** (2026-10-29): the consolidated
reranker record — architecture, artifact decisions, verification,
precision sweep, `score_long`, rejected alternatives (Qwen3, fp16, bf16),
and remaining work. Shipped 0.4.0/0.4.1 on `dev`; the production-bakeoff
decision lives there (§7.2), gated on okfgraph data.

## 3. Track B — typed decision (`src/decide.rs`)

**Moved to `docs/typed-decision-plan.md`** (2026-10-29): the consolidated
typed-decision record — verified contender repos, `Decider` architecture,
B0 bakeoff protocol (G0–G5) + execution logs (GLiClass-small G0–G3 done,
Laya fp16 killed / fp32 hunt / q8 measured), goldens, rejection log,
dual-access design (classifier + rerank-reduction), external corroboration,
and the reranker-learnings traceability table. Status: `decide.rs` +
GLiClass-small spike shipped 0.5.0 on `dev`; Laya + v4-edge bakeoffs queued.

## 4. Sequencing and releases

1. **Registry generalisation** (§1) — no model, no weights; pure refactor +
   `available_models`/`cache_info` extension. Ships in a patch.
2. **Twin MiniLM spike** (A0-spikes + A1 + A2) — proves pair-encoding →
   logits → goldens → `cache_info` across both arch/tokenizer families.
   Release **0.4.0** (new contract, two fixtures, compat.md rows, okfgraph
   floor-pin coordination).
3. **Production reranker decision** (A0 bakeoff v2-m3-export vs
   jina-v2-multilingual vs status-quo mMARCO on okfgraph's language mix;
   Qwen3-seq-cls rejected, reranker-plan §6) → second registry entry,
   same 0.4.x mechanics. If the winner needs a new input contract (32K
   context handling), it takes its own minor.
4. **Decider bakeoff → `decide.rs`** (B0 + B1 + B2) — DONE on `dev`
   (0.5.0: GLiClass-small spike, goldens, e2e, PyO3). Release **0.5.0**
   (merge → tag → publish, pending go-ahead; supersedes the 0.4.1
   release train).
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
