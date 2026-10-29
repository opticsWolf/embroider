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
