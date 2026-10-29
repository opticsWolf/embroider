# Typed-decision plan (embroider)

**Status:** B0 (artifact bakeoff) in progress — repos verified, no bytes
downloaded yet. Supersedes Track B of
`docs/rerank-typed-decision-plan.md` (which keeps the registry record and
the sequencing overview).

## 0. Goal and non-goals

**Goal:** serve Jev-shaped typed decisions — Choice / Score / Noul
questions over a state, one calibrated distribution per question, single
forward pass — from embroider's ORT runtime, reusing the `acquire` /
`policy` / `providers` / `cache_info` seams. Same posture as the reranker:
no torch anywhere in the chain, ONNX graph is the contract.

**Non-goals:** generative judges, per-label NLI forwards (defeated by
design — the single-pass property is the whole point), threshold policy
(consumer side, re-fit per model), training/fine-tuning.

## 1. Contenders (verified 2026-10-29 via HF API, no downloads yet)

### 1a. Laya — native typed decisions (lead)

- Base `convaiinnovations/laya` @ `7b928d8`: custom `LayaTypedDecisions`,
  Apache-2.0, three checkpoint variants (default / typed-decisions /
  multilingual), tokenizers in per-checkpoint subfolders, **no in-repo
  ONNX** (39 files, none `.onnx`).
- Export `inferenceprince/laya-onnx` (Apache-2.0, English): single
  `model.onnx` (3.3 MB graph) + `model.onnx.data` (842 MB **fp16**
  weights — external-data sidecar, the `Artifact.sidecar` path exists for
  exactly this), `tokenizer/` subfolder, `rl_agent_config.json` carrying
  `max_len`, `head_max_len`, and **fitted temperatures**.
- Contract per the export README: 5 inputs (`input_ids`,
  `attention_mask`, `marker_pos`, +2), outputs `logits` + `act_logits`;
  encoder ModernBERT-large (421M); `max_len` 512 covering
  instructions+options+state with options sharing `head_max_len` 192;
  question types `choice`/`score`/`noul`; prompt format fully documented
  (CLS/SEP markers, temperature application).
- Torch-side parity needs `trust_remote_code` against the base repo —
  the ONNX graph is the contract; pin bytes + fixture it (the custom-arch
  risk from the old plan, unchanged).

### 1b. GLiClass-small — general classifier (fallback)

- `knowledgator/gliclass-small-v1.0` (lowercased canonical id):
  `GLiClassModel`, Apache-2.0, card explicitly allows commercial use.
- **In-repo** `onnx/model.onnx` + `onnx/model-int8-quantized.onnx`, root
  `tokenizer.json` — the easiest acquisition of any candidate so far.
- Labels-spliced input format (ported via the `gliclass` pipeline or
  `trust_remote_code`); pipeline thresholds (`threshold=0.5`) stay
  consumer-side per §0 conventions. 144M single-pass.

### Decision rule (unchanged)

Laya if its ONNX graph loads on the 1.29 pin at parity — native typed
questions outrank GLiClass's smaller size. GLiClass-small stays the
fallback (and its in-repo int8 makes it the precision-sweep second
source either way).

## 2. Architecture to build (`src/decide.rs`)

Mirrors `Reranker` seam-for-seam (deliberately — same reviewers, same
e2e shape):

```rust
pub enum Question { Choice { instructions: String, criteria: Vec<(String, String)> },
                    Score  { instructions: String, /* … */ },
                    Noul   { instructions: String, /* … */ } }
pub struct Decision { pub label: String, pub probabilities: Vec<(String, f32)>, /* … */ }
```

- `open` / `open_files` (explicit tokenizer path — Laya's subfolders and
  GLiClass's root file both fit), `decide(state, &[Question])`
  single-pass batches all questions for one state,
  `decide_batch(states, …)` for throughput. Registry `Decide` entries +
  `PyDecider` + `cache_info` coverage + env-gated `tests/decide_e2e.rs`.
- **Temperature:** the fitted per-type temperatures ship *with the model*
  (`rl_agent_config.json`) — they are model parameters, not policy. So:
  applied inside embroider, pinned in goldens. Thresholds stay outside.
  (If a contender ships no fitted temps, identity temperature + a note.)
- **Truncation — truncate-state-side-first.** Instructions + options are
  the question; they are never cut to fit the state. An option block
  alone exceeding `head_max_len` errors. Truncated state tokens are
  counted and reported (the `count_tokens` pattern), not silently dropped.
- **`decide_long` explicitly deferred.** The reranker got MaxP because
  relevance is existential (any part answers). Decisions have no such
  property — a Choice distribution over chunks has no principled
  aggregation — so v1 truncates-with-accounting and chunking waits for
  bakeoff evidence, not analogy. (Per-label Score max-pooling is the
  obvious first experiment if okfgraph needs it; still an experiment.)

## 3. B0 bakeoff protocol (the A0 gates, ported)

Same gate shapes that admitted the rerank spikes, with two additions
(cost fail-fast first, fp16 scrutiny throughout):

- **G0 cost fail-fast (new, the Qwen3 lesson).** Before any parity work:
  time one decision CPU on the ORT 1.29 pin. Anything ~100x the rerank
  spikes (~2 ms/pair) needs a quality case *in the same week* or it dies
  like Qwen3-seq-cls. No 100x-cost model gets a golden fixture.
- **G1 graph inspection:** opset within the 1.29 pin, standard ops only
  (no custom/contrib burden), declared inputs feedable (5-input Laya form
  incl. `marker_pos`; GLiClass spliced form), outputs as documented.
- **G2 torch↔ONNX parity** on a fixed state×question set covering all
  three question types: probabilities abs-tolerated, labels exact-match.
- **G3 discrimination:** planted relevant/irrelevant (or correct/incorrect
  option) spreads per question type — the +7.8/−11.1 sanity, ported.
- **G4 fp16 scrutiny (the reranker lesson).** Laya's only export is fp16:
  fp16 must pass *both* bars that killed it for rerank — parity drift
  within tolerance AND faster-or-equal on CPU (it was slower). If fp16
  fails: self-export fp32 from base safetensors via optimum, re-run G1–G3
  on the self-export. If that fails: Laya dies, GLiClass-small leads.
- **G5 slices, not aggregates:** per-question-type and per-language golden
  cases (the befailproof discipline — weak slices hide in averages).

Revisions pinned at first download; licences re-checked at adoption
(the Jina-v3 survey-vs-API precedent); Apache-2.0 entries update
okfgraph's README model-licence section (the registry's commercial story
changes the moment the first one lands — flag, don't bury).

## 4. Verification & goldens

- `fixtures/golden_decide_<model>.json`: fixed state×question sets,
  expected labels + probability vectors (post-temperature), abs 1e-6 on
  probs, exact-match on labels — **plus pinned token streams** (input
  construction is contract; a prompt-format regression fails loudly).
- e2e asserts parity + per-type relevant-beats-irrelevant + validation
  order, same shape as `rerank_e2e.rs`.
- Precision sweep reuses the §5 matrix from `reranker-plan.md` verbatim
  (fp32 default claim, CUDA batch curve, fp16/int8/bf16 fates) — GLiClass's
  in-repo int8 gets a real harness this time, not a 12-pair probe.

## 5. Rejection log

- **Per-label NLI forwards** (DeBERTa-v3-large-zeroshot, bart-large-mnli):
  rejected by design — one forward per label per question cannot serve
  okfgraph's routing loop; single-pass is a requirement, not a preference.
- **Jev API / Clef / hosted judges:** non-goals (§0) — privacy posture of
  the graph forbids state exfiltration; recorded, not revisited.
- **Qwen3-seq-cls-as-decider:** inherits the §6 rejection — 100x CPU cost
  with no serving relief; revival needs the same two conditions.
- Open (not rejected): `decide_long` chunking, multilingual Laya
  checkpoint, GLiClass-base/large size-up if small underperforms.

## 6. Remaining work

1. **B0 execution:** download pinned revisions → G0 → G1 → G2/G3 →
   G4 → decision. GLiClass-small first (cheapest acquisition, unblocks
   the `decide.rs` shape even if Laya wins — same seam either way).
2. **`src/decide.rs` + goldens + e2e + PyO3** → release **0.5.0** (minor:
   new contract). okfgraph floor-pin coordination as with 0.4.x.
3. **okfgraph side** (separate plan when 0.5.0 exists): confidence-gated
   routing (cheap local decide → rerank → LLM/human escalation),
   threshold re-fit harness, regime curves before promotion.

## 7. Reranker learnings applied (traceability)

| # | Learning (`reranker-plan.md`) | Application here |
|---|---|---|
| 1 | Measured artifacts only | G0–G5; nothing listed without parity |
| 2 | Goldens are the contract proof | §4, plus token streams pinned |
| 3 | Same-API ≠ same-quality; thresholds consumer-side | §0, §2 temperature/threshold split |
| 4 | Twin spikes absorb arch+tokenizer variance | Laya (5-input marker form) vs GLiClass (spliced form) through one seam |
| 5 | Cost measured before adoption (Qwen3) | G0 fail-fast gate |
| 6 | fp16 needs both bars (rerank rejection) | G4 + fp32 self-export fallback |
| 7 | Slices, not aggregates | G5 |
| 8 | Plan-doc-only stays out of CHANGELOG | This doc ships no CHANGELOG entry |
| 9 | Know when the analogy breaks | `decide_long` deferred (§2), not ported |
| 10 | Licence re-check at adoption | §3 closing paragraph |
