# Typed-decision plan (embroider)

**Status:** B0 executing — small-v1.0 G0–G3 done (see §8); Laya +
v4-edge queued. Supersedes Track B of
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

## 1. Contenders (repos verified 2026-10-29 via HF API; small-v1.0 bytes downloaded + G0–G3 done, §8)

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

### 1b. GLiClass family — general classifiers (fallback pool)

All `GLiClassModel`, all Apache-2.0, single-pass labels-spliced input
(ported via the `gliclass` pipeline or `trust_remote_code`); pipeline
thresholds stay consumer-side per §0 conventions. Family survey 2026-10-29
(repo storage bundles all files; downloads/month trail the id):

| Repo | Gen / encoder era | Repo size | In-repo ONNX | Role |
|---|---|---|---|---|
| `knowledgator/gliclass-small-v1.0` | v1.0 DeBERTa, 144M (574 MB weights, 575 MB onnx — measured) | 1.3 GB | ✅ `onnx/model.onnx` + int8, root tokenizer | **B0-first**: cheapest acquisition, unblocks the `decide.rs` shape |
| `knowledgator/gliclass-base-v1.0` / `-large-v1.0` | v1.0, 186M / 438M | — | verify (`-lw` + `-init` variants also exist) | size-up if small underperforms |
| `knowledgator/gliclass-edge-v3.0` | v3.0, ~32M est. (131 MB repo — smallest official) | 131 MB | ❌ none | tiny candidate, needs self-export |
| `knowledgator/gliclass-base-v3.0` | v3.0 (most-pulled GLiClass, 6.7k dl) | 749 MB | ❌ none | quality candidate, needs self-export |
| `knowledgator/gliclass-modern-base/large-v3.0` | v3.0 ModernBERT | 606 MB / 1.6 GB | ❌ none | modern-encoder variants, self-export |
| `knowledgator/gliclass-large-v3.0` | v3.0, ~440M est. | 1.8 GB | ❌ none | top-size v3, self-export |
| `knowledgator/gliclass-v4-edge` | v4, **~440M** (882 MB safetensors fp16 / 1761 MB onnx fp32 — measured; "edge" is branding, not a size claim — same class as large-v1.0) | 3.3 GB (bundles onnx/) | ✅ `onnx/model.onnx` + `model_q8.onnx` | newest official ONNX; quality/size-up candidate, second source |
| `cnmoro/gliclass-*-v3.0-onnx` (edge/base/large/modern-large) | community v3.0 exports (`model.onnx` + `model_i8.onnx`) | — | ⚠️ **no licence tag** — cross-check only, never the pin | external corroboration |

Out of scope: `-init` checkpoints, `-lw` lite variants (revisit if the
base pick is close but heavy), bi-encoder/fused variants (different
tradeoff, breaks the single-pass requirement), backbone swaps
(llama-1.3B, qwen-0.5B/1.5B, v4-2b — GPU-class), third-party finetunes
(BioMike polished etc.).

### Decision rule (unchanged)

Laya if its ONNX graph loads on the 1.29 pin at parity — native typed
questions outrank GLiClass's smaller size. Inside the GLiClass pool:
B0-first is small-v1.0 (in-repo ONNX unblocks `decide.rs` even if Laya
wins — same seam either way); v4-edge is the official-ONNX second source
(newest graph + q8 for the precision sweep); v3-base/edge enter only via
self-export if the v1/v4 graphs disappoint on quality. Community cnmoro
exports are corroboration, never registry pins (untagged licence).

## 2. Architecture to build (`src/decide.rs`)

Mirrors `Reranker` seam-for-seam (deliberately — same reviewers, same
e2e shape):

```rust
// Shipped shape (0.5.0) — the sketch above grew `criteria` into `options`:
// instructions travel as the prompt suffix (prompt_first=false).
pub enum Question { Choice { instructions: String, options: Vec<String> },
                    Score  { instructions: String, options: Vec<String> },
                    Noul   { instructions: String, options: Vec<String> } } // loud-unsupported
pub struct Decision { pub label: String, pub probabilities: Vec<(String, f32)> }
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

1. **B0 execution (small-v1.0 done, §8):** Laya + v4-edge bakeoffs
   queued — same G0–G5 gates. GLiClass-small unblocked the `decide.rs`
   shape as planned (same seam either way).
2. **`src/decide.rs` + goldens + e2e + PyO3 — DONE on `dev` (0.5.0).**
   Release (merge → tag `v0.5.0` → publish) pending go-ahead; okfgraph
   floor-pin coordination as with 0.4.x.
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

## 8. B0 execution log — small-v1.0 (2026-10-29)

Pinned revision `21edefaf` (verified via HF API). 575 MB `onnx/model.onnx`
+ tokenizer + config downloaded; torch side `gliclass` 0.1.21 sdist +
torch 2.15 CPU + `trust_remote_code` for the parity reference.

- **G0 cost: PASS.** ~10.5 ms/decision CPU (seq ~20, ORT 1.29, mean of
  20) vs the ~2 ms/pair rerank spikes — same order of magnitude, 10x
  under the Qwen3-kill line. Cold load 0.8 s (torch: 20.9 s — the
  no-framework dividend, quantified).
- **G1 graph: PASS.** opset 14 (safe on the 1.29 pin), 2497 nodes /
  110 initializers, **zero** non-`ai.onnx` ops. 2 inputs
  (`input_ids`, `attention_mask`, int64 [batch, seq]), 1 output
  (`logits` float; symbolic shape `[position, batch]` resolves to
  `[batch, classes]` at runtime). Self-contained, no external data.
- **G2 parity: PASS, exact.** torch pipeline vs ORT on identical inputs
  match to 3 decimals on all 8 probes (2 README + 6 discrimination ×
  multi/single). The in-repo ONNX is faithful to the safetensors.
  Recipe confirmed: `text + (<<LABEL>>label)* + <<SEP>>`
  (`prompt_first=false`, `label_token_first=True`); special tokens
  verified single-token (128001/128002), not OOV-split.
- **G3 discrimination: SPLIT — the finding of the day.** Single-label
  (softmax): 3/3 correct (sports 0.944, cooking 0.564, politics 0.998).
  Multi-label (sigmoid): systematically over-confident — `quantum
  physics` scores 1.0 on a football text, `knitting` ≥0.98 everywhere,
  `weather` 0.999 on election news. Raw multi-label sigmoids are NOT
  decision-ready; they need the repo's calibrator and/or re-fit
  thresholds — exactly the consumer-policy split §0/§2 mandates. The
  README's discriminating example does not reproduce at this revision
  (vendor numbers are claims until reproduced — old plan §5, confirmed).
- **G4 fp16: N/A** (export is fp32; the in-repo int8 enters the §4
  precision sweep later). **G5 slices:** deferred to goldens (per-type
  cases pinned there).
- **Decision:** small-v1.0 proceeds to `decide.rs` — cheapest faithful
  graph, seam-proving role fulfilled. Multi-label goes behind
  calibration + thresholds (consumer side); single-label is shippable
  native. Laya and v4-edge bakeoffs unchanged in queue (Laya log below).

## 8b. B0 execution log — v4-edge (2026-10-29)

Pinned revision `885af1af` (Apache-2.0). 1.76 GB `onnx/model.onnx` +
647 MB official `model_q8.onnx`, DeBERTa-v3-large backbone (~440M —
"edge is branding, not a size claim").

- **G1: PASS.** 9307 nodes / 414 inits, zero non-standard ops,
  2-input / 1-output like small-v1.0.
- **G2: CONDITIONAL PASS.** torch↔ORT max|Δ| **0.028** (10 probes)
  vs small-v1.0's 3-decimal exactness. Proven recipe-correct, not
  sloppy: the ONNX embeds the **calibrator** — ORT matches the
  pipeline's calibrated logits (d=0.021) and mismatches raw
  `uncalibrated_logits` (d=7.9). The residual is fp32 kernel-order
  accumulation on a 3x-deeper model; argmax 10/10. Goldens pin the
  ORT values and the 0.03 tolerance, not 1e-6.
- **G0: PASS both providers.** CPU ~38 ms/question (slow-tier for
  routing loops); **CUDA ~2 ms/question** (20 ms/10q batch) —
  routing-viable. Torch reference needs `.float()` on CPU (weights
  are BF16, no CPU kernels).
- **G3: PASS, and it closes the §8 finding.** Multi-label is
  calibrated now: football→sports 0.96, quantum→technology 0.96
  (v1.0 claimed quantum=1.0 on *football*), knitting→weather 0.62
  with honest second-best sports 0.41 (v1.0: ≥0.98 everywhere).
- **Recipe change recorded: `prompt_first: true`.** v4-edge splices
  labels-first (`<<LABEL>>* + <<SEP>> + prompt? + text`) where
  small-v1.0 is text-first. A v4 registry entry needs its own input
  layout in `decide.rs`, not a parameter flip. Also new: `prompt`
  (instruction/policy), 2048 ctx, few-shot, hierarchical labels,
  and a documented rerank recipe (passages × [query]) — the
  dual-access pattern, upstream-confirmed.
- **q8: REJECTED.** Official quant drifts **5.5** on the logit
  scale; argmax survives 10/10 on easy probes but the probabilities
  (the contract) are garbage. Same verdict as Laya int8, third
  confirmation that quant needs the harness, not armchair.

Standing: v4-edge is the better *model* (calibrated, instruction-
 following); small-v1.0 is the shipped *spike* (simpler layout,
 already golden-pinned). Both can coexist as registry entries; the
 v4 layout is B1 follow-up work.

## 9. B0 execution log — Laya (2026-10-29, via gossamer)

Goal refinement: one weights family, two contracts — Laya as classifier
(native Choice/Score/Noul) AND as reranker (binary-relevance Choice
reduction: state=doc, question=query, P(relevant) as the score). The
reduction mapping is contract and will be golden-pinned like any other.

- **fp16 export (inferenceprince/laya-onnx @ `dd0f4db`): G1 pass, G0 KILL.**
  General 5-input marker form, `logits` + `act_logits`, loads 0.9 s;
  recipe reproduces the README (billing 0.97). But **~2.6 s/forward CPU**
  (seq 44) — ~1000x the rerank spikes, the fp16-on-CPU pathology at
  421M scale. Dead for CPU serving; CUDA unmeasured (box has a 3090).
  Base-checkpoint caveat (Ollaya): base `laya` is near-chance zero-shot
  on typed-decisions (0.357) — classifier duty wants the
  `laya-typed-decisions` checkpoint (0.767 vs Jev 0.727), not the base
  export. Revival conditions: CUDA numbers OR fp32 graph.
- **Non-fp16 hunt (all Apache-2.0-tagged or base-model-derived):**
  Mattepiu/laya-onnx @ `a7f385b` root `laya.onnx` (1.7 GB, true fp32) is
  **binary-only** (`marker_pos [B,2]`) — fits Noul true/false and the
  binary rerank reduction, not general Choice. Same for their int8.
  wexare/laya-onnx @ `2ac7176` `laya_q8.onnx` (633 MB, opset 18) is the
  **general form**, loads 0.6 s on the 1.29 pin.
- **wexare q8 on CPU: functional, slow, thin.** Billing-choice correct
  (0.969) at **~450 ms/decision** (45x GLiClass, 200x spikes — fails
  G0 for CPU routing loops, viable as a slow-tier/GPU-standby).
  Relevance reduction discriminates the right direction (pos 0.81 vs
  neg 0.69 P(relevant)) but the gap is thin — rerank duty needs the
  eval harness, not armchair (Mattepiu's own table: search relevance
  0.628, weakest family alongside sentiment 0.442; intent/routing 0.991
  and moderation 0.967 are the strong suits).
- **Sanctioned pattern (Ollaya, corroboration not measurement):** each
  checkpoint carries fp16 + fp32 graphs over one weights file, CPU
  loads fp32 — 100% answer agreement vs torch fp32 ref (Δp ≤ 1.1e-4)
  over 2,383 questions × 3 checkpoints. Our fp32 comes from a
  third-party binary-only export; the general-fp32 path is Ollaya's
  export script or a self-export (optimum) from `laya-typed-decisions`.
  Their serving numbers: 14–20 ms/5-question request on RTX 5090 fp16.
- **Standing order:** (1) GLiClass-small native (done, both paths —
  decide + relevance reduction — work today); (2) Laya classifier via
  typed-decisions checkpoint in fp32-graph or int8 form, G0–G5 on our
  hardware — **bakeoff done, see §9b**; verdict: adopt (CUDA) / slow-tier
  (CPU); (3) Laya-as-reranker iff the eval harness shows a gap worth
  the provider-appropriate cost (15 ms CUDA, 225 ms CPU). Binary
  Mattepiu fp32 stays the Noul/binary workbench.

## 9b. B0 execution — laya-typed-decisions self-export fp32 (2026-10-29)

Upstream: `convaiinnovations/laya-typed-decisions` @ `1a793eb` (the
revision desplega validated; HEAD is `e929ae5`). Apache-2.0. This
checkpoint's config — **max_len 1024 / head_max_len 256**, fitted
per-bucket temperatures (`temperature_by_options`) — differs from the
base export's 512/192; the registry entry carries 1024, not the
decide.rs default.

Export (self, recipe mirrors MstyAI): torch `DecisionModel` via
`torch.onnx.export(dynamo=True, opset_version=18)` → single graph,
5 inputs (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`,
`qtype`), 2 outputs (`logits`, `act_logits`). Two pitfalls recorded:
(1) `dynamic_shapes=True` is rejected — pass an explicit Dim dict
(batch/seq≤1024/k≤256) or the trace silently bakes static dims
(first export scored until a 1-row input failed `Expected: 4`);
(2) run the console UTF-8 (`PYTHONUTF8=1`), the export logger prints
non-cp1252 glyphs. Result: 3.3 MB graph + 1.69 GB `.onnx.data`
sidecar (the `Artifact.sidecar` shape, as with the fp16 export).

- **G1: PASS.** 1606 nodes, 32 distinct ops, zero non-standard.
- **G2: PASS.** torch↔ORT max|Δ| **2.0e-6** on `logits` (4-row batch
  covering choice/score/noul); `act_logits` 1.5e-3 on a ±4887 scale
  (relative 3e-7 — negligible; act head feeds escalation, not answers).
- **G0: SPLIT by provider (py-ort 1.31; 1.29-pin number pending).**
  CPU: ~65 ms/single-question, ~225 ms/4-row batch — 30x the spikes,
  slow-tier only. **CUDA (RTX 3090): ~15 ms/4-row batch** (~3.6 ms/row,
  15x over CPU) — routing-viable. Verdict: adopt for CUDA serving,
  slow-tier on CPU; the q8 (450 ms) is retired by this graph on both.
- **G3: PASS.** 4/4 probes correct (billing-choice, boiler-choice,
equipment-noul, sentiment-score) with correct-option margins.
- **Temperature application verified exact.** `t_scale =
  temperature_by_options.get(temp_bucket(qtype, k), temperature[qtype])`;
billing softmax(L/1.7601) reproduces the agent's [0.7463, 0.2086,
  0.0451] to 4 decimals. embroider must replicate three details:
  bucket key `f"{type}:{size}"` (2 / 3-5 / 6-10 / 11+), **clamp
  [0.5, 5.0] with rejection warnings** (`clamp_temperature` /
  `temperature_rejected`), and the `choice:11+` 0.1006 entry which the
  clamp refuses (it would sharpen ~10x — the checkpoint's own code
  calls that dishonest). No `binning_map` ships by default (`None`).
- **G4: fp16 adopted (CUDA-only), int8 rejected.** fp16 cast (IO fp32)
  drifts 1.5e-3 (td) / 3.9e-3 (base) / 4.3e-3 (multi) with no CPU
  relief — the rerank lesson verbatim — so the fp16 graphs ship as
  CUDA-serving variants (half download/VRAM), not CPU upgrades.
  int8 is **dead on three arms** (td representative): dynamic
  full-graph drifts 2.8–5.4 with flips; static full-graph (30-input
  calibration) drifts 9997 — the −1e4 mask sentinel is unquantizable;
  static **encoder-only** (fp32 head, split verified exact at 0.0)
  still drifts 2.85 at 2/4 with no CPU speedup (241 vs 234 ms).
  Not hosted on any checkpoint. Two converter pitfalls recorded:
  `onnx.save_model` appends to an existing `.data` path (delete
  first), and naive all-initializer fp16 casting breaks shape-sensitive
  nodes (use the Cast-inserting converter, not a dtype loop).
  **G5:** per-type cases pin in goldens.

- **Siblings (same recipe, 2026-10-29):** base/English @ `55cf4c4e`
  (G1 1833 nodes, G2 4.7e-6; CPU ~220 ms / CUDA ~15 ms; quality caveat:
  near-chance zero-shot per Ollaya, export faithful but not for
  classifier duty) and multilingual @ `e4e9ddf2` (G1 1479 nodes, G2
  7.9e-6; CPU ~97 ms / CUDA ~10 ms, fastest of the three on CPU).
  All three publish as public single-graph fp32 repos under
  `opticsWolf/laya-{typed-decisions,english,multilingual}-fp32-onnx`
  (manifest-pinned SHA-256). Served in-embroider by `LayaDecider`
  (0.7.0: `src/laya.rs` ports the marker contract 1:1 — the earlier
  "serve via `ONNXAgent`" decision is superseded; `ONNXAgent` remains
  the golden-builder the e2e pins against).

## 10. External corroboration (2026-10-29, via gossamer)

Three sources, one consistent story — classifier-as-reranker is a
validated pattern, and the reduction is TWO mappings, not one.

- **hev, "Jev as a Reranker" (BEIR: SciFact/NFCorpus/FiQA):** the
  canonical method is one **Noul per candidate** ("this document is
  relevant to the query"), sort by P(true). Jev-batch 0.501 mean
  nDCG@10 — alongside Voyage 0.504, Cohere 0.486, Jina 0.492; ahead of
  MiniLM-L6 0.447, bge-v2-m3 0.471, Qwen3-0.6B 0.478, BM25 0.404.
  Latency 223 ms p50 / 1.4 s p95 batch, $0.54/1k queries. Open-weights
  latencies underline our G0 rule: Qwen3 5.5 s, v2-m3 2.2 s local.
  Lesson for us: our Noul is the rerank primitive (currently
  loud-unsupported — Laya serves it, GLiClass approximates it via
  binary Choice); the okfgraph eval harness should reuse hev's axis
  (same 3 corpora, nDCG@10, paired per-query) so our spikes compare
  against published numbers, not vibes.
- **hotchpotch/jev-reranker (Python lib):** `rerank()` (order,
  threshold 0.0) vs `relevance_rerank()` (filter with a relevance
  prompt that pushes weak candidates toward zero, threshold 0.2) —
  **different prompts, because the goals conflict** (filtering crushes
  relative order). Ships built-in prompts
  (`src/jev_reranker/instructions.py`) + an eval harness vs
  Sentence-Transformers CrossEncoder. Lesson for us: the reduction is
  two golden-pinned mappings (rank-mode, filter-mode), not one; and
  the instruction text is contract — version it with the goldens.
- **Pinecone, "Rerankers and Two-Stage Retrieval":** standard framing
  (bi-encoder recall → cross-encoder reorder; accuracy for time). No
  new facts; confirms the retrieve-then-rerank placement (§1.5 of the
  okfgraph plan) is the industry default, not our invention.

