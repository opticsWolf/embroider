# Rerank + typed decisions — consolidated record

**Status 2026-11-07:** the single record for rerank and typed-decision
work in embroider. Everything below shipped: rerank 0.4.0/0.4.1,
decide 0.5.0, labels-first layout 0.6.0, Laya 0.7.0 (+0.8.0 long-scan,
binning, Noul fix), runtime inventory 0.9.0, Decider-2B 0.10.0, q8 tier
0.11.0, ORT 1.31.1 goldens 0.13.0. Supersedes and replaces three
documents, whose content is preserved verbatim below with status
annotations: `docs/reranker-plan.md` (Part II), `docs/typed-decision-plan.md`
(Parts III–IV), and `docs/rerank-typed-decision-plan.md` (Part I).
Sections marked **Status 2026-11-07** are the only new prose; everything
else is carried over. Open work lives in Part V, rejections in Part VI.

---

# Part I — Shared foundation (from `rerank-typed-decision-plan.md`)

Goal: serve **cross-encoder rerankers** and **typed-decision models** (Jev-shaped
Choice/Score/Noul) out of embroider's ORT runtime, reusing the `acquire` /
`policy` / `providers` / `cache_info` seams — same posture as the Jina and
vision contracts. Research background (model shortlists, ONNX verification,
licences) is in the session notes; this doc is the build order.

Non-goals: generative/decoder judges (Qwen causal-LM ONNX, small-LLM +
constrained decoding), hosted APIs (Jev, Clef, OpenAI Decisions, Perplexity
Decider), training/fine-tuning code, embedding-space changes. The frozen
Jina/vision contracts are untouched — new modules only.

## I.0. Conventions this plan inherits

- **Pinned stack**: ORT 1.29 (single binary, `ORT_DYLIB_PATH`-overridable),
  `tokenizers` 0.21, Rust core + opt-in PyO3 (`extension-module`).
  (**Status 2026-11-07:** historical — the pin moved to ORT 1.31.1 at
  0.13.0 with goldens re-recorded; the single-binary and override rules
  are unchanged.)
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
  (**Status 2026-11-07:** done — the table carries per-entry licences;
  Apache-2.0 entries (rerank spikes, GLiClass, Laya self-exports,
  Decider-2B, Gemma 2) duly flagged on arrival.)
- **Thresholds are consumer policy**: embroider returns native scores /
  probabilities; allow/review/block cutoffs live in okfgraph and are re-fit
  per model on held-out labels (same-API ≠ same-quality).

## I.1. Registry generalisation (do first, unblocks both tracks)

**Status 2026-11-07: DONE (0.3.4 registry, 0.4.0 contract).** `TaskSpec` /
`TaskKind` (`Rerank`/`Decide`) shipped as the non-embedding sibling of
`ModelSpec`; `builtin_tasks()` / `lookup_task()` / `artifact_for_task()`
mirror the embedding functions with identical fallback semantics;
`available_models()` carries the additive `kind` key; `cache_info` covers
task ids. Original text preserved below.

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

## I.2. Sequencing and releases

**Status 2026-11-07: items 1–4 DONE; item 5 is okfgraph-side (Part V).**

1. **Registry generalisation** (§I.1) — no model, no weights; pure refactor +
   `available_models`/`cache_info` extension. Ships in a patch. → **Done (0.3.4/0.4.0).**
2. **Twin MiniLM spike** (A0-spikes + A1 + A2) — proves pair-encoding →
   logits → goldens → `cache_info` across both arch/tokenizer families.
   Release **0.4.0** (new contract, two fixtures, compat.md rows, okfgraph
   floor-pin coordination). → **Done (0.4.0, 2026-10-28).**
3. **Production reranker decision** (A0 bakeoff v2-m3-export vs
   jina-v2-multilingual vs status-quo mMARCO on okfgraph's language mix;
   Qwen3-seq-cls rejected, Part II §6) → second registry entry,
   same 0.4.x mechanics. If the winner needs a new input contract (32K
   context handling), it takes its own minor. → **Open (Part V.1).**
4. **Decider bakeoff → `decide.rs`** (B0 + B1 + B2) — DONE on `dev`
   (0.5.0: GLiClass-small spike, goldens, e2e, PyO3). Release **0.5.0**
   (merge → tag → publish, pending go-ahead; supersedes the 0.4.1
   release train). → **Done (0.5.0 shipped; Laya 0.7.0, v4 layout 0.6.0,
   Decider-2B 0.10.0 followed).**
5. okfgraph-side (out of scope here, listed so it isn't dropped):
   confidence-gated routing (cheap local decide → rerank → LLM/human
   escalation), threshold re-fit harness on held-out labels, regime curves
   (labels × traffic × review coverage) before promoting any candidate.
   → **Consumer-side, partly done (Part V.6).**

## I.3. Risks

- Custom architectures (`LayaTypedDecisions`, `GLiClassModel`) — torch side
  needs `trust_remote_code`; the ONNX graph is the contract, pin bytes +
  fixture it. If a graph needs custom ops on the pin, kill it.
  (**Status 2026-11-07:** held — every shipped graph has zero
  non-standard ops.)
- Tokenizer layouts that aren't repo-root `tokenizer.json` (Laya
  subfolders, DeBERTa SentencePiece) — `open_files` takes explicit paths;
  registry entries record the exact file.
- Publisher numbers (Laya 33 ms, multilingual coverage, Clef benchmarks)
  are vendor results until reproduced on our hardware — the e2e tests are
  the reproduction.
- Licence drift: re-check card licences at adoption (Jina v3 family showed
  survey-vs-API disagreement); Apache-2.0 entries change the registry's
  commercial story — update okfgraph's README model-licence section too.

---
# Part II — Rerank (from `docs/reranker-plan.md`)

**Status 2026-11-07:** shipped 0.4.0 (contract) + 0.4.1 (`score_long`),
merged, tagged, published; q8 tier added 0.11.0; goldens re-recorded onto
ORT 1.31.1 at 0.13.0 (max drift 2.9e-6, mMARCO q8 bound raised 0.9 to 1.0).
The "unmerged, untagged" header below is the 2026-10-28 record, kept
verbatim. Remaining work moved to Part V.

**Status (as written 2026-10-28):** shipped on `dev` (0.4.0 contract +
0.4.1 `score_long`), unmerged, untagged. This document is the consolidated
record: what was built, what was measured, what was decided, and what
remains. Supersedes Track A of `docs/rerank-typed-decision-plan.md`
(which keeps the typed-decision track).

## II.0. Goal and non-goals

**Goal:** serve cross-encoder rerankers from embroider's ONNX Runtime —
pair in, relevance logit out — reusing the `acquire` / `policy` /
`providers` / `cache_info` seams, same posture as the Jina and vision
contracts.

**Non-goals:** generative/decoder judges (causal-LM ONNX, small-LLM +
constrained decoding), hosted APIs (Jev, Clef, OpenAI Decisions,
Perplexity Decider), training/fine-tuning, embedding-space changes. The
frozen Jina/vision contracts are untouched — new modules only.

## II.1. Architecture (as built)

- **Registry** (`src/acquire.rs`, 0.3.4): `TaskSpec` / `TaskKind`
  (`Rerank`/`Decide`) — the non-embedding sibling of `ModelSpec` (no
  `native_dim`/`ladder`; scores have no vector space). `builtin_tasks()` /
  `lookup_task()` / `artifact_for_task()` mirror the embedding functions
  with identical fallback semantics (unlisted pairs → fp32, unknown ids →
  legacy `onnx/model.onnx` layout). `cache_info` resolves task ids through
  the same delegation + id/tier patching; `available_models()` gains an
  additive `kind` key (`embed`/`rerank`/`decide`).
- **`Reranker`** (`src/rerank.rs`, 0.4.0): `open/open_files` (same
  device→precision ordering and registry ceilings as `JinaV5`, minus
  `truncate_dim`), `score_one` / `score_many` (one padded session run per
  batch call, order-preserving), `count_tokens` (truncation-free handle,
  shared helper with JinaV5), `score_long` (0.4.1, §II.4).
- **Pair encoding:** `[query block, doc block]` with
  truncate-doc-side-first (`OnlySecond`) and per-batch longest padding.
  `token_type_ids` fed only when the graph declares them (MiniLM 3-input
  vs mMARCO 2-input, discovered at load); `logits` output preferred when
  declared. Over-long queries error instead of scoring garbage.
- **Session policy:** `SessionPolicy::rerank_pair()` (starts from the
  text-embedding tuning; named to diverge independently).
- **Python:** `embroider.Reranker` with
  `open/open_files/score/score_batch/score_long/count_tokens` + getters.

## II.2. Artifact decisions

Twin spikes — one English, one multilingual — chosen to absorb arch *and*
tokenizer variance through the same seam early. Production slot stays open
pending okfgraph's language mix. (**Status 2026-11-07:** still the shipped
pair; bakeoff in Part V.1.)

| Model | Params | Licence | ONNX | Verdict |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (WordPiece, Bert seq-cls) | 22.7M | Apache-2.0 | in-repo `onnx/model.onnx` | **spike EN, shipped** |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Unigram, XLM-R seq-cls) | 118M | Apache-2.0 | in-repo `onnx/model.onnx` (not O1–O4) | **spike ML, shipped** |
| `BAAI/bge-reranker-v2-m3` (568M) | 568M | Apache-2.0 | self-export | unexamined; bakeoff candidate |
| `jinaai/jina-reranker-v2-base-multilingual` (278M) | 278M | CC BY-NC-4.0 | in-repo (+fp16/int8) | unexamined; bakeoff candidate |
| `Qwen/Qwen3-Reranker-0.6B` family | 596M | Apache-2.0 | none usable (Route B needs template machinery) | rejected (§II.6) |
| `jinaai/jina-reranker-v3/v3.5` | 597M | CC BY-NC-4.0 | none verifiable | rejected (§II.6) |
| `cross-encoder/ettin-reranker-{17,32,68,150}m` | 17–150M | verify | verify | reserve tiny siblings |

## II.3. Verification (A0 gates, all passed 2026-10-28)

Pinned revisions: MiniLM-L-6 @ `233902d`, mMARCO @ `1427fd6` (Apache-2.0
confirmed on both cards).

- Graphs: opset 14 (safe on the ORT 1.29 pin), `logits [B,1]`, standard
  ops only (MiniLM 785 nodes; mMARCO 1532 nodes). No custom ops, no
  contrib-op burden.
- torch↔ONNX max|Δ|: **2.86e-6** (L-6) / **7.63e-6** (mMARCO)
  (sentence-transformers 6.2.0, torch 2.15.0+cpu, same revision).
- Discrimination on the fixture set: +7.8 vs −11.1 (EN, L-6),
  +4.8/+5.8 vs −6.4/−5.7 (EN/DE, mMARCO).
- Goldens `fixtures/golden_rerank_{minilm,mmarco}.json` (ORT 1.29.0 CPU,
  abs 1e-6, per-pair token counts pinned) + ignored `tests/rerank_e2e.rs`
  (goldens + per-language relevant-beats-irrelevant + single-window
  equivalence + MaxP sanity + validation order) — green on the 1.29 dylib.
  (**Status 2026-11-07:** re-recorded onto ORT 1.31.1 CPU at 0.13.0, max
  drift 2.9e-6; still green.)
## II.4. `score_long` (0.4.1)

Documents past the 512 pair cap: sliding windows over the document with
MaxP aggregation (a document is relevant if any part answers).
`window` caps the whole pair (default: session `max_len`);
`stride` steps the windows (default: 50% overlap); larger-than-budget
strides and query-fills-window fail loudly. Chunk planning
(`plan_windows`) is pure and weight-free unit-tested (single docs, exact
multiples, ragged tails, no-gaps/no-overruns property). Template
overhead (BERT 3 vs XLM-R 4 specials) is derived per architecture by
diffing a probe encode — never hardcoded. Chunks decode→re-encode through
the standard pair seam, so single-window inputs score exactly like
`score_one` (e2e-pinned ≤1e-6).

## II.5. Precision sweep (2026-10-28/29, RTX 3090 + Ryzen 9 5950X, ORT 1.29.0)

Latency in ms/pair; drift = max|Δ| vs fp32 golden.

| Precision | L-6 speed | mMARCO speed | L-6 drift | mMARCO drift | Fate |
|---|---|---|---|---|---|
| fp32 CPU | 1.3–1.6 | 2.1–3.2 | — (golden) | — (golden) | ✅ the default |
| fp32 CUDA b1 → b16 → b32 | 3.6 → 1.2 → 1.0 | 5.3 → 2.1 → 1.7 | ~1e-3 | ~1e-3 | ✅ opportunistic |
| fp16 (optimum export) | 2.9–3.0 slower | 5.2 slower | 3.3e-3 | 5.3e-3 | rejected |
| int8 (vendor qint8/quint8, 4x smaller) | 1.1–1.2 | 1.8–2.2 | ≤0.55 | ≤1.10 | gated (§II.7) |
| bf16 (optimum export) | won't load | won't load | n/a | n/a | killed at validation |

Notes: CUDA ≈ CPU or worse at b1 (launch overhead), ~1.3–1.9x better at
batch, saturating ~b16 — short forwards are overhead-bound on both
(A 14x/8x reading failed to reproduce and was retracted). CUDA↔CPU drift
is kernel-order noise; goldens stay CPU-only. fp16 fails both bars (drift
*and* slower). int8 keeps rank 6/6 on a 12-pair probe but the probe is
too thin for a ship decision and int8 needs its own golden set. bf16
export produces graphs ORT 1.29 refuses on every provider (`Pow` fed
bf16 in LayerNorm — naive cast, no op-level keep-list); fixing that is
converter surgery for a dtype that couldn't beat fp32 on launch-bound
graphs anyway. Long-batch CUDA degraded anomalously on the desktop rig —
no claim until a controlled rerun. (**Status 2026-11-07:** the int8 gate
resolved — weight-only q8 shipped 0.11.0 with its own e2e legs: MiniLM
bound 0.2, mMARCO bound 1.0. The controlled long-batch CUDA rerun never
happened — Part V.3.)
## II.6. Rejected: Qwen3-0.6B-seq-cls (proposal 2026-10-29)

Verified then declined: `shawnw3i/...-seq-cls-ONNX` @ `e5d273d` (single
1.1 GB model.onnx, opset 17, 2-input, logits [B,1], Apache-2.0) reproduces
the README's Mars/Venus example modulo sigmoid. Integration would fit
(left padding + formatter helper + 32K ceiling — small change). Killer:
**~120–150 ms/pair CPU (~100x the spikes) with no demonstrated CUDA
relief on the 1.29 pin**, while mMARCO already covers 100+ langs at
~2 ms/pair. Adopt IFF a bakeoff shows a quality gap worth 100x AND
serving closes the cost gap. (The decoder-input protocol — pre-formatted
instruction-bearing pair blocks — is documented in the investigation;
the formatter design carries over if ever revived.)

Also rejected earlier: Jina v3/v3.5 (no verifiable ONNX, custom modeling,
non-commercial licence, late-interaction engine mismatch).
(**Status 2026-11-07:** both still rejected; revival conditions in Part VI.)

## II.7. Remaining work (as of 2026-10-28 — resolutions in Part V)

1. **Release (needs go-ahead):** merge `dev` → `main`, tag `v0.4.1`,
   publish (PyPI + crates.io, wheels via release.yml). okfgraph's floor
   pin (`<0.4`) can't move until a 0.4.x release exists. Same
   irreversibility rule as okfgraph releases: tag last, explicit approval.
   → **Done (0.4.1 shipped; the train continued through 0.14.0).**
2. **Production bakeoff (blocked on okfgraph data):** bge-v2-m3 export vs
   jina-v2-multilingual vs status-quo mMARCO, decided by held-out labels
   and language mix — not by priors. Includes the int8 rank-parity
   harness gate. → **Bakeoff still open (Part V.1); the int8 gate is
   done (0.11.0 q8 legs).**
3. **okfgraph integration:** separate plan
   (`OKFgraph/docs/reranker-integration-plan.md`) — lazy session,
   `rerank_with_cross_encoder`, dispatch, eval harness. Starts after (1).
   → **Done in okfgraph (Phases 0–2 on dev_0.11.0; see Part V.6).**
4. **Conditional:** Qwen3 left-padding + formatter (only if §II.6 reopens);
   long-batch CUDA rerun on a controlled rig; `score_long` chunk
   attribution (which chunk won) if okfgraph needs citations.
   → **All still conditional/open (Part V.2, V.3).**

## II.8. Conventions this work follows (for the next track)

- Measured artifacts only (quant-spike precedent).
- Goldens are the contract proof; consumers assert live outputs.
- Same-API ≠ same quality: thresholds are consumer policy, re-fit per
  model on held-out labels.
- Plan-doc-only changes stay out of the CHANGELOG; shipped behavior only.

---
# Part III — Typed decisions (from `docs/typed-decision-plan.md`)

**Status 2026-11-07:** shipped — decide 0.5.0 (GLiClass-small spike),
labels-first v4 layout 0.6.0, Laya 0.7.0 (+0.8.0 `predict_long`, binning
map, Noul confidence fix), runtime inventory 0.9.0, Decider-2B 0.10.0,
q8 tier 0.11.0. The "B0 executing / queued" language below is the
2026-10-29 record, kept verbatim.

**Status (as written 2026-10-29):** B0 executing — small-v1.0 G0–G3 done
(see §III.8); Laya + v4-edge queued. Supersedes Track B of
`docs/rerank-typed-decision-plan.md` (which keeps the registry record and
the sequencing overview).

## III.0. Goal and non-goals

**Goal:** serve Jev-shaped typed decisions — Choice / Score / Noul
questions over a state, one calibrated distribution per question, single
forward pass — from embroider's ORT runtime, reusing the `acquire` /
`policy` / `providers` / `cache_info` seams. Same posture as the reranker:
no torch anywhere in the chain, ONNX graph is the contract.

**Non-goals:** generative judges, per-label NLI forwards (defeated by
design — the single-pass property is the whole point), threshold policy
(consumer side, re-fit per model), training/fine-tuning.

## III.1. Contenders (repos verified 2026-10-29 via HF API; small-v1.0 bytes downloaded + G0–G3 done, §III.8)

### III.1a. Laya — native typed decisions (lead)

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
### III.1b. GLiClass family — general classifiers (fallback pool)

All `GLiClassModel`, all Apache-2.0, single-pass labels-spliced input
(ported via the `gliclass` pipeline or `trust_remote_code`); pipeline
thresholds stay consumer-side per §III.0 conventions. Family survey
2026-10-29 (repo storage bundles all files; downloads/month trail the id):

| Repo | Gen / encoder era | Repo size | In-repo ONNX | Role |
|---|---|---|---|---|
| `knowledgator/gliclass-small-v1.0` | v1.0 DeBERTa, 144M (574 MB weights, 575 MB onnx — measured) | 1.3 GB | yes `onnx/model.onnx` + int8, root tokenizer | **B0-first**: cheapest acquisition, unblocks the `decide.rs` shape |
| `knowledgator/gliclass-base-v1.0` / `-large-v1.0` | v1.0, 186M / 438M | — | verify (`-lw` + `-init` variants also exist) | size-up if small underperforms |
| `knowledgator/gliclass-edge-v3.0` | v3.0, ~32M est. (131 MB repo — smallest official) | 131 MB | none | tiny candidate, needs self-export |
| `knowledgator/gliclass-base-v3.0` | v3.0 (most-pulled GLiClass, 6.7k dl) | 749 MB | none | quality candidate, needs self-export |
| `knowledgator/gliclass-modern-base/large-v3.0` | v3.0 ModernBERT | 606 MB / 1.6 GB | none | modern-encoder variants, self-export |
| `knowledgator/gliclass-large-v3.0` | v3.0, ~440M est. | 1.8 GB | none | top-size v3, self-export |
| `knowledgator/gliclass-v4-edge` | v4, **~440M** (882 MB safetensors fp16 / 1761 MB onnx fp32 — measured; "edge is branding, not a size claim" — same class as large-v1.0) | 3.3 GB (bundles onnx/) | yes `onnx/model.onnx` + `model_q8.onnx` | newest official ONNX; quality/size-up candidate, second source |
| `cnmoro/gliclass-*-v3.0-onnx` (edge/base/large/modern-large) | community v3.0 exports (`model.onnx` + `model_i8.onnx`) | — | **no licence tag** — cross-check only, never the pin | external corroboration |

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
(**Status 2026-11-07:** played out as written — small-v1.0 unblocked
`decide.rs` 0.5.0; v4-edge layout shipped 0.6.0 as a second registry
entry; Laya serves three self-exported fp32 checkpoints since 0.7.0.)

## III.2. Architecture to build (`src/decide.rs`)

**Status 2026-11-07: BUILT — `src/decide.rs` + `PyDecider` shipped 0.5.0;
v4 labels-first layout added 0.6.0.** Mirrors `Reranker` seam-for-seam
(deliberately — same reviewers, same e2e shape):

```rust
// Shipped shape (0.5.0) — the sketch above grew `criteria` into `options`:
// instructions travel as the prompt suffix (prompt_first=false).
pub enum Question { Choice { instructions: String, options: Vec<String> },
                    Score  { instructions: String, options: Vec<String> },
                    Noul   { instructions: String, options: Vec<String> } }
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
  (**Status 2026-11-07:** still deferred as designed; note 0.8.0 added
  `LayaDecider::predict_long` — a long-document scan for the Laya slot —
  not a general `decide_long`. See Part V.4.)

## III.3. B0 bakeoff protocol (the A0 gates, ported)

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

## III.4. Verification and goldens

- `fixtures/golden_decide_<model>.json`: fixed state×question sets,
  expected labels + probability vectors (post-temperature), abs 1e-6 on
  probs, exact-match on labels — **plus pinned token streams** (input
  construction is contract; a prompt-format regression fails loudly).
- e2e asserts parity + per-type relevant-beats-irrelevant + validation
  order, same shape as `rerank_e2e.rs`.
- Precision sweep reuses the §II.5 matrix from the reranker record verbatim
  (fp32 default claim, CUDA batch curve, fp16/int8/bf16 fates) — GLiClass's
  in-repo int8 gets a real harness this time, not a 12-pair probe.
## III.5. Rejection log (as of 2026-10-29 — consolidated with resolutions in Part VI)

- **Per-label NLI forwards** (DeBERTa-v3-large-zeroshot, bart-large-mnli):
  rejected by design — one forward per label per question cannot serve
  okfgraph's routing loop; single-pass is a requirement, not a preference.
- **Jev API / Clef / hosted judges:** non-goals (§III.0) — privacy posture of
  the graph forbids state exfiltration; recorded, not revisited.
- **Qwen3-seq-cls-as-decider:** inherits the §II.6 rejection — 100x CPU cost
  with no serving relief; revival needs the same two conditions.
- Open (not rejected): `decide_long` chunking, multilingual Laya
  checkpoint, GLiClass-base/large size-up if small underperforms.
  (**Status 2026-11-07:** multilingual Laya shipped 0.7.0 — done. The rest
  stand: Part V.4, V.5.)

## III.6. Remaining work (as of 2026-10-29 — resolutions in Part V)

1. **B0 execution (small-v1.0 done, §III.8):** Laya + v4-edge bakeoffs
   queued — same G0–G5 gates. GLiClass-small unblocked the `decide.rs`
   shape as planned (same seam either way). → **Done — execution logs in
   §§III.8–III.9b; v4 layout shipped 0.6.0, Laya 0.7.0.**
2. **`src/decide.rs` + goldens + e2e + PyO3 — DONE on `dev` (0.5.0).**
   Release (merge → tag `v0.5.0` → publish) pending go-ahead; okfgraph
   floor-pin coordination as with 0.4.x. → **Done (0.5.0 shipped).**
3. **okfgraph side** (separate plan when 0.5.0 exists): confidence-gated
   routing (cheap local decide → rerank → LLM/human escalation),
   threshold re-fit harness, regime curves before promotion.
   → **Consumer-side (Part V.6).**

## III.7. Reranker learnings applied (traceability)

| # | Learning (Part II) | Application here |
|---|---|---|
| 1 | Measured artifacts only | G0–G5; nothing listed without parity |
| 2 | Goldens are the contract proof | §III.4, plus token streams pinned |
| 3 | Same-API ≠ same-quality; thresholds consumer-side | §III.0, §III.2 temperature/threshold split |
| 4 | Twin spikes absorb arch+tokenizer variance | Laya (5-input marker form) vs GLiClass (spliced form) through one seam |
| 5 | Cost measured before adoption (Qwen3) | G0 fail-fast gate |
| 6 | fp16 needs both bars (rerank rejection) | G4 + fp32 self-export fallback |
| 7 | Slices, not aggregates | G5 |
| 8 | Plan-doc-only stays out of CHANGELOG | This doc ships no CHANGELOG entry |
| 9 | Know when the analogy breaks | `decide_long` deferred (§III.2), not ported |
| 10 | Licence re-check at adoption | §III.3 closing paragraph |
## III.8. B0 execution log — small-v1.0 (2026-10-29)

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
  thresholds — exactly the consumer-policy split §III.0/§III.2 mandates.
  The README's discriminating example does not reproduce at this revision
  (vendor numbers are claims until reproduced — old plan §5, confirmed).
- **G4 fp16: N/A** (export is fp32; the in-repo int8 enters the §III.4
  precision sweep later). **G5 slices:** deferred to goldens (per-type
  cases pinned there).
- **Decision:** small-v1.0 proceeds to `decide.rs` — cheapest faithful
  graph, seam-proving role fulfilled. Multi-label goes behind
  calibration + thresholds (consumer side); single-label is shippable
  native. Laya and v4-edge bakeoffs unchanged in queue (Laya log below).

## III.8b. B0 execution log — v4-edge (2026-10-29)

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
- **G3: PASS, and it closes the §III.8 finding.** Multi-label is
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
v4 layout is B1 follow-up work. (**Status 2026-11-07:** done — the
v4 labels-first layout + second registry entry shipped 0.6.0.)
## III.9. B0 execution log — Laya (2026-10-29, via gossamer)

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
  hardware — **bakeoff done, see §III.9b**; verdict: adopt (CUDA) /
  slow-tier (CPU); (3) Laya-as-reranker iff the eval harness shows a gap
  worth the provider-appropriate cost (15 ms CUDA, 225 ms CPU). Binary
  Mattepiu fp32 stays the Noul/binary workbench.

## III.9b. B0 execution — laya-typed-decisions self-export fp32 (2026-10-29)

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

## III.10. External corroboration (2026-10-29, via gossamer)

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
  against published numbers, not vibes. (**Status 2026-11-07:** the
  okfgraph harness exists (`scripts/eval_search.py`, Phase 3); whether
  it reuses hev's axis is okfgraph-side — Part V.6.)
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
## III.11. decider-2b in-embroider (0.10.0, supersedes the Ollaya-side note)

The §III.8b Evaluator footnote said decider-2b stays Ollaya-side. That was
a serving gap, not a verdict: the user asked for it in the bench, and the
bench drives embroider. So `Decider2B` (`src/decider2b.rs`) ports
`decider-slots-v1` 1:1 — lettered/wide option pieces, isolated yes/no score
rows, ort_rows batching, per-qtype temps from `calibration.json` clamped
to the checkpoint's own [0.05, 5.0]. Noul carries BOTH criteria (every
S1MB noul ships both); no `option_order` (the reference has none). CUDA
runs the parallel executor with default thread pools — the sequential
planner aliases a shape buffer (ORT #4829 class, gossamer-sourced),
`enable_mem_reuse=false` is unreachable from the C API, and
`inter_op=1` re-triggers it (all probed). Goldens
`fixtures/golden_decider2b.json` (8 cases) pin exact ids + 5e-4 on CPU
and CUDA. Bench (GPU-only, same axis as §III.10): classify agree 0.696
full-coverage (800 rows incl. banking77) at ~87 ms/row; noul-as-reranker
nDCG@10 0.944/0.992/0.947/0.954 across the four Nano sets — the best
reranker in the building, ahead of the dedicated cross-encoders on three
of four.

**q8 (0.11.0)**: the 186 decoder linears go to int8 MatMulNBits
(block 32, fp32 scales to match the fp32 activations) via a
hand-rolled swap — the vendored `.safetensors` (bf16 bytes, declared
fp16) is unreadable to the ORT quantizer, so each MatMul B-path is
traced to its initializer, true weights read via `safetensors`, and
orphans dropped with a graph-output-seeded fixpoint. Dynamic-B
MatMuls (slot head) skip themselves; embeddings/head/norms stay in
the sidecar (`Decider2B::open` fetches it alongside the q8 artifact).
Gate: worst 0.0020, 0 flips, 10/10 golden agree (incl. the
0.05-temp noul). Publish: `model_q8.onnx(.data)` next to the fp32 in
`opticsWolf/decider-2b-fp32-onnx`.

---
# Part V — Open and deferred topics (new 2026-11-07)

Accepted work that is queued, conditional, or lives consumer-side. Each
item names its trigger so it can be picked up without re-derivation.

## V.1. Production reranker bakeoff (open, blocked on okfgraph data)

The twin spikes still hold the production slot. The bakeoff from
§I.2.3 — bge-v2-m3 self-export vs jina-v2-multilingual vs status-quo
mMARCO — never ran: none of the candidates is in the registry
(verified 2026-11-07), and the decision was gated on okfgraph's
language mix and held-out labels, not priors. Trigger: okfgraph
supplies the mix + labels; run the A0 gates (same seam) and promote
the winner to a second registry entry under the same 0.4.x mechanics.
The ettin tiny siblings (§II.2) stay the reserve bench.

## V.2. `score_long` chunk attribution (open, conditional on consumer need)

`score_long` aggregates with MaxP but never reports *which chunk won*.
Needs citations in okfgraph would trigger this: return the winning
window (indices, not just the max) alongside the score, e2e-pinned the
same way. Trigger: okfgraph asks for citations over long documents.

## V.3. Long-batch CUDA rerun on a controlled rig (open)

The §II.5 note — long-batch CUDA degrading anomalously on the desktop
rig — was never re-measured on controlled hardware, so no claim ships
with it. Trigger: a controlled rerun (pinned clocks, warm cache) before
any batch-size guidance for CUDA rerank serving.

## V.4. `decide_long` chunking (deferred by design)

`decide_long` stays deferred per §III.2: a Choice distribution over
chunks has no principled aggregation, so the analogy with MaxP breaks
and chunking waits for bakeoff evidence, not enthusiasm. The obvious
first experiment — per-label Score max-pooling — is recorded and still
unrun. Note: 0.8.0 shipped `LayaDecider::predict_long` (long-document
scan for the Laya slot), which covers the long-document need for Laya
checkpoints but is not a general `decide_long`. Trigger: bakeoff
evidence or an okfgraph need that `predict_long` cannot cover.

## V.5. GLiClass size-ups and lite variants (deferred)

GLiClass-base/large-v1.0 size-up, `-lw` lite variants, v3 self-exports
(base/edge), and the multilingual Laya follow-ups beyond the shipped
trio enter only if the shipped graphs disappoint on quality — the
decision rule in §III.1 has not been revisited because the trigger
(a quality gap on held-out labels) has not fired. The v3 self-export
path is the expensive one (no official ONNX); prefer re-measuring the
shipped entries first.

## V.6. okfgraph-side integration and promotion (consumer-side, partly done)

embroider's side is shipped; the loop closes in okfgraph:
confidence-gated routing (cheap local decide → rerank → LLM/human
escalation), threshold re-fit harness on held-out labels, regime curves
before promotion. Status per `OKFgraph/docs/reranker-integration-plan.md`
(2026-11-01): Phases 0–2 implemented on `dev_0.11.0`, Phase 3 harness
(`scripts/eval_search.py`) built with a first run, promotion still open —
with the recorded deviations (`rerank` not honoured by `concepts:none`;
no combination with `recency`). Whether the harness reuses hev's axis
(same 3 corpora, nDCG@10, paired per-query — the §III.10 lesson) is
visible only in the okfgraph tree. Trigger: okfgraph promotion decision.

## V.7. Qwen3 and Laya-as-reranker revivals (conditional)

Both carry explicit revival conditions, neither met: Qwen3-seq-cls
needs a bakeoff quality gap worth 100x *and* serving that closes the
cost gap (§II.6); Laya-as-reranker needs the eval harness to show a gap
worth the provider-appropriate cost, 15 ms CUDA / 225 ms CPU (§III.9).
The decoder-input formatter design and the Noul/binary-Choice reduction
mappings are preserved in §§II.6/III.9–III.10 so a revival starts from
the design, not from zero.

---
# Part VI — Rejected items (new 2026-11-07)

Decided-no, consolidated from the rejection logs in Parts II–III plus
the verdicts the execution logs produced. Each names the killer and the
revival condition where one exists; the rest are recorded, not revisited.

## VI.1. Models declined

- **Qwen3-0.6B-seq-cls (rerank and decider).** Killer: ~120–150 ms/pair
  CPU (~100x the spikes) with no demonstrated CUDA relief, while mMARCO
  covers 100+ langs at ~2 ms/pair. Revival: bakeoff gap worth 100x AND
  serving closes the cost gap. (Part II §II.6, Part III §III.5.)
- **Jina reranker v3/v3.5.** No verifiable ONNX, custom modeling,
  non-commercial licence, late-interaction engine mismatch. No revival
  condition stated — recorded, not revisited.
- **GLiClass `-init` checkpoints, `-lw` lite variants, bi-encoder/fused
  variants, backbone swaps (llama-1.3B, qwen-0.5B/1.5B, v4-2b),
  third-party finetunes.** Out of scope per §III.1b: `-init`/`-lw`
  revisit only if the base pick is close but heavy; bi-encoder/fused
  breaks the single-pass requirement; backbone swaps are GPU-class.
  (The `-lw` and size-up triggers also appear as deferred items in
  Part V.5 — the distinction is deliberate: revisit conditions live
  there, the no-go stands here.)

## VI.2. Precisions declined

- **Rerank fp16 (optimum export).** Fails both bars: drift 3.3e-3/5.3e-3
  AND slower than fp32 (2.9–5.2 ms vs 1.1–3.2 ms). Killed, not gated.
- **Rerank bf16 (optimum export).** Won't load on any provider: naive
  cast feeds bf16 `Pow` into LayerNorm with no op-level keep-list. Fixing
  it is converter surgery for a dtype that couldn't beat fp32 on
  launch-bound graphs anyway.
- **GLiClass-small int8 / v4-edge official q8.** Small stays fp32: the
  in-repo int8 path was not adopted (small's e2e pins fp32; the q8 fleet
  work in 0.11.0 explicitly records small-v1.0 failing the 0-flip gate —
  knife-edge travel cases, encoder-sensitive — stays fp32). v4-edge
  official `model_q8.onnx`: drifts 5.5 on the logit scale; argmax
  survives easy probes but the probabilities (the contract) are garbage.
- **Laya int8, all three arms (td representative).** Dynamic full-graph:
  drift 2.8–5.4 with flips. Static full-graph (30-input calibration):
  drift 9997 — the −1e4 mask sentinel is unquantizable. Static
  encoder-only (fp32 head, split verified exact): still 2.85 with no CPU
  speedup. Dead; fp16 CUDA variants and the fp32 self-exports cover the
  serving need instead.
- **Embeddings int8/int4.** Never shipped: quant shifts the vector space,
  which needs retrieval re-eval, not parity. (fp16 mirrors for CUDA are
  adopted, parity-gated — see the conversion playbook §5, not a
  rejection.)

## VI.3. Approaches declined by design

- **Per-label NLI forwards** (DeBERTa-v3-large-zeroshot, bart-large-mnli).
  One forward per label per question cannot serve a routing loop;
  single-pass is a requirement, not a preference.
- **Generative/decoder judges and hosted APIs** (Qwen causal-LM ONNX,
  small-LLM + constrained decoding; Jev, Clef, OpenAI Decisions,
  Perplexity Decider). Non-goals: training/serving shape mismatch plus
  the graph's privacy posture forbids state exfiltration.
- **Serving decider-2b Ollaya-side.** Superseded, not rejected on merit:
  it was a serving gap and 0.10.0 closed it (`src/decider2b.rs`).
- **Serving Laya via `ONNXAgent`.** Superseded by `LayaDecider` (0.7.0:
  marker contract ported 1:1); `ONNXAgent` remains the golden-builder
  the e2e pins against.
- **`decide_long` by analogy with MaxP.** Deferred by design (§III.2,
  Part V.4) — relevance is existential, decisions are not.

## VI.4. Non-goals carried forward

Training/fine-tuning, embedding-space changes, threshold policy (consumer
side, re-fit per model), per-label NLI, hosted judges — unchanged since
the original plan headers. Anything here reopens only through a new
proposal with gates, never by drift.
