# Reranker plan (embroider)

**Status:** shipped on `dev` (0.4.0 contract + 0.4.1 `score_long`), unmerged,
untagged. This document is the consolidated record: what was built, what was
measured, what was decided, and what remains. Supersedes Track A of
`docs/rerank-typed-decision-plan.md` (which keeps the typed-decision track).

## 0. Goal and non-goals

**Goal:** serve cross-encoder rerankers from embroider's ONNX Runtime —
pair in, relevance logit out — reusing the `acquire` / `policy` /
`providers` / `cache_info` seams, same posture as the Jina and vision
contracts.

**Non-goals:** generative/decoder judges (causal-LM ONNX, small-LLM +
constrained decoding), hosted APIs (Jev, Clef, OpenAI Decisions, Perplexity
Decider), training/fine-tuning, embedding-space changes. The frozen
Jina/vision contracts are untouched — new modules only.

## 1. Architecture (as built)

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
  shared helper with JinaV5), `score_long` (0.4.1, §4).
- **Pair encoding:** `[query block, doc block]` with
  truncate-doc-side-first (`OnlySecond`) and per-batch longest padding.
  `token_type_ids` fed only when the graph declares them (MiniLM 3-input
  vs mMARCO 2-input, discovered at load); `logits` output preferred when
  declared. Over-long queries error instead of scoring garbage.
- **Session policy:** `SessionPolicy::rerank_pair()` (starts from the
  text-embedding tuning; named to diverge independently).
- **Python:** `embroider.Reranker` with
  `open/open_files/score/score_batch/score_long/count_tokens` + getters.

## 2. Artifact decisions

Twin spikes — one English, one multilingual — chosen to absorb arch *and*
tokenizer variance through the same seam early. Production slot stays open
pending okfgraph's language mix.

| Model | Params | Licence | ONNX | Verdict |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (WordPiece, Bert seq-cls) | 22.7M | Apache-2.0 | in-repo `onnx/model.onnx` | **spike EN, shipped** |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Unigram, XLM-R seq-cls) | 118M | Apache-2.0 | in-repo `onnx/model.onnx` (not O1–O4) | **spike ML, shipped** |
| `BAAI/bge-reranker-v2-m3` (568M) | 568M | Apache-2.0 | self-export | unexamined; bakeoff candidate |
| `jinaai/jina-reranker-v2-base-multilingual` (278M) | 278M | CC BY-NC-4.0 | in-repo (+fp16/int8) | unexamined; bakeoff candidate |
| `Qwen/Qwen3-Reranker-0.6B` family | 596M | Apache-2.0 | none usable (Route B needs template machinery) | rejected (§6) |
| `jinaai/jina-reranker-v3/v3.5` | 597M | CC BY-NC-4.0 | none verifiable | rejected (§6) |
| `cross-encoder/ettin-reranker-{17,32,68,150}m` | 17–150M | verify | verify | reserve tiny siblings |

## 3. Verification (A0 gates, all passed 2026-10-28)

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

## 4. `score_long` (0.4.1)

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

## 5. Precision sweep (2026-10-28/29, RTX 3090 + Ryzen 9 5950X, ORT 1.29.0)

Latency in ms/pair; drift = max|Δ| vs fp32 golden.

| Precision | L-6 speed | mMARCO speed | L-6 drift | mMARCO drift | Fate |
|---|---|---|---|---|---|
| fp32 CPU | 1.3–1.6 | 2.1–3.2 | — (golden) | — (golden) | ✅ the default |
| fp32 CUDA b1 → b16 → b32 | 3.6 → 1.2 → 1.0 | 5.3 → 2.1 → 1.7 | ~1e-3 | ~1e-3 | ✅ opportunistic |
| fp16 (optimum export) | 2.9–3.0 ❌ slower | 5.2 ❌ slower | 3.3e-3 | 5.3e-3 | ❌ rejected |
| int8 (vendor qint8/quint8, 4x smaller) | 1.1–1.2 | 1.8–2.2 | ≤0.55 | ≤1.10 | ⏸ gated (§7) |
| bf16 (optimum export) | won't load | won't load | n/a | n/a | ❌ killed at validation |

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
no claim until a controlled rerun.

## 6. Rejected: Qwen3-0.6B-seq-cls (proposal 2026-10-29)

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

## 7. Remaining work

1. **Release (needs go-ahead):** merge `dev` → `main`, tag `v0.4.1`,
   publish (PyPI + crates.io, wheels via release.yml). okfgraph's floor
   pin (`<0.4`) can't move until a 0.4.x release exists. Same
   irreversibility rule as okfgraph releases: tag last, explicit approval.
2. **Production bakeoff (blocked on okfgraph data):** bge-v2-m3 export vs
   jina-v2-multilingual vs status-quo mMARCO, decided by held-out labels
   and language mix — not by priors. Includes the int8 rank-parity
   harness gate.
3. **okfgraph integration:** separate plan
   (`OKFgraph/docs/reranker-integration-plan.md`) — lazy session,
   `rerank_with_cross_encoder`, dispatch, eval harness. Starts after (1).
4. **Conditional:** Qwen3 left-padding + formatter (only if §6 reopens);
   long-batch CUDA rerun on a controlled rig; `score_long` chunk
   attribution (which chunk won) if okfgraph needs citations.

## 8. Conventions this work follows (for the next track)

- Measured artifacts only (quant-spike precedent).
- Goldens are the contract proof; consumers assert live outputs.
- Same-API ≠ same quality: thresholds are consumer policy, re-fit per
  model on held-out labels.
- Plan-doc-only changes stay out of the CHANGELOG; shipped behavior only.
