# EG2 precision × width × database-size experiment (2026-11-14)

A first real-text run of the matrix, on the RTX 3090 machine, in an
isolated scratch area. Nothing here touches the live `kb.db`.

The okfgraph-side record of the same run (precision pins, width, storage
and `db-size` implications) is `OKFgraph/docs/bench-eg2-precision-width.md`.

## Setup

- **Corpus**: 1,500 long Wikipedia articles (50 curated topics fetched
  through the REST API, 1,450 from `wikimedia/wikipedia` 20231101.en),
  capped at 4,000 chars and split into ~1,000-char chunks:
  5.2 M chars, mean 3,477 chars/doc, 3.72 chunks/doc.
- **Queries**: 50 hand-written questions, one target article each
  (e.g. "which planet is closest to the Sun" → Mercury (planet)).
  The other 1,450 articles are distractors.
- **Prompts**: `task: search result | query: …` for queries,
  `title: {title} | text: {body}` for documents — the publisher's Gemma
  contract, identical in every tier.
- **Tiers**:
  - `fp32`, `fp16`: embroider `open("embed", …, device="cuda")`
  - `int8` (q8), `q4`, `q4f16`: the publisher's `model_quantized.onnx`,
    `model_q4.onnx`, `model_q4f16.onnx` through ONNX Runtime 1.31 CUDA
    (embroider's `prepare_cuda_runtime()` loads cuDNN), text graph only
    with empty `image/video/audio_features`.
- **Retrieval**: exact cosine over the concept vectors (no HNSW), so the
  numbers isolate the embedding tier and width. The 20 DBs also carry
  real HNSW + FTS indexes and were spot-checked through
  `QUERY_VECTOR_INDEX` (Mercury first for the Sun query).
- **DBs**: one real OKFgraph schema per tier × width, 1,500 concepts +
  5,587–5,623 chunks, indexes built.

## Parity vs fp32 (same documents, 768d, cosine drift)

| Tier | mean 1−cos | p99 | max |
|---|---|---|---|
| fp16 | 6.5e-7 | 1.7e-6 | 3.9e-6 |
| int8 (q8) | 8.5e-4 | 2.9e-2 | 6.6e-2 |
| q4 | 2.2e-2 | 4.7e-2 | 8.5e-2 |
| q4f16 | 2.2e-2 | 4.7e-2 | 8.4e-2 |

fp16 is effectively exact here; int8 is three orders of magnitude closer
than the 4-bit tiers.

## Retrieval (50 queries, 1,500 docs)

| Tier | 768 R@1 / R@5 / NDCG | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .80 / .96 / .905 |
| fp16 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .80 / .96 / .905 |
| int8 | .88 / 1.0 / .956 | .88 / 1.0 / .956 | .84 / 1.0 / .941 | .82 / 1.0 / .929 |
| q4 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .84 / 1.0 / .941 | .82 / .98 / .918 |
| q4f16 | .88 / 1.0 / .956 | .86 / 1.0 / .948 | .84 / 1.0 / .941 | .80 / .98 / .911 |

One query is 2 points, so only differences above ~4–6 points are
meaningful with this set. Within that: width matters more than precision;
int8 is indistinguishable at 768/512; q4 loses a step at 512/256 and is
10–20× further from fp32 than int8.

## Database size (real schema, 1,500 concepts, 5,587–5,623 chunks)

| Tier | 768 | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | 79.7 MB | 73.5 MB | 59.1 MB | 59.0 MB |
| fp16 | 80.4 MB | 73.6 MB | 59.1 MB | 59.0 MB |
| int8 | 80.0 MB | 73.6 MB | 64.3 MB | 59.1 MB |
| q4 | 80.0 MB | 73.6 MB | 64.3 MB | 59.1 MB |
| q4f16 | 80.0 MB | 73.6 MB | 64.3 MB | 59.1 MB |

The int8/q4/q4f16 DBs have 36 more chunks (0.6%): the corpus file was
rewritten by a second builder between the fp32/fp16 and quantized runs.
The concept documents are identical (fp16 parity is 1e-6 and int8 8e-4),
so the retrieval comparison is unaffected.

Composition (reporter, `fp32_768` vs `fp32_256` vs `fp32_128`):

| Part | 768 | 256 | 128 |
|---|---|---|---|
| File | 76.0 MB | 56.3 MB | 56.2 MB |
| Vector columns | 20.8 MB | 6.9 MB | 3.5 MB |
| Text | 9.6 MB | 9.6 MB | 9.6 MB |
| Index/pages | ~45 MB | ~40 MB | ~43 MB |

So the 768 → 256 change buys ~21 MB (26% of the file), and 256 → 128 buys
**nothing measurable**: the vector column halves (6.9 → 3.5 MB), but the
HNSW/FTS indexes and page overhead do not scale with width. At this
corpus (≈3.5 KB/doc) the index and text are the file; precision tiers do
not move it at all (re-embedding under int8 leaves the same
`FLOAT[dim]` payload).

## Ranking agreement with fp32@768 (top-1 / top-10 overlap)

| Tier | 768 | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | 1.00 / 1.00 | 1.00 / 0.87 | 0.94 / 0.76 | 0.92 / 0.50 |
| fp16 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| int8 | 1.00 / 0.88 | 1.00 / 0.83 | 0.96 / 0.75 | 0.94 / 0.53 |
| q4 | 1.00 / 0.80 | 0.98 / 0.78 | 0.96 / 0.73 | 0.94 / 0.55 |
| q4f16 | 1.00 / 0.81 | 0.98 / 0.78 | 0.96 / 0.72 | 0.92 / 0.55 |

The first column is a same-tier comparison, so fp16's rows are the parity
story (rankings identical at every width); the truncation columns show
how much the width, not the tier, moves the neighbours.

## Embedding throughput (RTX 3090)

| Tier | docs/s | chunks/s | queries/s |
|---|---|---|---|
| fp32 | 38.4 | 43.5 | 36.7 |
| fp16 | 33.2 | 45.8 | 40.9 |
| int8 | 38.6 | 162.5 | 740.3 |
| q4 | 35.0 | 134.0 | 380.4 |
| q4f16 | 56.7 | 215.2 | 240.8 |

Queries are short (a sentence), so their numbers are higher than docs.

The quantized graphs run on CUDA with ORT 1.31 + cuDNN and are not
slower than fp32; q4f16 is the fastest. On CPU the same int8 graph ran
at ~0.6 docs/s (≈60× slower) before the CUDA path was used — so "int8 is
a CPU tier" is a deployment statement, not a hardware limit.

## Model artifact sizes (text tower, publisher's ONNX repo)

| Tier | File | Note |
|---|---|---|
| fp32 | 1,085 MB | base |
| fp16 | 542 MB | CUDA mirror |
| int8 (q8) | 314 MB | dynamic int8 |
| q4 | 175 MB | 4-bit weight-only (`MatMulNBits`) |
| q4f16 | 157 MB | 4-bit weights, fp16 compute |

These are model footprint, not database size.

## Findings

1. **int8 (q8) is free at 768 and 512** on this corpus: identical
   retrieval, mean cosine drift 8.5e-4, and 3.4× smaller weights
   (314 MB vs 1.08 GB). The remaining question is a larger query set,
   not a quality cliff.
2. **q4/q4f16 drift 2.2%** and lose 2 points at 512/256. They are
   deployment tiers for size, not quality tiers; nothing here justifies
   putting them in the embedding lane.
3. **Width**: 512 is free, 256 costs ~1–2 points, 128 costs 4–8 points
   (R@1 .88 → .80). For images/audio, 128 is worse than the text numbers
   suggest (the official multimodal drop is much larger).
4. **Database size**: the 768 → 256 step saves 26% of the file;
   128 saves ~0 because indexes/pages dominate. Precision tiers change
   nothing (same f32 column). The real levers are width, index shape,
   and what is stored beside the vectors.
5. **CUDA quantized graphs work** with this runtime; the plan's
   "int8 is CPU-only" rule should become "CPU is the portable lane;
   CUDA was measured and is fine with cuDNN loaded".

## Caveats

- 50 queries, one target each: a 2-point difference is one query. Treat
  the quality ordering as a signal, and re-run on a labelled set before
  shipping a decision.
- Curated queries are relatively easy (baseline R@1 .88 at 768), so the
  floor is high and precision differences compress.
- Exact cosine, not HNSW: the DBs have indexes and were spot-checked,
  but index recall is not measured across the matrix.
- One run, one machine, no warm-up control for the throughput column;
  fp32/fp16 and the quantized tiers were embedded in overlapping jobs.
- No image/audio vectors; this is the text tower only.

## Earlier synthetic dummy (for comparison)

Before the real corpus, two schema-only dummy DBs were built to size the
levers: 2,000 concepts, 20,000 chunks, 300 images (~220 KB each), no
model. Same content at two widths:

| Part | 768d | 256d |
|---|---|---|
| File | 260.5 MB | 214.1 MB |
| Vectors | 65.3 MB (25%) | 21.8 MB (10%) |
| Image BLOBs | 64.9 MB (25%) | 64.3 MB (30%) |
| Text | 26.8 MB (10%) | 26.8 MB (13%) |
| Indexes / pages | 103.5 MB (40%) | 101.2 MB (47%) |

The same conclusion as the real run: vectors are a minority of the file,
and 256d buys ~18% there (with images) / ~26% in the text-only run.

## Reproduce

```
cd C:/Users/Main/AppData/Local/Temp/okf-eg2-bench
python build_corpus.py 1500        # curated REST + wikimedia/wikipedia rows
python embed_tiers.py              # all five tiers, GPU
python build_dbs.py                # 768 512 256 128 DBs
python eval_matrix.py              # results.json + report.md
```

`embed_tiers.py` takes tier names to run a subset; `build_dbs.py` takes
widths. Both skip anything already on disk. The corpus builder needs
network for the Wikipedia API and the HF datasets-server; the model
artifacts are in the HF cache.

## Where things are

`C:/Users/Main/AppData/Local/Temp/okf-eg2-bench/`

- `corpus.jsonl`, `queries.json` — the corpus and the 50 queries
- `emb_<tier>.npz` — 768d doc/chunk/query vectors per tier
- `db/<tier>_<dim>.db` — 20 real DBs (HNSW + FTS)
- `results.json`, `report.md` — the matrix as data / markdown
- `build_corpus.py`, `embed_tiers.py`, `build_dbs.py`, `eval_matrix.py`
