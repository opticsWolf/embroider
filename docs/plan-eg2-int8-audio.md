# Plan — EmbeddingGemma 2: int8, the audio slot, and mixed precision

Status: plan, 2026-11-14; implementation status 2026-11-15. Owner repos:
**embroider** (artifacts, lanes, sessions, host), **okfgraph** (pins,
re-embed, image lane), **Silk** (rows only). bobine is not involved.
Execution plan, target architecture and next steps:
`docs/plan-eg2-rollout.md`.

## 0. What EG2 is, in the terms this stack uses

EmbeddingGemma 2 is one 768-d space fed by three towers:

| Piece | Params | Artifact (publisher repo, pinned `daa72c5` by embroider) |
|---|---|---|
| Text tower (24 layers, dim 512, mean pooling, 512→768 projection) | 270M (130M backbone + 140M embedder, vocab 262,144) | `onnx/model.onnx` 1.08 GB, `model_fp16.onnx` 542 MB, `model_quantized.onnx` 314 MB |
| Vision tower (16 layers, patch 16, pool 3, 280 soft tokens) | 170M | `onnx/vision_encoder.onnx` 671 MB, `_fp16` 336 MB, `_quantized` 195 MB |
| Audio encoder (12 layers, hidden 1024, conv 5, 25 tok/s) | 300M | `onnx/audio_encoder.onnx` 1.17 GB, `_fp16` 587 MB, `_quantized` 340 MB |

Shared contract: 8,192-token context, MRL 128/256/512/768, task prefixes
on text only (`task: search result | query:`, `title: {title} | text:`),
media placed by `<|image|>` / `<|video|>` / `<|audio|>` tokens. The
publisher supports **per-component dtypes** (text, vision, audio chosen
separately), which is the upstream basis for mixed precision.

The text graph is the **fusor**: `gemma.rs::encode_image` runs the vision
tower, then the text session with `image_features`, `video_features` and
`audio_features` inputs (the empty arrays are passed today), and reads the
`native sentence_embedding` head. So:

- text vector = text tower at `P_text`
- image vector = vision tower at `P_vision`, fused by the text tower at `P_text`
- video vector = frames through the same vision tower at `P_vision`, fused
  through the `video_features` input at `P_text` (the graph already
  declares it; only the `<|video|>` framing differs)
- audio vector = audio tower at `P_audio`, fused by the text tower at `P_text`

### Current state

- `EMBEDGEMMA2`: fp32 + fp16 + int8 (self-hosted mirror
  `opticsWolf/embeddinggemma-2-onnx-int8` @ `52614d7`, upstream
  `model_quantized.onnx`); `auto` never resolves to int8, and an
  unmeasured request still falls back to fp32. 8K, ladder
  128/256/512/768. Text fp16 is parity-gated: 2.1e-4 max abs vs fp32,
  cosine ~1.0 (`docs/compat.md`).
- `EMBEDGEMMA2_VISION`: fp32 + fp16 + int8 (same mirror,
  `vision_encoder_quantized.onnx`); `text_partner` set; fp16-on-CPU
  refused (the tower NaNs under emulated fp16 — `docs/compat.md`). The
  image gates are still open; the artifact alone changes no default.
- No audio slot, no audio artifacts, no `/v1/audio/embeddings` route.
- Precision policy: `auto` = CUDA→fp16, CPU→fp32; `auto` never resolves
  to int8; unlisted tiers fall back to fp32 and `describe()` reports what
  loaded.
- okf pins: text model + precision `{int8: 8, fp16: 16, fp32: 32}` +
  space; image model + precision (same codec) + image space; `reembed`
  refuses when `ImageAsset` rows exist; `space` =
  `embroider/<model>@<rev>:<precision>:<dim>:<max_length>:<tuning>:<contract>`.

### Decisions this plan makes

1. **The artifact is registered; the pin and retrieval gates are the
   guards.** The graphs are on the self-hosted mirror; the codec keeps
   them honest, and the labelled text gate passed 2026-11-16. Vision
   int8 has its own cross-modal gate.
2. **Mixed precision is legal per lane.** The unit of a vector space is
   the `(text fusion, tower)` pair; media pins must name the fusion
   identity, not only the tower.
3. **Audio is a third embroider slot**, built like the vision slot. No
   okfgraph or Silk work until an audio consumer exists.
4. **No bf16.** The overflow fallback is fp32, which is already shipped
   and pinned.
5. **Video needs no lane**: frames reuse the vision tower and the
   `video_features` input.
6. **The embedding int8 policy changes**, deliberately: from "never"
   (`docs/ARCHITECTURE.md`) to "never automatic; explicit, measured,
   pinned, and retrieval-evaluated". That is a policy edit, not a code
   detail, and this plan is the record for it.
7. **Model tiers do not shrink the graph.** Re-embedding under int8
   keeps the same `FLOAT[dim]` payload, so database size is W4's budget:
   256d is an option, int8 storage is a bigger option, and the image-bytes
   decision stands. int8 shrinks the model download (314 MB vs 1.08 GB);
   that is its only size claim. `okf db-size` is an option, not a
   commitment.

### Status (2026-11-15)

- **Artifact**: text and vision int8 registered on the self-hosted
  mirror (`opticsWolf/embeddinggemma-2-onnx-int8` @ `52614d7`); `auto`
  still never selects it. A live `open(..., precision="int8")` reports
  `int8` and encodes.
- **okf pin codec**: landed in `4eaf1e2` — `int8: 8` in the codec, both
  decoders and enforcers, unknown codes fail closed, the image
  caller/settings/hash paths accept int8, `pin_info` compares image pins,
  and reembed validates the precision before writing (clearing an unnamed
  pin instead of leaving a lie). Tests cover the cross-refusals.
- **Silk rows**: landed in `d455bc2` / `b38784a` — EG2 text and vision
  offer `int8`; the ONNX Loader and KG Q5 checks pick it up through the
  existing helpers.
- **Live proof**: a temp graph on CPU reports `precision=int8`, `space`
  `…@52614d7…:int8:768:8192:text_embed:v1`, reembed keeps int8, and an
  fp32 reopen raises `PRECISION_PIN_MISMATCH`.
- **Still open**: the vision cross-modal gate, the mixed-pair product
  rule, and the audio slot. The labelled text-retrieval gate passed on
  2026-11-16 (scifact + nfcorpus, 768/512; `OKFgraph/docs/eg2-int8-gates.md`).
  The image lane has no okf/Silk consumer yet beyond the pin. See
  `plan-eg2-rollout.md` §3–§4.

## 1. Non-goals

- **bf16** (checked 2026-11-14). Four facts, in order of weight:
  1. The publisher ships no bf16 ONNX: `onnx-community/embeddinggemma-2-ONNX`
     holds `model`/`vision_encoder`/`audio_encoder` each in fp32, fp16, q8,
     q4 and q4f16 only.
  2. The one public "bf16" branch, `Synaptics/Google-EmbeddingGemma-2`
     (Astra SL2610 / Torq NPU), is a **`.vmfb` NPU build** with
     `token_embeddings.npy` and mel filters — no ONNX files; its `onnx/`
     directory is a copy of the community fp32/fp16/q* export.
  3. `optimum-cli export onnx --dtype bf16` exists (CLI choices
     `fp32|fp16|bf16`), but optimum's own exporter warns: *"After the
     export, ONNX Runtime InferenceSession with CPU/CUDA execution
     provider likely does not implement all operators for the bfloat16
     data type, and the loading is likely to fail."* ORT's tracker agrees:
     issue #28768 (open since 2026-06) is a plan to add *comprehensive*
     bf16 support; issue #33874 (2026-11) shows a bf16 graph failing at
     session init because `Pad` has no bf16 kernel on CPU or CUDA.
  4. `embedding_gemma2` has no optimum `OnnxConfig` (only `gemma`,
     `gemma2`, `gemma3_text`, `gemma3` are registered), so a self-export
     is a custom-config project — and it must reproduce the fusion
     signature (`image_features`/`video_features`/`audio_features` →
     `sentence_embedding`), which the publisher's graph has by hand.

  Add the new pin codec (as int8 needs), and there is still no measured
  throughput gain over the already parity-gated fp16 lane on the same
  tensor cores. The overflow fallback is fp32, which is already shipped
  and pinned. **Should the decision be revisited**, the probe is: a
  custom `OnnxConfig` bf16 export, open with CUDA EP, profile for fp32
  islands / CPU fallback, then parity and latency against the existing
  fp16 lane — a failing load or a fp32 island is the likely outcome, and
  either ends the probe.
- **Re-exporting.** Use the publisher's pinned artifacts.
- **Storage-format quantization.** okf stores f32 vectors; int8/binary
  *storage* is a future index-format project, not a precision setting
  (W4 is where that conversation lives).
- **An image-bytes policy.** How image assets are stored is a settled
  decision: `ImageAsset.data` keeps the original bytes. W4 reports their
  size so the trade-off is visible; it does not propose dropping,
  recompressing or externalising them.
- **CUDA graphs.** Variable-length text plus a serialized request pool;
  ORT requires fixed shapes/addresses and IOBinding. Measure steady-state
  latency before reopening this.
- **q4 / q4f16.** Published (0.975 / 0.988 text cosine) but well below
  the embedding bar this stack holds.
- **okf audio search** until an audio-asset consumer exists (the image
  lane is the precedent: lane first, consumer later).

## 2. Mixed precision — the actual question

### 2.1 It is structurally supported, per lane

Each slot has its own `Precision` (`ModelSpec.fp32/fp16/int8`, resolved
by `policy.rs`), and the publisher supports per-component dtypes. The
stack already runs text and vision at independent precisions. There is no
technical obstacle to text fp16 + vision fp32, text int8 + vision int8,
or text fp16 + audio q8.

### 2.2 But the unit of a vector space is the *pair*

Because the text tower fuses every modality, a media vector's space is
`(text fusion identity, tower identity, width)` — not just the tower.
Concretely, an image vector built with text fp16 is not comparable to a
text vector from a graph at text int8, even if the vision tower and width
are identical.

Today this is safe **by construction, not by an explicit pin**:

- the text precision pin is fail-closed, so a graph can never reopen at a
  different `P_text`;
- `reembed` refuses when `ImageAsset` rows exist, so image vectors cannot
  survive a text-space change.

That is an invariant by omission: two independent mechanisms happen to
cover it. The int8 work must not weaken either, and the audio lane needs
the same treatment before it ships.

### 2.3 What is safe / unsafe

| Case | Verdict |
|---|---|
| text fp16 + vision fp16 (today) | safe, pinned |
| text int8 + vision int8 | safe *after* the pin codec learns int8 |
| text fp16 + vision fp32 (or the reverse) | safe; each tower is pinned separately; quality is the tower's own parity question |
| text int8 + vision fp16 | safe mechanically (both pinned); quality is dominated by the int8 fusion — accept or reject on retrieval eval, not on cosine |
| reopen a graph at another `P_text` | unsafe; must never be possible |
| re-embed text under `ImageAsset` rows | unsafe; refused today, keep it refused (fresh graph or re-encode media) |
| image vectors carried into a graph embedded at another text precision | unsafe; blocked by the text pin + reembed refusal |
| assuming publisher whole-model dtype numbers transfer to arbitrary combinations | unsafe; each combination needs its own end-to-end gate |

### 2.4 What to add (W3, small, precedes the UI)

1. Extend the precision codec to int8 **once**, for text and image pins
   (§3.3) — `_PRECISION_CODE`, `stored_precision`,
   `_image_pin_precision`, `pin_info`.
2. Make the fusion identity explicit for media: store the text space an
   image was fused under (e.g. `embedding_image_text_space`) or fold it
   into the image space string; compare at open and refuse a mismatch
   with a "re-encode images / import a fresh graph" remedy.
3. Add the audio pin trio (`embedding_audio_model`,
   `embedding_audio_precision`, `embedding_audio_space`) mirroring the
   image trio, plus the same fusion-identity field.
4. Keep `reembed` as the only text-space migration, and keep its
   ImageAsset refusal; update the remedy text to name the media lanes.

This turns "mixed precision is safe because the text pin is fail-closed"
into "mixed precision is safe because the (text, tower) pair is pinned".

## 3. W1 — int8 for the text tower

### 3.1 Artifact — registered

The text and vision q8 graphs are registered on the self-hosted mirror
(`opticsWolf/embeddinggemma-2-onnx-int8`, commit `52614d7`), byte-identical
to upstream `daa72c5`:

```rust
int8: Some(Artifact {
    repo: EMBEDGEMMA2_INT8_REPO,          // opticsWolf/embeddinggemma-2-onnx-int8
    file: "onnx/model_quantized.onnx",    // q8 / dynamic int8
    sidecar: Some("onnx/model_quantized.onnx_data"),
    revision: Some("52614d767029dd71817689c0bde0a0146c5cc0f0"),
}),
```

The vision twin is `EMBEDGEMMA2_VISION.int8` →
`onnx/vision_encoder_quantized.onnx` + `_data` on the same commit; the
audio q8 graph is mirrored too but has no slot yet. `auto` still never
resolves to int8, and `EMBEDGEMMA2.int8` is explicit-only.

**q8 is int8.** transformers.js names the 8-bit tier `q8` and the file
`model_quantized.onnx`; ONNX Runtime's name for the same recipe is
dynamic-int8 quantization (`QuantType.QInt8`). So "q8" and "int8" are one
tier, not two alternatives, and W1's gate applies to it. A *different*
int8 recipe is a separate candidate: **static/QDQ int8** (calibrated
activations, no per-batch quantization) can be faster on CPU than the
dynamic recipe but needs its own calibration set, artifact and gate;
**weight-only q8** (`MatMulNBits`, 8-bit) is a third. None of these is
4-bit: q4/q4f16 stay out.

`Auto` must still never resolve to int8; an unmeasured request still
falls back to fp32; `describe()` keeps reporting the loaded precision.

### 3.2 Gate (before the registry entry ships)

External corroboration: Synaptics' int8 build of EG2 (8-bit, block 32,
bf16 activations, on their NPU) reports cosine 0.997 vs fp32 over a media
library, and their 4-bit build 0.976. Different quantization path from
ORT dynamic int8, so it is a signal, not a gate — but it supports "int8
is close; the question is retrieval, not cosine".

- Parity vs fp32: ≥ 4 probes, worst-case cosine ≥ 0.995 (publisher's
  whole-model figure is 0.9997).
- **Retrieval**: NDCG@10 and recall@20 on the repo eval set; accept only
  if the delta is ≤ 0.5 pt. This is the policy's actual bar — int8 is
  refused for embeddings today because parity does not prove retrieval.
- Non-finite check: zero NaN/inf across the probe set.
- Throughput: record CPU and CUDA separately. CPU int8 was ~0.6 docs/s;
  the CUDA EP measured **≈fp32** on 2026-11-14 (38.6 vs 38.4 docs/s;
  q4f16 56.7). int8 is a **size** choice, not a speed one.
- CUDA: the same q8/q4/q4f16 graphs **do run on the CUDA EP** here
  (ORT 1.31 + cuDNN via `prepare_cuda_runtime()`): 38.6 / 35.0 / 56.7
  docs/s against fp32's 38.4, while the same int8 graph on CPU was
  ~0.6 docs/s. So "int8 is a CPU tier" is a portability statement, not a
  hardware one: gate quality first, then record CPU and CUDA throughput
  separately. The CPU lane stays the one every deployment has.
- Memory: confirm the 314 MB text artifact.

**Evidence already in hand (2026-11-14).** On 1,500 long Wikipedia
articles (5.2 M chars) q8 measured mean 1−cos 8.5e-4 (p99 2.9e-2)
against fp32, and on 50 curated queries it matched fp32's retrieval at
768/512 (R@1 .88, NDCG .956) and stayed within ±2 points at 256/128.
The model footprint is 314 MB vs 1.08 GB; the graph file is unchanged
(same `FLOAT[dim]`). It supported the gate but did not replace it; the
labelled eval set was the decision, and it passed on 2026-11-16.

### 3.3 Pin codec (okfgraph) — landed (`4eaf1e2`)

- `_PRECISION_CODE = {"int8": 8, "fp16": 16, "fp32": 32}` in
  `components/embedding.py`; `stored_precision`, both enforcers,
  `pin_info` and `doctor` decode 8.
- `_enforce_pins` enforces int8 like any other precision. Unknown stored
  codes are now **fail-closed** in both enforcers instead of being
  treated as unpinned.
- `reembed` validates the engine's precision **before** rewriting
  vectors, writes a named pin, clears an unnamed one (no lying pin), and
  re-reads the landed precision after the encoder opens. The
  `ImageAsset` refusal stays.
- Image paths learned int8 end to end: `router` and `settings` accept
  `image_precision="int8"`; the vision pin caller and stored decode use
  the codec; `image_assets` content-hash and `doctor` decode int8.
- `pin_info` compares image model/precision/space, reports
  `image_mismatch` (and prefixed entries in `mismatch`), and uses an
  external host's `identity.precision`.
- **The model pin cannot cover this.** int8 is the same
  `google/embeddinggemma-2` id; `enforce_model_pin` compares ids, so an
  int8 graph looks model-identical to fp16. The precision pin is the only
  guard. Do not mint a fake `-int8` id: it would split one vector space
  across two ids and break the one-id-per-space convention.
- **Image lane:** an `ImageAsset` graph is fresh DB + re-ingest +
  re-encode, never an in-place reembed. The default product rule is text
  int8 + vision int8 (or no image lane); mixed vision precision is
  mechanically legal only with the fusion identity pinned (§2.4), and
  the quality gate decides.
- **Database size:** int8 vectors keep `FLOAT[dim]`; the graph file does
  not shrink. 314 MB is model footprint, not DB.
- Tests: int8 set/match and int8↔fp16/fp32 refusals; empty-graph
  adoption; reembed writes 8 and `pin_info` reports it; image graph
  reembed still refused; the int8 vision caller actually pins; router
  and settings accept image int8; `pin_info` reports an image mismatch;
  an unknown stored code refuses; a reembed whose encoder cannot name
  its precision refuses without touching either pin.

### 3.4 Rows (Silk) — landed (`d455bc2`)

- `MODEL_PRECISIONS["google/embeddinggemma-2"] += ("int8",)` — the ONNX
  Loader's precision row and the KG's Q5 precision check pick it up via
  `precisions_for()` / `landing_precision()` with no UI code.
- Tooltip: "int8 is a size choice (~0.3 GB text vs 1.08 GB; the graph
  file does not shrink). CPU ~0.6 docs/s; CUDA measured at fp32 speed.
  Never automatic."
- **No CUDA refusal.** The earlier "int8 is a CPU lane" rule was a
  portability assumption, not a measurement: the 2026-11-14 run put the
  q8 graph on the CUDA EP at 38.6 docs/s against fp32's 38.4 (q4f16
  56.7), with cuDNN loaded. `wiring_problem()` must not refuse
  `device=cuda` + `precision=int8`; if a deployment wants the CPU lane,
  it says so.
- `IMAGE_PRECISIONS` includes int8; the vision q8 artifact exists and
  okf's `image_precision` accepts it.

### 3.5 Vision int8 (only if image search needs the size)

`EMBEDGEMMA2_VISION.int8` is registered on the mirror
(`onnx/vision_encoder_quantized.onnx` + `_data`). The gate must include
**cross-modal end-to-end** (image → text query retrieval),
not just tower parity. Note the existing measured fact that the vision
tower NaNs under emulated fp16 on CPU — int8 is the only quantized CPU
path here, so its gate runs natively and must be at least as strict.

### 3.6 Rollout

artifact + gate → codec → Silk rows → re-embed a **copy** of a real
graph (fresh DB for image graphs) → retrieval eval vs fp32 → ship or
revert. Never before the pin.

## 4. W2 — the audio slot

### 4.1 Artifacts

`audio_encoder.onnx` (1.17 GB), `audio_encoder_fp16.onnx` (587 MB),
`audio_encoder_quantized.onnx` (340 MB), pinned revision + sha256. The
text graph already declares `audio_features` (vision passes an empty
array), so no new fusor graph is needed.

### 4.2 Contract

- **Input**: 16 kHz mono PCM. Callers do not have a processor; accepting
  PCM is the friendly contract.
- **Front end**: the processor's feature extraction (subsampling conv
  128/32, kernel 5, chunked attention 12, hidden 1024, 12 layers,
  `output_proj_dims` 1536) is the risky part. Port it in Rust and
  golden-gate against `transformers`' audio processor, or (fallback
  split, like vision's caller-side resize) accept precomputed features
  with a documented shape. Decide by the probe: if the mel front end
  matches to ≤1e-4 within a day, port it; otherwise ship the
  features-in contract first.
- **Encoder output → fusion**: read the text graph's `audio_features`
  input shape at open and refuse a mismatch loudly (the vision K check is
  the template). Determine the framing ids from config (`audio_token_id`
  258881, `boa` 256000, `eoa` 258883) and pin them with a golden run:
  `[bos, boa, audio × K, eoa, eos]`.
- **Budget**: 25 tokens/s, shared with text in the 8K window (~327 s of
  audio alone); cap at `min(8192, encoder capacity)`.

### 4.3 Structure

New `src/gemma_audio.rs` mirroring `gemma.rs`: audio session + text
fusion session, one space, per-slot `Precision`, `describe()` names the
lane. New `ModelSpec` audio slot: `text_partner = EMBEDGEMMA2_MODEL`,
ladder 128/256/512/768, `max_tokens` 8192.

### 4.4 Host and consumers

- `KINDS` gains `"audio"`; route `/v1/audio/embeddings`; `serve_args` /
  `handle_for` mirror vision; `/v1/models` reports it.
- okf/Silk consumers are deferred until an audio-asset story exists
  (import, storage, search). This plan stops at the host route.

### 4.5 Video

No new encoder and no new graph input: frames are sampled at 1 fps (the
publisher subsamples >32 frames to 32), run through the vision tower, and
are fused through the text graph's existing `video_features` input with
`<|video|>` (258884) framing instead of `<|image|>`. Video therefore
inherits `P_vision` and the vision pins. Document it next to the vision
slot rather than adding a lane.

## 5. W4 — database size (adjacent; okfgraph, not a precision decision)

The tiers in W1 do not change the graph file. Re-embedding under int8
keeps `FLOAT[dim]` vectors at the same width, so the vector payload is
byte-identical; only the values move. Database size is a separate budget
with its own levers.

### 5.1 What the file is made of

From `okfgraph/components/schema.py` and `image_assets.py`:

| Part | Stored as | At 768d |
|---|---|---|
| Concept embedding | `FLOAT[dim]` | 3,072 B/row |
| Chunk embedding | `FLOAT[dim]` | 3,072 B/row |
| ImageAsset embedding | `FLOAT[dim]` | 3,072 B/row |
| Image bytes | `ImageAsset.data BLOB` | the original file |
| Text | Concept title/description/body, Chunk text | 1× |
| Search indexes | HNSW (`mu 30, ml 60, efc 200`) + 2 FTS | ~15–30% + tokenized text |

Measured on a schema-faithful dummy (2,000 concepts, 20,000 chunks, 300
images averaging ~220 KB), same content at two widths:

| Part | 768d | 256d |
|---|---|---|
| File | 260.5 MB | 214.1 MB |
| Vectors (22,300 × dim) | 65.3 MB (25%) | 21.8 MB (10%) |
| Image BLOBs | 64.9 MB (25%) | 64.3 MB (30%) |
| Text | 26.8 MB (10%) | 26.8 MB (13%) |
| Indexes / pages | 103.5 MB (40%) | 101.2 MB (47%) |

The vector share grows with chunk count and falls with image bytes, so
the first step is a measurement, not a change. Note the index/page
residual: at this scale it is as large as the vectors, so a vector-only
lens overstates what a width change buys.

**First real-text run (2026-11-14).** 1,500 long Wikipedia articles
(5.2 M chars, 3.5 KB/doc, 3.7 chunks/doc), 50 curated queries, 20 real
DBs (every tier × width, HNSW + FTS). File sizes, 1,500 concepts +
~5,600 chunks:

| Tier | 768 | 512 | 256 | 128 |
|---|---|---|---|---|
| fp32 | 79.7 MB | 73.5 | 59.1 | 59.0 |
| fp16 | 80.4 MB | 73.6 | 59.1 | 59.0 |
| int8 | 80.0 MB | 73.6 | 64.3 | 59.1 |

On this corpus, 768 → 256 saves 26% of the file and 256 → 128 saves
nothing measurable: the vector column halves (6.9 → 3.5 MB) but HNSW,
FTS and page overhead do not scale with width. Precision tiers change the
file not at all — re-embedding under int8 keeps the same `FLOAT[dim]`
column. Retrieval: int8 matched fp32 at 768/512 (R@1 .88, NDCG .956) and stayed
within ±2 points at 256/128 (one query = 2 points at this size); q4/q4f16
drifted 2.2% cosine and lost 2 points at 512/256. The full matrix, scripts and
caveats live in `docs/eg2-int8-audio-results.md` (the 20 DBs and the
npz vectors are in the temp bench it names); the okfgraph-side record —
pins, width, storage and `db-size` implications — is
`OKFgraph/docs/bench-eg2-precision-width.md`. This section is the
summary, not a shipping decision.

### 5.2 Option A — 256d (supported today, no engine work)

The MRL ladder 768 → 512 → 256 → 128 is already offered by the ONNX
Loader and pinned by okf. At 256d a vector is 1,024 B instead of 3,072 B
(3×); the official quality is ~98.5% of text MTEB and ~95% of multimodal
retrieval, with 128d text-only (~94% / ~77%). On the dummy that is
65.3 → 21.8 MB of vectors and 260.5 → 214.1 MB of file: 18% total,
because images and text dominate that mix — a text-heavy graph gains
more.

Cost: **a fresh database and a re-import**. okf adopts the column width
at boot; a width change is a new graph, not a re-embed. The image lane is
capped at 768, so 256 fits comfortably. This is an option on the table,
not a scheduled change.

### 5.3 Option B — int8 vector storage (engine work, only if 256d is not enough)

4× on vectors (3,072 → 768 B at 768d; 1,024 → 256 B at 256d, 12× vs
today). It is not a precision setting: it changes the column type
(`FLOAT` → `INT8` plus a per-vector scale), the distance path (int8 dot
product with an f32 rescore of the top-k), the schema/migration and the
re-embed. It only makes sense after a measurement shows vectors dominate
*and* 256d has been taken.

### 5.4 `okf db-size` (an option, not scheduled)

A read-only report of exactly the fields above: row counts, vector bytes
by table at the graph's own width, BLOB bytes (with the largest assets),
text bytes, index/page overhead — plus the same numbers at a candidate
width so a 256d decision is immediate. Uses:

1. deciding whether 256d pays for a given graph;
2. spotting index bloat after churn, deletes or a re-import;
3. capacity planning and backup sizing;
4. comparing two graphs before/after an import or a width change.

It must run against a **copy** (or through the router that already holds
the lock), because opening a Ladybug file takes its process lock — the
same rule the rest of this stack follows.

Until it exists: `make-dummy-db.py` (schema-faithful, synthetic, no
model) and `okf-db-size.py` in the temp directory reproduce the numbers
above. The dummy is not an ingest artifact (no Meta/FileHash rows), so
use it for sizing, not for search.

## 6. Gates, in one table

| Gate | Where | Pass |
|---|---|---|
| fp16 text at 8K tokens vs fp32 | embroider probe | mean cosine ≥ 0.999, no NaN, retrieval delta ≤ 0.5 pt |
| int8 text parity | embroider gate | worst cosine ≥ 0.995, no non-finite |
| int8 text retrieval | repo eval set | NDCG@10 / recall@20 delta ≤ 0.5 pt |
| int8 throughput | probe | CPU and CUDA recorded; CPU ~0.6 docs/s, CUDA ≈fp32 (38.6 vs 38.4); size, not speed |
| vision fp16 e2e | existing | already 1.0 CUDA, refused on CPU |
| vision int8 e2e + cross-modal | gate | text-query→image retrieval delta ≤ 0.5 pt |
| audio front end parity | golden features | ≤ 1e-4 vs processor |
| audio e2e cross-modal | probe | text-query→audio Hit@1 sanity, no NaN |
| pin codec | okf tests | int8/fp16/fp32 cross-refusals, reembed re-pin |
| int8 image path | okf tests | router/settings accept int8; vision pin writes 8; fp16 reopen refused |
| unknown precision code | okf tests | graph refuses, never adopts |
| reembed unnamed precision | okf tests | refuses, leaves no mixed model/precision pin |
| mixed-precision pair | okf tests | tower change refused; text change refused with media present |
| 256d rebuild (option) | real copy | vectors −3×; file −26% on the long-text run (−18% on the dummy); retrieval delta ≤ 0.5 pt |

## 7. Sequence and current status

1. **M0 — probes.** Long-context fp16 parity and audio artifact smoke are
   still open. The int8/q4/q4f16 real-text matrix was taken on 2026-11-14
   (1,500 long articles, 50 queries, 20 DBs;
   `docs/eg2-int8-audio-results.md`) and is the reference for W1/W4; the
   labelled query set passed the shipping gate on 2026-11-16.
2. **M1 — int8 text.** Artifact, okf codec, Silk row and the text-only
   reembed path are **landed** (`95ff50c`, `4eaf1e2`, `d455bc2`); the
   labelled retrieval gate passed 2026-11-16. The reembed-a-copy check
   remains open; image graphs need a fresh DB + re-encode.
3. **M2 — pins for mixed precision (W3).** Fusion identity for media and
   the audio pin trio are **not yet implemented**; they must precede any
   audio UI and the int8 image lane. The mixed-pair default is still a
   product decision.
4. **M2b — database size (option).** Measure a real graph with the temp
   reporter; if vectors dominate and quality allows, plan a fresh 256d
   import. No engine work, no image-policy change.
5. **M3 — audio slot (embroider).** Encoder + front end + fusion + host
   route. **Not started**; independent of M1.
6. **M4 — consumers.** okf audio asset/search, Silk nodes. Only after a
   real use case.

The ordered execution list with owners, acceptance gates and open
questions is `docs/plan-eg2-rollout.md` §4.

## 8. Files

- embroider: `src/acquire.rs` (specs, artifacts, revisions),
  `src/policy.rs` (int8 docstring), `src/gemma.rs` (fusor + audio),
  `src/providers.rs` if a session knob is needed,
  `python/embroider/serve.py` (kind, route), `docs/compat.md`,
  `tests/` (e2e gates).
- okfgraph: `okfgraph/components/embedding.py` (`_PRECISION_CODE`,
  `stored_precision`, pins), `okfgraph/router.py` (pin calls, image
  validation), `okfgraph/ops/admin.py` (reembed ordering,
  `_image_pin_precision`, `pin_info`), `okfgraph/settings.py` (text and
  image choices), `okfgraph/components/image_assets.py` (content-hash
  precision), `okfgraph/components/doctor.py` (pin display),
  `tests/test_precision.py`, `tests/test_reembed*.py`,
  `tests/test_vision.py`, `tests/test_settings.py`; for the optional
  `db-size` command, `okfgraph/cli.py` + a small read-only component
  (the temp reporter is its prototype).
- Silk: `functions/okf_graph.py` (`MODEL_PRECISIONS`, `IMAGE_PRECISIONS`),
  `functions/onnx_roles.py` (`precisions_for` path; `landing_precision`
  unchanged), `docs/NODES.md`.

## 9. Risks and open questions

- **int8 retrieval drift** is the real gate; cosine will pass long before
  ranking does. If the eval set is small, say so in the record.
- **Downgrade back-compat**: tested 2026-11-16 on okfgraph 0.14.2; the
  space pin refused the new int8 graph (`OKFgraph/docs/compat.md`). Do not
  serve int8 graphs to clients that predate the space pin.
- **Audio front end** is the only genuinely uncertain piece; the
  features-in fallback keeps the lane shippable if the port slips.
- **int8 speed**: measured ~0.6 docs/s on CPU and ≈fp32 on CUDA
  (38.6 vs 38.4 docs/s); if the goal is throughput rather than model
  size, stop after M0.
- **Space-pin back-compat**: a new key must mean "absent = old
  behaviour", or every existing graph trips on the new check.
- **Per-component dtype measurements** in the publisher's table are
  whole-model; combinations (e.g. text fp16 + audio q8) are untested
  upstream and need their own e2e gate.
- **Vision fp16 NaN on CPU emulation** is already a documented refusal;
  audio should be assumed to behave the same until measured, so never
  offer fp16-on-CPU for the towers.
- **The dummy is synthetic.** Its mix (images ~25%, text ~10%) is a
  sizing sandbox, not a prediction for any real graph; the 2026-11-14
  long-text run is the first real measurement, and its 50 queries are a
  signal. The labelled retrieval gate passed 2026-11-16.
- **The index/page residual is large at small scale** (~40% of the
  dummy). Do not read the 768 → 256 file delta as a general ratio; in a
  graph with far more chunks per image the vector share is higher, and in
  one with many images the ratio is lower.
- **256d still costs a full re-import.** The width is adopted at boot; a
  re-embed cannot change it. Budget that as a migration, not a setting.
