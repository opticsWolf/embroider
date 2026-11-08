# embroider — Quick Reference

Copy-paste surface. Design rationale lives in `architecture.md`, release matrix in
[compat.md](compat.md).

## Install

```bash
pip install embroider
```

```rust
embroider = "0.14"  // shared ORT plumbing; embedding (Jina v5 + Gemma 2) + rerank/decide engines included
```

Needs exactly one ONNX Runtime: `onnxruntime==1.31.1` (or
`-gpu[cuda,cudnn]==1.31.1`), auto-located on import. Override:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime.dll   # e.g. <venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll

from embroider import prepare_cuda_runtime
prepare_cuda_runtime()   # 0.14.0+: self-service CUDA loader PATH (venv
                         # nvidia dists + capi dirs + SDK roots), idempotent;
                         # every CUDA open already runs this itself
```

## Text embeddings (Python)

```python
from embroider import JinaV5

m = JinaV5.open("jinaai/jina-embeddings-v5-text-small-retrieval")
v = m.encode("what do badgers eat?", task="query")       # task: query | document
vs = m.encode_batch(["a", "b"], task="document")         # sequential by design
m.dim(), m.used_cuda(), m.precision(), m.model_id()     # introspection
m.count_tokens("long text...")                           # session's max_length applies

# nano (compact) / pinned revision / cache dir / dim / device / ceiling / weights / arena
m = JinaV5.open("jinaai/jina-embeddings-v5-text-nano-retrieval",
                truncate_dim=512, device="auto", precision="auto")

# air-gapped: local files only, zero network (both or neither)
m = JinaV5.open_files("model.onnx", "tokenizer.json")

# what the registries hold (ids, kind=embed|rerank|decide, ladders,
# precisions, vision text_partner; task rows leave vector-space keys empty;
# + source=builtin|registered, measured=true|false on every row)
from embroider import available_models
available_models()

# register a new model under a known kind (no source change):
# kinds are rerank | decide | decide-laya | decide-2b (+ embed below)
from embroider import register_task
register_task("myorg/my-reranker", "rerank", 256,
              {"repo": "myorg/my-reranker-onnx", "file": "onnx/model.onnx"},
              q8={"repo": "myorg/my-reranker-q8-onnx", "file": "model_q8.onnx",
                  "sidecar": "model_q8.onnx.data"},
              provenance={"measured": False, "notes": "self-quant, bench pending"},
              probe=True)
# file form (strict schema, version-checked) + overlay export (sorted, round-trippable)
from embroider import load_registry_file, export_registry
load_registry_file("models/extra.json")   # {"registry_version": 1, "tasks": [...], "models": [...]}
export_registry()
```

`task` is load-bearing and per-model: Jina ids map `query` ↔ `Query:`
prefix / `document` ↔ `Document:` (last-token pooling), while
`google/embeddinggemma-2` maps them onto Gemma prompts (`task: search
result | query:` / `title: none | text:`, mean pooling) — swapped
prefixes silently fork the space.

```python
# EmbeddingGemma 2 (Apache-2.0): same calls, Gemma contract inside
g = JinaV5.open("google/embeddinggemma-2", truncate_dim=512)
g.encode("what do badgers eat?", task="query")
g.encode("a badger diet overview", task="document")
g.encode("what do badgers eat?", task="query")  # fp16 on CUDA via auto
```

```python
# pin weights + tokenizer to an exact commit (reproducibility)
m = JinaV5.open("jinaai/jina-embeddings-v5-text-small-retrieval",
                revision="abc123...", cache_dir="~/.cache/myapp")

# someone else's compatible repo (Jina layout: onnx/model.onnx + tokenizer.json)
m = JinaV5.open("myorg/my-encoder")

# int8: nano only, explicit opt-in (auto never resolves here)
m = JinaV5.open("jinaai/jina-embeddings-v5-text-nano-retrieval", precision="int8")
```

`open` is the single expensive step (download + session build); a failed
open is cached and re-raised, so config errors fail fast once. `max_length`
caps encodes at 1..=32768 (`None` = 8192 compat default; `.max_length`
reports the effective limit). `cpu_arena=False` (default) disables the CPU
arena allocator — 8x lower peak RSS (15.3 → 1.9 GB on FP32) for ~1.4x encode
time; pass `True` only when peak throughput beats memory pressure.

### Devices and precisions

| `device=` | Meaning |
|---|---|
| `"auto"` (default) | CUDA when the loaded ORT exposes a usable EP, else CPU |
| `"cpu"` / `"cuda"` (`"gpu"` alias) | Pin it; CUDA-requested-but-missing degrades to CPU with a stderr warning |

| `precision=` | Meaning |
|---|---|
| `"auto"` (default) | Follows the *landed* device: CUDA→fp16, CPU→fp32 |
| `"fp32"` / `"fp16"` | Pinned (fp16 selects the mirror artifact for the default id) |
| `"int8"` | Nano only; deployment choice, never automatic |
| `"q8"` | Weight-only int8 where listed (decider-2b, v4-edge, Laya×3, rerankers); deployment choice, never automatic |

`truncate_dim`: Matryoshka ladder only, 32–1024 (nano tops at 768).
`max_length`: 1–32768, default 8192 (compat); same (text, `max_length`)
→ same vector, short inputs bit-identical at any limit.

`auto` follows the *resolved* device, so CUDA-requested-but-missing degrades
to FP32 weights instead of stranding FP16 on CPU (FP16-on-CPU runs >40x
slower than FP32-CPU — measured). `fp16` maps the default id to the published
FP16 mirror repo; explicit model ids always win untouched. An explicit
`fp16`-on-CPU warns loudly but is honoured for text. Do not mix precisions
in one index — FP32 vs FP16 weights shift vectors, same as mixing tuning
levels.

## Vision embeddings (Python)

Two slots with the same encode shape. The Jina slot maps image bytes into
the text-nano space (text-nano graphs only — the registry's `text_partner`
pin refuses anything else); the Gemma slot (`google/embeddinggemma-2-vision`,
Apache-2.0) maps them cross-modally into the `google/embeddinggemma-2` text
space. Each fuses its tower with a text tower at the same precision tier.
Resize on the caller side to exactly the slot's target size with Pillow
bicubic — anything else is rejected rather than embedded under the wrong
resolution contract. `auto` precision → fp16 on CUDA / fp32 on CPU;
explicit fp16-on-CPU is an error in both slots (the Jina graph stalls on
CPU; the Gemma tower NaNs under emulated fp16 — fail fast either way).

```python
from embroider import JinaV5Vision, vision_target_size
from PIL import Image
import numpy as np

v = JinaV5Vision.open("jina-v5-omni-nano-retrieval-vision")
h, w = vision_target_size(img.height, img.width)   # contract target; anything else is rejected
rgb = np.asarray(img.resize((w, h), Image.BICUBIC)).tobytes()
vec = v.encode_image(rgb, h, w)   # compares against text-NANO vectors only
```

The dynamic-grid graph takes six inputs; the three grid tensors
(`vision_pos_ids`, `interp_indices`, `interp_weights`) are computed on the
host, bit-identical to transformers and fixture-pinned. Every encode
re-checks `seq == image_tokens + 15`, so a wrong tokenizer fails loudly
instead of shifting the space. Vision slots run the vision-slot session
policy (ORT defaults, never the text tuning; CUDA
`arena_extend_strategy=kSameAsRequested`, optional `gpu_mem_limit`, CPU
arena off).

```python
# local weight files (sidecar .onnx_data must sit next to model.onnx)
v = JinaV5Vision.open_files("model.onnx", "tokenizer.json")

# cap CUDA arena growth (bytes); CPU arena is always off here
v = JinaV5Vision.open("jina-v5-omni-nano-retrieval-vision",
                      gpu_mem_limit=2_000_000_000)
```

```python
from embroider import GemmaVision, gemma_vision_target_size
# Gemma vision slot: tower + text-tower fusion, same encode shape
g = GemmaVision.open("google/embeddinggemma-2-vision")   # resize to gemma_vision_target_size(h, w) first (Pillow bicubic)
v = g.encode_image(rgb_bytes, h, w)                        # unified with google/embeddinggemma-2 text space
v = GemmaVision.open("google/embeddinggemma-2-vision", truncate_dim=128, device="cuda").encode_image(rgb_bytes, h, w)
# explicit files: vision tower + text tower + tokenizer
g = GemmaVision.open_files("vision_encoder.onnx", "model.onnx", "tokenizer.json")
```

Two sessions run per Gemma encode: the SigLIP-style vision tower
(`pixel_values` + `pixel_position_ids` in, `image_features` `[K,512]` out,
K = pooled real-patch count) and the text tower from `text_partner`, fusing
with `[bos, boi, image × K, eoi, eos]` ids and the native
`sentence_embedding` head. Host math ports the transformers image processor
exactly (aspect-preserving resize to multiples of 48, rescale /255, no
normalize, patch 16, `[col,row]` positions, `[-1,-1]` pad).

Fail-fast surfaces: wrong resolution (target-size mismatch), wrong
tokenizer (Jina `seq` check), non-nano text partner — all raise, never
embed.

## Rerank (Python)

```python
from embroider import Reranker

r = Reranker.open("cross-encoder/ms-marco-MiniLM-L-6-v2")
r.score("What is the capital of France?", "Paris is the capital of France.")  # logit, higher = more relevant
r.score_batch([(q, d1), (q, d2)])   # one padded session run per call, order-preserving

# multilingual twin (mMARCO, XLM-R — token_type_ids not fed, discovered from the graph)
r = Reranker.open("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# air-gapped: local files only, zero network
r = Reranker.open_files("model.onnx", "tokenizer.json")

# documents past the 512-token pair cap: sliding windows + max-pool
r.score_long(query, long_doc)                    # window=max_len, 50% overlap
r.score_long(query, long_doc, window=512, stride=128)
r.count_tokens(long_doc)   # true length, no specials; budgets build on this
```

Pair cap: `max_length` (default 512, the model ceiling); truncation is
truncate-doc-side-first — the query is never cut to fit the document, and
a query that alone exceeds the cap raises instead of scoring garbage. The
loader feeds `token_type_ids` only when the graph declares them (MiniLM
3-input vs mMARCO 2-input, discovered at load) and reads the `logits`
output. `score_long` slides windows with MaxP aggregation (a document is
relevant if any part answers); single-window inputs score exactly like
`score_one` (e2e-pinned ≤1e-6). Scores reorder candidates — they never
enter an index, and rerankers take no `truncate_dim`. Precision notes from
the spike: fp16 rejected (slower + drift), CUDA opportunistic (~1.5x at
batch, worse at b1). Goldens in `fixtures/golden_rerank_*.json` (torch
parity 2.9e-6/7.7e-6 at pinned revisions).

## Decide (Python)

```python
from embroider import Decider

d = Decider.open("knowledgator/gliclass-small-v1.0")
d.decide("The football match ended 2-1.", [("choice", "", ["sports", "cooking"])])
# → [("sports", [("sports", 0.94...), ("cooking", 0.0...)])]  — softmax, exactly one wins
d.decide(state, [("score", "", options)])   # sigmoid per option, native unthresholded
# d.decide(state, [("noul", "", options)])  → RuntimeError: no registered model serves Noul

d4 = Decider.open("knowledgator/gliclass-v4-edge")   # labels-first layout, prompts, 2048 ctx
d4.decide(state, [("choice", "Apply the refund policy.", ["approved", "denied", "escalate"])])

d.count_tokens(state)   # true length, no specials
```

`Choice` is single-label (stable softmax — exactly one option wins);
`Score` is multi-label sigmoid, native and unthresholded — the consumer fits
thresholds, because raw multi-label sigmoids are systematically
over-confident (the B0 probe: `quantum physics` at 1.0 on a football text).
`Noul` is declared but loud-unsupported — no registered model serves it.
Small-v1.0 is text-first (`state + labels + SEP + instructions`,
`prompt_first=false`); v4-edge is labels-first (`prompt_first=true`, per
model in the registry — `open()` resolves it, `open_files` takes it
caller-side). Temperature is identity — neither model ships fitted temps
(v4's calibration lives in its graph). Goldens in
`fixtures/golden_decide_gliclass_{small,v4edge}.json` (small abs 1e-6, v4
abs 0.03 — the embedded calibrator accumulates kernel-order drift).

## Laya (Python)

```python
from embroider import LayaDecider

d = LayaDecider.open("convaiinnovations/laya-typed-decisions")
d.decide("My invoice shows a double charge.", [
    ("choice", "Route the ticket.", [("billing", "invoices and payments"), ("technical", "login and crashes")]),
    ("noul", "Does the passage answer the query: How do I reset my password?",
     "Hold the reset button for 10 seconds."),
])
# -> [(label, probs, value|None, confidence, answer_confidence, act), ...]
d.temperature_report()   # (triple, per-bucket map, clamp rejections)
d.decide(state, [("score", "Rate the sentiment.", ["negative", "neutral", "positive"])])
# -> value = expected level index, label = top level text
d.predict_long(long_doc, [("noul", "Does it answer Q?", "passage holds the answer.")], window=128, stride=64)
# -> one shared run, strongest-P(true) window decides; spans reported
d.load_binning_map("calibration.json")   # fitted overlay only; nothing shipped
```

`open_files` takes the config path as required, never optional — it carries
the checkpoint truth (ceilings, temperatures, layout check), and
caller-supplied temps would silently fork the calibration. The four
structural tokens resolve from the sibling `tokenizer_config.json` exactly
the way HF resolves `cls_token_id` (the multilingual checkpoint maps them
onto `<bos>`/`<eos>`/`<mask>`/`<pad>`); registry opens fetch that file
automatically. Choice carries labels with optional descriptions (softmax);
Score carries ordered LEVELS (one distribution — expected value in `.value`,
top level in `.label`); Noul carries one statement (P(true) — the rerank
primitive) and must not be confused with GLiClass multi-label Score, so each
loader refuses the other's ids. Temperature is fitted per-bucket from the
config, clamped to [0.5, 5.0] with rejections in `temperature_report()`.
Decode is `logits/t → softmax → unpermute → per-type shape` in float64 (the
agent's numpy arithmetic), cast at the boundary; the act head reads
`softmax(act_logits)[0]` (`.act_probability`, feeds escalation).

## Decider-2B (Python)

```python
from embroider import Decider2B

d = Decider2B.open("Mapika/decider-2b")   # slot layout, 32k ctx, fp32; CUDA = parallel executor
d.decide("My card was charged twice.", [
    ("choice", "Which department?", [("billing", "invoices and payments"), ("sales", None)]),
    ("noul", "Does this need a refund?", "yes", "no"),   # BOTH criteria (unlike Laya)
    ("score", "How urgent?", ["low", "medium", "high"]),
])
# -> [(label, probs, value|None, confidence, answer_confidence), ...]  (no act head)
d.temperature_report()   # ([2.465, 1.0, 0.05], clamp rejections) — [0.05, 5.0] range
```

One row per question: `Context:` + state, then lettered choices (`(A) …`,
one label token each past 10 options) and a trailing `Answer: (` slot scored
over the 255 label ids; score questions become one yes/no row per level.
Two inputs (`input_ids`, `slot_pos` — causal, no attention mask), one output
(`label_logits`). Temperatures are the checkpoint's own triple from
`calibration.json` — the fitted noul temp IS 0.05, so a Laya-style 0.5 floor
would corrupt it. Noul carries BOTH criteria (`yes:` / `no:`); there is no
`option_order` (the reference has none). CUDA runs the parallel executor
with default thread pools (the sequential planner aliases a shape buffer on
this graph — probed). 32k context, fp32 + q8 (worst 0.0020, 0 flips).
Goldens in `fixtures/golden_decider2b.json` (exact ids, 5e-4 on CPU *and*
CUDA).

Sequence cap: `max_length` (default 512, the model ceiling); truncation is
truncate-state-side-first — the options are the question and are never cut
to fit the state, and options that alone fill the cap raise instead of
deciding garbage. Multi-label sigmoids run systematically over-confident
(B0 §8) — thresholds and calibration live in the consumer, never here.

## Tokenizer-only counts (Python)

```python
from embroider import JinaTokenizer

t = JinaTokenizer.open("jinaai/jina-embeddings-v5-text-small-retrieval")
t = JinaTokenizer.open_files("tokenizer.json")   # no download at all
t.count_tokens("...")   # true length, never truncated (~0.5 s cold)
```

## Cache inspection (Python)

```python
from embroider import cache_info

info = cache_info("jinaai/jina-embeddings-v5-text-small-retrieval", precision="fp16")
info["cached"], info["repo"], info["snapshot_path"], info["disk_usage_bytes"]
info["files"]   # per-file paths: model, sidecar (if any), tokenizer.json

# full signature: cache_info(model_id, revision=None, cache_dir=None, precision=None)
# offline only: never downloads, never opens a session, never probes the device
```

Resolves the artifact exactly like `open()` (`lookup_model` →
`artifact_for`; task ids via `lookup_task` → `artifact_for_task`): the default id at fp16 reports the mirror repo,
multi-file repos list their `.onnx_data` sidecar. `cached=True` means an
offline `open()` would succeed — model file + `tokenizer.json` present;
a missing sidecar only warns at open and keeps `cached` true.
`precision=None` reads fp32 (side-effect free by contract); 'auto' and
bare legacy ids raise `ValueError` before any I/O.

Non-registry layouts (converter models, custom repos) use the generic
surface — same offline mechanics, same report contract:

```python
from embroider import cache_info_files

info = cache_info_files("SWHL/RapidOCR",
                        ["PP-OCRv4/ch_PP-OCRv4_det_infer.onnx",  # bare name = required
                         ("optional-sidecar.onnx", False)],      # (name, required) pair
                        cache_dir="/shared/hf-hub")
info["cached"]   # True when every required file is present
# info["model_id"] echoes the repo; info["precision"] is None (no tier here)
```

## Runtime inventory (which dylib, which CUDA)

```python
from embroider import runtime_info, save_runtime_info, prepare_cuda_runtime
info = runtime_info()
info["ort_dylib"], info["sibling_provider_dlls"], info["cuda_usable"]
save_runtime_info("bench/run-001/runtime.json")   # pretty JSON next to the numbers

prepare_cuda_runtime()   # idempotent, never throws; the same PATH prep
                         # every CUDA open already runs itself
from embroider import configure_runtime
configure_runtime(dll_dirs=info["load_dirs"])   # reproduce a saved layout
```

One dict inventories the ORT situation: the loaded dylib (`ORT_DYLIB_PATH`,
with existence check), sibling provider DLLs, every `onnxruntime*` install in
the venv (CPU and GPU side by side), the CUDA/cuDNN DLLs the `nvidia*` dists
ship, and the usability probe. `configure_runtime(dylib_path=None,
dll_dirs=None)` is the write half — `dylib_path` is effective only before the
first `open()` in the process (ort pins the first library it initializes, and
even the probe counts as a first touch), `dll_dirs` prepend to `PATH`
immediately (this is what makes `cudnn64_9.dll` resolve for cuDNN-backed ops
like GLiClass's `ReduceMax`). Keys are stable (`ort_dylib`,
`ort_dylib_exists`, `sibling_provider_dlls`, `cuda_usable`, `ort_installs`,
`cuda_runtime_dlls`, `load_dirs`, `virtual_env`).

## Explicit local files (air-gapped)

```python
# every loader has an open_files seam: downloads skipped, sidecar
# (model.onnx_data-style) must sit next to the ONNX file — ORT resolves
# it relative to the model path, same as the HF cache layout
m = JinaV5.open_files("model.onnx", "tokenizer.json")
# precision selection does not apply here — the files are what they are
# (reported precision reads fp32; pass FP16 files explicitly for FP16)
```

Same bytes in → same vectors out (test-pinned against HF acquisition).
OKFgraph's `OKFRouter(model_path=..., tokenizer_path=...)` uses these (both
or neither; missing files raise `FileNotFoundError` at construction).

## Rust: sessions without Jina

```rust
// tuned session over your own model file (cf. bobine's session_builder):
let session = ort::session::Session::builder()?;
let tuned = embroider::SessionPolicy::ort_defaults().apply(builder)?;
let session = embroider::apply_providers(tuned, providers)?.commit_from_file(path)?;

// text-tuned profile (Level3, measured) — Jina path only; never mix tunings in one index
let policy = embroider::SessionPolicy::text_embed();

// environment diagnostics
let rep = embroider::report();   // dylib path + CUDA usable?
let cuda = embroider::cuda_available();   // OnceLock-cached EP check, not a registration probe
let _dirs = embroider::prepare_cuda_dirs();   // 0.14.0+: PATH prep every CUDA open runs itself; idempotent
let (owner, name) = embroider::parse_owner_name("org/repo")?;

// request parsing (same strings as the Python kwargs)
let device = embroider::DeviceReq::parse("auto")?;        // auto | cpu | cuda (gpu alias)
let prec = embroider::Precision::parse("fp16")?;          // auto | fp32 | fp16 | int8
let landed = prec.resolve(used_cuda);   // pass the LANDED device, never the request

// offline cache inspection (Python `embroider.cache_info` underneath):
// never the network, never a session, no device probe — precision must be explicit
let rep = embroider::cache_info(
    "jinaai/jina-embeddings-v5-text-small-retrieval",
    None,                       // revision: None = main
    None,                       // cache_dir: None = env-resolved hub cache
    embroider::Precision::Fp32,
)?;
// rep: CacheReport { repo, precision, cache_dir, files, cached,
//                    snapshot_path, disk_usage_bytes }
```

## Errors

Python: bad arguments → `ValueError` (firing before any I/O);
load/encode failures → `RuntimeError` with the full anyhow chain
(`{e:#}`). A failed `open` is cached — config errors raise once, not
per encode. Rust: `anyhow::Result` everywhere, `ort` errors stringified
at API boundaries.

## Conformance (for consumers)

Vendor `fixtures/golden_jina_v5_text_small.json` (12 vectors: 4 texts ×
Query/Document × dims 64/512 + token counts) and assert live vectors at
`abs=1e-6` — catches wrong model, pooling, prefix, or truncation, immune
to cross-CPU noise. Vision host math: `fixtures/vision/`
(`target_sizes.json`, `host_tensors.json`, `pixel_cases.json`) asserts
bitwise equality. Regenerate only on an intentional contract change =
new minor version + re-index-everything notice.

## Testing

```bash
cargo test --locked                      # 111 pure unit tests (no net/dylib)
# vision e2e (ignored): needs ORT 1.31 dylib + weight files + CUDA ideally
ORT_DYLIB_PATH=.../onnxruntime.dll VISION_E2E_FP32=.../model.onnx \
  VISION_E2E_FP16=.../model.onnx VISION_E2E_TOK=.../tokenizer.json \
  VISION_E2E_CUDA=1 cargo test --offline --test vision_e2e -- --ignored
```

Python parity lives with okfgraph (`tests/test_parity.py`, slow):
Rust vs numpy/transformers across dims × tasks × texts at ≤ 1e-5.

## Rules

- Same (model, precision, tuning, `max_length`) or re-embed. Never mix
  precisions or opt levels in one index — FP32 vs FP16 and Level1 vs
  Level3 both shift bits.
- `truncate_dim`: Matryoshka ladder only (nano tops at 768).
- Encode fails fast: no fallback, no silent space fork.
- Stale `C:\Windows\System32\onnxruntime.dll` (1.17.x in the wild) kills
  the load — point `ORT_DYLIB_PATH` at the venv build.
- GIL is released during encode; batch encoding stays sequential
  (padded batches waste attention compute on variable-length docs).
