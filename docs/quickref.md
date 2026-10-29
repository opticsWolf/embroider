# embroider — Quick Reference

Copy-paste surface. Rationale lives in `README.md`, release matrix in
[compat.md](compat.md).

## Install

```bash
pip install embroider
```

```rust
embroider = "0.4"   // shared ORT plumbing; Jina + rerank engines included
```

Needs exactly one ONNX Runtime: `onnxruntime==1.29.0` (or
`-gpu[cuda,cudnn]==1.29.0`), auto-located on import. Override:

```bash
export ORT_DYLIB_PATH=/path/to/onnxruntime.dll   # e.g. <venv>/Lib/site-packages/onnxruntime/capi/onnxruntime.dll
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
# precisions, vision text_partner; task rows leave vector-space keys empty)
from embroider import available_models
available_models()
```

`task` is load-bearing: `query` ↔ `Query:` prefix, `document` ↔
`Document:` — swapped prefixes silently fork the space.

```python
# pin weights + tokenizer to an exact commit (reproducibility)
m = JinaV5.open("jinaai/jina-embeddings-v5-text-small-retrieval",
                revision="abc123...", cache_dir="~/.cache/myapp")

# someone else's compatible repo (Jina layout: onnx/model.onnx + tokenizer.json)
m = JinaV5.open("myorg/my-encoder")

# int8: nano only, explicit opt-in (auto never resolves here)
m = JinaV5.open("jinaai/jina-embeddings-v5-text-nano-retrieval", precision="int8")
```

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

`truncate_dim`: Matryoshka ladder only, 32–1024 (nano tops at 768).
`max_length`: 1–32768, default 8192 (compat); same (text, `max_length`)
→ same vector, short inputs bit-identical at any limit.

## Vision embeddings (Python)

```python
from embroider import JinaV5Vision, vision_target_size
from PIL import Image
import numpy as np

v = JinaV5Vision.open()   # default: jina-v5-omni-nano-retrieval-vision
h, w = vision_target_size(img.height, img.width)   # contract target; anything else is rejected
rgb = np.asarray(img.resize((w, h), Image.BICUBIC)).tobytes()
vec = v.encode_image(rgb, h, w)   # compares against text-NANO vectors only
```

fp32 on CPU, fp16 on CUDA (`auto` default). Explicit fp16-on-CPU is an
error (the graph stalls — fail-fast, not slow).

```python
# local weight files (sidecar .onnx_data must sit next to model.onnx)
v = JinaV5Vision.open_files("model.onnx", "tokenizer.json")

# cap CUDA arena growth (bytes); CPU arena is always off here
v = JinaV5Vision.open(gpu_mem_limit=2_000_000_000)
```

Fail-fast surfaces: wrong resolution (`vision_target_size` mismatch),
wrong tokenizer (`seq == image_tokens + 15` re-checked every encode),
non-nano text partner — all raise, never embed.

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
a query that alone exceeds the cap raises instead of scoring garbage.

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
cargo test --locked                      # 49 pure unit tests (no net/dylib)
# vision e2e (ignored): needs ORT 1.29 dylib + weight files + CUDA ideally
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
