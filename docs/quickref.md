# embroider — Quick Reference

Copy-paste surface. Rationale lives in `README.md`, release matrix in
`COMPAT.md`.

## Install

```bash
pip install embroider
```

```rust
embroider = "0.3"   // shared ORT plumbing; Jina engine included
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

# what the registry holds (ids, ladders, precisions, vision text_partner)
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

## Tokenizer-only counts (Python)

```python
from embroider import JinaTokenizer

t = JinaTokenizer.open("jinaai/jina-embeddings-v5-text-small-retrieval")
t = JinaTokenizer.open_files("tokenizer.json")   # no download at all
t.count_tokens("...")   # true length, never truncated (~0.5 s cold)
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
cargo test --locked                      # 42 pure unit tests (no net/dylib)
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
