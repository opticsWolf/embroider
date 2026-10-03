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
let cuda = embroider::cuda_available();
let (owner, name) = embroider::parse_owner_name("org/repo")?;
```

## Rules

- Same (model, precision, tuning, `max_length`) or re-embed. Never mix
  precisions or opt levels in one index — FP32 vs FP16 and Level1 vs
  Level3 both shift bits.
- `truncate_dim`: Matryoshka ladder only (nano tops at 768).
- Encode fails fast: no fallback, no silent space fork.
- Stale `C:\Windows\System32\onnxruntime.dll` (1.17.x in the wild) kills
  the load — point `ORT_DYLIB_PATH` at the venv build.
