# Serving a model: `python -m embroider.serve` (0.15)

One embroider model behind HTTP, in its own process. Use it when the
consumer must not load ONNX Runtime itself (a second runtime beside one
already loaded takes an interpreter down), or when several consumers
should share one loaded model.

The routes answer in the shapes the llama.cpp servers answer, so a client
written for those takes an embroider host unchanged. `GET /v1/models` adds
the model's identity, so a graph can pin a hosted model as strictly as one
it runs itself.

```bash
python -m embroider.serve --kind embed --model google/embeddinggemma-2 --dim 512 --port 8811
# stderr: listening http://127.0.0.1:8811
```

## Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--kind` | required | `embed`, `vision`, `rerank`, `decide`, `decide-laya`, `decide-2b` (`embroider.KINDS`) |
| `--model` | required | registry id or hub repo |
| `--revision` | `None` | branch, tag or commit sha |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--precision` | `auto` | `auto`, `fp32`, `fp16`, `int8`, `q8` (what the model ships) |
| `--cache-dir` | hub default | Hugging Face hub cache |
| `--dim` | `0` | embed/vision width; `0` = the model's native width |
| `--max-length` | model's | token ceiling (not for vision) |
| `--host` / `--port` | `127.0.0.1` / `0` | `0` picks a free port; read it from the `listening` line |

Exit codes: `2` bad arguments, `3` the model did not open (reason on
stderr). Before opening, `import embroider` has already run
`locate_runtime()`, so the pip-installed ONNX Runtime is found without
importing the Python `onnxruntime` module.

## Routes

| Route | Kinds | Body | Answer |
|---|---|---|---|
| `GET /health` | all | | `{"status": "ok"}` |
| `GET /v1/models` | all | | `{"object": "list", "data": [identity]}` |
| `POST /v1/embeddings` | embed | `{input, task?, input_type?, dimensions?}` | OpenAI embeddings shape |
| `POST /v1/images/embeddings` | vision | `{image \| input: base64 PNG/JPEG (or a list)}` | OpenAI embeddings shape |
| `POST /v1/rerank` | rerank | `{query, documents, top_n?}` | `{results: [{index, relevance_score}], usage}`, best first |
| `POST /v1/systemone` | decide-* | `{state, questions}` | `{answers, usage}` (llama.cpp's typed decisions) |
| `POST /tokenize` | text kinds | `{content}` | `{tokens, count}` |

A route of another kind answers `501`; a malformed body `400`.

- **Embeddings.** `task` is embroider's (`Document`, the default, or
  `Query`; lower case is fine), and `input_type` (`query`, `document`,
  `search_query`, `passage`, ...) is read as `task`. **The host applies
  the model's prompts** (Jina's `Query:`/`Document:`, EmbeddingGemma's
  task prompts): a client sends plain text and never prefixes.
  `dimensions` is accepted only when it equals the opened width; anything
  else is `400`, so a vector is never silently truncated.
- **Images.** The host decodes and resizes with `embroider.prepare_image`
  (Pillow bicubic to the slot's grid, needs `embroider[image]`), the same
  pixels an in-process caller gets.
- **Decisions.** `choice` (criteria `{label: description|null}`),
  `score` (criteria `[level, ...]`, at least two) and `noul` questions,
  answered as llama.cpp answers them: `{type, choice, probabilities,
  confidence}`, `{type, score, legend, probabilities, confidence}`,
  `{type, noul}`. A noul question may carry `criteria` (a string, or
  `{"true": ..., "false": ...}`); without it the instructions are the
  statement. GLiClass (`decide`) has no noul head: `501`.
- **Tokenize.** `count` is the model's own count (what a budget reads).
  `tokens` are the ids from `Tokenizer.open(model)`; a model whose
  tokenizer doesn't open by id gets `count` placeholders.

## Identity and `space`

`GET /v1/models` returns `Model.describe()`:

```json
{"id": "google/embeddinggemma-2", "object": "model", "kind": "embed",
 "engine": "embroider", "engine_version": "0.15.0", "contract": "0.15",
 "revision": "<snapshot sha>", "device": "cpu", "precision": "fp32",
 "max_length": 8192, "family": "embeddinggemma-2",
 "dim": 512, "native_dim": 768, "ladder": [128, 256, 512, 768],
 "tuning": "text_embed",
 "space": "embroider/google/embeddinggemma-2@<sha>:fp32:512:8192:text_embed:v1"}
```

`space` (embed and vision kinds) is the vector-space identity: **equal
`space`, identical vectors**. It is `embroider.space_id(model, revision,
precision, dim, max_length, tuning)`, with the resolved snapshot sha as
revision (`?` when unknown, which never matches) and the vector contract
(`embroider.VECTOR_CONTRACT`, changed only when vectors change). The
device is not part of it: CPU and CUDA sessions at one precision serve
one space, as graph pins always treated them.

The host opens the model with `embroider.open`, exactly as an in-process
caller does, so the vectors match bit for bit across processes;
`tests/test_serve_live.py` pins that at `abs=0` on CPU.

Rerank and decide kinds carry the same fields minus `dim`, `ladder` and
`space`: they are reported, never pinned. `precision` is always the
weights that loaded: `auto` resolved on the landed device, and a tier
the model doesn't ship reads as the `fp32` it fell back to.

## From Python

```python
import embroider
model = embroider.open("embed", "google/embeddinggemma-2", truncate_dim=512, device="cpu")
model.describe()["space"]                   # what a host of the same settings publishes
model.encode_batch(["text"], task="Query")  # the native JinaV5 call
```
