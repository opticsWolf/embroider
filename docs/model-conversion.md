# Model conversion playbook

How we get weights from a HuggingFace checkpoint into a served,
parity-gated ONNX artifact — and everything that bit us along the way.
Covers the three conversions done to date (decider-2b weightless,
strands pointer-head fused, sauhsoj verification) plus the quantization
ladder and the ORT execution gotchas. Read this before touching a new
checkpoint.

## 0. Principles (non-negotiable)

1. **License first.** Only Apache-2.0 (or equivalent permissive) flows
   into `opticsWolf/*`. Check `cardData.license` via the HF API *and*
   the repo's `LICENSE.md` before downloading weights. CC BY-NC models
   (Jina embeds/vision) are referenced upstream, never mirrored.
2. **Measured artifacts only.** Nothing ships without a parity number
   against an independent reference (torch engine, upstream agent, or
   the exporter's own selftest). Tolerances are per-family (see §6).
3. **Reference before export.** A conversion without an independent
   truth source is guesswork. Find the upstream inference
   implementation first (§2); if none exists, the model is not
   convertible yet.
4. **Pin everything.** Source revision, base revision, adapter config,
   file SHA-256 in `manifest.json`. A re-run a year from now must
   reproduce the bytes.
5. **One weights contract per serving path.** The fused single-graph
   artifact is what ships (one session, no numpy sidecar). Unfused
   intermediates are construction proofs, not deliverables.

## 1. Recon: reading a checkpoint repo (30 minutes, no downloads)

Start with the HF API — never the web UI:

```python
import json, urllib.request
u = 'https://huggingface.co/api/models/<owner>/<repo>'
d = json.load(urllib.request.urlopen(u))
# d['siblings'] -> file list; d['sha'] -> pinned revision
# (d.get('cardData') or {}).get('license'); d['tags'] (peft? lora? base_model:*?)
```

Then fetch the small files (all <100 KB, raw URLs):
`.../resolve/main/<config>.json`, `adapter_config.json`,
`provenance.json`, `train_config.json`, `MANIFEST.sha256`, `README.md`.

What you're looking for, in order:

| question | where | why it matters |
|---|---|---|
| merged or adapter? | siblings (`adapter_model.safetensors` + `head.safetensors` = LoRA) | adapter needs a merge step (§3b) |
| which architecture? | `config.json` / `strands_decider_config.json` (`head_type`, `num_slots`, `pointer_dim`) | slot head vs pointer head = different export entirely |
| temperature semantics | `temperature` / `temperature_by_kind` / `calibration.json` | divide-outside vs inside; clamp range (decider-2b's noul IS 0.05 — a 0.5 floor corrupts it) |
| max context, option caps | `max_length`, `max_options`, `max_levels` | registry ceilings, bench filters |
| base + revision | `base_model`, `base_model_revision`, provenance | the merge target; "inferred: Hub main at training time" = re-verify, don't trust |
| evals shipped? | `eval/` (`jevbench*`, `summary.json`) | free acceptance criteria + reference numbers |
| xet or git? | `Content-Length: ~1 KB` on a GB file = xet pointer | needs `huggingface_hub` (xet-capable), not curl, for weights |

Beware `Content-Length: 1038` on a gigabyte file — that's an xet
pointer, not the weights. Download weights only via `huggingface_hub`
(`snapshot_download`), which resolves xet automatically.

## 2. Reference implementations (find these first)

| model | reference | what it pins |
|---|---|---|
| Mapika/decider-2b | torch `decider` package (`decider.infer.Decider`, ollaya-convert cache) | system_one answers; 3/3 parity at 1e-4 CPU |
| strands v1-2610 / hobson-v19 | `strands_decider` PyPI (`infer.load_engine`, `prompting`, `schema`) + `fixtures/requests.json` | per-question probs, option positions, shared-prefix deltas |
| sauhsoj q4f16 | `selftest.json` in the repo (ids + torch probs + q4f16 probs) | third-party cross-check, no torch needed |

Version drift is real: `strands_decider` 0.2.0 changed
`engine._fit` (2-tuple → 3-tuple with offsets) and `_option_idx`
(grew `base` + `offsets` params). Pin the reference package version
next to the checkpoint revision; when the API drifts, adapt the
*caller* (`reference.py`), never the golden bytes.

## 3. Conversion recipes

### 3a. Weightless graph + layout port (Mapika/decider-2b pattern)

Use when: torch `dynamo`/`torch.onnx.export` hangs or produces
contrib-ops (the decider-2b graph did — structurally hung), but the
model is a standard decoder with a simple head.

1. **Build the graph algebraically** (Ollaya `weightless_sharded.py`
   pattern): empty ONNX shell with the right inputs/outputs, weights
   vendored alongside as `.safetensors` (external data). ORT resolves
   external data by filename next to the `.onnx` — keep them together.
2. **Port the layout 1:1** (`render_state`, lettered/wide option
   pieces, isolated score rows, row batching with budget 8192 and
   seq-multiple-64 padding, temperatures + clamp). The port *is* the
   product; the graph is dumb weights.
3. **Serve it** (embroider `Decider2B`): own session policy, own
   question type where the reference demands it (decider's noul carries
   BOTH criteria — sharing Laya's single-statement enum would silently
   drop the false side on the whole bench).
4. Gate: reference-built goldens (exact input ids = construction pin)
   + CPU *and* CUDA e2e at 5e-4.

### 3b. LoRA-merge + OPT-graph swap + fuse (strands/hobson pattern)

Use when: the checkpoint is LoRA + head, and an `onnx-community`
OPT export exists for the base architecture (Qwen3.5-2B-OPT here).

Pipeline (`conversion/`, adapted per source — the v1-2610 work kept
every script, changed only ids/config filename):

1. `download.py` — decider (minus `eval/*`, `training/*`), base torso
   at the **pinned** revision, OPT graphs (`decoder fp32 + q4f16`,
   `embed fp32 + q4f16`). Check what the base repo costs first; the
   torso is the biggest download.
2. `reference.py` — torch reference outputs over `fixtures/requests.json`.
   Needs the upstream engine installed (`strands_decider` from PyPI
   worked at 0.2.0). This is the truth everything gates against.
3. `build_onnx.py` — merge LoRA into fp32 base (`W + alpha/r * B@A`,
   read `r`/`alpha`/targets from `adapter_config.json`, assert
   `not use_dora and not use_rslora`), map onto graph initializers by
   name with transforms (transpose / plus_one / neg_exp) via the
   value-checked mapping, **cosine-check every swapped weight against
   the instruct weights it replaces** (gate: worst > 0.5; we measured
   0.993). Drop `lm_head` → expose `last_hidden_state`. Re-quantize
   MatMulNBits for q4 (`BITS=4`) / q8 (`BITS=8`); embeds stay q4
   (see §5). Head → `head.npy` + `decider_config.json`.
4. `fuse.py` — one graph per precision: embeds + decoder + pointer
   head in fp32 (`LayerNorm → q/k MatMuls → scaled dot`), empty caches
   as in-graph zeros, positions as `arange(L)`, dead-node elimination.
   Contract: `input_ids/attention_mask/answer_pos/option_pos -> logits`.
   `PRECISION` + `EMBED_SUFFIX` envs select the parts; `NAME` the
   output. (Default `EMBED_SUFFIX` = decoder suffix — the one
   mis-settable knob; q8 uses q4 embeds deliberately.)
5. `parity.py` (unfused, fp32 default) and `parity_fused.py` (one
   padded batch per record). Fused-vs-unfused must agree to ~5e-5
   (we measured 4.9e-05); fused-vs-torch is the ship gate.

### 3c. Verifying someone else's export (sauhsoj pattern)

Use when: a community ONNX build already exists. Don't rebuild what
you can verify.

1. Confirm the contract from `config.json` (`inputs`/`outputs` names,
   prompt layout, temperature table) and the method from the README.
2. Run their selftest ids through ORT (CPU *and* CUDA) and reproduce
   *their* q-column first (we hit max|d| <= 6e-4 on all 7) — that
   validates the artifact is what it claims.
3. Then compare against torch (theirs or yours) and record the gaps
   honestly, flips included (`tabs/dev` flips under q4 — documented,
   not hidden).
4. Note what it is and isn't: their q4f16 complemented but never
   replaced our fp32 build.

## 4. ORT execution gotchas (all measured, all on RTX3090 + ORT 1.29/1.31)

- **#4829-class shape aliasing (decider-2b, CUDA).** The sequential
  memory planner reuses a shape buffer; a chunk-scan Reshape reads a
  zeroed/garbage dim and the run fails on the *first* batch.
  Deterministic, not a race. The strands graphs don't hit it (shape
  ops partition to CPU by default — ORT even logs it).
- **`enable_mem_reuse=false` is unreachable from the C API.**
  No setter in `ort-sys` *or* the 1.29 C headers (only
  Enable/DisableMemPattern + CpuMemArena exist). The
  `session.enable_mem_reuse` config key is **not honored** (probed).
  Python reaches it only because pybind writes the C++ struct
  directly. Rust cannot set it — stop looking.
- **Parallel executor is the sanctioned equivalent** (decider-2b):
  `SetSessionExecutionMode(ORT_PARALLEL)` is C-API-backed
  (`SessionBuilder::with_parallel_execution`, already in ort 2.0).
  Proven parity-identical on CPU + CUDA.
- **...but only with default thread pools.** An explicit
  `inter_op_num_threads=1` re-triggers the corruption under the
  parallel executor (probed 2x2: inter=1 fails, inter=default
  matches, either intra). `SessionPolicy::decider2b` sets NO thread
  counts. Do not "tune" this without re-running parity.
- **`with_memory_pattern(false)` is NOT an equivalent** (probed:
  still corrupt). Neither is `use_env_allocators`.
- **cuDNN DLL search.** ORT's CUDA EP dlopens `cudnn64_9.dll` by bare
  name; it is invisible inside `site-packages` until the dirs hit
  `PATH`/`add_dll_directory`. `embroidery.configure_runtime(dll_dirs=
  runtime_info()["load_dirs"])` is the sanctioned exposure — bench
  and e2e processes both need it. (`_9` is the major, not age:
  cuDNN 9.24 is current; only ReduceMax/Einsum graphs load it, which
  is why rerank/Laya ran while GLiClass failed.)
- **ORT_DYLIB_PATH is first-touch.** The dylib binds on first `open()`
  (the probe counts). Set it before anything opens a session.

## 5. Quantization (what works, what doesn't, and why)

Measured on our hardware — not theory. The ladder for a 2B decision
model now reads: **fp32** (faithful) -> **q8** (no flips, ~1/4 size)
-> **q4f16** (one known flip class, ~1/7 size). Built with
`quantize_matmul_4bits` / `quantize_matmul_8bits` (block 32) from the
`onnxruntime` package itself.

| route | mechanism | house verdict |
|---|---|---|
| dynamic int8 | int8 weights + fp32 activations, one call | REJECTED — drift 2.8-5.4 + flips (Laya); the -1e4 mask sentinel is unquantizable |
| static int8 | + calibration data | REJECTED — 9997 on the sentinel |
| encoder-only int8 | partial | REJECTED — 2.85, no CPU relief |
| fp16 | halves download/VRAM, CUDA-only (CPU >40x slower, measured) | ADOPTED (Laya CUDA variants); the sane next step for decider-2b |
| MatMulNBits q4 (w:4b, s:fp16) | weight-only, activations fp32 | ADOPTED with a documented flip (strands q4f16: worst 0.2074, 1 flip) |
| MatMulNBits q8 (w:8b, s:fp16) | weight-only | ADOPTED (strands q8: worst 0.0504, 0 flips) |
| int4/GPTQ/AWQ, QAT | — | untested / unavailable post-hoc |

Why weight-only survives where dynamic int8 died: activations stay
fp32, so the mask sentinel and the prompt text are exact — only the
matmul weights lose precision. Two consequences that follow everywhere:

1. **Fitted temperatures amplify weight noise.** Dividing logits by
   0.05 (decider-2b noul) multiplies any quant error by 20. Gate the
   *calibrated* outputs (noul P(true)), not raw logits — q4 is likely
   too noisy for 0.05-temp heads, q8 survives.
2. **Embeds follow the hobson recipe, not the decoder BITS.**
   `GatherBlockQuantized` blocks are 4-bit (`bits=4`, q4f16 embeds)
   even in the "q8" model (int8 matmuls + int4 embeds) — verified
   against the published hobson graph. Re-quantizing embeds to 8-bit
   crashes the reshape (different block layout); don't.

Per-model outlook (see the q8/q4 discussion): decider-2b q8 first
(same decoder family, gate noul), v4-edge q8 second (weight-only
dodges the sentinel kill), Laya q8 third (temps amplify; fp16 covers
CUDA), rerankers only for bundle size, embeddings never (quant shifts
the vector space — needs retrieval re-eval, not parity).

## 6. Parity methodology (tolerances are per-family)

- **Torch reference on CPU fp32** (`reference.py` over
  `fixtures/requests.json`): the truth. Record ids + option
  positions + probs; the same fixture pins the ONNX, the TypeScript
  port, and the Rust e2e independently.
- **Unfused first, fused second.** Unfused parity isolates the weight
  mapping (cosine gate 0.5, we see 0.993); fused-vs-unfused must agree
  to ~5e-5 before torch numbers mean anything.
- **Report worst |d| AND flip count.** Drift without flips is a
  calibration question; a flip is a behavior change. Ship gates: fp32
  0 flips; q8 0 flips on the reference set; q4 flips documented by
  case in the README, never averaged away.
- **e2e pins construction, not just numbers.** Exact input-id
  equality (the layout proof) plus 5e-4 on probs (fp32) — above the
  1e-4 torch-vs-ORT gap so the fixture stays torch-consistent.
  CUDA legs gate executor switches (decider-2b's CUDA test exists
  because the planner bug lives there).

## 7. Publishing (`opticsWolf/*`, public, Apache-2.0)

Per-model repo (house style, cf. `opticsWolf/laya-*-fp32-onnx`):

- `onnx/model.onnx` (+`.onnx_data`, fp32) and/or `model_q{4f16,8}.onnx`
  — fused single-graph artifacts only. Unfused intermediates,
  `eval/`, `training/` never upload.
- `config.json` with a `decider` section (contract: inputs, prompt
  layout, temperatures) + `transformers.js_config` for the browser
  path; `tokenizer.json` + `tokenizer_config.json` from the source
  checkpoint (tokenizer must match the training one, not the base's).
- `README.md`: method (loRA rank/merge, mapping, fuse), parity table
  with worst-|d| AND flips per precision, usage snippet, license.
- `manifest.json`: SHA-256 of every file (hash the 8 GB — minutes,
  worth it). `.gitattributes`: `*.onnx*` → LFS.
- `upload_folder` after `create_repo(exist_ok=True)`; needs
  `hf auth login` (token with write). Never publish without an
  explicit go-ahead — tags and uploads don't un-happen.

## 8. Environment and tooling

- **Venv**: `D:/User/Documents/Python/OKFgraph/.venv` (torch 2.15 CPU
  via the PyTorch CPU index, `onnx`, `safetensors`,
  `strands_decider`, `huggingface_hub` for xet). `pip install` works;
  `pip show` throws (broken pipe in this venv — ignore it, verify
  with an import instead). The embroider `.pyd` sometimes needs a
  manual copy from `target/wheels/` when pip's force-reinstall
  silently keeps the stale build (check `hasattr`, don't trust pip).
- **Disk**: D: has room (446 GB free at last check), but keep scratch
  in `D:/tmp/<model>-<tag>/` and weights in the HF cache — never in a
  repo. Only `tools/download.py --tier` touches network for eval
  data; model weights are scratch by definition.
- **Windows notes**: no symlinks without dev mode (hf_hub warns,
  still works); `PATH` vs `add_dll_directory` for cuDNN (§4); POSIX
  `/tmp` in shell ≠ `D:/tmp` — use absolute Windows paths in scripts.
- **Where things live**: conversions in `D:/tmp/<name>/`
  (`conversion/` scripts adapted per source, `out/` artifacts,
  `upload_repo.py` for publish); goldens in
  `embroider/fixtures/golden_*.json`; generators in
  `embroider/tools/gen_golden_*.py`; e2e in `embroider/tests/*_e2e.rs`.

## 9. New-model checklist

1. API recon (§1): license, adapter-vs-merged, head type, temps,
   base+revision, shipped evals.
2. Reference implementation installed and running (§2); torch
   reference generated over the request fixture.
3. Recipe picked (§3a/b/c); LoRA merged with cosine gate; weights
   swapped; head handled (fused fp32, never quantized).
4. Parity: unfused → fused (~5e-5) → torch (worst |d| + flips).
5. Precision ladder as needed (fp32 → q8 → q4f16), each gated.
6. CUDA smoke on the serving box (mem aliasing, provider
   assignment, latency) before any serving claim.
7. Publish (§7) only on go-ahead; bench into the tables; serve in
   embroider with goldens + CPU/CUDA e2e.
