# EmbeddingGemma 2 — target architecture, rollout plan, and next steps

Status: **2026-11-16**, after the int8 text/vision landing and the N1/N3
gates. This is the
consolidated execution plan; the detailed research and workstreams stay in
`plan-eg2-int8-audio.md`, and the measurements stay in
`eg2-int8-audio-results.md` (embroider) and
`OKFgraph/docs/bench-eg2-precision-width.md`.

Owner repos: **embroider** (artifacts, registry, runtime), **okfgraph**
(pins, spaces, reembed), **Silk/Weave** (rows and UI). bobine is not
involved.

---

## 1. Target architecture

```
                       Hugging Face
  opticsWolf/embeddinggemma-2-onnx-int8 @ 52614d7
    onnx/model_quantized.onnx          (text)
    onnx/vision_encoder_quantized.onnx (vision)
    onnx/audio_encoder_quantized.onnx  (audio, mirrored only)
    tokenizer.json + configs + manifest
                           │
                           ▼
  embroider  src/acquire.rs  (ModelSpec)
    fp32 / fp16 / int8 artifact + revision + sidecar
    text_partner; MRL ladder; max_tokens
                           │
                           ▼
  embroider  src/policy.rs   (precision policy)
    auto: CUDA→fp16, CPU→fp32; auto NEVER int8
    explicit int8: artifact if registered, else fp32 fallback
                           │
                           ▼
  embroider runtime lanes
    text tower ─┐
    vision tower├─► text fusor ─► 768-d sentence_embedding
    audio tower ┘   (image_features / video_features / audio_features)
                           │
                           ▼
  okfgraph graph state
    model pin  : embedding_model        (MetaText)
    precision  : embedding_precision    (Meta, INT64 8|16|32)
    space      : embedding_space        (MetaText, model@rev:precision:dim:...)
    image pins : embedding_image_*      (model + precision + space)
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    text reembed (in place)      image graph: fresh DB + re-ingest
    writes model/space pin       + re-encode (in-place reembed refused)
             │                           │
             ▼                           ▼
  Silk / Weave rows + ONNX Loader + KG pin display
```

### Invariants

1. **One model id per vector space.** int8 is a precision tier of
   `google/embeddinggemma-2`, not a new id. The model pin cannot
   distinguish tiers; the precision pin is the guard. Do not create a
   fake `-int8` id.
2. **`auto` never resolves to int8.** An unmeasured request falls back to
   fp32, and `describe()` reports what actually loaded.
3. **The precision codec is `{int8: 8, fp16: 16, fp32: 32}`.** Unknown
   stored codes fail closed; they are never treated as “unpinned”.
4. **The space string is the second guard.** It records the model
   revision, precision, width, max length, tuning and contract. A
   precision-pin miss that somehow slips through still has to pass the
   space pin.
5. **Media precision is a pair.** The vector space is
   `(text fusion precision, tower precision)`. Image vectors are produced
   by the vision tower but fused by the text tower; the image space pin
   must name the text fusion identity, not only the tower.
6. **Image graphs are not text-reembedded.** `reembed` refuses when
   `ImageAsset` rows exist; int8 on an image graph means fresh DB +
   re-ingest + re-encode.
7. **int8 shrinks the model download, not the database.** Vectors stay
   `FLOAT[dim]`; DB size is a separate lever (width, storage
   quantization).
8. **Explicit and measured.** int8 is enabled only by an explicit
   setting, never by a default, and only after the retrieval gate.

---

## 2. What is landed

| Layer | State | Commit / artifact |
|---|---|---|
| Mirror | text + vision + audio q8, byte-identical to upstream `daa72c5` | `opticsWolf/embeddinggemma-2-onnx-int8` @ `52614d7` |
| embroider registry | text int8 + vision int8, pinned revisions/sidecars; `auto` never int8 | `95ff50c` |
| embroider live open | `open(..., precision="int8")` reports `int8`; 768-d encode works on CPU | verified 2026-11-15 |
| okf precision codec | `int8: 8`, both decoders/enforcers, fail-closed unknown codes | `4eaf1e2` |
| okf image lane | router/settings accept `image_precision="int8"`; caller, stored decode, content-hash, doctor | `4eaf1e2` |
| okf `pin_info` | text + image pins; `image_mismatch`; external host precision | `4eaf1e2` |
| reembed | validates precision before rewriting; writes a named pin; clears an unnamed pin; reads the landed precision after open | `4eaf1e2` |
| Silk/Weave | EG2 text and vision offer int8; loader rows and Q5 pick them up | `d455bc2` / `b38784a` |
| Tests | okf full suite 1339 passed / 7 skipped; Silk 1505 passed / 3 skipped / 1 deselected | 2026-11-15 |

Live text proof: `pin_info` reports `int8`, `space` is
`…@52614d7…:int8:768:8192:text_embed:v1`, `reembed` keeps int8, and an
fp32 reopen raises `PRECISION_PIN_MISMATCH`.

---

## 3. What is still open

| # | Item | Why it matters | Blocking? |
|---|---|---|---|
| 1 | **Labelled retrieval eval** for int8 text — **passed 2026-11-16** | 50-query run is a signal, not the policy bar (NDCG@10 / recall@20 delta ≤ 0.5 pt) | no (gate passed) |
| 2 | **Vision int8 live + cross-modal gate** | vision q8 is registered but no image has been encoded; image→text retrieval is unmeasured | yes, for image int8 |
| 3 | **Mixed-pair product rule** | code allows text int8 + vision fp16; plan default is text int8 + vision int8 (or no image lane) | yes, for image int8 |
| 4 | **Audio slot (W2)** | audio q8 is mirrored but has no `ModelSpec`, route, front end or consumer | no, separate feature |
| 5 | **Database size** | `okf db-size` is a prototype; 256d fresh-import and int8 storage are options, not scheduled | no, optional |
| 6 | **Downgrade/back-compat test** — **passed 2026-11-16** | older okfgraph (0.14.2 codec) does not decode 8; the space pin should catch it | no (space pin refused) |
| 7 | **Silk UI / preflight** | `image_mismatch` is server-side; the KG pin display only reads text precision; no image precision preflight | no, polish |
| 8 | **Release/rollout** | commits are local, no version bump/changelog/tag; Weave is far ahead of its remote | operational |
| 9 | **Known flaky test** | `test_only_one_agent_wins_a_contested_claim` hangs in the 8-thread race (pre-existing) | no, deselected |

---

## 4. Next steps, in order

### N1 — Retrieval gate (owner: embroider + okfgraph) — **done 2026-11-16**

Build or adopt a labelled retrieval set (the cached BeIR suites are a
candidate; a hand-reviewed in-domain set is better). Run fp32 vs int8 at
768 and 512 (optionally 256), same prompts, same corpus. Report
NDCG@10, recall@20, R@1/R@5, top-1 agreement and latency.

**Accept only if** the NDCG@10 and recall@20 deltas are ≤ 0.5 pt. If any
calibrated thresholds exist, re-tune and record them. If the set is
small, say so in the record.

Files: `scripts/eval_search.py`-style harness in okfgraph, or the temp
bench's `eval_matrix.py` lifted into a tracked script. No engine change.

**Result (2026-11-16):** pass on BEIR scifact (300 queries) and
nfcorpus (323 queries) at 768 and 512. NDCG@10 deltas are +0.16/−0.13
and +0.12/−0.03 pt; the NDCG@10 and recall@20 deltas are inside the
0.5 pt bar (the largest R@1 move is +0.62 pt).
Harness: `OKFgraph/scripts/eval_precision.py`; record:
`OKFgraph/docs/eg2-int8-gates.md`.

### N2 — Vision int8 end-to-end (owner: embroider + Silk)

1. Live encode one image through `EMBEDGEMMA2_VISION.int8` on CPU and
   CUDA; verify finite output, shape and the fusion path.
2. Run the cross-modal gate: text query → image retrieval vs fp16, same
   corpus; delta ≤ 0.5 pt.
3. Decide the mixed-pair rule. Default: text int8 + vision int8, or no
   image lane. If mixed text-int8 + vision-fp16 is allowed, gate that
   pair explicitly and ensure the image space names the text fusion.
4. If the default is enforced, add the refusal in `okfgraph` (pre-boot
   or first vision open) and a Silk row-level check.

Files: `src/vision.rs` / `src/gemma.rs`, `okfgraph/components/embedding.py`,
`okfgraph/router.py`, `weave/plugins/silk/functions/okf_wiring.py`,
`nodes/knowledge_graph.py`, tests.

### N3 — Downgrade / mixed-version check (owner: okfgraph) — **done 2026-11-16**

Open a new int8-pinned graph with the previous okfgraph release
(0.14.2). Confirm the space pin refuses a fp16/fp32 session. If it does
not, either require a space pin for int8 graphs or add a version/schema
floor; record the result in `docs/compat.md`. Add a regression test that
reads a hand-written code 8 with the old decoder path if feasible.

**Result (2026-11-16):** okfgraph 0.14.2 landed fp32 and the space pin
refused the int8 graph (`StateError`, pinned int8 / landed fp32); no
floor was needed. Record: `OKFgraph/docs/compat.md`. The current suite
already covers the fail-closed unknown-code path.

### N4 — Release (owner: all)

- Changelog entries: embroider (artifact/mirror), okfgraph (pin codec,
  image lane, reembed), Silk (rows).
- Version decision and tag; push the local commits.
- Weave: decide the `dev` push (it was already ahead of `origin/dev`
  before this work).
- Do **not** tag/publish without an explicit go.

### N5 — Audio slot (owner: embroider, then okf/Silk)

Build it like the vision slot, after a consumer exists:
`audio_encoder_quantized.onnx` is already mirrored. Port the front end
(PCM 16 kHz) or use the features-in fallback; add
`/v1/audio/embeddings`, the `[bos, boa, audio×K, eoa, eos]` framing, the
audio pin trio, and the cross-modal gate. The front end is the only
genuinely uncertain piece; the fallback keeps it shippable.

### N6 — Database size (owner: okfgraph, optional)

Productize the read-only `okf db-size` report (counts, vector bytes by
table, BLOB bytes + largest, text bytes, index/page overhead, same at a
candidate width). Use it on a copy before deciding 256d; 128d saves
nothing beyond the index/text floor on the measured corpus. Storage
quantization is a separate engine project, only after measurement.

### N7 — UI / preflight polish (owner: Silk)

- Surface `image_mismatch` in the KG pin display.
- Add an image-precision preflight (server refuses at first image encode
  today).
- Keep the warm-pool node UI and reranker `max_tokens` wording on the
  earlier S7/S8 list; unrelated to int8.

### N8 — Cleanup

- The known ledger race test (pre-existing) hangs when run; either fix
  the race or keep it explicitly deselected in the release gate.
- Pre-existing Weave layout-lint findings remain if that gate is in
  scope.

---

## 5. Gates and status

| Gate | Status | Evidence |
|---|---|---|
| int8 artifact registered, explicit-only | **done** | `95ff50c`, mirror `52614d7` |
| text parity | **done** | mean 1−cos 8.5e-4, p99 2.9e-2 |
| text retrieval (labelled) | **pass 2026-11-16** | scifact + nfcorpus, 768/512; NDCG delta ≤ 0.16 pt (`eg2-int8-gates.md`) |
| text pin cross-refusals | **done** | `4eaf1e2` + tests |
| image pin paths | **done** | router/settings/caller/decode/hash/pin_info |
| vision int8 e2e / cross-modal | **open** | no live image encode yet |
| mixed-pair rule | **open** | code allows; product default undecided |
| audio front end / e2e | **open** | W2 not started |
| DB width (256d) | **option** | 26% smaller file; fresh DB needed |
| downgrade from 0.14.2 | **pass 2026-11-16** | space pin refused the int8 graph (`compat.md`) |
| release/push | **open** | local commits, no tag |

---

## 6. Decisions locked by this work

- **int8 is explicit and measured**, never automatic; the policy line
  changes from “never” to “explicit, measured, pinned, retrieval-evaluated”.
- **int8 is the canonical name**; “q8” is the publisher/transformers.js
  label for the same dynamic-int8 `model_quantized.onnx`. Embroider’s
  `Q8` is a different weight-only recipe and is not used for EG2
  embeddings.
- **The model pin cannot distinguish tiers** (same model id); the
  precision pin is the only guard, so no fake id.
- **int8 does not shrink the database**; it shrinks the model download.
- **Image graphs are fresh-DB + re-encode**, never in-place reembed.
- **bf16 and q4/q4f16 stay rejected**; see `plan-eg2-int8-audio.md`.

---

## 7. File map

- **embroider**: `src/acquire.rs` (registry/mirror), `src/policy.rs`
  (auto/explicit), `src/gemma.rs` (fusor, media), `src/vision.rs`
  (vision), `python/embroider/serve.py` (host routes), `docs/compat.md`,
  `docs/quickref.md`, `docs/plan-eg2-int8-audio.md`,
  `docs/eg2-int8-audio-results.md`, tests.
- **okfgraph**: `components/embedding.py` (codec/pins/spaces),
  `router.py` (pin calls, image validation), `ops/admin.py` (reembed,
  `pin_info`), `settings.py` (choices), `components/image_assets.py`
  (content hash), `components/doctor.py`, `tests/test_precision.py`,
  `test_reembed*.py`, `test_vision.py`, `test_settings.py`,
  `docs/bench-eg2-precision-width.md`.
- **Silk/Weave**: `functions/okf_graph.py` (precision rows),
  `functions/onnx_roles.py`, `nodes/onnx_loader.py`,
  `nodes/knowledge_graph.py`, `functions/okf_wiring.py`, tests.
- **Bench (not versioned)**: `%TEMP%/okf-eg2-bench/` — corpus, npz
  vectors, 20 DBs, `results.json`, scripts.

---

## 8. Reproduce / verify

```bash
# okfgraph suite (Python 3.14 probe venv)
cd OKFgraph
./.okf-p0-venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider

# silk suite (developenv), with the known racy test excluded
cd Weave
PYTHONPATH="$PWD" /c/Users/Main/AppData/Local/Python/developenv/Scripts/python.exe \
  -m pytest weave/plugins/silk/tests -q -p no:cacheprovider \
  --ignore=weave/plugins/silk/tests/test_silk_task_ledger.py
PYTHONPATH="$PWD" /c/Users/Main/AppData/Local/Python/developenv/Scripts/python.exe \
  -m pytest weave/plugins/silk/tests/test_silk_task_ledger.py -q \
  -p no:cacheprovider -k "not contested"

# embroider registry
cd embroider
cargo test --lib registry_embedgemma2
```

Text int8 live check (temp DB, CPU):

```python
import embroider
m = embroider.open("embed", "google/embeddinggemma-2",
                   precision="int8", device="cpu", truncate_dim=768)
print(m.describe()["precision"], m.describe()["revision"])
print(m.encode("which planet is closest to the Sun", task="query").shape)
```

---

## 9. Open questions to settle before shipping

1. ~~Which labelled set is the retrieval gate, and what is the acceptable
   delta on it?~~ Closed: scifact + nfcorpus, NDCG@10/recall@20 ≤ 0.5 pt,
   both passed 2026-11-16.
2. Is text int8 + vision fp16 allowed, or is the product rule “both int8
   or no image lane”?
3. ~~Does the previous okfgraph release refuse a new int8 graph via the
   space pin, or does it need a version floor?~~ Closed: 0.14.2 refuses
   through the space pin; no floor needed.
4. Is `okf db-size` worth productizing now, or only when a real graph
   hits a size problem?
5. Is the audio slot waiting for a consumer, or is there a concrete
   use case that justifies W2 now?
