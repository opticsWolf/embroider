# Compatibility matrix

One pinned stack in practice. A new `embroider` minor without an okfgraph
update is a **supported** state (the floor pin allows it); the vendored
golden vectors (`fixtures/golden_jina_v5_text_small.json`) are the contract
proof — okfgraph's suite asserts live vectors against them.

| okfgraph | bobine | embroider | onnxruntime (pip) | embroider wheels |
|---|---|---|---|---|
| 0.2.12 | 0.5.11 | 0.1.3 | 1.29.0 | py3.11–3.13 × linux / win / macOS-arm64 |

Rules:

- **Single pinned ORT**: one `onnxruntime` binary shared by every Rust
  consumer (`ORT_DYLIB_PATH`-overridable). Never float it per project.
- **Floor pin**: okfgraph requires `embroider>=0.1,<0.2` — 0.2.x stays on
  embroider 0.1.x so the Jina contract moves only with okfgraph releases.
- **Frozen vector space**: prefixes, last-token pooling, truncation order
  never change inside a minor. A contract change is a new minor plus a
  re-index-everything notice, and the golden fixture is regenerated.
- **Truncation limit is a parameter, not the contract** (0.1.4): same
  (text, `max_length`) always yields the same vector; the default stays
  8192 so existing graphs are unaffected. Raising the limit changes
  vectors for inputs longer than the old truncation only — reimport
  long docs after changing it, don't mix limits in one graph.
- **Wheel matrix**: embroider ships cp311/cp312/cp313 wheels per platform;
  okfgraph requires Python ≥ 3.11, matching exactly.

## Model licences

embroider's code is MIT OR Apache-2.0. The weights its registry
(`src/acquire.rs`) downloads are licensed separately:

| Registry model | Artifact repo | Licence |
|---|---|---|
| `jina-embeddings-v5-text-small-retrieval` fp32 | `jinaai/jina-embeddings-v5-text-small-retrieval` | CC BY-NC 4.0 |
| `jina-embeddings-v5-text-small-retrieval` fp16 | `opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16` | CC BY-NC 4.0 (ONNX conversion of the above) |
| `jina-embeddings-v5-text-nano-retrieval` fp32 / fp16 / int8 | `jinaai/jina-embeddings-v5-text-nano-retrieval` | CC BY-NC 4.0 |

- **Every built-in model is non-commercial.** Any consumer (okfgraph,
  bobine's callers) that runs them commercially needs a commercial licence
  from Jina AI.
- **Mirrors and conversions** we publish (e.g. the fp16 mirror) keep the
  upstream licence. Their model card must carry `license`, `base_model`,
  attribution to the upstream authors and a list of the changes made.
- **New registry entries** record the artifact's licence in this table in
  the same change. A model whose licence forbids redistributing adapted
  weights is referenced by upstream repo only, never mirrored.
- bobine's PDF models (layout, OCR, tables, formulas) are fetched by bobine,
  not by this registry. They are listed with their licences in okfgraph's
  README, "Model licences" section.
- Licences are as declared on each Hugging Face model card when checked
  (2026-09-27). The cards are authoritative; this is not legal advice.
