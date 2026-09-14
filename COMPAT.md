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
