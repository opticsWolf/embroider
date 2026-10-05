# Fix: offline cache inspection (`cache_info`)

## Problem

Consumers have no way to ask embroider "is this model already cached, and
where?" without opening a session (which downloads on a miss).

okfgraph 0.10.0 works around this in `EmbeddingEngine.model_info()`
(`okfgraph/components/embedding.py`) with
`huggingface_hub.snapshot_download(..., local_files_only=True)`. That breaks
in two ways:

1. **Crash:** `huggingface_hub` is not a dependency of okfgraph or embroider
   (embroider acquires through the Rust `hf-hub` crate), so `okf model-info`
   dies with `ModuleNotFoundError`.
2. **Wrong answer even when installed:** it checks the repo named by
   `model_id`, but embroider fetches `artifact_for(model_id, precision)`. The
   default text id at FP16 resolves to the FP16 mirror repo, and multi-file
   repos also need the `.onnx_data` sidecar plus `tokenizer.json`. Only
   embroider knows that layout.

## Fix in embroider

Add a Python-level function in `src/lib.rs`, backed by a helper in
`src/acquire.rs`:

```python
embroider.cache_info(model_id, revision=None, cache_dir=None,
                     precision=None) -> dict
```

- Resolve the artifact exactly as `JinaV5.open` / `JinaV5Vision.open` do:
  `lookup_model` → `artifact_for(model_id, precision)`. Use the same
  precision resolution, but **without** probing the device: `precision=None`
  means fp32 and is documented as such, so the call stays side-effect free.
- Look up each required file (`artifact.file`, `artifact.sidecar()` if any,
  `tokenizer.json`) in the hf-hub cache **offline only**. Use the crate's
  cache API (`hf_client(cache_dir)` + cache/offline lookup). It must never
  hit the network; if the crate cannot do offline lookups, read the standard
  layout directly:
  `<cache>/models--{owner}--{name}/refs/{revision|main}` →
  `snapshots/<sha>/<file>`.
- Return:

  ```python
  {
    "model_id": str,           # as passed
    "repo": str,               # artifact.repo actually used
    "precision": str,          # resolved tier
    "cache_dir": str,          # effective hub cache dir
    "files": {name: path | None},  # one entry per required file
    "cached": bool,            # all required files present (missing sidecar is OK, matching open())
    "snapshot_path": str | None,
    "disk_usage_bytes": int,   # sum over present files (follow symlinks/blobs)
  }
  ```

- Unknown ids follow the legacy path (repo = id, `onnx/model.onnx`), same as
  `open()`. A bad id (`no-slash`) raises `ValueError` from
  `parse_owner_name`, before any I/O.

### Tests

- Rust: an artifact-resolution unit test (default id + fp16 → mirror repo; an
  official multi-file repo lists its sidecar). A temp-dir fake cache with
  `refs/main` + `snapshots/<sha>/...` covers hit, partial miss and full miss,
  and asserts no network (unset `HF_ENDPOINT` / point it at an unroutable
  host).
- Python: `cache_info` on an empty temp cache returns `cached=False`,
  `files` all `None`, and raises nothing.
- Add it to `docs/quickref.md` and note the version in `docs/compat.md`.

## Follow-up in okfgraph (after the embroider release)

- `EmbeddingEngine.model_info()` calls `embroider.cache_info(model_id,
  cache_dir=cache_dir)` and drops the `huggingface_hub` import; the output
  keys stay compatible (`model_id, cache_dir, cached, snapshot_path,
  disk_usage_bytes`) plus `repo` and `files`.
- On an older embroider without `cache_info`, refuse with a typed
  `OKFError` (`NO_ORT_RUNTIME`-style, remedy: upgrade embroider), not a
  crash.
- Raise the embroider version requirement in `pyproject.toml`.
- `tests/test_parity.py` also imports `huggingface_hub`. Keep that as an
  explicitly skipped optional dependency (`pytest.importorskip`).
