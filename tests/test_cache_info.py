"""`embroider.cache_info`: offline cache inspection (no session, no download).

Covers the okfgraph-relevant contract: an empty cache reports
`cached=False` with `files` all `None` and raises nothing; a fake
`refs/main` + `snapshots/<sha>/...` layout is recognized offline, with a
missing sidecar tolerated (matching `open()`); validation errors are
`ValueError`s raised before any I/O.

Needs a built extension module (`maturin develop`, or a wheel install) —
the pure-Rust unit tests in `src/acquire.rs` cover the same logic without
one, so this file skips cleanly when the module isn't importable.
"""

import pytest

embroider = pytest.importorskip("embroider")

NANO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
SMALL = "jinaai/jina-embeddings-v5-text-small-retrieval"


def test_cache_info_empty_cache_raises_nothing(tmp_path):
    info = embroider.cache_info(NANO, cache_dir=str(tmp_path))
    assert info["model_id"] == NANO
    assert info["repo"] == NANO
    assert info["precision"] == "fp32"  # precision=None reads fp32, no device probe
    assert info["cache_dir"] == str(tmp_path)
    assert info["cached"] is False
    assert set(info["files"]) == {
        "onnx/model.onnx",
        "onnx/model.onnx_data",  # derived sidecar, probed like open() does
        "tokenizer.json",
    }
    assert all(p is None for p in info["files"].values())
    assert info["snapshot_path"] is None
    assert info["disk_usage_bytes"] == 0


def test_cache_info_pinned_precision_selects_the_mirror_repo(tmp_path):
    info = embroider.cache_info(SMALL, cache_dir=str(tmp_path), precision="fp16")
    assert info["repo"] == "opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16"
    assert info["precision"] == "fp16"
    assert info["cached"] is False


def test_cache_info_validation_before_io():
    with pytest.raises(ValueError, match="owner/name"):
        embroider.cache_info("no-slash")
    with pytest.raises(ValueError, match="precision='auto'"):
        embroider.cache_info(NANO, precision="auto")


def test_cache_info_fake_cache_hit(tmp_path):
    """The standard refs/ + snapshots/ layout is recognized offline."""
    sha = "0123456789abcdef0123456789abcdef01234567"
    repo_dir = tmp_path / "models--jinaai--jina-embeddings-v5-text-nano-retrieval"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(sha)
    snap = repo_dir / "snapshots" / sha
    (snap / "onnx").mkdir(parents=True)
    (snap / "onnx" / "model.onnx").write_bytes(b"x" * 128)
    (snap / "tokenizer.json").write_bytes(b"y" * 32)

    info = embroider.cache_info(NANO, cache_dir=str(tmp_path))
    assert info["cached"] is True
    assert info["files"]["onnx/model.onnx"] == str(snap / "onnx" / "model.onnx")
    assert info["files"]["tokenizer.json"] == str(snap / "tokenizer.json")
    assert info["files"]["onnx/model.onnx_data"] is None  # missing sidecar is OK
    assert info["snapshot_path"] == str(snap)
    assert info["disk_usage_bytes"] == 128 + 32
