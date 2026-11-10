"""The 0.15 Python layer: import surface, runtime locator, open(), identity.

Needs the built extension (`maturin develop`); no model or runtime: the
live checks are in `test_serve_live.py`.
"""
import sys
from pathlib import Path

import pytest

embroider = pytest.importorskip("embroider")

NATIVE_NAMES = [
    "JinaV5", "JinaV5Vision", "GemmaVision", "Reranker", "Decider",
    "LayaDecider", "Decider2B", "Tokenizer", "available_models",
    "register_model", "register_task", "load_registry_file", "export_registry",
    "unregister_model", "unregister_task", "cache_info", "cache_info_files",
    "runtime_info", "configure_runtime", "save_runtime_info",
    "prepare_cuda_runtime", "vision_target_size", "gemma_vision_target_size",
    "VISION_NANO_MODEL", "NATIVE_DIM", "MAX_LENGTH", "MODEL_MAX_TOKENS",
    "FP32_TEXT_MODEL", "NANO_TEXT_MODEL",
]


# -- E1: the package re-exports the native module --------------------------

def test_every_native_name_is_where_it_was():
    from embroider import _native
    for name in NATIVE_NAMES:
        assert getattr(embroider, name) is getattr(_native, name), name
        assert name in embroider.__all__, name


def test_the_package_adds_its_python_layer():
    for name in ("locate_runtime", "open", "Model", "space_id", "prepare_image",
                 "KINDS", "OPEN_KWARGS", "CONTRACT", "VECTOR_CONTRACT"):
        assert name in embroider.__all__ and hasattr(embroider, name), name
    assert embroider.__version__.startswith("0.15")


def test_the_old_tokenizer_name_still_resolves_with_a_warning():
    with pytest.warns(DeprecationWarning, match="use embroider.Tokenizer"):
        assert embroider.JinaTokenizer is embroider.Tokenizer
    with pytest.raises(AttributeError):
        embroider.NoSuchName  # noqa: B018


def test_importing_embroider_does_not_import_onnxruntime():
    assert "onnxruntime" not in sys.modules


# -- E2: one runtime locator -------------------------------------------------

def _capi(tmp_path, *names):
    capi = tmp_path / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    for name in names:
        (capi / name).write_bytes(b"")
    return capi.parent


@pytest.mark.parametrize("os_name, platform, files, want", [
    ("nt", "win32", ["onnxruntime.dll"], "onnxruntime.dll"),
    ("posix", "darwin", ["libonnxruntime.dylib"], "libonnxruntime.dylib"),
    ("posix", "linux", ["libonnxruntime.so", "libonnxruntime.so.1.31.1",
                        "libonnxruntime.so.1.29.0"], "libonnxruntime.so.1.31.1"),
    ("posix", "linux", ["libonnxruntime.so"], "libonnxruntime.so"),
    ("nt", "win32", [], None),
])
def test_the_library_is_found_in_the_package(tmp_path, os_name, platform, files, want):
    from embroider._runtime import library_in_package
    found = library_in_package(_capi(tmp_path, *files), os_name, platform)
    assert (found.name if found else None) == want


def test_an_explicit_path_always_wins(monkeypatch):
    monkeypatch.setenv("ORT_DYLIB_PATH", "/custom/onnxruntime.dll")
    assert embroider.locate_runtime() == {
        "dylib": "/custom/onnxruntime.dll", "source": "env", "version": None}


def test_a_pip_runtime_is_found_without_importing_it(tmp_path, monkeypatch):
    from embroider import _runtime
    package = _capi(tmp_path, "onnxruntime.dll", "libonnxruntime.so.1.31.1",
                    "libonnxruntime.dylib")
    # The GPU wheel installs the `onnxruntime` package; its dist-info
    # folder names the build (read by the Rust locator).
    (tmp_path / "onnxruntime_gpu-1.31.1.dist-info").mkdir()
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    monkeypatch.setattr(_runtime, "_package_dir", lambda: package)
    found = embroider.locate_runtime(set_env=False)
    assert found["source"] == "pip:onnxruntime-gpu" and found["version"] == "1.31.1"
    assert Path(found["dylib"]).parent == package / "capi"
    assert "ORT_DYLIB_PATH" not in __import__("os").environ, "set_env=False"
    embroider.locate_runtime()
    assert __import__("os").environ["ORT_DYLIB_PATH"] == found["dylib"]
    assert "onnxruntime" not in sys.modules


def test_no_runtime_reports_the_os_path(monkeypatch):
    from embroider import _runtime
    monkeypatch.delenv("ORT_DYLIB_PATH", raising=False)
    monkeypatch.setattr(_runtime, "_package_dir", lambda: None)
    assert embroider.locate_runtime() == {"dylib": None, "source": "os", "version": None}


def test_every_model_class_reports_its_landed_precision():
    for cls in (embroider.JinaV5, embroider.JinaV5Vision, embroider.GemmaVision,
                embroider.Reranker, embroider.Decider, embroider.LayaDecider,
                embroider.Decider2B):
        assert hasattr(cls, "precision"), cls.__name__


# -- E4: open(kind, ...) and the registry kinds ------------------------------

def test_the_registry_lists_every_kind_open_takes():
    kinds = {row["kind"] for row in embroider.available_models()}
    assert kinds == set(embroider.KINDS)
    assert embroider.kind_of(embroider.VISION_NANO_MODEL) == "vision"
    assert embroider.kind_of("google/embeddinggemma-2") == "embed"
    assert embroider.kind_of("google/embeddinggemma-2-vision") == "vision"
    assert embroider.kind_of("no/such-model") is None


def test_open_refuses_what_the_kind_does_not_take():
    with pytest.raises(TypeError, match="does not take truncate_dim"):
        embroider.open("rerank", "cross-encoder/ms-marco-MiniLM-L-6-v2", truncate_dim=256)
    with pytest.raises(ValueError, match="unknown kind"):
        embroider.open("chat", "x/y")
    with pytest.raises(ValueError, match="not in embroider's registry"):
        embroider.open(None, "no/such-model")


def test_open_dispatches_on_the_kind(monkeypatch):
    from embroider import _models, _native
    opened = []

    class Fake:
        def __init__(self, name):
            self.name = name

        def open(self, model_id, **options):
            opened.append((self.name, model_id, options))
            return self

    for name in ("JinaV5", "JinaV5Vision", "GemmaVision", "Reranker", "Decider",
                 "LayaDecider", "Decider2B"):
        monkeypatch.setattr(_native, name, Fake(name))
    embroider.open("embed", "google/embeddinggemma-2", truncate_dim=256,
                   precision="auto", cache_dir=None)
    embroider.open(None, "google/embeddinggemma-2-vision")
    embroider.open("vision", embroider.VISION_NANO_MODEL)
    embroider.open("decide-2b", "Mapika/decider-2b", device="cpu")
    model = embroider.open("rerank", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert opened == [
        ("JinaV5", "google/embeddinggemma-2", {"truncate_dim": 256}),
        ("GemmaVision", "google/embeddinggemma-2-vision", {}),
        ("JinaV5Vision", embroider.VISION_NANO_MODEL, {}),
        ("Decider2B", "Mapika/decider-2b", {"device": "cpu"}),
        ("Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2", {}),
    ], "auto precision and None options are dropped"
    assert isinstance(model, _models.Model) and model.name == "Reranker", \
        "attributes fall through to the native model"


# -- E3: describe() and space ------------------------------------------------

def test_the_space_format_is_pinned():
    assert embroider.space_id("google/embeddinggemma-2", "a" * 40, "fp32", 512, 8192,
                              "text_embed") == (
        "embroider/google/embeddinggemma-2@" + "a" * 40 + ":fp32:512:8192:text_embed:v1")
    assert ":?:" not in embroider.space_id("m/x", "", "fp16", 256, 512, "text_embed")
    assert "@?:" in embroider.space_id("m/x", "", "fp16", 256, 512, "text_embed")


class _FakeNative:
    dim = 512
    used_cuda = False
    max_length = 8192
    precision = "fp32"


def test_describe_reads_the_landed_settings_and_the_snapshot(monkeypatch):
    from embroider import _models
    monkeypatch.setattr(_models._native, "cache_info", lambda *a, **k: {
        "snapshot_path": "/hub/models--google--embeddinggemma-2/snapshots/" + "b" * 40})
    model = _models.Model("embed", "google/embeddinggemma-2", _FakeNative(),
                          {"truncate_dim": 512})
    identity = model.describe()
    assert identity["engine"] == "embroider" and identity["contract"] == embroider.CONTRACT
    assert (identity["device"], identity["precision"], identity["dim"]) == ("cpu", "fp32", 512)
    assert identity["native_dim"] == 768 and identity["ladder"][-1] == 768
    assert identity["revision"] == "b" * 40
    assert identity["family"] == "embeddinggemma-2"
    assert identity["space"] == embroider.space_id(
        "google/embeddinggemma-2", "b" * 40, "fp32", 512, 8192, "text_embed")
    assert model.space == identity["space"]


def test_a_task_model_has_no_space(monkeypatch):
    from embroider import _models

    class Native:
        used_cuda = True
        max_length = 512
        precision = "q8"

    model = _models.Model("rerank", "cross-encoder/ms-marco-MiniLM-L-6-v2", Native(), {})
    identity = model.describe()
    assert identity["device"] == "cuda" and identity["precision"] == "q8"
    assert "space" not in identity and "dim" not in identity


def test_a_pinned_sha_is_the_revision():
    from embroider._models import resolved_revision
    assert resolved_revision("x/y", "c" * 40, None, "fp32") == "c" * 40
