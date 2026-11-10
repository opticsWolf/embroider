"""embroider: ONNX embeddings, rerankers and decision models (Rust core).

The models live in the native module (:mod:`embroider._native`); this
package re-exports every name from it and adds the Python layer shared by
every consumer (okfgraph, bobine, Silk):

- :func:`locate_runtime` finds the ONNX Runtime library without importing
  the Python ``onnxruntime`` module; ``import embroider`` runs it once.
- :func:`open` opens any registry kind and returns a :class:`Model` that
  describes itself (:meth:`Model.describe`), including the vector-space
  identity :func:`space_id`.
- :func:`prepare_image` is the image resize contract of the vision slots.
- ``python -m embroider.serve`` serves one model over HTTP.
"""
from __future__ import annotations

import warnings as _warnings

from embroider._runtime import locate_runtime

# Before any session: ort reads ORT_DYLIB_PATH when the first one builds
# (importing the native module doesn't touch it), and an explicit setting
# always wins (locate_runtime never overrides it).
try:
    locate_runtime()
except Exception:  # noqa: BLE001 - never break the import over discovery
    pass

from embroider._native import (  # noqa: E402
    FP32_TEXT_MODEL,
    MAX_LENGTH,
    MODEL_MAX_TOKENS,
    NANO_TEXT_MODEL,
    NATIVE_DIM,
    VISION_NANO_MODEL,
    Decider,
    Decider2B,
    GemmaVision,
    JinaV5,
    JinaV5Vision,
    LayaDecider,
    Reranker,
    Tokenizer,
    available_models,
    cache_info,
    cache_info_files,
    configure_runtime,
    export_registry,
    gemma_vision_target_size,
    load_registry_file,
    prepare_cuda_runtime,
    register_model,
    register_task,
    runtime_info,
    save_runtime_info,
    unregister_model,
    unregister_task,
    vision_target_size,
)
from embroider._models import (  # noqa: E402
    CONTRACT,
    KINDS,
    OPEN_KWARGS,
    VECTOR_CONTRACT,
    Model,
    kind_of,
    open,
    registry_row,
    space_id,
)
from embroider._image import prepare_image  # noqa: E402

try:
    from importlib.metadata import version as _version
    __version__ = _version("embroider")
except Exception:  # noqa: BLE001 - a source checkout without metadata
    __version__ = "0.0.0"

__all__ = [
    # native: models
    "JinaV5", "JinaV5Vision", "GemmaVision", "Reranker", "Decider",
    "LayaDecider", "Decider2B", "Tokenizer",
    # native: registry, cache, runtime
    "available_models", "register_model", "register_task",
    "load_registry_file", "export_registry", "unregister_model",
    "unregister_task", "cache_info", "cache_info_files", "runtime_info",
    "configure_runtime", "save_runtime_info", "prepare_cuda_runtime",
    "vision_target_size", "gemma_vision_target_size",
    # native: constants
    "VISION_NANO_MODEL", "NATIVE_DIM", "MAX_LENGTH", "MODEL_MAX_TOKENS",
    "FP32_TEXT_MODEL", "NANO_TEXT_MODEL",
    # Python layer (0.15)
    "locate_runtime", "open", "Model", "KINDS", "OPEN_KWARGS", "CONTRACT",
    "VECTOR_CONTRACT", "kind_of", "registry_row", "space_id",
    "prepare_image", "__version__",
]


def __getattr__(name: str):
    """``JinaTokenizer``: the pre-0.14 name of :class:`Tokenizer`."""
    if name == "JinaTokenizer":
        _warnings.warn(
            "embroider.JinaTokenizer is deprecated; use embroider.Tokenizer "
            "(works for every tokenizer, not just Jina)",
            DeprecationWarning, stacklevel=2)
        return Tokenizer
    raise AttributeError(f"module 'embroider' has no attribute {name!r}")
