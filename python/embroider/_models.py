"""One way to open any embroider model, and how a model describes itself.

:func:`open` dispatches on the registry kind and returns a :class:`Model`:
the native model plus :meth:`Model.describe`, the identity a host
publishes on ``GET /v1/models`` and a graph pins. Every other attribute
(``encode_batch``, ``score_batch``, ``decide``, ``encode_image``,
``count_tokens``, ...) is the native model's own.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from embroider import _native

#: The kinds :func:`open` takes, as ``available_models()`` lists them.
KINDS = ("embed", "vision", "rerank", "decide", "decide-laya", "decide-2b")

#: The ``open`` arguments each kind takes (besides ``model_id``).
OPEN_KWARGS: Dict[str, Tuple[str, ...]] = {
    "embed": ("revision", "cache_dir", "truncate_dim", "device",
              "max_length", "precision", "cpu_arena"),
    "vision": ("revision", "cache_dir", "truncate_dim", "device",
               "precision", "gpu_mem_limit"),
    "rerank": ("revision", "cache_dir", "device", "max_length",
               "precision", "cpu_arena"),
    "decide": ("revision", "cache_dir", "device", "max_length",
               "precision", "cpu_arena"),
    "decide-laya": ("revision", "cache_dir", "device", "max_length",
                    "precision", "cpu_arena"),
    "decide-2b": ("revision", "cache_dir", "device", "max_length",
                  "precision", "cpu_arena"),
}

#: Version of the host wire and the ``describe()`` schema.
CONTRACT = "0.15"

#: Version of the vector contract (prompts, pooling, truncation order).
#: It changes only when the same model and settings would give different
#: vectors, so a release that leaves vectors alone keeps every graph's
#: ``space`` valid.
VECTOR_CONTRACT = "v1"

#: Session tuning per kind, part of ``space`` (see ``SessionPolicy``).
TUNING = {"embed": "text_embed", "vision": "vision"}

_SHA = re.compile(r"^[0-9a-f]{40}$")


def registry_row(model_id: str) -> Dict[str, str]:
    """``available_models()``'s row for *model_id*, or ``{}``."""
    for row in _native.available_models():
        if row.get("id") == model_id:
            return dict(row)
    return {}


def kind_of(model_id: str) -> Optional[str]:
    """The registry kind of *model_id*, or ``None`` when it is unlisted."""
    return registry_row(model_id).get("kind") or None


def _native_class(kind: str, model_id: str) -> Any:
    if kind == "embed":
        return _native.JinaV5
    if kind == "vision":
        # The gemma-2 vision slot rides GemmaVision; the jina omni slot
        # JinaV5Vision (okfgraph's rule, now here).
        return (_native.GemmaVision if model_id.startswith("google/")
                else _native.JinaV5Vision)
    return {"rerank": _native.Reranker, "decide": _native.Decider,
            "decide-laya": _native.LayaDecider,
            "decide-2b": _native.Decider2B}[kind]


def space_id(model_id: str, revision: str, precision: str, dim: int,
             max_length: int, tuning: str,
             vector_contract: str = VECTOR_CONTRACT) -> str:
    """The vector-space identity: equal strings mean identical vectors.

    ``embroider/<model>@<revision>:<precision>:<dim>:<max_length>:<tuning>:<contract>``.
    *revision* is the resolved snapshot sha (``?`` when unknown, which
    never equals a known one). The device is not part of it: a CPU and a
    CUDA session at the same precision serve one space, as graph pins
    always treated them.
    """
    return (f"embroider/{model_id}@{revision or '?'}:{precision}:{int(dim)}:"
            f"{int(max_length)}:{tuning}:{vector_contract}")


def resolved_revision(model_id: str, revision: Optional[str],
                      cache_dir: Optional[str], precision: str) -> str:
    """The snapshot sha the model was opened from, or ``""``."""
    if revision and _SHA.match(revision):
        return revision
    try:
        report = _native.cache_info(model_id, revision=revision,
                                    cache_dir=cache_dir,
                                    precision=precision or None)
    except Exception:  # noqa: BLE001 - task ids and air-gapped files
        return ""
    snapshot = report.get("snapshot_path") or ""
    return Path(snapshot).name if snapshot else ""


def _ints(value: str) -> List[int]:
    return [int(x) for x in str(value or "").split(",") if x.strip()]


class Model:
    """An opened embroider model of one kind.

    Attribute access falls through to the native model, so a ``Model`` is
    used exactly like the class :func:`open` picked. :meth:`describe`
    and :meth:`tokenize` are what it adds.
    """

    def __init__(self, kind: str, model_id: str, native: Any,
                 options: Dict[str, Any]) -> None:
        self.kind = kind
        self.model_id = model_id
        self.native = native
        self.options = dict(options)
        self._identity: Optional[Dict[str, Any]] = None
        self._tokenizer: Any = None
        self._tokenizer_lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name == "native":   # not set yet: during unpickling or __init__
            raise AttributeError(name)
        return getattr(self.native, name)

    def __repr__(self) -> str:
        return f"Model({self.kind!r}, {self.model_id!r})"

    def describe(self) -> Dict[str, Any]:
        """The model's identity, as a host publishes it and a graph pins it.

        Every kind: ``id``, ``object``, ``kind``, ``engine``,
        ``engine_version``, ``contract``, ``revision``, ``device``,
        ``precision``, ``max_length``, ``family``. Embed and vision kinds
        add ``dim``, ``native_dim``, ``ladder``, ``tuning`` and ``space``;
        vision adds ``text_partner``.
        """
        if self._identity is None:
            self._identity = self._describe()
        return dict(self._identity)

    def _describe(self) -> Dict[str, Any]:
        from embroider import __version__
        native = self.native
        row = registry_row(self.model_id)
        # The weights the session loaded: auto resolved on the landed
        # device, an unshipped tier read as the fp32 it fell back to.
        precision = str(getattr(native, "precision", "") or "")
        identity: Dict[str, Any] = {
            "id": self.model_id, "object": "model", "kind": self.kind,
            "engine": "embroider", "engine_version": __version__,
            "contract": CONTRACT,
            "revision": resolved_revision(
                self.model_id, self.options.get("revision"),
                self.options.get("cache_dir"), precision),
            "device": "cuda" if getattr(native, "used_cuda", False) else "cpu",
            "precision": precision,
            "max_length": int(getattr(native, "max_length", 0) or
                              row.get("max_tokens") or 0),
            "family": self.model_id.rsplit("/", 1)[-1],
        }
        if self.kind in ("embed", "vision"):
            dim = int(getattr(native, "dim", 0) or 0)
            ladder = _ints(row.get("ladder", ""))
            identity.update({
                "dim": dim,
                "native_dim": int(row.get("native_dim") or 0) or dim,
                "ladder": ladder,
                "tuning": TUNING[self.kind],
            })
            if self.kind == "vision":
                identity["text_partner"] = row.get("text_partner", "")
                identity["max_length"] = 0
            identity["space"] = space_id(
                self.model_id, identity["revision"], precision, dim,
                identity["max_length"], TUNING[self.kind])
        return identity

    @property
    def space(self) -> Optional[str]:
        """The vector-space identity (embed and vision kinds), else ``None``."""
        return self.describe().get("space")

    def tokenize(self, text: str) -> Optional[List[int]]:
        """Token ids for *text*, or ``None`` when the tokenizer won't open.

        Opens a :class:`Tokenizer` for the model once (the model's own
        handle counts but hands out no ids).
        """
        with self._tokenizer_lock:
            if self._tokenizer is None:
                try:
                    self._tokenizer = _native.Tokenizer.open(
                        self.model_id,
                        revision=self.options.get("revision"),
                        cache_dir=self.options.get("cache_dir"))
                except Exception:  # noqa: BLE001 - short ids, files
                    self._tokenizer = False
            tokenizer = self._tokenizer
        if tokenizer is False:
            return None
        return list(tokenizer.encode(text))


def open(kind: Optional[str], model_id: str, **options: Any) -> Model:  # noqa: A001
    """Open *model_id* as a model of *kind* and return a :class:`Model`.

    *kind* is one of :data:`KINDS`, or ``None`` to take it from the
    registry. *options* are the kind's ``open`` arguments
    (:data:`OPEN_KWARGS`); one that the kind doesn't take is a
    ``TypeError`` naming the ones it does, and ``None`` values are
    dropped, so callers can pass a full settings row.
    """
    if kind is None:
        kind = kind_of(model_id)
        if kind is None:
            raise ValueError(
                f"{model_id!r} is not in embroider's registry; pass its kind "
                f"(one of {', '.join(KINDS)})")
    if kind not in OPEN_KWARGS:
        raise ValueError(f"unknown kind {kind!r} (one of {', '.join(KINDS)})")
    options = {k: v for k, v in options.items() if v is not None}
    allowed = OPEN_KWARGS[kind]
    extra = sorted(set(options) - set(allowed))
    if extra:
        raise TypeError(
            f"a {kind} model does not take {', '.join(extra)} "
            f"(it takes {', '.join(allowed)})")
    if options.get("precision") == "auto":
        options.pop("precision")
    native = _native_class(kind, model_id).open(model_id, **options)
    return Model(kind, model_id, native, options)
