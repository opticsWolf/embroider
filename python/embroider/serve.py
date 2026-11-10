"""One embroider model behind HTTP: ``python -m embroider.serve``.

A model a consumer wants out of its own process (a second ONNX Runtime
beside one already loaded takes an interpreter down) runs here. The
routes answer in the shapes the llama.cpp servers answer, so a client
cannot tell the runtimes apart, and ``GET /v1/models`` adds the model's
identity (:meth:`embroider.Model.describe`) so a graph can pin it.

======================== =========== =========================================
Route                    Kinds       Body -> answer
======================== =========== =========================================
``GET /health``          all         ``{"status": "ok"}``
``GET /v1/models``       all         ``{"object": "list", "data": [identity]}``
``POST /v1/embeddings``  embed       ``{input, task?, input_type?,
                                     dimensions?}`` -> the OpenAI shape
``POST /v1/images/       vision      ``{image | input: base64 PNG/JPEG,
embeddings``                         ...}`` -> the OpenAI shape
``POST /v1/rerank``      rerank      ``{query, documents, top_n?}`` ->
                                     ``{results: [{index, relevance_score}]}``
``POST /v1/systemone``   decide-*    llama.cpp's typed decisions
``POST /tokenize``       text kinds  ``{content}`` -> ``{tokens, count}``
======================== =========== =========================================

- **Embeddings**: *task* is embroider's (``Document``, the default, or
  ``Query``); ``input_type`` (``query`` / ``document``) is read as *task*
  too. The host applies the model's prompts, so a client never prefixes.
  ``dimensions`` must equal the opened width (no silent truncation).
- **Images**: the host decodes and resizes (:func:`embroider.prepare_image`).
- **Tokenize**: ``count`` is the model's own count; ``tokens`` are the
  ids (placeholders when the tokenizer can't be opened by id).

Exit codes: 2 for bad arguments, 3 when the model does not open (the
reason on stderr). Once listening it prints ``listening <url>`` on stderr.
"""
from __future__ import annotations

import os
import sys

if sys.path and os.path.abspath(sys.path[0] or ".") == os.path.dirname(
        os.path.abspath(__file__)):
    sys.path.pop(0)   # run as a script: its folder must not shadow stdlib

import argparse  # noqa: E402
import base64  # noqa: E402
import binascii  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

#: Largest request body accepted, in bytes.
MAX_BODY = 32 * 1024 * 1024

#: ``input_type`` and lower-case *task* spellings read as embroider's tasks.
TASKS = {"query": "Query", "search_query": "Query",
         "document": "Document", "passage": "Document",
         "search_document": "Document"}

TEXT_KINDS = ("embed", "rerank", "decide", "decide-laya", "decide-2b")


class Host:
    """One opened model; calls serialise on it (one ONNX session)."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.kind = model.kind
        self.model_id = model.model_id
        self._lock = threading.Lock()

    def describe(self) -> Dict[str, Any]:
        return self.model.describe()

    def tokenize(self, text: str) -> Dict[str, Any]:
        with self._lock:
            count = int(self.model.count_tokens(text))
        ids = self.model.tokenize(text)
        return {"tokens": ids if ids is not None else [0] * count, "count": count}

    def embed(self, texts: List[str], task: str,
              dimensions: Optional[int] = None) -> Dict[str, Any]:
        dim = int(self.describe().get("dim") or 0)
        if dimensions not in (None, 0, dim):
            raise ValueError(f"dimensions={dimensions}: this host serves "
                             f"{dim}-wide vectors (open it with --dim {dimensions})")
        with self._lock:
            vectors = self.model.encode_batch(texts, task=task)
            used = sum(int(self.model.count_tokens(t)) for t in texts)
        return _embedding_list(self.model_id, vectors, used)

    def embed_images(self, images: List[bytes]) -> Dict[str, Any]:
        from embroider import prepare_image
        prepared = [prepare_image(data, self.model_id) for data in images]
        with self._lock:
            vectors = [self.model.encode_image(rgb, h, w) for rgb, h, w in prepared]
        return _embedding_list(self.model_id, vectors, 0)

    def rerank(self, query: str, documents: List[str],
               top_n: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            scores = self.model.score_batch([(query, d) for d in documents])
            used = sum(int(self.model.count_tokens(d)) for d in documents)
        results = [{"index": i, "relevance_score": float(s)}
                   for i, s in enumerate(scores)]
        results.sort(key=lambda row: row["relevance_score"], reverse=True)
        if top_n is not None and top_n > 0:
            results = results[:top_n]
        return {"model": self.model_id, "object": "list", "results": results,
                "usage": {"prompt_tokens": used, "total_tokens": used}}

    def decide(self, state: Any, questions: Any) -> Dict[str, Any]:
        from embroider._systemone import decide
        with self._lock:
            answer = decide(self.model, self.kind, state, questions)
        answer["model"] = self.model_id
        return answer


def _embedding_list(model_id: str, vectors: Any, used: int) -> Dict[str, Any]:
    return {"object": "list", "model": model_id,
            "data": [{"object": "embedding", "index": i,
                      "embedding": [float(x) for x in vector]}
                     for i, vector in enumerate(vectors)],
            "usage": {"prompt_tokens": used, "total_tokens": used}}


def _texts(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("input must be a string or a list of strings")
    return list(value)


def _images(data: Dict[str, Any]) -> List[bytes]:
    value = data.get("image", data.get("input"))
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(v, str) for v in items):
        raise ValueError("image (or input) must be a base64 string or a list of them")
    out = []
    for item in items:
        if item.startswith("data:"):
            item = item.split(",", 1)[-1]
        try:
            out.append(base64.b64decode(item, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image is not base64: {exc}") from exc
    return out


def _documents(value: Any) -> List[str]:
    """Plain strings, or ``{text}`` rows as Cohere-style clients send."""
    if not isinstance(value, list):
        raise ValueError("documents must be a list")
    return [str(item.get("text", "") if isinstance(item, dict) else item)
            for item in value]


def _task(data: Dict[str, Any]) -> str:
    task = str(data.get("task") or "")
    if task:
        return TASKS.get(task.lower(), task)
    return TASKS.get(str(data.get("input_type") or "").lower(), "Document")


def make_handler(host: Host) -> type:
    from embroider._systemone import Unsupported

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:   # quiet: callers log
            pass

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("request body must be a JSON object")
            return data

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("/health", ""):
                self._send(200, {"status": "ok"})
            elif path in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [host.describe()]})
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            kind = host.kind
            try:
                data = self._body()
                if path in ("/v1/embeddings", "/embeddings") and kind == "embed":
                    dims = data.get("dimensions")
                    self._send(200, host.embed(_texts(data.get("input")), _task(data),
                                               int(dims) if dims else None))
                elif path in ("/v1/images/embeddings",) and kind == "vision":
                    self._send(200, host.embed_images(_images(data)))
                elif path in ("/v1/rerank", "/rerank") and kind == "rerank":
                    top_n = data.get("top_n")
                    self._send(200, host.rerank(
                        str(data.get("query") or ""),
                        _documents(data.get("documents", data.get("texts"))),
                        int(top_n) if top_n else None))
                elif path in ("/v1/systemone", "/systemone") and kind.startswith("decide"):
                    self._send(200, host.decide(data.get("state"), data.get("questions")))
                elif path == "/tokenize" and kind in TEXT_KINDS:
                    self._send(200, host.tokenize(str(data.get("content") or "")))
                else:
                    status = 501 if path in (
                        "/v1/embeddings", "/v1/images/embeddings", "/v1/rerank",
                        "/v1/systemone", "/tokenize") else 404
                    self._send(status, {"error": f"{self.path} is not served by "
                                                 f"a {kind} model"})
            except Unsupported as exc:
                self._send(501, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - reported to the client
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def build_parser() -> argparse.ArgumentParser:
    from embroider._models import KINDS
    parser = argparse.ArgumentParser(
        prog="python -m embroider.serve",
        description="Serve one embroider model over HTTP.")
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--model", required=True, help="registry id or hub repo")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--precision", default="auto",
                        choices=("auto", "fp32", "fp16", "int8", "q8"))
    parser.add_argument("--cache-dir", "--cache_dir", dest="cache_dir", default=None)
    parser.add_argument("--dim", type=int, default=0,
                        help="embed/vision width; 0 = the model's native width")
    parser.add_argument("--max-length", dest="max_length", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = any free port")
    return parser


def open_model(args: argparse.Namespace) -> Any:
    """Open the model the arguments name (``--dim 0`` = native width)."""
    import embroider
    options: Dict[str, Any] = {
        "revision": args.revision, "cache_dir": args.cache_dir or None,
        "device": args.device, "precision": args.precision,
    }
    if args.kind in ("embed", "vision"):
        native = int(embroider.registry_row(args.model).get("native_dim") or 0)
        options["truncate_dim"] = args.dim or native or None
    elif args.dim:
        raise ValueError(f"--dim applies to embed and vision models, not {args.kind}")
    if args.kind != "vision":
        options["max_length"] = args.max_length
    elif args.max_length:
        raise ValueError("--max-length does not apply to a vision model")
    return embroider.open(args.kind, args.model, **options)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)   # exits 2 on bad arguments
    try:
        model = open_model(args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - the reason goes to the caller
        print(f"embroider.serve: could not open {args.model}: {exc}", file=sys.stderr)
        return 3
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Host(model)))
    server.daemon_threads = True
    host, port = server.server_address[:2]
    print(f"listening http://{host}:{port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
