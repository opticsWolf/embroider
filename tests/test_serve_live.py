"""Live: a hosted model gives the in-process model's vectors bit for bit.

The 0.15 identity (`space`) promises that a graph built in-process can be
reopened through `python -m embroider.serve` without re-embedding. This
runs both in separate processes and compares at abs=0. Skipped unless an
ONNX Runtime resolves and the model is cached.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import pytest

embroider = pytest.importorskip("embroider")

MODEL = "jinaai/jina-embeddings-v5-text-nano-retrieval"
TEXTS = [f"sentence number {i} about embroidery and onnx runtimes" * (1 + i % 3)
         for i in range(20)]

IN_PROCESS = r"""
import json, sys, embroider
model = embroider.open("embed", sys.argv[1], device="cpu", truncate_dim=int(sys.argv[2]))
texts = json.loads(sys.stdin.read())
print(json.dumps({"space": model.space,
                  "query": model.encode_batch(texts, task="Query"),
                  "document": model.encode_batch(texts, task="Document")}))
"""


def _ready():
    if not os.environ.get("ORT_DYLIB_PATH") and embroider.locate_runtime()["dylib"] is None:
        return "no ONNX Runtime (set ORT_DYLIB_PATH or install onnxruntime)"
    if not embroider.cache_info(MODEL)["cached"]:
        return f"{MODEL} is not cached"
    return ""


pytestmark = pytest.mark.skipif(bool(_ready()), reason=_ready() or "ready")


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def test_the_host_and_the_library_share_one_vector_space():
    dim = 256
    direct = json.loads(subprocess.run(
        [sys.executable, "-c", IN_PROCESS, MODEL, str(dim)], input=json.dumps(TEXTS),
        capture_output=True, text=True, check=True).stdout)
    host = subprocess.Popen(
        [sys.executable, "-m", "embroider.serve", "--kind", "embed", "--model", MODEL,
         "--device", "cpu", "--dim", str(dim), "--port", "0"],
        stderr=subprocess.PIPE, text=True)
    try:
        line = host.stderr.readline()
        assert line.startswith("listening "), line
        url = line.split()[1]
        identity = json.loads(urllib.request.urlopen(url + "/v1/models").read())["data"][0]
        assert identity["space"] == direct["space"]
        assert identity["dim"] == dim and identity["revision"]
        for task in ("query", "document"):
            hosted = _post(url + "/v1/embeddings", {"input": TEXTS, "input_type": task})
            vectors = [row["embedding"] for row in hosted["data"]]
            assert vectors == direct[task], f"{task} vectors differ across processes"
        tokens = _post(url + "/tokenize", {"content": TEXTS[0]})
        assert tokens["count"] == len(tokens["tokens"]) and any(tokens["tokens"])
    finally:
        host.terminate()
        host.wait(timeout=10)
