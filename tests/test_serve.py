"""`python -m embroider.serve`: every route against a fake model, no runtime."""
import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

embroider = pytest.importorskip("embroider")
from embroider import serve  # noqa: E402
from embroider._systemone import native_question  # noqa: E402


class FakeModel:
    def __init__(self, kind, model_id="org/model", dim=4):
        self.kind, self.model_id, self.dim = kind, model_id, dim
        self.calls = []

    def describe(self):
        row = {"id": self.model_id, "object": "model", "kind": self.kind,
               "engine": "embroider"}
        if self.kind in ("embed", "vision"):
            row.update(dim=self.dim, space=f"embroider/{self.model_id}@x")
        return row

    def count_tokens(self, text):
        return len(text.split())

    def tokenize(self, text):
        return list(range(len(text.split())))

    def encode_batch(self, texts, task="Document"):
        self.calls.append(("encode_batch", texts, task))
        return [[float(i)] * self.dim for i, _ in enumerate(texts)]

    def encode_image(self, rgb, h, w):
        self.calls.append(("encode_image", len(rgb), h, w))
        return [1.0] * self.dim

    def score_batch(self, pairs):
        return [float(len(doc)) for _, doc in pairs]

    def decide(self, state, questions):
        self.calls.append(("decide", state, questions))
        out = []
        for q in questions:
            if q[0] == "choice":
                labels = [o[0] if isinstance(o, tuple) else o for o in q[2]]
                out.append((labels[1], [(labels[0], 0.2), (labels[1], 0.8)], None, 0.5, 0.8))
            elif q[0] == "score":
                out.append((q[2][2], [(lv, p) for lv, p in zip(q[2], (0.1, 0.2, 0.7))],
                            1.6, 0.4, 0.7))
            else:
                out.append(("true", [("false", 0.25), ("true", 0.75)], None, 0.5, 0.75))
        return out


@pytest.fixture
def serve_model():
    servers = []

    def start(model):
        server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(serve.Host(model)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_and_identity(serve_model):
    url = serve_model(FakeModel("embed"))
    assert call(url, "/health") == (200, {"status": "ok"})
    status, body = call(url, "/v1/models")
    assert status == 200 and body["data"][0]["space"] == "embroider/org/model@x"


def test_embeddings_carry_the_task_and_refuse_another_width(serve_model):
    model = FakeModel("embed")
    url = serve_model(model)
    status, body = call(url, "/v1/embeddings", {"input": ["a b", "c"], "input_type": "query"})
    assert status == 200 and len(body["data"]) == 2 and len(body["data"][0]["embedding"]) == 4
    assert body["usage"]["prompt_tokens"] == 3
    assert model.calls[-1] == ("encode_batch", ["a b", "c"], "Query")
    call(url, "/v1/embeddings", {"input": "x", "task": "document"})
    assert model.calls[-1][2] == "Document", "lower-case task spellings are read"
    assert call(url, "/v1/embeddings", {"input": "x", "dimensions": 4})[0] == 200
    status, body = call(url, "/v1/embeddings", {"input": "x", "dimensions": 2})
    assert status == 400 and "4-wide" in body["error"]
    assert call(url, "/v1/embeddings", {"input": [1]})[0] == 400


def test_tokenize_gives_ids_and_the_count(serve_model):
    url = serve_model(FakeModel("rerank"))
    assert call(url, "/tokenize", {"content": "one two three"}) == (
        200, {"tokens": [0, 1, 2], "count": 3})


def test_rerank_answers_best_first(serve_model):
    url = serve_model(FakeModel("rerank"))
    status, body = call(url, "/v1/rerank", {"query": "q", "documents": ["a", {"text": "ccc"}, "bb"],
                                            "top_n": 2})
    assert status == 200
    assert [r["index"] for r in body["results"]] == [1, 2]


def test_a_route_of_another_kind_is_not_implemented(serve_model):
    url = serve_model(FakeModel("rerank"))
    assert call(url, "/v1/embeddings", {"input": "x"})[0] == 501
    assert call(url, "/v1/nothing", {})[0] == 404


def test_images_are_decoded_and_resized_by_the_host(serve_model):
    pil = pytest.importorskip("PIL.Image")
    import io
    buffer = io.BytesIO()
    pil.new("RGB", (64, 48), (200, 10, 10)).save(buffer, format="PNG")
    model = FakeModel("vision", model_id=embroider.VISION_NANO_MODEL)
    url = serve_model(model)
    status, body = call(url, "/v1/images/embeddings",
                        {"image": base64.b64encode(buffer.getvalue()).decode()})
    assert status == 200 and len(body["data"]) == 1
    _, size, h, w = model.calls[-1]
    assert (h, w) == embroider.vision_target_size(48, 64) and size == h * w * 3
    assert call(url, "/v1/images/embeddings", {"image": "not base64!"})[0] == 400


QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "payments", "shipping": None}},
    "urgency": {"type": "score", "instructions": "How soon?",
                "criteria": ["can wait", "this week", "today"]},
    "angry": {"type": "noul", "instructions": "Is the writer angry?"},
}


def test_systemone_answers_in_llama_cpps_shape(serve_model):
    model = FakeModel("decide-laya")
    url = serve_model(model)
    status, body = call(url, "/v1/systemone", {"state": "charged twice", "questions": QUESTIONS})
    assert status == 200, body
    answers = body["answers"]
    assert answers["route"] == {"type": "choice", "choice": "shipping",
                                "probabilities": {"billing": 0.2, "shipping": 0.8},
                                "confidence": 0.8}
    assert answers["urgency"]["score"] == pytest.approx(1.6)
    assert answers["urgency"]["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert answers["angry"] == {"type": "noul", "noul": 0.75}
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 0}
    _, state, native = model.calls[-1]
    assert native[0] == ("choice", "Which team?", [("billing", "payments"), ("shipping", None)])
    assert native[2] == ("noul", "Is the writer angry?", "Is the writer angry?")


def test_systemone_refuses_bad_questions_and_unsupported_ones(serve_model):
    url = serve_model(FakeModel("decide"))
    assert call(url, "/v1/systemone", {"state": "x", "questions": {}})[0] == 400
    bad = {"q": {"type": "score", "instructions": "i", "criteria": ["one"]}}
    assert call(url, "/v1/systemone", {"state": "x", "questions": bad})[0] == 400
    status, body = call(url, "/v1/systemone", {"state": "x", "questions": QUESTIONS})
    assert status == 501 and "noul" in body["error"]


def test_a_gliclass_score_is_the_expected_level_of_its_normalised_scores():
    from embroider._systemone import _answer
    row = ("today", [("can wait", 0.5), ("this week", 1.0), ("today", 1.0)])
    answer = _answer("score", ["can wait", "this week", "today"], row)
    assert answer["score"] == pytest.approx((0 * 0.5 + 1 + 2) / 2.5)
    assert 0 <= answer["score"] <= 2


def test_each_decider_gets_its_own_question_tuple():
    noul = {"type": "noul", "instructions": "Angry?", "criteria": {"true": "angry", "false": "calm"}}
    assert native_question("decide-2b", noul) == ("noul", "Angry?", "angry", "calm")
    assert native_question("decide-laya", noul) == ("noul", "Angry?", "angry")
    choice = QUESTIONS["route"]
    assert native_question("decide", choice) == ("choice", "Which team?", ["billing", "shipping"])


def test_the_command_line(capsys):
    with pytest.raises(SystemExit) as stop:
        serve.main(["--kind", "chat", "--model", "x/y"])
    assert stop.value.code == 2
    with pytest.raises(SystemExit) as stop:
        serve.main(["--kind", "rerank", "--model", "x/y", "--dim", "256"])
    assert stop.value.code == 2 and "--dim applies" in capsys.readouterr().err
