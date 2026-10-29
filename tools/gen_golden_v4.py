"""Generate fixtures/golden_decide_gliclass_v4edge.json with the PINNED runtime.

Two-step (keeps the 1.29 provenance honest):
 1. s1mb-env (has tokenizers): build the labels-first input ids, save.
 2. OKFgraph venv (py-ort 1.29.0): run model.onnx, save logits.
Then this script assembles the fixture (softmax/sigmoid + winners).
"""
import json
import subprocess
import sys

SCRATCH = "D:/tmp/gli4-scratch"
OUT = "fixtures/golden_decide_gliclass_v4edge.json"

CASES = [
    {"key": "sport_single", "kind": "choice", "state": "The football match ended 2-1 after extra time.",
     "options": ["sports", "cooking", "quantum physics", "knitting"], "instructions": ""},
    {"key": "vote_single", "kind": "choice", "state": "The election results will be announced at midnight.",
     "options": ["sports", "politics", "cooking", "weather"], "instructions": ""},
    {"key": "quantum_score", "kind": "score", "state": "The quantum computer solved the optimization problem.",
     "options": ["sports", "cooking", "politics", "technology", "weather"], "instructions": ""},
    {"key": "knit_score", "kind": "score", "state": "Knitting patterns for winter scarves.",
     "options": ["sports", "cooking", "politics", "technology", "weather"], "instructions": ""},
    {"key": "refund_prompt", "kind": "choice",
     "state": "Our refund window is 30 days. The customer bought the item 45 days ago.",
     "options": ["refund approved", "refund denied", "escalate to a human"],
     "instructions": "Apply the refund policy to the customer's request."},
    {"key": "nano_relevance", "kind": "score",
     "state": "Hold the reset button for 10 seconds to restore factory settings.",
     "options": ["relevant"],
     "instructions": "Does the passage answer the query: How do I reset a forgotten router password?"},
]

BUILDER = """
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file("D:/tmp/gli4-scratch/tokenizer.json")
cases = json.load(open("D:/tmp/gli4-scratch/v4-cases.json"))
out = []
for c in cases:
    label_text = "".join(f"<<LABEL>>{o}" for o in c["options"]) + "<<SEP>>" + c["instructions"]
    lb = tok.encode(label_text, add_special_tokens=False).ids
    st = tok.encode(c["state"], add_special_tokens=False).ids
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    seq = [cls] + lb + st + [sep]
    out.append({"key": c["key"], "ids": seq, "state_tokens": len(st)})
json.dump(out, open("D:/tmp/gli4-scratch/v4-ids.json", "w"))
print("built", len(out))
"""

RUNNER = """
import json
import numpy as np
import onnxruntime as ort
print("ort", ort.__version__)
items = json.load(open("D:/tmp/gli4-scratch/v4-ids.json"))
s = ort.InferenceSession("D:/tmp/gli4-scratch/model.onnx", providers=["CPUExecutionProvider"])
print("rev", s.get_modelmeta().custom_metadata_map)
out = []
for it in items:
    n = len(it["ids"])
    feed = {"input_ids": np.array([it["ids"]], dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64)}
    o = s.run(None, feed)[0][0]
    out.append({"key": it["key"], "logits": o.tolist(), "state_tokens": it["state_tokens"]})
json.dump(out, open("D:/tmp/gli4-scratch/v4-logits.json", "w"))
print("ran", len(out))
"""


def softmax(xs):
    m = max(xs)
    es = [__import__("math").exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def sigmoid(xs):
    import math
    return [1.0 / (1.0 + math.exp(-x)) for x in xs]


if __name__ == "__main__":
    json.dump(CASES, open(f"{SCRATCH}/v4-cases.json", "w"))
    r = subprocess.run([sys.executable, "-c", BUILDER], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-200:] if r.returncode else "")
    assert r.returncode == 0, "builder failed"
    OKF = "D:/User/Documents/Python/OKFgraph/.venv/Scripts/python.exe"
    r = subprocess.run([OKF, "-c", RUNNER], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-200:] if r.returncode else "")
    assert r.returncode == 0, "runner failed"
    logits = {o["key"]: o for o in json.load(open(f"{SCRATCH}/v4-logits.json"))}
    cases = []
    counts = {}
    for c in CASES:
        lg = logits[c["key"]]["logits"][:len(c["options"])]
        probs = softmax(lg) if c["kind"] == "choice" else sigmoid(lg)
        best = max(range(len(probs)), key=lambda i: probs[i])
        cases.append({"key": c["key"], "kind": c["kind"], "state": c["state"],
                      "options": c["options"], "instructions": c["instructions"],
                      "label": c["options"][best], "probs": [round(p, 6) for p in probs]})
        counts[c["key"]] = logits[c["key"]]["state_tokens"]
    fixture = {
        "contract": "decide-gliclass",
        "model": "knowledgator/gliclass-v4-edge",
        "revision": "885af1af",
        "provenance": "onnxruntime 1.29.0 CPU; truncate-state-side-first max_length=2048 (labels-first: (<<LABEL>>option)* + <<SEP>> + instructions + state); torch parity max|D|=0.028 on 10 probes vs calibrated branch (gliclass 0.1.21 fp32-cast, torch 2.10, same revision); goldens pin ORT values at 0.03 tolerance",
        "max_length": 2048,
        "token_counts": counts,
        "cases": cases,
    }
    with open(OUT, "w") as fh:
        json.dump(fixture, fh, indent=1)
        fh.write("\n")
    print("wrote", OUT)
