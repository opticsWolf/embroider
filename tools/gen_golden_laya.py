"""Generate fixtures/golden_laya_td.json via the UPSTREAM agent.

The true-behavior source is `laya.ONNXAgent` itself (s1mb-env, laya
0.4.17): `_encode_state` builds the marker rows, the agent's own
session (py-ort) runs them. The generator decodes in float64 with the
documented formulas (temperature lookup + clamp, stable softmax,
slot->caller unpermute, per-type shapes) — the same math `src/laya.rs`
ports — and stores full-precision numbers plus the agent-built input
ids. The Rust e2e then proves two things: (1) embroider rebuilds the
agent's ids exactly (construction pin), (2) embroider's session outputs
match at 1e-6.

Tolerance note: goldens pin the ORT values, not torch parity; the
agent rounds presentation to 4 decimals but the fixture keeps raw
float64 (toward Rust f32 at 1e-6).
"""
import json
import math
import subprocess
import sys

SCRATCH = "D:/tmp/laya-scratch/stage"
OUT = "fixtures/golden_laya_td.json"
ONNX = f"{SCRATCH}/laya-td-fp32.onnx"

CASES = [
    {"key": "billing_choice", "kind": "choice",
     "state": "My invoice shows a double charge for last month.",
     "instructions": "Route the ticket to the right team.",
     "options": [["billing", "invoices, charges and payments"],
                 ["technical", "login, errors and crashes"],
                 ["shipping", "delivery tracking and returns"]]},
    {"key": "sport_choice", "kind": "choice",
     "state": "The football match ended 2-1 after extra time.",
     "instructions": "Decide the topic.",
     "options": [["sports", None], ["cooking", None]]},
    {"key": "sentiment_score", "kind": "score",
     "state": "I love this product, it works perfectly.",
     "instructions": "Rate the sentiment from negative to positive.",
     "levels": ["negative", "neutral", "positive"]},
    {"key": "relevance_true", "kind": "noul",
     "state": "supporting text unused",
     "instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
     "statement": "Hold the reset button for 10 seconds to restore factory settings, then set a new password."},
    {"key": "relevance_false", "kind": "noul",
     "state": "supporting text unused",
     "instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
     "statement": "Preheat the oven to 200 degrees and fold the dough gently."},
    {"key": "ordered_choice", "kind": "choice",
     "state": "The election results will be announced at midnight.",
     "instructions": "Decide the topic.",
     "options": [["sports", None], ["politics", None], ["weather", None]],
     "option_order": [2, 0, 1]},
    {"key": "solo_choice", "kind": "choice",
     "state": "The football match ended 2-1 after extra time.",
     "instructions": "Decide the topic.",
     "options": [["sports", None]]},
]

GEN = """
import json, math
import numpy as np
from laya.onnx_agent import ONNXAgent
from laya.common import QTYPES, temp_bucket, clamp_temperature

agent = ONNXAgent.__new__(ONNXAgent)
import onnxruntime as ort
print("ort", ort.__version__)
from transformers import AutoTokenizer
agent.tok = AutoTokenizer.from_pretrained("D:/tmp/laya-scratch/stage")
agent.session = ort.InferenceSession("GRAPH_PATH", providers=["CPUExecutionProvider"])
cfg = json.load(open("D:/tmp/laya-scratch/stage/rl_agent_config.json"))
agent.cfg = cfg
agent.temperature_raw = cfg["temperature"]
agent.temperature_by_options_raw = cfg.get("temperature_by_options", {})
agent.temperature = [clamp_temperature(t) for t in agent.temperature_raw]
agent.temperature_by_options = {k: clamp_temperature(v) for k, v in agent.temperature_by_options_raw.items()}

def to_internal(c):
    t = c["kind"]
    if t == "choice":
        crit = {o[0]: o[1] for o in c["options"]}
    elif t == "score":
        crit = list(c["levels"])
    else:
        crit = {"false": None, "true": c["statement"]}
    q = {"t": t, "ins": c["instructions"], "crit": crit}
    if "option_order" in c:
        q["option_order"] = c["option_order"]
    return q

cases = json.load(open("D:/tmp/laya-scratch/stage/laya-cases.json"))
out = []
for c in cases:
    q = to_internal(c)
    items = agent._encode_state(c["state"], [c["key"]], {c["key"]: q},
                                cfg["max_len"], cfg["head_max_len"])
    it = items[0]
    n, L = 1, len(it["ids"])
    kcols = max(len(it["markers"]), 2)
    ids = np.full((1, L), agent.tok.pad_token_id, dtype=np.int64); ids[0, :] = it["ids"]
    att = np.zeros((1, L), dtype=np.int64); att[0, :L] = 1
    mpos = np.zeros((1, kcols), dtype=np.int64); mpos[0, :len(it["markers"])] = it["markers"]
    mmask = np.zeros((1, kcols), dtype=bool); mmask[0, :len(it["markers"])] = True
    qt = np.array([QTYPES[q["t"]]], dtype=np.int64)
    logits, act_logits = agent.session.run(["logits", "act_logits"],
        {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
         "marker_mask": mmask, "qtype": qt})
    k = len(it["markers"])
    t = agent.temperature_by_options.get(temp_bucket(QTYPES[q["t"]], k),
                                         agent.temperature[QTYPES[q["t"]]])
    z = (logits[0, :k] / t).astype(np.float64)
    z = z - z.max()
    p = np.exp(z); p = p / p.sum()
    order = c.get("option_order")
    if order and len(order) == len(p):
        canon = np.empty_like(p); canon[np.asarray(order)] = p; p = canon
    e = -(np.clip(p, 1e-12, 1.0) * np.log(np.clip(p, 1e-12, 1.0))).sum()
    if q['t'] == 'noul':
        conf = max(float(p[1]), 1.0 - float(p[1]))
    else:
        conf = 1.0 if k < 2 else min(1.0, max(0.0, 1.0 - float(e) / math.log(k)))
    aconf = min(1.0, max(0.0, float(p.max())))
    az = act_logits[0].astype(np.float64); az = az - az.max()
    ae = np.exp(az); act = float(ae[0] / ae.sum())
    if q["t"] == "choice":
        keys = list(q["crit"].keys())
        label = keys[int(p.argmax())]
        probs = [[kk, float(v)] for kk, v in zip(keys, p)]
        value = None
    elif q["t"] == "score":
        value = float((np.arange(k) * p).sum())
        label = c["levels"][int(p.argmax())]
        probs = [[lv, float(v)] for lv, v in zip(c["levels"], p)]
    else:
        label = "true" if p[1] >= p[0] else "false"
        probs = [["false", float(p[0])], ["true", float(p[1])]]
        value = None
    out.append({"key": c["key"], "ids": it["ids"], "markers": it["markers"],
                "qtype": int(qt[0]), "k": k, "label": label, "probs": probs,
                "value": value, "confidence": conf,
                "answer_confidence": aconf, "act_probability": act,
                "state_tokens": len(it["ids"])})
json.dump(out, open("D:/tmp/laya-scratch/stage/laya-ref.json", "w"))
print("generated", len(out))

# Long-document scan through the AGENT's predict_long (window/stride
# identical to the Rust e2e call below). Answers come back rounded to
# 4 decimals with window attribution -- the e2e pins them at 1e-4.
LONG_Q = {
    "long_billing": {"type": "choice",
                     "instructions": "Route the ticket to the right team.",
                     "criteria": {"billing": "invoices, charges and payments",
                                  "technical": "login, errors and crashes"}},
    "long_relevance": {"type": "noul",
                       "instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
                       "criteria": {"false": None,
                                    "true": "Hold the reset button for 10 seconds to restore factory settings."}},
}
LONG_STATE = json.load(open("D:/tmp/laya-scratch/stage/laya-long.json"))
scan = agent.predict_long(LONG_STATE, LONG_Q, window=128, stride=64)
long_out = {"window": 128, "stride": 64, "state": LONG_STATE, "answers": {}}
for qid, ans in scan["answers"].items():
    a = dict(ans)
    w = a.pop("window", None)
    probs = a.get("probabilities")
    if isinstance(probs, dict):
        probs = [[k, float(v)] for k, v in probs.items()]
    action = a.get("action") or {}
    long_out["answers"][qid] = {
        "type": a.get("type"), "label": a.get("choice", a.get("noul", a.get("score"))),
        "probs": probs, "value": a.get("score"),
        "confidence": a.get("confidence"), "answer_confidence": a.get("answer_confidence"),
        "act_probability": action.get("act_probability"), "window": w,
        "windows": scan["usage"].get("windows"),
    }
json.dump(long_out, open("D:/tmp/laya-scratch/stage/laya-long-ref.json", "w"))
print("scanned", len(long_out["answers"]), "windows:", scan["usage"].get("windows"))
""".replace("GRAPH_PATH", ONNX)

def long_cases(long_ref):
    """Reformat the agent scan answers into fixture entries. The agent
    reports noul as a bare P(true) number (no label, no prob pair), so
    the pair and the true/false label are reconstructed from it --
    formatting only, the numbers stay the agent's (rounded to 4)."""
    out = []
    for qid, a in long_ref["answers"].items():
        if a["type"] == "choice":
            kind, label = "choice", a["label"]
            probs = a["probs"]
            options = [[k, v] for k, v in LONG_Q_DEF[qid].get("criteria", {}).items()]
            statement = None
        else:
            kind = "noul"
            p_true = float(a["label"])
            probs = [["false", round(1.0 - p_true, 6)], ["true", round(p_true, 6)]]
            label = "true" if p_true >= 1.0 - p_true else "false"
            options, statement = None, LONG_Q_DEF[qid]["criteria"]["true"]
        out.append({"key": qid, "kind": kind,
                    "instructions": LONG_Q_DEF[qid]["instructions"],
                    "options": options, "statement": statement,
                    "label": label, "value": a["value"], "probs": probs,
                    "confidence": a["confidence"],
                    "answer_confidence": a["answer_confidence"],
                    "act_probability": a["act_probability"],
                    "window": a["window"], "windows": a["windows"]})
    return out


if __name__ == "__main__":
    json.dump(CASES, open(f"{SCRATCH}/laya-cases.json", "w"))
    filler = "Preheat the oven to 200 degrees and fold the dough gently until it rises. "
    signal_billing = "My invoice shows a double charge for last month and I want a refund. "
    signal_noul = "Hold the reset button for 10 seconds to restore factory settings. "
    long_state = filler * 14 + signal_billing + filler * 4 + signal_noul + filler * 10
    json.dump(long_state, open(f"{SCRATCH}/laya-long.json", "w"))
    LONG_Q_DEF = {
        "long_billing": {"instructions": "Route the ticket to the right team.",
                         "criteria": {"billing": "invoices, charges and payments",
                                      "technical": "login, errors and crashes"}},
        "long_relevance": {"instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
                           "criteria": {"false": None,
                                        "true": "Hold the reset button for 10 seconds to restore factory settings."}},
    }
    r = subprocess.run([sys.executable, "-c", GEN], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("generator failed")
    ref = {o["key"]: o for o in json.load(open(f"{SCRATCH}/laya-ref.json"))}
    long_ref = json.load(open(f"{SCRATCH}/laya-long-ref.json"))
    cases = []
    counts = {}
    for c in CASES:
        o = ref[c["key"]]
        entry = {"key": c["key"], "kind": c["kind"], "state": c["state"],
                 "instructions": c["instructions"], "label": o["label"],
                 "probs": [[k, round(v, 6)] for k, v in o["probs"]],
                 "confidence": round(o["confidence"], 6),
                 "answer_confidence": round(o["answer_confidence"], 6),
                 "act_probability": round(o["act_probability"], 6),
                 "agent_ids": o["ids"], "markers": o["markers"],
                 "state_tokens": o["state_tokens"]}
        if c["kind"] == "choice":
            entry["options"] = c["options"]
            if "option_order" in c:
                entry["option_order"] = c["option_order"]
        elif c["kind"] == "score":
            entry["levels"] = c["levels"]
            entry["value"] = round(o["value"], 6)
        else:
            entry["statement"] = c["statement"]
        cases.append(entry)
        counts[c["key"]] = o["state_tokens"]
    fixture = {
        "contract": "decide-laya",
        "model": "convaiinnovations/laya-typed-decisions",
        "revision": "1a793eb568e6718f15941d08f85432581df534e3",
        "artifact": "opticsWolf/laya-typed-decisions-fp32-onnx",
        "provenance": "laya 0.4.17 ONNXAgent construction + py-ort CPU session; Rust e2e pins ids exactly and probs at abs 1e-6; temperature math unit-pinned against the billing vector (softmax(L/1.7601) == [0.7463, 0.2086, 0.0451])",
        "max_length": 1024,
        "head_max_len": 256,
        "token_counts": counts,
        "cases": cases,
        "long_window": long_ref["window"],
        "long_stride": long_ref["stride"],
        "long_state": long_ref["state"],
        "long_cases": long_cases(long_ref),
    }
    with open(OUT, "w") as fh:
        json.dump(fixture, fh, indent=1)
        fh.write("\n")
    print("wrote", OUT)
