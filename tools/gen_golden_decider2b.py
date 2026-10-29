"""Generate fixtures/golden_decider2b.json via the reference layout runner.

The true-behavior source is `ollaya_convert.families.decider.layout`
(the port source, torch-paritied 3/3 at 1e-4 CPU in parity3.log) driving
plain onnxruntime: `DeciderLayout.encode` builds the slot rows, one
session run per ort_rows batch runs them, `option_logits` + checkpoint
temperatures + float64 softmax decode. The fixture keeps raw float64
plus the reference-built input ids. The Rust e2e then proves two
things: (1) embroider rebuilds the reference ids exactly (construction
pin), (2) embroider's answers match at 5e-4 (above the 1e-4 torch gap,
so the fixture stays torch-consistent too).

Stage default: D:/tmp/decider-scratch/stage (override DECIDER2B_STAGE).
CPU session (the CUDA shape-aliasing bug needs no flag on CPU).
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

STAGE = os.environ.get("DECIDER2B_STAGE", "D:/tmp/decider-scratch/stage")
OUT = "fixtures/golden_decider2b.json"

sys.path.insert(0, "D:/tmp/ollaya-repo/convert")
from ollaya_convert.families.decider.layout import DeciderLayout, softmax
from ollaya_convert.families.llm_common import ort_rows as R

CASES = [
    {"key": "billing_choice", "kind": "choice",
     "state": "My card was charged twice for the same purchase.",
     "instructions": "Which department should handle this?",
     "options": [["billing", "invoices, charges and payments"],
                 ["technical support", "login, errors and crashes"],
                 ["sales", "plans, upgrades and quotes"]]},
    {"key": "sport_choice", "kind": "choice",
     "state": "The football match ended 2-1 after extra time.",
     "instructions": "Decide the topic.",
     "options": [["sports", None], ["cooking", None]]},
    {"key": "sentiment_score", "kind": "score",
     "state": "I love this product, it works perfectly.",
     "instructions": "Rate the sentiment from negative to positive.",
     "levels": ["negative", "neutral", "positive"]},
    {"key": "relevance_true", "kind": "noul",
     "state": "Hold the reset button for 10 seconds to restore factory settings, then set a new password.",
     "instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
     "true_criterion": "The passage answers the query.",
     "false_criterion": "The passage does not answer the query."},
    {"key": "relevance_false", "kind": "noul",
     "state": "Preheat the oven to 200 degrees and fold the dough gently.",
     "instructions": "Does the passage answer the query: How do I reset a forgotten router password?",
     "true_criterion": "The passage answers the query.",
     "false_criterion": "The passage does not answer the query."},
    {"key": "topic_choice", "kind": "choice",
     "state": "The election results will be announced at midnight.",
     "instructions": "Decide the topic.",
     "options": [["sports", None], ["politics", None], ["weather", None]]},
    {"key": "wide_choice", "kind": "choice",
     "state": "The quarterly ledger needs reconciling before Friday.",
     "instructions": "Route to the right department.",
     "options": [[f"dept-{c}", f"handles matter {c}"] for c in
                 ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]]},
    {"key": "multi_mixed", "kind": "multi",
     "state": "My card was charged twice for the same purchase.",
     "questions": [
         {"kind": "choice", "instructions": "Which department should handle this?",
          "options": [["billing", None], ["sales", None]]},
         {"kind": "noul", "instructions": "Does this need a refund action?",
          "true_criterion": "yes", "false_criterion": "no"},
         {"kind": "score", "instructions": "How urgent is this?",
          "levels": ["low", "medium", "high"]}]},
]


def to_spec(c):
    t = c["kind"]
    if t == "choice":
        crit = {o[0]: o[1] for o in c["options"]}
    elif t == "score":
        crit = list(c["levels"])
    else:
        crit = {"true": c["true_criterion"], "false": c["false_criterion"]}
    return {"type": t, "instructions": c["instructions"], "criteria": crit}


def main():
    decision = json.load(open(f"{STAGE}/decision.json", encoding="utf-8"))
    calib = json.load(open(f"{STAGE}/calibration.json", encoding="utf-8"))["temperature"]
    lay = DeciderLayout(Tokenizer.from_file(f"{STAGE}/tokenizer.json"), decision)
    pad = decision["special_tokens"]["pad"]
    sess = ort.InferenceSession(f"{STAGE}/model.onnx", providers=["CPUExecutionProvider"])
    QT = {"choice": 0, "score": 1, "noul": 2}

    out = {"temps": calib, "cases": []}
    for c in CASES:
        if c["kind"] == "multi":
            specs = {f"q{j}": to_spec({**q, "kind": q["kind"]}) for j, q in enumerate(c["questions"])}
            orders = [None] * len(c["questions"])
        else:
            specs = {"q0": to_spec(c)}
            orders = [None]
        rows, plan = lay.encode(c["state"], specs)

        def extra(idx, L):
            return {"slot_pos": np.array([rows[i]["slot"] for i in idx], dtype=np.int64)}

        outs, _ = R.run_rows(sess, [r["ids"] for r in rows], extra, pad, budget=8192)
        k_of_row = {}
        for item in plan:
            # Per-ROW option count: iso rows are yes/no pairs (k=2)
            # under a k=levels plan item.
            rk = 2 if item["kind"] == "iso" else item["k"]
            for r in item["rows"]:
                k_of_row[r] = rk
        answers = []
        for pi, item in enumerate(plan):
            t_idx = QT[item["type"]]
            p = softmax([x / calib[t_idx] for x in lay.option_logits(item, outs)])
            order = orders[pi]
            if order and len(order) == len(p):
                canon = [0.0] * len(p)
                for s_i, o in enumerate(order):
                    canon[o] = p[s_i]
                p = canon
            if item["type"] == "choice":
                keys = list(specs[item["qid"]]["criteria"].keys())
                top = int(np.argmax(p))
                answers.append({"label": keys[top],
                                "probs": [[k, float(v)] for k, v in zip(keys, p)],
                                "value": None})
            elif item["type"] == "score":
                levels = list(specs[item["qid"]]["criteria"])
                top = int(np.argmax(p))
                answers.append({"label": levels[top],
                                "probs": [[lv, float(v)] for lv, v in zip(levels, p)],
                                "value": float(sum(i * v for i, v in enumerate(p)))})
            else:
                answers.append({"label": "true" if p[1] >= p[0] else "false",
                                "probs": [["false", float(p[0])], ["true", float(p[1])]],
                                "value": None})
        out["cases"].append({
            "key": c["key"], "kind": c["kind"],
            "state": c["state"],
            "spec": c,
            "rows": [{"ids": rows[i]["ids"], "slot": rows[i]["slot"],
                      "k": k_of_row[i]} for i in range(len(rows))],
            "answers": answers,
        })
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, f"({len(out['cases'])} cases)")


if __name__ == "__main__":
    main()
