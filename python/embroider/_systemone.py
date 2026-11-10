"""llama.cpp's ``/v1/systemone`` decision wire, on embroider's deciders.

A request names typed questions about a *state*::

    {"state": "...", "questions": {
        "route":   {"type": "choice", "instructions": "...",
                    "criteria": {"billing": "payments ...", "shipping": null}},
        "urgency": {"type": "score", "instructions": "...",
                    "criteria": ["can wait", "this week", "today"]},
        "angry":   {"type": "noul", "instructions": "..."}}}

and the answer is ``{"answers": {qid: ...}, "usage": {...}}`` with
``choice`` / ``score`` / ``noul`` rows as llama.cpp answers them, so a
client of that server takes an embroider decider unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

TYPES = ("choice", "score", "noul")


class Unsupported(Exception):
    """The loaded model cannot answer this question (HTTP 501)."""


def _criteria_pairs(criteria: Any) -> List[Tuple[str, Any]]:
    if isinstance(criteria, dict):
        return [(str(k), None if v is None else str(v)) for k, v in criteria.items()]
    return [(str(c), None) for c in criteria]


def _noul_sides(spec: Dict[str, Any]) -> Tuple[str, str]:
    """``(true, false)`` criteria of a noul question."""
    criteria = spec.get("criteria")
    if isinstance(criteria, str) and criteria.strip():
        return criteria, ""
    if isinstance(criteria, dict):
        return (str(criteria.get("true") or spec["instructions"]),
                str(criteria.get("false") or ""))
    if isinstance(criteria, (list, tuple)) and criteria:
        return str(criteria[0]), str(criteria[1]) if len(criteria) > 1 else ""
    return str(spec["instructions"]), ""


def validate(questions: Any) -> None:
    """Raise ``ValueError`` for a request llama.cpp answers 400 to."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object of id -> spec")
    for qid, spec in questions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"question {qid!r} must be an object")
        qtype = spec.get("type")
        if qtype not in TYPES:
            raise ValueError(f"question {qid!r} has unknown type {qtype!r} "
                             f"(one of {', '.join(TYPES)})")
        if not str(spec.get("instructions") or "").strip():
            raise ValueError(f"question {qid!r} needs instructions")
        criteria = spec.get("criteria")
        if qtype == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise ValueError(f"choice question {qid!r} needs a non-empty criteria object")
        if qtype == "score" and (not isinstance(criteria, (list, tuple))
                                 or len(criteria) < 2):
            raise ValueError(f"score question {qid!r} needs at least two levels")


def native_question(kind: str, spec: Dict[str, Any]) -> tuple:
    """One question as *kind*'s native ``decide`` takes it."""
    qtype, instructions = spec["type"], str(spec["instructions"])
    criteria = spec.get("criteria")
    if kind == "decide":
        # GLiClass: plain labels, no descriptions, no noul head.
        if qtype == "noul":
            raise Unsupported("a decide (GLiClass) model cannot answer noul "
                              "questions; use a decide-laya or decide-2b model")
        return (qtype, instructions, [label for label, _ in _criteria_pairs(criteria)])
    if qtype == "choice":
        return (qtype, instructions, list(_criteria_pairs(criteria)))
    if qtype == "score":
        return (qtype, instructions, [str(c) for c in criteria])
    true_side, false_side = _noul_sides(spec)
    if kind == "decide-laya":
        return (qtype, instructions, true_side)
    return (qtype, instructions, true_side, false_side)


def _answer(qtype: str, levels: List[str], row: tuple) -> Dict[str, Any]:
    label, probabilities = row[0], [(str(k), float(p)) for k, p in row[1]]
    probs = dict(probabilities)
    value = row[2] if len(row) > 2 else None
    # Laya / 2B: the calibrated max(p); GLiClass: the plain max(p).
    confidence = float(row[4]) if len(row) > 4 else max(probs.values(), default=0.0)
    if qtype == "choice":
        return {"type": "choice", "choice": label, "probabilities": probs,
                "confidence": confidence}
    if qtype == "score":
        if value is None:
            # GLiClass scores each level on its own (sigmoids, not a
            # distribution): normalise before taking the expected index.
            total = sum(p for _, p in probabilities) or 1.0
            value = sum(i * p for i, (_, p) in enumerate(probabilities)) / total
        return {"type": "score", "score": float(value),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": probs, "confidence": confidence}
    return {"type": "noul", "noul": float(probs.get("true", 0.0))}


def decide(model: Any, kind: str, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
    """Answer a ``/v1/systemone`` request with *model* (a decider of *kind*)."""
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ValueError("state must be a non-empty string or object")
    validate(questions)
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    ids = list(questions)
    native = [native_question(kind, questions[qid]) for qid in ids]
    rows = model.decide(text, native)
    answers = {}
    for qid, row in zip(ids, rows):
        spec = questions[qid]
        levels = [str(c) for c in spec.get("criteria") or []] if spec["type"] == "score" else []
        answers[qid] = _answer(spec["type"], levels, row)
    used = int(model.count_tokens(text))
    return {"answers": answers, "usage": {"input_tokens": used, "output_tokens": 0}}
