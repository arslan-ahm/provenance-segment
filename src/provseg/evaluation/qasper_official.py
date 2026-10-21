"""Answer and evidence scoring, following the official QASPER evaluator
(github.com/allenai/qasper-led-baseline, ``qasper_evaluator.py``): SQuAD-style
normalisation, token F1, and the maximum over annotators."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(ground_truth).split()
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def set_f1(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    """(precision, recall, F1); two empty sets score 1.0 like the official paragraph F1."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    same = len(predicted & gold)
    if same == 0:
        return 0.0, 0.0, 0.0
    p = same / len(predicted)
    r = same / len(gold)
    return p, r, 2 * p * r / (p + r)
