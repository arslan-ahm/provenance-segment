"""Claim-versus-source verifiers.

Both verifiers judge a claim against the ORIGINAL text of the cited chunks, never against
notes or summaries, so compression errors cannot launder themselves into "support".

* ``NLIVerifier``: a local NLI cross-encoder. A chunk is longer than the model's input,
  so the premise is scored as sliding windows of sentences and the best window wins.
  When a claim cites several chunks, the best window of each is also concatenated, so
  claims that combine facts from two places can still be entailed.
* ``LLMVerifier``: a second model with a strict three-way rubric.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np

from .chunking import split_sentences
from .llm import LLMClient, UsageMeter, call_structured
from .prompts import VERIFY_SYSTEM, VerifyOut, verify_user

Label = Literal["entailed", "contradicted", "not_enough_evidence"]


@dataclass
class Verdict:
    label: Label
    entail: float
    contradict: float
    detail: dict


class Verifier(Protocol):
    name: str

    def verify(self, claim: str, sources: list[str], meter: UsageMeter | None = None) -> Verdict: ...


def strip_headings(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.startswith("§ ")).strip()


def windows(text: str, size: int | tuple[int, ...] = 3) -> list[str]:
    """Sliding windows of ``size`` sentences; several sizes give the union (deduplicated, in order)."""
    sizes = (size,) if isinstance(size, int) else size
    sents = split_sentences(strip_headings(text).replace("\n\n", " "))
    out: list[str] = []
    for k in sizes:
        if len(sents) <= k:
            out.extend([" ".join(sents)] if sents else [])
        else:
            out.extend(" ".join(sents[i : i + k]) for i in range(len(sents) - k + 1))
    return list(dict.fromkeys(out))


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


class NLIVerifier:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.5,
        window_sentences: int | tuple[int, ...] = 3,
    ):
        self.model_name = model_name
        self.name = f"nli:{model_name}"
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self.window_sentences = window_sentences

    def _label_index(self) -> dict[str, int]:
        cfg = _cross_encoder(self.model_name).model.config
        return {v.lower(): int(k) for k, v in cfg.id2label.items()}

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Probabilities with columns (entailment, contradiction) for (premise, hypothesis) pairs."""
        if not pairs:
            return np.zeros((0, 2))
        logits = np.asarray(_cross_encoder(self.model_name).predict(pairs, batch_size=16, show_progress_bar=False))
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        li = self._label_index()
        return np.stack([probs[:, li["entailment"]], probs[:, li["contradiction"]]], axis=1)

    def verify(self, claim: str, sources: list[str], meter: UsageMeter | None = None, with_windows: bool = False) -> Verdict:
        """``with_windows`` adds the best-scoring window text per source to the detail (used by the demo app)."""
        per_source = [windows(s, self.window_sentences) for s in sources]
        flat = [(w, claim) for ws in per_source for w in ws]
        scores = self.score_pairs(flat)
        best_windows: list[str] = []
        detail: dict = {"per_source": []}
        offset = 0
        best_e = best_c = 0.0
        for ws in per_source:
            sc = scores[offset : offset + len(ws)]
            offset += len(ws)
            if len(ws) == 0:
                detail["per_source"].append({"entail": 0.0, "contradict": 0.0})
                continue
            i = int(sc[:, 0].argmax())
            best_windows.append(ws[i])
            e, c = float(sc[:, 0].max()), float(sc[:, 1].max())
            detail["per_source"].append({"entail": round(e, 4), "contradict": round(c, 4)})
            if with_windows:
                detail["per_source"][-1]["window"] = ws[i]
            best_e, best_c = max(best_e, e), max(best_c, c)
        if len(best_windows) > 1:
            joint = self.score_pairs([(" ".join(best_windows), claim)])[0]
            detail["joint"] = {"entail": round(float(joint[0]), 4), "contradict": round(float(joint[1]), 4)}
            best_e = max(best_e, float(joint[0]))
        if best_e >= self.entail_threshold:
            label: Label = "entailed"
        elif best_c >= self.contradict_threshold:
            label = "contradicted"
        else:
            label = "not_enough_evidence"
        return Verdict(label=label, entail=best_e, contradict=best_c, detail=detail)


class LLMVerifier:
    def __init__(self, client: LLMClient):
        self.client = client
        self.name = f"llm:{client.spec.model}"

    def verify(self, claim: str, sources: list[str], meter: UsageMeter | None = None) -> Verdict:
        passages = "\n\n".join(f"Passage {i + 1}:\n{strip_headings(s)}" for i, s in enumerate(sources))
        out = call_structured(self.client, meter, "verify", VERIFY_SYSTEM, verify_user(claim, passages), VerifyOut, max_tokens=1024)
        return Verdict(
            label=out.label,
            entail=1.0 if out.label == "entailed" else 0.0,
            contradict=1.0 if out.label == "contradicted" else 0.0,
            detail={"rationale": out.rationale},
        )
