"""Model-free checks the explorer can run on request: no LLM, no API key, no writes.

- ``budget_contexts``: the chunks the ``direct`` and ``rag_bm25`` systems would put into the
  final context for an instance, built with the same ranking, costs and budget as the systems.
- ``featured``: a small, deterministic set of questions to start from, chosen by a fixed rule
  from the committed retrieval diagnostic (never hand-picked).
- ``check_claim``: exact quote anchoring and the in-system NLI verifier for a claim against
  chosen source chunks, i.e. the verification step of Provenance-Segment, run live.
"""

from __future__ import annotations

import threading
from collections import defaultdict

from ..quotes import locate
from ..retrieval import bm25_rank, fill_budget
from ..schemas import Chunk
from ..systems import chunk_costs, make_handle
from ..verification import NLIVerifier

MAX_CLAIM_CHARS = 600
MAX_SOURCES = 4


class LiveError(ValueError):
    """Bad request input (maps to HTTP 400)."""


def budget_contexts(question: str, chunks: list[Chunk], budget: int, gold: set[str]) -> dict:
    costs = chunk_costs(chunks)
    ranking = bm25_rank(question, chunks)
    orders = {"direct": list(range(len(chunks))), "rag_bm25": [i for i, _ in ranking]}
    scores = {i: s for i, s in ranking}
    out = {}
    for name, order in orders.items():
        picked = sorted(fill_budget(order, costs, budget))
        ids = [chunks[i].chunk_id for i in picked]
        out[name] = {
            "chunk_ids": ids,
            "tokens": sum(costs[i] for i in picked),
            "distractor_tokens": sum(costs[i] for i in picked if chunks[i].is_distractor),
            "gold_in_context": len(gold & set(ids)),
            "gold_total": len(gold),
        }
    out["rag_bm25"]["scores"] = {chunks[i].chunk_id: round(scores[i], 3) for i in range(len(chunks))}
    out["budget"] = budget
    return out


def featured(retrieval_rows: list[dict], questions: dict, n_answerable: int = 4, n_unanswerable: int = 2, noise: int = 50) -> list[dict]:
    """Answerable questions whose gold evidence BM25 fits fully into the budget without noise
    but loses at ``noise`` distractors (the effect the project studies), then unanswerable ones.
    Ties are broken by qid so the selection is stable."""
    full = defaultdict(dict)
    for r in retrieval_rows:
        if r["method"] == "bm25":
            full[r["qid"]][r["noise_level"]] = r["all_gold_in_context"]
    lost = sorted(qid for qid, by in full.items() if by.get(0) == 1.0 and by.get(noise) == 0.0 and qid in questions)
    picks = [{"qid": q, "reason": f"BM25 keeps all gold evidence with no noise and loses it with {noise} distractors"} for q in lost[:n_answerable]]
    unans = sorted(qid for qid, q in questions.items() if q.unanswerable_all)
    picks += [{"qid": q, "reason": "Every annotator marked it unanswerable: a faithful system should abstain"} for q in unans[:n_unanswerable]]
    return picks


class ClaimChecker:
    """Wraps the run's NLI verifier; one check at a time so a small CPU host stays responsive."""

    def __init__(self, model: str, entail_threshold: float, contradict_threshold: float, window_sentences: tuple[int, ...]):
        self.verifier = NLIVerifier(model, entail_threshold, contradict_threshold, window_sentences)
        self._lock = threading.Semaphore(1)

    def check(self, claim: str, sources: list[Chunk], handles: dict[str, str], nli: bool = True) -> dict:
        claim = (claim or "").strip()
        if not claim:
            raise LiveError("claim is empty")
        if len(claim) > MAX_CLAIM_CHARS:
            raise LiveError(f"claim is longer than {MAX_CLAIM_CHARS} characters")
        if not sources:
            raise LiveError("choose at least one source passage")
        if len(sources) > MAX_SOURCES:
            raise LiveError(f"choose at most {MAX_SOURCES} source passages")
        per = []
        for c in sources:
            hit = locate(claim, c.text)
            per.append(
                {
                    "chunk_id": c.chunk_id,
                    "handle": handles.get(c.chunk_id),
                    "is_distractor": c.is_distractor,
                    "quote": {"found": hit is not None, "start": hit[0] if hit else None, "end": hit[1] if hit else None, "elided": bool(hit and hit[2])},
                }
            )
        result = {"claim": claim, "sources": per, "nli": None}
        if not nli:
            return result
        with self._lock:
            v = self.verifier.verify(claim, [c.text for c in sources], with_windows=True)
        for p, c, d in zip(per, sources, v.detail["per_source"]):
            w = d.get("window")
            span = locate(w, c.text) if w else None
            p["nli"] = {"entail": d["entail"], "contradict": d["contradict"], "window": w, "window_start": span[0] if span else None, "window_end": span[1] if span else None}
        result["nli"] = {
            "label": v.label,
            "entail": round(v.entail, 4),
            "contradict": round(v.contradict, 4),
            "joint": v.detail.get("joint"),
            "model": self.verifier.model_name,
            "entail_threshold": self.verifier.entail_threshold,
            "window_sentences": list(self.verifier.window_sentences),
        }
        return result


def handles_for(instance_ids: list[str]) -> dict[str, str]:
    return {cid: make_handle(i) for i, cid in enumerate(instance_ids)}
