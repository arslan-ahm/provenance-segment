"""Within-document retrievers used by the RAG baseline and the retrieval diagnostics."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .schemas import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been by with as at from that this these those it its "
    "which what who whom how why when where do does did their they them we our you your not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def bm25_rank(question: str, chunks: list[Chunk]) -> list[tuple[int, float]]:
    """Indices of ``chunks`` sorted by BM25 score (ties broken by document order)."""
    if not chunks:
        return []
    bm25 = BM25Okapi([tokenize(c.section_heading + " " + c.text) or ["_"] for c in chunks])
    scores = bm25.get_scores(tokenize(question))
    return sorted(((i, float(s)) for i, s in enumerate(scores)), key=lambda t: (-t[1], t[0]))


@lru_cache(maxsize=2)
def _encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


class DenseRanker:
    """Cosine-similarity ranking with a sentence-transformers bi-encoder (optional extra ``nli``)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str], keys: list[str]) -> np.ndarray:
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self._cache]
        if missing:
            vecs = _encoder(self.model_name).encode([t for _, t in missing], normalize_embeddings=True, batch_size=16)
            for (k, _), v in zip(missing, vecs):
                self._cache[k] = v
        return np.stack([self._cache[k] for k in keys])

    def rank(self, question: str, chunks: list[Chunk]) -> list[tuple[int, float]]:
        if not chunks:
            return []
        q = _encoder(self.model_name).encode([question], normalize_embeddings=True)[0]
        m = self._embed([c.section_heading + ". " + c.text for c in chunks], [c.chunk_id for c in chunks])
        scores = m @ q
        return sorted(((i, float(s)) for i, s in enumerate(scores)), key=lambda t: (-t[1], t[0]))


def fill_budget(order: list[int], costs: list[int], budget_tokens: int, max_items: int | None = None) -> list[int]:
    """Take items in ``order`` while their token ``costs`` fit the budget (skipping ones that do not).

    Costs must be measured on the exact text the model will see, so every system is held
    to the same budget.
    """
    picked, used = [], 0
    for i in order:
        if max_items is not None and len(picked) >= max_items:
            break
        t = costs[i]
        if used + t <= budget_tokens:
            picked.append(i)
            used += t
    return picked
