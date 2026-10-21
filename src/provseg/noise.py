"""Matched distractor injection.

For a document and a maximum noise level K_max, we draw K_max distractor chunks from a
pool of *other* papers (by default QASPER train, disjoint from the evaluation split) and
give each a fixed insertion slot. The variant with K distractors uses the first K of
them at the same slots, so noise levels are nested: every distractor present at K=10 is
also present, at the same position, at K=25 and K=50. Seeds are derived from the
document ID so the same document always receives the same noise.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from rank_bm25 import BM25Okapi

from .retrieval import tokenize
from .schemas import Chunk, Document


def _rng(seed: int, doc_id: str) -> random.Random:
    h = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()[:16], 16)
    return random.Random(h)


class DistractorPool:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = [c for c in chunks if c.section_heading not in ("Abstract",)]
        self._bm25: BM25Okapi | None = None

    def _index(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        return self._bm25

    def draw(
        self,
        doc: Document,
        k_max: int,
        seed: int,
        mode: Literal["random", "hard"] = "random",
    ) -> list[Chunk]:
        rng = _rng(seed, doc.doc_id)
        candidates = [c for c in self.chunks if c.doc_id != doc.doc_id]
        if mode == "random":
            picked = rng.sample(candidates, k_max)
        else:
            # "Hard" distractors: topically similar passages from other papers,
            # ranked by BM25 against the target paper's title and abstract.
            query = tokenize(doc.title + " " + " ".join(doc.sections[0].paragraphs))
            scores = self._index().get_scores(query)
            ranked = sorted(
                (i for i, c in enumerate(self.chunks) if c.doc_id != doc.doc_id),
                key=lambda i: -scores[i],
            )
            picked = [self.chunks[i] for i in ranked[: k_max * 3]]
            rng.shuffle(picked)
            picked = picked[:k_max]
        out = []
        for i, c in enumerate(picked):
            out.append(
                c.model_copy(
                    update={
                        # The distractor keeps its true origin in its ID; models never see IDs
                        # directly (they cite neutral handles), so this cannot leak the label.
                        "chunk_id": f"noise{i:02d}__{c.chunk_id}",
                        "is_distractor": True,
                    }
                )
            )
        return out


def inject(
    doc_chunks: list[Chunk],
    distractors: list[Chunk],
    k: int,
    seed: int,
    doc_id: str,
) -> list[Chunk]:
    """Insert the first ``k`` distractors at slots fixed per (seed, doc)."""
    if k == 0:
        return list(doc_chunks)
    rng = _rng(seed + 1, doc_id)
    # One slot per distractor in the K_max ordering, drawn once so levels stay nested.
    slots = [rng.randint(0, len(doc_chunks)) for _ in range(len(distractors))]
    merged: list[tuple[float, int, Chunk]] = [(float(i), 0, c) for i, c in enumerate(doc_chunks)]
    for j in range(k):
        # slot s means "before original chunk s"; ties broken by draw order
        merged.append((slots[j] - 0.5, j + 1, distractors[j]))
    merged.sort(key=lambda t: (t[0], t[1]))
    return [c for _, _, c in merged]
