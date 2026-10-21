"""Chunking with stable provenance IDs.

Paragraphs are packed greedily, in document order, up to ``target_tokens`` (hard cap
``max_tokens``). Section boundaries are preferred break points: a chunk ends at a
section boundary once it holds at least ``min_tokens``, so short adjacent sections can
share a chunk while long sections are split. A paragraph longer than the cap is split
on sentence boundaries. Each section start inside a chunk is marked by a heading line
(``§ Heading``) so the model sees the structure.

When a chunk starts in the middle of a section it is prefixed with up to
``overlap_tokens`` of trailing sentences from the previous chunk; that prefix is
recorded in ``overlap_chars`` and is not owned evidence.

``spans`` records exactly which (section, first paragraph, last paragraph) each chunk
owns. IDs look like ``doc_1912_01214_sec_04_chunk_01``: the section the chunk starts in
and a running index of chunks starting in that section.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel

from .data.qasper import normalize_paragraph
from .schemas import Chunk, Document, Question
from .tokens import count_tokens

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


class ChunkingConfig(BaseModel):
    target_tokens: int = 550
    max_tokens: int = 700
    min_tokens: int = 300
    overlap_tokens: int = 50


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if s.strip()]


def _split_long(paragraph: str, max_tokens: int) -> list[str]:
    pieces, cur = [], []
    for sent in split_sentences(paragraph):
        if cur and count_tokens(" ".join(cur + [sent])) > max_tokens:
            pieces.append(" ".join(cur))
            cur = [sent]
        else:
            cur.append(sent)
    if cur:
        pieces.append(" ".join(cur))
    # A single sentence longer than max_tokens stays whole: we never cut inside a sentence.
    return pieces


def _tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    out: list[str] = []
    for sent in reversed(split_sentences(text)):
        if count_tokens(" ".join([sent] + out)) > overlap_tokens:
            break
        out.insert(0, sent)
    return " ".join(out)


def heading_line(heading: str) -> str:
    return f"§ {heading}"


def chunk_document(doc: Document, cfg: ChunkingConfig, id_prefix: str | None = None) -> list[Chunk]:
    prefix = id_prefix or doc.slug
    # unit = (section idx, paragraph idx, text, first unit of its section?)
    units: list[tuple[int, int, str, bool]] = []
    for sec in doc.sections:
        first = True
        for p_idx, para in enumerate(sec.paragraphs):
            pieces = _split_long(para, cfg.max_tokens) if count_tokens(para) > cfg.max_tokens else [para]
            for piece in pieces:
                units.append((sec.idx, p_idx, piece, first))
                first = False
    headings = {s.idx: s.heading for s in doc.sections}

    groups: list[list[tuple[int, int, str, bool]]] = []
    cur: list[tuple[int, int, str, bool]] = []
    cur_tokens = 0
    for u in units:
        t = count_tokens(u[2]) + (count_tokens(heading_line(headings[u[0]])) if u[3] else 0)
        boundary_break = u[3] and cur_tokens >= cfg.min_tokens
        if cur and (boundary_break or cur_tokens + t > cfg.target_tokens):
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += t
    if cur:
        groups.append(cur)

    chunks: list[Chunk] = []
    per_section_count: dict[int, int] = {}
    prev_body = ""
    for group in groups:
        start_sec = group[0][0]
        lines: list[str] = []
        for i, (s, _, text, first) in enumerate(group):
            # Always show the heading at the top of a chunk, and at each new section inside it.
            if first or i == 0:
                lines.append(heading_line(headings[s]))
            lines.append(text)
        body = "\n\n".join(lines)
        overlap = "" if group[0][3] else _tail(prev_body, cfg.overlap_tokens)
        if overlap:
            # The heading stays first; the carried-over sentences follow it.
            head, rest = body.split("\n\n", 1)
            text = f"{head}\n\n{overlap}\n\n{rest}"
        else:
            text = body
        spans: list[tuple[int, int, int]] = []
        for s, p, _, _ in group:
            if spans and spans[-1][0] == s:
                spans[-1] = (s, spans[-1][1], p)
            else:
                spans.append((s, p, p))
        c_idx = per_section_count.get(start_sec, 0)
        per_section_count[start_sec] = c_idx + 1
        chunks.append(
            Chunk(
                chunk_id=f"{prefix}_sec_{start_sec:02d}_chunk_{c_idx:02d}",
                doc_id=doc.doc_id,
                section_idx=start_sec,
                section_heading=headings[start_sec],
                chunk_idx=c_idx,
                text=text,
                spans=spans,
                overlap_chars=len(overlap) + 2 if overlap else 0,
                n_tokens=count_tokens(text),
            )
        )
        prev_body = "\n\n".join(u[2] for u in group)
    return chunks


def gold_chunk_ids(question: Question, doc: Document, chunks: Iterable[Chunk]) -> tuple[list[set[str]], int, int]:
    """Map each annotator's evidence paragraphs to the chunks that own them.

    Returns one set of chunk IDs per reference, plus (#evidence strings matched,
    #evidence strings total) so coverage of the mapping can be reported.
    """
    owner: dict[tuple[int, int], str] = {}
    for c in chunks:
        if c.is_distractor:
            continue
        for s, p0, p1 in c.spans:
            for p in range(p0, p1 + 1):
                owner.setdefault((s, p), c.chunk_id)
    index: dict[str, tuple[int, int]] = {}
    for sec in doc.sections:
        for p_idx, para in enumerate(sec.paragraphs):
            index.setdefault(normalize_paragraph(para), (sec.idx, p_idx))

    per_ref: list[set[str]] = []
    matched = total = 0
    for ref in question.references:
        ids: set[str] = set()
        for ev in ref.evidence:
            total += 1
            key = normalize_paragraph(ev)
            loc = index.get(key)
            if loc is None:
                # Evidence sometimes differs from full_text by truncation; accept containment of long strings.
                loc = next((v for k, v in index.items() if min(len(k), len(key)) > 40 and (key in k or k in key)), None)
            if loc is not None and loc in owner:
                ids.add(owner[loc])
                matched += 1
        per_ref.append(ids)
    return per_ref, matched, total
