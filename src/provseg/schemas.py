"""Data model shared by ingestion, systems, evaluation and the demo app."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Section(BaseModel):
    idx: int
    heading: str
    paragraphs: list[str]


class Document(BaseModel):
    doc_id: str
    title: str
    sections: list[Section]
    source: str = "qasper"
    license: str = "CC-BY-4.0"

    @property
    def slug(self) -> str:
        return doc_slug(self.doc_id)


def doc_slug(doc_id: str) -> str:
    return "doc_" + "".join(ch if ch.isalnum() else "_" for ch in doc_id)


class Chunk(BaseModel):
    """A contiguous span of one section. ``text`` is the permanent source of truth."""

    chunk_id: str
    doc_id: str
    section_idx: int
    section_heading: str
    chunk_idx: int
    text: str
    # Owned paragraphs as (section idx, first paragraph, last paragraph), inclusive.
    # Overlap text copied from the previous chunk is not owned.
    spans: list[tuple[int, int, int]]
    overlap_chars: int = 0
    n_tokens: int = 0
    is_distractor: bool = False


class GoldReference(BaseModel):
    """One annotator's answer, in the shape of the official QASPER evaluator."""

    answer: str
    type: Literal["extractive", "abstractive", "boolean", "none"]
    evidence: list[str] = Field(default_factory=list)
    highlighted_evidence: list[str] = Field(default_factory=list)


class Question(BaseModel):
    qid: str
    doc_id: str
    question: str
    references: list[GoldReference]

    @property
    def unanswerable_all(self) -> bool:
        return bool(self.references) and all(r.type == "none" for r in self.references)

    @property
    def unanswerable_any(self) -> bool:
        return any(r.type == "none" for r in self.references)


# ---- pipeline artefacts -------------------------------------------------------


class QuoteSpan(BaseModel):
    text: str
    chunk_id: str
    start_char: int | None = None
    end_char: int | None = None
    verified: bool = False
    elided: bool = False


class Note(BaseModel):
    """Structured per-chunk note (evidence kept separate from reasoning, as in Segment+)."""

    chunk_id: str
    relevance: Literal["high", "medium", "low", "none"]
    evidence_quotes: list[QuoteSpan] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    reasoning_note: str = ""
    uncertainty: str = ""
    # Merged notes cover several chunks.
    source_chunk_ids: list[str] = Field(default_factory=list)
    fabricated_quotes: int = 0


class ContextItem(BaseModel):
    """One labelled block of the final answer context."""

    handle: str
    chunk_ids: list[str]
    text: str


class Statement(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)  # resolved stable chunk IDs
    raw_citations: list[str] = Field(default_factory=list)  # handles as emitted by the model
    invalid_citations: list[str] = Field(default_factory=list)  # handles not in the final context
    verdict: Literal["entailed", "contradicted", "not_enough_evidence", "unverified"] = "unverified"
    verifier_scores: dict[str, Any] = Field(default_factory=dict)
    action: Literal["kept", "revised", "dropped", "none"] = "none"
    original_text: str | None = None


class SystemOutput(BaseModel):
    qid: str
    system: str
    noise_level: int
    answer: str
    abstained: bool
    statements: list[Statement]
    dropped_statements: list[Statement] = Field(default_factory=list)
    context: list[ContextItem]
    context_tokens: int
    error: str | None = None
    trace: dict = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)
