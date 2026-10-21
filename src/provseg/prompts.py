"""All prompts in one place. ``PROMPT_VERSION`` is written into every run manifest."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PROMPT_VERSION = "2026-10-21.1"

ABSTAIN = "Unanswerable"

# ---- per-chunk structured note ---------------------------------------------------

NOTE_SYSTEM = """You read ONE segment of a long document and record material that may help answer a question.
You will not see the rest of the document. Be precise and conservative.

Fields:
- relevance: "high" if the segment directly answers or is required for the answer; "medium" if it gives useful context, definitions or partial information; "low" if only loosely related; "none" if unrelated.
- evidence_quotes: up to 4 exact quotations copied character-for-character from the segment that directly bear on the question (each at most 60 words). Do not paraphrase, fix typos or merge sentences. Use [] if nothing applies.
- facts: up to 5 short facts stated explicitly in the segment that bear on the question.
- reasoning_note: one or two sentences on how this segment could help answer the question (connections, implications, entities or events worth keeping).
- uncertainty: what this segment does NOT establish about the question."""


def note_user(question: str, heading: str, segment: str) -> str:
    return f"Question: {question}\n\nSegment (section: {heading}):\n\"\"\"\n{segment}\n\"\"\""


class NoteOut(BaseModel):
    relevance: Literal["high", "medium", "low", "none"]
    evidence_quotes: list[str]
    facts: list[str]
    reasoning_note: str
    uncertainty: str


# ---- merging ---------------------------------------------------------------------

MERGE_SYSTEM = """You consolidate research notes about a question into fewer, shorter notes so they fit a limited context window.
Each input note is labelled with source handles like [P07].

Rules:
- Keep every evidence quote that bears on the question EXACTLY as written, with the single handle it came from. Never edit, shorten or combine quotes. Drop quotes that are irrelevant.
- Combine and compress facts and reasoning; each fact must list the handles it is based on.
- Never introduce information that is not in the input notes.
- Produce at most {max_out} merged notes."""


def merge_user(question: str, rendered_notes: str) -> str:
    return f"Question: {question}\n\nNotes:\n{rendered_notes}"


class MergedQuote(BaseModel):
    handle: str
    quote: str


class MergedFact(BaseModel):
    text: str
    handles: list[str]


class MergedNote(BaseModel):
    evidence: list[MergedQuote]
    facts: list[MergedFact]
    reasoning_note: str


class MergeOut(BaseModel):
    notes: list[MergedNote]


# ---- answering (identical for every system) --------------------------------------

ANSWER_SYSTEM = f"""You answer a question about a document using ONLY the context provided. The context consists of items labelled with handles such as [P03].

Return:
- answer: a concise answer (a short phrase or one sentence). For yes/no questions answer exactly "Yes" or "No". If the context does not establish the answer, answer exactly "{ABSTAIN}".
- statements: the factual sentences that justify the answer. Each statement must be supported by the context and list the handle(s) of the item(s) that support it in "citations". Cite only handles that appear in the context. If the answer is "{ABSTAIN}", statements may be empty.

Do not use outside knowledge."""


def answer_user(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}"


class AnswerStatement(BaseModel):
    text: str
    citations: list[str]


class AnswerOut(BaseModel):
    answer: str
    statements: list[AnswerStatement]


# ---- verification and repair -----------------------------------------------------

VERIFY_SYSTEM = """You check whether a claim is supported by source passages. Judge strictly from the passages; ignore outside knowledge.

Labels:
- "entailed": the passages state or directly imply the entire claim.
- "contradicted": the passages state something incompatible with the claim.
- "not_enough_evidence": anything else, including partial support."""


def verify_user(claim: str, passages: str) -> str:
    return f"Passages:\n{passages}\n\nClaim: {claim}"


class VerifyOut(BaseModel):
    label: Literal["entailed", "contradicted", "not_enough_evidence"]
    rationale: str


REVISE_SYSTEM = """A statement in an answer was not supported by the passages it cites. Rewrite it so that every part is supported by those passages, staying as close to the original as possible. If no supported version is possible, set drop to true."""


def revise_user(question: str, statement: str, passages: str) -> str:
    return f"Question: {question}\n\nPassages:\n{passages}\n\nUnsupported statement: {statement}"


class ReviseOut(BaseModel):
    drop: bool
    statement: str


REANSWER_SYSTEM = f"""Some statements in a draft answer failed source verification and were removed or revised. Given the question and the verified statements that remain, give the final concise answer. For yes/no questions answer exactly "Yes" or "No". If the verified statements do not establish the answer, answer exactly "{ABSTAIN}"."""


def reanswer_user(question: str, draft_answer: str, statements: str) -> str:
    return f"Question: {question}\n\nDraft answer: {draft_answer}\n\nVerified statements:\n{statements or '(none)'}"


class ReanswerOut(BaseModel):
    answer: str


# ---- claim decomposition (optional) ----------------------------------------------

DECOMPOSE_SYSTEM = """Split the statement into minimal, self-contained factual claims (resolve pronouns). If it is already atomic, return it unchanged as the only claim."""


class DecomposeOut(BaseModel):
    claims: list[str] = Field(default_factory=list)


# ---- evaluation judge (never used inside a system) --------------------------------

JUDGE_SYSTEM = """You are auditing citations in an answer about a scientific paper. Decide whether the cited source text supports the claim. Judge only from the source text.

Labels:
- "supported": the source text states or directly implies the whole claim.
- "partially_supported": some but not all of the claim is supported.
- "not_supported": the source text does not support the claim.
- "contradicted": the source text conflicts with the claim."""


def judge_user(question: str, claim: str, source: str) -> str:
    return f"Question being answered: {question}\n\nSource text:\n\"\"\"\n{source}\n\"\"\"\n\nClaim: {claim}"


class JudgeOut(BaseModel):
    label: Literal["supported", "partially_supported", "not_supported", "contradicted"]
    rationale: str


CORRECTNESS_SYSTEM = """You grade an answer to a question about a scientific paper against reference answers written by expert annotators. The answer is correct if it conveys the same information as at least one reference (wording may differ; extra correct detail is fine). If a reference is "Unanswerable", an answer of "Unanswerable" matches it."""


def correctness_user(question: str, references: list[str], answer: str) -> str:
    refs = "\n".join(f"- {r}" for r in references)
    return f"Question: {question}\n\nReference answers:\n{refs}\n\nAnswer to grade: {answer}"


class CorrectnessOut(BaseModel):
    correct: bool
    rationale: str
