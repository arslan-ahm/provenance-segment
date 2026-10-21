"""The four compared systems behind one interface.

=============  ===============================================================
``direct``     first chunks in document order until the context budget is full
``rag``        top-ranked chunks (BM25 or dense) until the budget is full
``segment``    Segment+-style notes for every chunk -> filter -> merge to budget
``provseg``    ``segment`` + quote anchoring + provenance-checked merging +
               out-of-context citation stripping + claim verification/repair
=============  ===============================================================

Every system ends with the same answer prompt, the same budget and the same citation
format, so differences in citation quality come from the context each system builds
and from verification, not from different instructions.

The model never sees chunk IDs. Each chunk of the (possibly noisy) document gets a
neutral handle ``P01``, ``P02``... in document order; citations are resolved back to
stable chunk IDs deterministically. This keeps distractor chunks indistinguishable by
their labels.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from .llm import LLMClient, LLMError, UsageMeter, call_structured
from .prompts import (
    ABSTAIN,
    ANSWER_SYSTEM,
    DECOMPOSE_SYSTEM,
    MERGE_SYSTEM,
    NOTE_SYSTEM,
    REANSWER_SYSTEM,
    REVISE_SYSTEM,
    AnswerOut,
    DecomposeOut,
    MergeOut,
    NoteOut,
    ReanswerOut,
    ReviseOut,
    answer_user,
    merge_user,
    note_user,
    reanswer_user,
    revise_user,
)
from .quotes import locate
from .retrieval import DenseRanker, bm25_rank, fill_budget
from .schemas import Chunk, ContextItem, Question, Statement, SystemOutput
from .tokens import count_tokens
from .verification import Verifier, strip_headings

RELEVANCE_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


class SystemConfig(BaseModel):
    name: str
    kind: Literal["direct", "rag", "segment", "provseg"]
    context_budget: int = 2048
    retriever: Literal["bm25", "dense"] = "bm25"
    rag_max_chunks: int | None = None
    # segment / provseg
    keep_relevance: list[str] = ["high", "medium"]
    merge_batch_tokens: int = 2048
    merge_max_rounds: int = 3
    merge_max_out: int = 4
    # provseg only
    anchor_quotes: bool = True
    verify: bool = True
    claim_mode: Literal["statement", "atomic"] = "statement"
    on_unsupported: Literal["drop", "revise"] = "revise"
    abstain_if_empty: bool = True


@dataclass
class Resources:
    base: LLMClient
    verifier: Verifier | None = None
    dense: DenseRanker | None = None
    note_workers: int = 8


# ---- handles ---------------------------------------------------------------------

_HANDLE = re.compile(r"[Pp]\s*0*(\d+)")


def make_handle(i: int) -> str:
    return f"P{i + 1:02d}"


def canonical_handle(raw: str) -> str | None:
    m = _HANDLE.search(raw or "")
    return f"P{int(m.group(1)):02d}" if m else None


@dataclass
class Workspace:
    question: Question
    chunks: list[Chunk]
    handle_to_idx: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.handle_to_idx = {make_handle(i): i for i in range(len(self.chunks))}

    def chunk(self, handle: str) -> Chunk:
        return self.chunks[self.handle_to_idx[handle]]


# ---- notes ---------------------------------------------------------------------


@dataclass
class MQuote:
    handle: str
    text: str
    verified: bool
    start: int | None = None
    end: int | None = None


@dataclass
class MNote:
    """A (possibly merged) note. Every piece of evidence and every fact carries handles."""

    handles: list[str]
    relevance: str
    evidence: list[MQuote]
    facts: list[tuple[str, list[str]]]
    reasoning: str
    uncertainty: str = ""

    def render(self) -> str:
        lines = [f"Note (sources: {', '.join(f'[{h}]' for h in self.handles)})"]
        if self.evidence:
            lines.append("Evidence:")
            lines += [f'- [{q.handle}] "{q.text}"' for q in self.evidence]
        if self.facts:
            lines.append("Facts:")
            lines += [f"- {t} [{', '.join(hs)}]" for t, hs in self.facts]
        if self.reasoning:
            lines.append(f"Reasoning: {self.reasoning}")
        if self.uncertainty:
            lines.append(f"Not established: {self.uncertainty}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "handles": self.handles,
            "relevance": self.relevance,
            "evidence": [q.__dict__ for q in self.evidence],
            "facts": [{"text": t, "handles": hs} for t, hs in self.facts],
            "reasoning": self.reasoning,
            "uncertainty": self.uncertainty,
        }


def anchor(handle: str, quote: str, ws: Workspace) -> MQuote:
    if handle not in ws.handle_to_idx:
        return MQuote(handle=handle, text=quote, verified=False)
    loc = locate(quote, ws.chunk(handle).text)
    if loc is None:
        return MQuote(handle=handle, text=quote, verified=False)
    return MQuote(handle=handle, text=quote, verified=True, start=loc[0], end=loc[1])


def make_note(i: int, ws: Workspace, res: Resources, meter: UsageMeter) -> MNote:
    c = ws.chunks[i]
    h = make_handle(i)
    out = call_structured(
        res.base, meter, "note", NOTE_SYSTEM, note_user(ws.question.question, c.section_heading, strip_headings(c.text)), NoteOut, max_tokens=2048
    )
    quotes = [anchor(h, q, ws) for q in out.evidence_quotes if q.strip()]
    return MNote(
        handles=[h],
        relevance=out.relevance,
        evidence=quotes,
        facts=[(f, [h]) for f in out.facts if f.strip()],
        reasoning=out.reasoning_note,
        uncertainty=out.uncertainty,
    )


def render_notes(notes: list[MNote]) -> str:
    return "\n\n".join(n.render() for n in notes)


def merge_notes(notes: list[MNote], cfg: SystemConfig, ws: Workspace, res: Resources, meter: UsageMeter, trace: dict) -> list[MNote]:
    rounds = []
    for r in range(cfg.merge_max_rounds):
        size = count_tokens(render_notes(notes))
        if size <= cfg.context_budget:
            break
        batches: list[list[MNote]] = []
        cur: list[MNote] = []
        cur_t = 0
        for n in notes:
            t = count_tokens(n.render())
            if cur and cur_t + t > cfg.merge_batch_tokens:
                batches.append(cur)
                cur, cur_t = [], 0
            cur.append(n)
            cur_t += t
        if cur:
            batches.append(cur)
        merged: list[MNote] = []
        stats = {"round": r, "tokens_before": size, "batches": len(batches), "dropped_quotes": 0, "dropped_fact_handles": 0}
        for batch in batches:
            if len(batch) == 1:
                merged.append(batch[0])
                continue
            batch_handles = sorted({h for n in batch for h in n.handles})
            out = call_structured(
                res.base,
                meter,
                "merge",
                MERGE_SYSTEM.format(max_out=cfg.merge_max_out),
                merge_user(ws.question.question, render_notes(batch)),
                MergeOut,
                max_tokens=4096,
            )
            relevance = max((n.relevance for n in batch), key=RELEVANCE_RANK.__getitem__)
            for mn in out.notes:
                quotes: list[MQuote] = []
                for mq in mn.evidence:
                    h = canonical_handle(mq.handle) or mq.handle
                    q = anchor(h, mq.quote, ws)
                    if cfg.kind == "provseg" and cfg.anchor_quotes and (not q.verified or h not in batch_handles):
                        stats["dropped_quotes"] += 1
                        continue
                    quotes.append(q)
                facts: list[tuple[str, list[str]]] = []
                for f in mn.facts:
                    hs = [canonical_handle(x) or x for x in f.handles]
                    if cfg.kind == "provseg":
                        valid = [x for x in hs if x in batch_handles]
                        stats["dropped_fact_handles"] += len(hs) - len(valid)
                        if not valid:
                            continue
                        hs = valid
                    facts.append((f.text, hs))
                handles = sorted({q.handle for q in quotes} | {h for _, hs in facts for h in hs}) or batch_handles
                merged.append(MNote(handles=handles, relevance=relevance, evidence=quotes, facts=facts, reasoning=mn.reasoning_note))
        stats["tokens_after"] = count_tokens(render_notes(merged))
        rounds.append(stats)
        if stats["tokens_after"] >= size:  # merging stopped making progress
            notes = merged
            break
        notes = merged
    trace["merge_rounds"] = rounds
    # Deterministic fallback: drop the least relevant notes (latest first) until the budget is met.
    truncated = 0
    while notes and count_tokens(render_notes(notes)) > cfg.context_budget:
        worst = min(range(len(notes)), key=lambda i: (RELEVANCE_RANK[notes[i].relevance], -i))
        notes.pop(worst)
        truncated += 1
    trace["truncated_notes"] = truncated
    return notes


# ---- answering -----------------------------------------------------------------


def _resolve(statements_raw, valid: set[str], ws: Workspace) -> list[Statement]:
    out = []
    for s in statements_raw:
        raw = [str(c) for c in s.citations]
        canon = [canonical_handle(c) for c in raw]
        good = [h for h in dict.fromkeys(c for c in canon if c) if h in valid]
        bad = [r for r, c in zip(raw, canon) if not c or c not in valid]
        out.append(
            Statement(
                text=s.text,
                raw_citations=raw,
                citations=[ws.chunk(h).chunk_id for h in good],
                invalid_citations=bad,
            )
        )
    return out


def is_abstention(answer: str) -> bool:
    return answer.strip().strip(".").lower() in {"unanswerable", "unanswerable."}


def _context_handles(context: str, ws: Workspace) -> set[str]:
    return {h for h in (canonical_handle(m) for m in re.findall(r"\[P\d+\]", context)) if h and h in ws.handle_to_idx}


CONTEXT_SEP = "\n\n"


def render_chunk(i: int, c: Chunk) -> str:
    return f"[{make_handle(i)}] (section: {c.section_heading})\n{strip_headings(c.text)}"


def chunk_costs(chunks: list[Chunk]) -> list[int]:
    """Tokens each chunk adds to the answer context, separator included."""
    sep = count_tokens(CONTEXT_SEP)
    return [count_tokens(render_chunk(i, c)) + sep for i, c in enumerate(chunks)]


def raw_chunk_context(indices: list[int], ws: Workspace) -> tuple[str, list[ContextItem]]:
    items = [ContextItem(handle=make_handle(i), chunk_ids=[ws.chunks[i].chunk_id], text=render_chunk(i, ws.chunks[i])) for i in indices]
    return CONTEXT_SEP.join(it.text for it in items), items


# ---- verification --------------------------------------------------------------


def verify_and_repair(
    statements: list[Statement], answer: str, cfg: SystemConfig, ws: Workspace, res: Resources, meter: UsageMeter, trace: dict
) -> tuple[str, list[Statement], list[Statement]]:
    assert res.verifier is not None
    id_to_chunk = {c.chunk_id: c for c in ws.chunks}
    kept: list[Statement] = []
    dropped: list[Statement] = []
    changed = False

    def check(text: str, chunk_ids: list[str]) -> tuple[str, dict]:
        if not chunk_ids:
            return "not_enough_evidence", {"reason": "no valid citation"}
        sources = [id_to_chunk[c].text for c in chunk_ids]
        claims = [text]
        if cfg.claim_mode == "atomic":
            claims = call_structured(res.base, meter, "decompose", DECOMPOSE_SYSTEM, text, DecomposeOut, max_tokens=1024).claims or [text]
        details = {}
        worst = "entailed"
        for cl in claims:
            v = res.verifier.verify(cl, sources, meter)
            details[cl] = {"label": v.label, "entail": round(v.entail, 4), "contradict": round(v.contradict, 4), **v.detail}
            if v.label != "entailed":
                worst = "contradicted" if v.label == "contradicted" or worst == "contradicted" else "not_enough_evidence"
        return worst, details

    for st in statements:
        label, details = check(st.text, st.citations)
        st.verdict = label  # type: ignore[assignment]
        st.verifier_scores = details
        if label == "entailed":
            st.action = "kept"
            kept.append(st)
            continue
        changed = True
        if cfg.on_unsupported == "revise" and st.citations:
            passages = "\n\n".join(strip_headings(id_to_chunk[c].text) for c in st.citations)
            rev = call_structured(res.base, meter, "revise", REVISE_SYSTEM, revise_user(ws.question.question, st.text, passages), ReviseOut, max_tokens=1024)
            if not rev.drop and rev.statement.strip():
                label2, details2 = check(rev.statement, st.citations)
                if label2 == "entailed":
                    kept.append(
                        st.model_copy(
                            update={"original_text": st.text, "text": rev.statement, "verdict": "entailed", "verifier_scores": details2, "action": "revised"}
                        )
                    )
                    continue
        st.action = "dropped"
        dropped.append(st)

    if changed:
        if kept:
            listing = "\n".join(f"- {s.text}" for s in kept)
            answer = call_structured(res.base, meter, "reanswer", REANSWER_SYSTEM, reanswer_user(ws.question.question, answer, listing), ReanswerOut, max_tokens=512).answer
        elif cfg.abstain_if_empty:
            answer = ABSTAIN
    elif not statements and not is_abstention(answer) and cfg.abstain_if_empty:
        # An answer with no supporting statements cannot be verified.
        answer = ABSTAIN
        trace["abstained_no_statements"] = True
    return answer, kept, dropped


# ---- main entry ----------------------------------------------------------------


def run_system(cfg: SystemConfig, question: Question, chunks: list[Chunk], noise_level: int, res: Resources) -> SystemOutput:
    ws = Workspace(question=question, chunks=chunks)
    meter = UsageMeter()
    trace: dict = {"n_chunks": len(chunks), "n_distractors": sum(c.is_distractor for c in chunks)}
    try:
        if cfg.kind == "direct":
            idx = fill_budget(list(range(len(chunks))), chunk_costs(chunks), cfg.context_budget)
            context, items = raw_chunk_context(idx, ws)
        elif cfg.kind == "rag":
            ranking = res.dense.rank(question.question, chunks) if cfg.retriever == "dense" and res.dense else bm25_rank(question.question, chunks)
            idx = fill_budget([i for i, _ in ranking], chunk_costs(chunks), cfg.context_budget, cfg.rag_max_chunks)
            trace["ranking"] = [(make_handle(i), round(s, 4)) for i, s in ranking[:20]]
            context, items = raw_chunk_context(idx, ws)
        else:
            with ThreadPoolExecutor(max_workers=res.note_workers) as pool:
                notes = list(pool.map(lambda i: make_note(i, ws, res, meter), range(len(chunks))))
            n_quotes = sum(len(n.evidence) for n in notes)
            n_fab = sum(1 for n in notes for q in n.evidence if not q.verified)
            trace["notes"] = [n.to_dict() for n in notes]
            trace["quote_stats"] = {"quotes": n_quotes, "unanchored": n_fab}
            if cfg.kind == "provseg" and cfg.anchor_quotes:
                for n in notes:
                    n.evidence = [q for q in n.evidence if q.verified]
            kept = [n for n in notes if n.relevance in cfg.keep_relevance]
            trace["filter"] = {make_handle(i): ("keep" if n.relevance == "high" else "maybe" if n.relevance in cfg.keep_relevance else "remove") for i, n in enumerate(notes)}
            kept = merge_notes(kept, cfg, ws, res, meter, trace)
            context = render_notes(kept)
            items = [ContextItem(handle=f"N{i + 1}", chunk_ids=[ws.chunk(h).chunk_id for h in n.handles if h in ws.handle_to_idx], text=n.render()) for i, n in enumerate(kept)]
            trace["final_notes"] = [n.to_dict() for n in kept]

        valid = _context_handles(context, ws)
        out = call_structured(res.base, meter, "answer", ANSWER_SYSTEM, answer_user(question.question, context or "(empty)"), AnswerOut, max_tokens=2048)
        statements = _resolve(out.statements, valid, ws)
        trace["raw_answer"] = out.model_dump()
        trace["raw_statements"] = [s.model_dump() for s in statements]
        answer = out.answer
        dropped: list[Statement] = []
        if cfg.kind == "provseg":
            # Citations that never entered the final context are removed before verification.
            for s in statements:
                s.invalid_citations = []
            if cfg.verify:
                answer, statements, dropped = verify_and_repair(statements, answer, cfg, ws, res, meter, trace)
        return SystemOutput(
            qid=question.qid,
            system=cfg.name,
            noise_level=noise_level,
            answer=answer,
            abstained=is_abstention(answer),
            statements=statements,
            dropped_statements=dropped,
            context=items,
            context_tokens=count_tokens(context),
            trace=trace,
            usage=meter.summary(),
        )
    except LLMError as e:
        return SystemOutput(
            qid=question.qid,
            system=cfg.name,
            noise_level=noise_level,
            answer="",
            abstained=False,
            statements=[],
            context=[],
            context_tokens=0,
            error=str(e),
            trace=trace,
            usage=meter.summary(),
        )
