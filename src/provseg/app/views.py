"""View models shared by the FastAPI explorer and the Gradio Space: plain dicts built from a
loaded run, no web framework involved."""

from __future__ import annotations

from ..runner import Experiment, read_jsonl
from ..systems import make_handle
from .live import budget_contexts, featured

TRACE_KEYS = ("filter", "quote_stats", "merge_rounds", "truncated_notes", "ranking", "notes", "final_notes", "abstained_no_statements")


class NotFound(KeyError):
    """Unknown question, noise level or chunk."""


def question_list(exp: Experiment) -> list[dict]:
    out = []
    for q in exp.questions.values():
        done = sorted({(s, k) for (s, k, qid) in exp._preds if qid == q.qid})
        out.append(
            {
                "qid": q.qid,
                "question": q.question,
                "doc_id": q.doc_id,
                "title": exp.documents.get(q.doc_id, {}).get("title", q.doc_id),
                "unanswerable": q.unanswerable_all,
                "answers": sorted({r.answer for r in q.references}),
                "predictions": [{"system": s, "noise_level": k} for s, k in done],
            }
        )
    return out


def question_detail(exp: Experiment, qid: str, noise: int) -> dict:
    q = exp.questions.get(qid)
    if q is None:
        raise NotFound("unknown question")
    ids = exp.instances.get((qid, noise))
    if ids is None:
        raise NotFound(f"no instance at noise level {noise}")
    gold = sorted(set().union(*exp.gold[qid])) if exp.gold[qid] else []
    chunks = []
    for i, cid in enumerate(ids):
        c = exp.chunk_by_id[cid]
        chunks.append({"handle": make_handle(i), "chunk_id": cid, "heading": c.section_heading, "is_distractor": c.is_distractor, "is_gold": cid in gold, "n_tokens": c.n_tokens})
    preds = []
    for s in exp.cfg.systems:
        o = exp._preds.get((s.name, noise, qid))
        if o is None:
            continue
        preds.append(
            {
                "system": s.name,
                "kind": s.kind,
                "answer": o.answer,
                "abstained": o.abstained,
                "error": o.error,
                "statements": [st.model_dump() for st in o.statements],
                "dropped_statements": [st.model_dump() for st in o.dropped_statements],
                "context": [c.model_dump() for c in o.context],
                "context_tokens": o.context_tokens,
                "usage": {k: v for k, v in o.usage.items() if k != "cache_keys"},
                "judgments": exp._judg.get((s.name, noise, qid)),
                "trace": {k: v for k, v in o.trace.items() if k in TRACE_KEYS},
            }
        )
    return {
        "qid": qid,
        "question": q.question,
        "doc_id": q.doc_id,
        "title": exp.documents.get(q.doc_id, {}).get("title", q.doc_id),
        "unanswerable": q.unanswerable_all,
        "references": [r.model_dump() for r in q.references],
        "gold_chunk_ids": gold,
        "noise_level": noise,
        "noise_levels": sorted({k for (qq, k) in exp.instances if qq == qid}),
        "chunks": chunks,
        "predictions": preds,
        "computed_contexts": budget_contexts(q.question, [exp.chunk_by_id[c] for c in ids], exp.cfg.context_budget, set(gold)),
    }


def featured_questions(exp: Experiment) -> list[dict]:
    rows = read_jsonl(exp.dir / "diagnostics" / "retrieval_rows.jsonl")
    out = []
    for f in featured(rows, exp.questions):
        q = exp.questions[f["qid"]]
        out.append({**f, "question": q.question, "title": exp.documents.get(q.doc_id, {}).get("title", q.doc_id), "unanswerable": q.unanswerable_all})
    return out
