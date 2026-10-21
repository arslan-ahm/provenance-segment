"""Pre-run call and cost estimate (no model calls).

Input tokens are counted exactly for note prompts (every chunk of every instance) and
approximated for the other stages; output tokens use stated assumptions, which matter
most for models that think before answering. Note calls are shared between noise
levels and between ``segment`` and ``provseg`` because their prompts are identical,
and the cache answers repeats, so only unique (question, chunk) pairs are billed.
"""

from __future__ import annotations

from pydantic import BaseModel

from .prompts import ANSWER_SYSTEM, MERGE_SYSTEM, NOTE_SYSTEM, note_user
from .tokens import count_tokens
from .verification import strip_headings


class Assumptions(BaseModel):
    note_output_tokens: int = 350
    answer_output_tokens: int = 500
    merge_calls_per_instance: float = 2.0
    merge_output_tokens: int = 1200
    revise_calls_per_instance: float = 1.0
    judge_calls_per_prediction: float = 5.0
    judge_input_tokens: int = 900
    judge_output_tokens: int = 250


def estimate(exp, a: Assumptions | None = None) -> dict:
    a = a or Assumptions()
    exp.load_built()
    cfg = exp.cfg
    base = cfg.models[cfg.base_model]
    judge = cfg.models.get(cfg.judge.model_ref or "judge") if cfg.judge.kind == "llm" else None
    sys_note = count_tokens(NOTE_SYSTEM)
    n_inst = len(exp.instances)
    note_pairs: dict[tuple[str, str], int] = {}
    for (qid, _k), ids in exp.instances.items():
        q = exp.questions[qid].question
        for cid in ids:
            if (qid, cid) not in note_pairs:
                c = exp.chunk_by_id[cid]
                note_pairs[(qid, cid)] = sys_note + count_tokens(note_user(q, c.section_heading, strip_headings(c.text)))
    answer_in = count_tokens(ANSWER_SYSTEM) + cfg.context_budget + 60
    merge_in = count_tokens(MERGE_SYSTEM) + 2 * cfg.context_budget

    rows = []
    kinds = {s.kind for s in cfg.systems}
    notes_billed = False
    for s in cfg.systems:
        calls = n_inst  # answer
        tin = n_inst * answer_in
        tout = n_inst * a.answer_output_tokens
        if s.kind in ("segment", "provseg"):
            calls += n_inst * a.merge_calls_per_instance
            tin += n_inst * a.merge_calls_per_instance * merge_in
            tout += n_inst * a.merge_calls_per_instance * a.merge_output_tokens
            if not notes_billed:
                calls += len(note_pairs)
                tin += sum(note_pairs.values())
                tout += len(note_pairs) * a.note_output_tokens
                notes_billed = True
        if s.kind == "provseg" and s.verify:
            calls += n_inst * a.revise_calls_per_instance * 2
            tin += n_inst * a.revise_calls_per_instance * 2 * 1200
            tout += n_inst * a.revise_calls_per_instance * 2 * 300
        cost = (tin * base.price_in_per_mtok + tout * base.price_out_per_mtok) / 1e6
        rows.append({"system": s.name, "calls": int(calls), "input_tokens": int(tin), "output_tokens": int(tout), "cost_usd": round(cost, 2)})
    judge_row = None
    if judge is not None:
        jc = len(cfg.systems) * n_inst * a.judge_calls_per_prediction
        jin, jout = jc * a.judge_input_tokens, jc * a.judge_output_tokens
        judge_row = {
            "model": judge.model,
            "calls": int(jc),
            "input_tokens": int(jin),
            "output_tokens": int(jout),
            "cost_usd": round((jin * judge.price_in_per_mtok + jout * judge.price_out_per_mtok) / 1e6, 2),
        }
    return {
        "base_model": base.model,
        "instances": n_inst,
        "unique_note_calls": len(note_pairs),
        "systems": rows,
        "judge": judge_row,
        "total_usd": round(sum(r["cost_usd"] for r in rows) + (judge_row["cost_usd"] if judge_row else 0), 2),
        "assumptions": a.model_dump(),
        "note_sharing": "segment and provseg share note calls" if {"segment", "provseg"} <= kinds else "n/a",
    }
