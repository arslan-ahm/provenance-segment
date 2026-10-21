"""Per-question metrics and their aggregation.

Definitions (all computed per question, then averaged over questions):

answer_f1               official QASPER token F1, max over annotators; abstention is the string "Unanswerable"
evidence_{p,r,f1}       cited chunk IDs vs gold evidence chunks, max F1 over annotators
citation_precision      ALCE-style: a (statement, citation) pair is precise if the cited chunk alone supports
                        the statement, or partially supports it while the statement is supported jointly
statement_support       fraction of statements supported by their citations (single pair or joint judgment)
unsupported_claim_rate  1 - statement_support; statements with no valid citation count as unsupported
out_of_context_rate     citations, as emitted by the model, that point at no item of the final context
abstained / gold_unanswerable   for abstention precision/recall/F1 over the question set
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..schemas import Question, SystemOutput
from .qasper_official import set_f1, token_f1

NAN = float("nan")


def is_distractor_id(chunk_id: str) -> bool:
    return chunk_id.startswith("noise")


def answer_f1(pred: str, q: Question) -> tuple[float, str]:
    scored = [(token_f1(pred, r.answer), r.type) for r in q.references]
    return max(scored, key=lambda t: t[0])


def evidence_scores(cited: set[str], gold_sets: list[set[str]], q: Question) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None
    for ref, gold in zip(q.references, gold_sets):
        if ref.type == "none":
            gold = set()
        elif not gold:
            continue  # evidence could not be mapped to chunks; this reference is not scorable
        scores = set_f1(cited, gold)
        if best is None or scores[2] > best[2]:
            best = scores
    return best if best is not None else (NAN, NAN, NAN)


def question_metrics(out: SystemOutput, q: Question, gold_sets: list[set[str]], judgments: list[dict] | None) -> dict:
    """``judgments`` is None when the prediction has not been judged yet (support metrics -> NaN)."""
    f1, atype = answer_f1(out.answer, q) if not out.error else (0.0, "error")
    cited = {c for s in out.statements for c in s.citations}
    ev_p, ev_r, ev_f = evidence_scores(cited, gold_sets, q)
    all_cits = [c for s in out.statements for c in s.citations]
    n_distractor_cits = sum(1 for c in all_cits if is_distractor_id(c))

    raw = out.trace.get("raw_statements", [])
    n_raw_cits = sum(len(s.get("raw_citations", [])) for s in raw)
    n_bad = sum(len(s.get("invalid_citations", [])) for s in raw)

    by_stmt: dict[int, dict[str, str]] = defaultdict(dict)
    dropped_labels: dict[str, dict[str, str]] = defaultdict(dict)
    for j in judgments or []:
        if isinstance(j["statement_idx"], str):
            dropped_labels[j["statement_idx"]][j["scope"]] = j["label"]
        else:
            by_stmt[j["statement_idx"]][j["scope"]] = j["label"]
    # A dropped statement the judge considers supported is a false rejection by the verifier.
    false_rejections = 0
    for i, st in enumerate(out.dropped_statements):
        labels = dropped_labels.get(f"d{i}", {})
        key = st.citations[0] if len(st.citations) == 1 else "joint"
        false_rejections += labels.get(key) == "supported"

    n_pairs = n_precise = 0
    n_supported = 0
    for si, st in enumerate(out.statements):
        labels = by_stmt.get(si, {})
        if len(st.citations) == 1:
            supported = labels.get(st.citations[0]) == "supported"
        elif st.citations:
            supported = labels.get("joint") == "supported"
        else:
            supported = False
        n_supported += supported
        for c in st.citations:
            n_pairs += 1
            lab = labels.get(c)
            if lab == "supported" or (lab == "partially_supported" and supported):
                n_precise += 1
    n_st = len(out.statements)
    # Statements without valid citations need no judge: they are unsupported by definition.
    judged = judgments is not None or all(not s.citations for s in out.statements)

    usage = out.usage or {}
    return {
        "qid": q.qid,
        "system": out.system,
        "noise_level": out.noise_level,
        "error": out.error,
        "answer_type": atype,
        "gold_unanswerable": q.unanswerable_all,
        "abstained": out.abstained,
        "answer_f1": f1,
        "evidence_precision": ev_p,
        "evidence_recall": ev_r,
        "evidence_f1": ev_f,
        "n_statements": n_st,
        "n_dropped": len(out.dropped_statements),
        "false_rejections": false_rejections if judgments is not None else NAN,
        "citation_precision": (n_precise / n_pairs) if (n_pairs and judged) else NAN,
        "statement_support": (n_supported / n_st) if (n_st and judged) else NAN,
        "unsupported_claim_rate": (1 - n_supported / n_st) if (n_st and judged) else NAN,
        # Defined only for answers with statements: a correct abstention is neither supported nor unsupported.
        "fully_supported_answer": float(n_supported == n_st) if (judged and n_st) else NAN,
        "out_of_context_rate": (n_bad / n_raw_cits) if n_raw_cits else NAN,
        "distractor_citation_rate": (n_distractor_cits / len(all_cits)) if all_cits else NAN,
        "context_tokens": out.context_tokens,
        "calls": usage.get("calls", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost_usd": usage.get("cost_usd", 0.0),
        "llm_latency_s": usage.get("llm_latency_s", 0.0),
        "quotes": out.trace.get("quote_stats", {}).get("quotes", NAN),
        "unanchored_quotes": out.trace.get("quote_stats", {}).get("unanchored", NAN),
    }


def abstention_scores(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["abstained"] and r["gold_unanswerable"])
    fp = sum(1 for r in rows if r["abstained"] and not r["gold_unanswerable"])
    fn = sum(1 for r in rows if not r["abstained"] and r["gold_unanswerable"])
    answerable = [r for r in rows if not r["gold_unanswerable"]]
    p = tp / (tp + fp) if tp + fp else NAN
    r = tp / (tp + fn) if tp + fn else NAN
    f = 2 * p * r / (p + r) if p and r and not math.isnan(p) and not math.isnan(r) and (p + r) else NAN
    return {
        "abstain_precision": p,
        "abstain_recall": r,
        "abstain_f1": f,
        "false_abstention_rate": (sum(r_["abstained"] for r_ in answerable) / len(answerable)) if answerable else NAN,
    }
