"""Turn predictions + judgments into per-question rows and aggregate tables."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .metrics import abstention_scores, question_metrics
from .stats import bootstrap_ci, paired_test

MEAN_METRICS = [
    "answer_f1",
    "evidence_precision",
    "evidence_recall",
    "evidence_f1",
    "citation_precision",
    "statement_support",
    "unsupported_claim_rate",
    "fully_supported_answer",
    "out_of_context_rate",
    "distractor_citation_rate",
    "n_statements",
    "n_dropped",
    "false_rejections",
    "context_tokens",
    "calls",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "llm_latency_s",
]
PAIRED_METRICS = ["answer_f1", "citation_precision", "unsupported_claim_rate", "evidence_f1"]


def _nan_to_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def per_question_rows(exp) -> list[dict]:
    judgments = exp.judgments()
    rows = []
    for out in exp.latest_predictions():
        key = (out.system, out.noise_level, out.qid)
        rows.append(question_metrics(out, exp.questions[out.qid], exp.gold[out.qid], judgments.get(key)))
    rows.sort(key=lambda r: (r["system"], r["noise_level"], r["qid"]))
    return rows


def aggregate(rows: list[dict], reference_system: str | None = "rag_bm25", focus_system: str | None = "provseg") -> dict:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["noise_level"])].append(r)

    table = []
    for (system, k), rs in sorted(groups.items()):
        answerable = [r for r in rs if not r["gold_unanswerable"]]
        entry = {"system": system, "noise_level": k, "n": len(rs), "n_answerable": len(answerable), "errors": sum(1 for r in rs if r["error"])}
        for m in MEAN_METRICS:
            # Answer quality on answerable questions; support metrics over all answered questions.
            pool = answerable if m in ("answer_f1", "evidence_precision", "evidence_recall", "evidence_f1") else rs
            mean, lo, hi = bootstrap_ci([r[m] for r in pool])
            entry[m] = {"mean": mean, "low": lo, "high": hi}
        entry["answer_f1_all"] = dict(zip(("mean", "low", "high"), bootstrap_ci([r["answer_f1"] for r in rs])))
        entry.update(abstention_scores(rs))
        table.append(entry)

    paired = []
    if reference_system and focus_system:
        for k in sorted({r["noise_level"] for r in rows}):
            a = {r["qid"]: r for r in groups.get((focus_system, k), [])}
            b = {r["qid"]: r for r in groups.get((reference_system, k), [])}
            common = sorted(set(a) & set(b))
            for m in PAIRED_METRICS:
                pool = [q for q in common if not (m.startswith(("answer", "evidence")) and a[q]["gold_unanswerable"])]
                res = paired_test([a[q][m] for q in pool], [b[q][m] for q in pool])
                paired.append({"metric": m, "noise_level": k, "a": focus_system, "b": reference_system, **res})

    robustness = []
    for system in sorted({r["system"] for r in rows}):
        base = {r["qid"]: r for r in groups.get((system, 0), [])}
        for k in sorted({r["noise_level"] for r in rows if r["noise_level"] > 0}):
            cur = {r["qid"]: r for r in groups.get((system, k), [])}
            common = sorted(set(base) & set(cur))
            for m in ("answer_f1", "citation_precision", "unsupported_claim_rate"):
                pool = [q for q in common if not (m == "answer_f1" and base[q]["gold_unanswerable"])]
                res = paired_test([cur[q][m] for q in pool], [base[q][m] for q in pool])
                robustness.append({"system": system, "metric": m, "noise_level": k, "delta_vs_clean": res["diff"], "low": res["low"], "high": res["high"], "n": res["n"]})

    clean = json.loads(json.dumps({"table": table, "paired": paired, "robustness": robustness}, default=float), parse_constant=lambda c: None)
    return _walk(clean)


def _walk(x):
    if isinstance(x, dict):
        return {k: _walk(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_walk(v) for v in x]
    return _nan_to_none(x)


def write_outputs(exp, rows: list[dict], agg: dict) -> dict[str, Path]:
    d = exp.dir
    csv_path = d / "per_question.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: _nan_to_none(v) for k, v in r.items()})
    m_path = d / "metrics.json"
    m_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return {"per_question": csv_path, "metrics": m_path}
