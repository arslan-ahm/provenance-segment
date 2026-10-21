"""Retrieval-only diagnostics (no LLM): how much gold evidence survives into a fixed
context budget, and how much of that budget distractors take, as noise increases.

This measures the input side of the hypothesis. Direct and RAG can only cite what they
are shown; the Segment-style systems read every chunk, so their recall ceiling is 1.0
and their losses happen later (filtering, merging, answering), which the main
experiment measures.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .evaluation.stats import bootstrap_ci
from .retrieval import DenseRanker, bm25_rank, fill_budget
from .systems import chunk_costs


def run(exp, use_dense: bool = False) -> dict:
    exp.load_built()
    budget = exp.cfg.context_budget
    dense = DenseRanker() if use_dense else None
    rows = []
    for (qid, k), chunk_ids in sorted(exp.instances.items()):
        q = exp.questions[qid]
        if q.unanswerable_any:
            continue
        gold_sets = [g for g, r in zip(exp.gold[qid], q.references) if r.type != "none" and g]
        if not gold_sets:
            continue
        chunks = [exp.chunk_by_id[c] for c in chunk_ids]
        orders = {"direct": list(range(len(chunks))), "bm25": [i for i, _ in bm25_rank(q.question, chunks)]}
        if dense is not None:
            orders["dense"] = [i for i, _ in dense.rank(q.question, chunks)]
        costs = chunk_costs(chunks)
        for method, order in orders.items():
            picked = fill_budget(order, costs, budget)
            picked_ids = {chunks[i].chunk_id for i in picked}
            used = sum(costs[i] for i in picked)
            distractor_tokens = sum(costs[i] for i in picked if chunks[i].is_distractor)
            recall = max(len(picked_ids & g) / len(g) for g in gold_sets)
            ranks = [r for r, i in enumerate(order, 1) if any(chunks[i].chunk_id in g for g in gold_sets)]
            rows.append(
                {
                    "qid": qid,
                    "noise_level": k,
                    "method": method,
                    "gold_recall_at_budget": recall,
                    "all_gold_in_context": float(any(g <= picked_ids for g in gold_sets)),
                    "distractor_token_share": distractor_tokens / used if used else 0.0,
                    "mrr_first_gold": 1.0 / ranks[0] if ranks else 0.0,
                    "chunks_in_context": len(picked),
                    "doc_chunks": len(chunks),
                }
            )
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["noise_level"])].append(r)
    table = []
    for (method, k), rs in sorted(groups.items()):
        entry = {"method": method, "noise_level": k, "n": len(rs)}
        for m in ("gold_recall_at_budget", "all_gold_in_context", "distractor_token_share", "mrr_first_gold", "chunks_in_context", "doc_chunks"):
            mean, lo, hi = bootstrap_ci([r[m] for r in rs])
            entry[m] = {"mean": mean, "low": lo, "high": hi}
        table.append(entry)
    out_dir = exp.dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "retrieval_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = {"context_budget": budget, "table": table}
    (out_dir / "retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Retrieval diagnostic: {exp.cfg.name}\n",
        f"Answerable questions with mapped gold evidence; context budget {budget} tokens; noise mode `{exp.cfg.noise.mode}`. Means with 95% bootstrap CIs.\n",
        "| Method | Noise | n | Gold recall @ budget | All gold in context | Distractor token share | MRR (first gold) | Chunks in context | Chunks in doc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in table:
        def f(m, pct=True, e=e):
            c = e[m]
            return f"{100 * c['mean']:.1f} [{100 * c['low']:.1f}, {100 * c['high']:.1f}]" if pct else f"{c['mean']:.1f}"

        lines.append(
            f"| {e['method']} | {e['noise_level']} | {e['n']} | {f('gold_recall_at_budget')} | {f('all_gold_in_context')} | "
            f"{f('distractor_token_share')} | {f('mrr_first_gold')} | {f('chunks_in_context', False)} | {f('doc_chunks', False)} |"
        )
    (out_dir / "RETRIEVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
