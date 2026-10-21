"""Deterministic extraction of failure examples for the report.

Categories
----------
unsupported_kept      a final statement the judge says is not supported (for provseg: a verifier miss)
false_rejection       a statement provseg dropped that the judge considers supported
distractor_cited      a statement citing an injected distractor chunk
noise_flip            answer F1 >= 0.5 without noise but < 0.5 at the highest noise level
wrong_abstention      abstained on an answerable question
hallucinated_answer   answered a question every annotator marked unanswerable
fabricated_quote      a note quote that does not occur in its source chunk (segment / provseg notes)
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .metrics import is_distractor_id


def _order(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def collect(exp, rows: list[dict], per_category: int = 8) -> dict[str, list[dict]]:
    preds = {(o.system, o.noise_level, o.qid): o for o in exp.latest_predictions()}
    judgments = exp.judgments()
    cats: dict[str, list[dict]] = defaultdict(list)

    for (system, k, qid), out in preds.items():
        q = exp.questions[qid]
        base = {"system": system, "noise_level": k, "qid": qid, "question": q.question, "answer": out.answer, "gold": [r.answer for r in q.references]}
        labels: dict = defaultdict(dict)
        for j in judgments.get((system, k, qid), []):
            labels[j["statement_idx"]][j["scope"]] = (j["label"], j.get("rationale", ""))
        for si, st in enumerate(out.statements):
            key = st.citations[0] if len(st.citations) == 1 else "joint"
            lab = labels.get(si, {}).get(key)
            if lab and lab[0] in ("not_supported", "contradicted"):
                cats["unsupported_kept"].append({**base, "statement": st.text, "citations": st.citations, "judge": lab[0], "judge_rationale": lab[1], "verifier": st.verdict})
            if any(is_distractor_id(c) for c in st.citations):
                cats["distractor_cited"].append({**base, "statement": st.text, "citations": st.citations})
        for i, st in enumerate(out.dropped_statements):
            key = st.citations[0] if len(st.citations) == 1 else "joint"
            lab = labels.get(f"d{i}", {}).get(key)
            if lab and lab[0] == "supported":
                cats["false_rejection"].append({**base, "statement": st.original_text or st.text, "citations": st.citations, "verifier_scores": st.verifier_scores, "judge_rationale": lab[1]})
        if out.abstained and not q.unanswerable_any:
            cats["wrong_abstention"].append(base)
        if not out.abstained and q.unanswerable_all and not out.error:
            cats["hallucinated_answer"].append({**base, "statements": [s.text for s in out.statements]})
        for n in out.trace.get("notes", []):
            for qt in n["evidence"]:
                if not qt["verified"]:
                    cats["fabricated_quote"].append({**base, "handle": qt["handle"], "quote": qt["text"]})

    by_key = {(r["system"], r["noise_level"], r["qid"]): r for r in rows}
    levels = sorted({r["noise_level"] for r in rows})
    if levels and levels[-1] > 0:
        for (system, k, qid), r in by_key.items():
            if k != 0 or r["gold_unanswerable"]:
                continue
            hi = by_key.get((system, levels[-1], qid))
            if hi and r["answer_f1"] >= 0.5 and hi["answer_f1"] < 0.5:
                o0, o1 = preds[(system, 0, qid)], preds[(system, levels[-1], qid)]
                cats["noise_flip"].append(
                    {"system": system, "qid": qid, "question": exp.questions[qid].question, "answer_clean": o0.answer, "answer_noisy": o1.answer, "noise_level": levels[-1]}
                )

    return {c: sorted(v, key=lambda e: _order(c, e.get("system"), e.get("noise_level"), e.get("qid"), e.get("statement", e.get("quote", ""))))[:per_category] for c, v in cats.items()} | {
        f"{c}__count": len(v) for c, v in cats.items()
    }


def write(exp, failures: dict) -> None:
    (exp.dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
