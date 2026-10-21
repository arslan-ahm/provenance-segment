"""Sanity check of the in-system NLI verifier on real QASPER text (no LLM calls).

For answerable questions in a built run, each annotator's *highlighted evidence*
sentence is used as a claim and checked against
  (a) the gold chunk that contains its evidence paragraph -> should be "entailed"
  (b) a distractor chunk from another paper           -> should not be "entailed"

This only checks that windowing and thresholds behave on real text with verbatim
claims. It does NOT measure how well the verifier judges paraphrased model output;
that is what the human audit of the main run is for.

    uv run python scripts/verifier_sanity.py --run runs/main_v1 --n 60
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from provseg.chunking import split_sentences
from provseg.verification import NLIVerifier


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main_v1")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    chunks = {r["chunk_id"]: r for r in rows(run / "doc_chunks.jsonl")}
    distractors = rows(run / "distractors.jsonl")
    rng = random.Random(a.seed)
    cases = []
    for q in rows(run / "questions.jsonl"):
        for ref, gold in zip(q["references"], q["gold_chunk_sets"]):
            if ref["type"] == "none" or len(gold) != 1:
                continue
            chunk = chunks[gold[0]]
            for hl in ref["highlighted_evidence"]:
                sents = [s for s in split_sentences(hl) if 8 <= len(s.split()) <= 60 and "FLOAT SELECTED" not in s]
                if sents and sents[0] in chunk["text"]:
                    cases.append((q["qid"], sents[0], chunk["text"]))
                    break
            break
    rng.shuffle(cases)
    cases = cases[: a.n]
    v = NLIVerifier(a.model)
    t0 = time.perf_counter()
    out = []
    for qid, claim, gold_text in cases:
        d = rng.choice(distractors)
        pos = v.verify(claim, [gold_text])
        neg = v.verify(claim, [d["text"]])
        out.append({"qid": qid, "claim": claim, "gold": pos.label, "gold_entail": round(pos.entail, 4), "distractor": neg.label, "distractor_entail": round(neg.entail, 4)})
    secs = time.perf_counter() - t0
    n = len(out)
    summary = {
        "model": a.model,
        "n": n,
        "gold_entailed_rate": sum(o["gold"] == "entailed" for o in out) / n,
        "distractor_entailed_rate": sum(o["distractor"] == "entailed" for o in out) / n,
        "seconds_per_pair_cpu": round(secs / (2 * n), 3),
    }
    dest = run / "diagnostics" / "verifier_sanity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": summary, "cases": out}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    misses = [o for o in out if o["gold"] != "entailed"][:3]
    for m in misses:
        print("MISS:", m["gold_entail"], m["claim"][:160])


if __name__ == "__main__":
    main()
