"""Evaluation-time citation judge.

This is deliberately separate from the verifier used inside ``provseg``: if the method
were scored by its own verifier, its gains would be circular. The default judge is a
different LLM; the NLI backend exists for cheap offline estimates and must use a
different checkpoint from the in-system verifier. A random sample of judgments is
exported for human audit (``provseg audit-export``).

For each statement with citations we judge every (statement, cited chunk) pair and, if
there are several citations, the statement against all cited chunks together.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ..llm import LLMClient, LLMError, call_structured
from ..prompts import JUDGE_SYSTEM, JudgeOut, judge_user
from ..schemas import SystemOutput
from ..verification import NLIVerifier, strip_headings


class Judge(Protocol):
    name: str

    def judge(self, question: str, claim: str, source: str) -> dict: ...


class LLMJudge:
    def __init__(self, client: LLMClient):
        self.client = client
        self.name = f"llm:{client.spec.model}"

    def judge(self, question: str, claim: str, source: str) -> dict:
        try:
            out = call_structured(self.client, None, "judge", JUDGE_SYSTEM, judge_user(question, claim, source), JudgeOut, max_tokens=1024)
        except LLMError as e:
            return {"label": "error", "rationale": str(e)}
        return {"label": out.label, "rationale": out.rationale}


class NLIJudge:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        self.nli = NLIVerifier(model_name=model_name)
        self.name = f"nli:{model_name}"

    def judge(self, question: str, claim: str, source: str) -> dict:
        v = self.nli.verify(claim, [source])
        label = {"entailed": "supported", "contradicted": "contradicted"}.get(v.label, "not_supported")
        return {"label": label, "entail": round(v.entail, 4), "contradict": round(v.contradict, 4)}


def judgment_tasks(out: SystemOutput, chunk_text: dict[str, str]) -> list[dict]:
    """Judge final statements (integer index) and statements the verifier dropped
    (index ``d<i>``, judged on their original text) so false rejections can be measured."""
    tasks = []
    items = list(enumerate(out.statements)) + [
        (f"d{i}", st.model_copy(update={"text": st.original_text or st.text})) for i, st in enumerate(out.dropped_statements)
    ]
    for si, st in items:
        cits = [c for c in st.citations if c in chunk_text]
        for c in cits:
            tasks.append({"statement_idx": si, "scope": c, "claim": st.text, "source": strip_headings(chunk_text[c])})
        if len(cits) > 1:
            joint = "\n\n".join(strip_headings(chunk_text[c]) for c in cits)
            tasks.append({"statement_idx": si, "scope": "joint", "claim": st.text, "source": joint})
    return tasks


def judge_output(out: SystemOutput, question: str, chunk_text: dict[str, str], judge: Judge, workers: int = 8) -> list[dict]:
    tasks = judgment_tasks(out, chunk_text)

    def run(t: dict) -> dict:
        res = judge.judge(question, t["claim"], t["source"])
        return {
            "qid": out.qid,
            "system": out.system,
            "noise_level": out.noise_level,
            "statement_idx": t["statement_idx"],
            "scope": t["scope"],
            "claim": t["claim"],
            "judge": judge.name,
            **res,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, tasks))
