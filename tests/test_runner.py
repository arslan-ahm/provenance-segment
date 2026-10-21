"""Whole pipeline on a hand-written two-paper corpus: build -> predict -> judge -> evaluate -> report.

The model, verifier and judge are scripted fakes; files go to pytest's tmp_path only.
"""

from __future__ import annotations

import csv
import json

import pytest

from provseg import corpus as corpus_mod
from provseg.chunking import ChunkingConfig, chunk_document
from provseg.config import ExperimentConfig
from provseg.evaluation import failures
from provseg.evaluation.aggregate import aggregate, per_question_rows, write_outputs
from provseg.llm import ModelSpec
from provseg.prompts import ANSWER_SYSTEM, JUDGE_SYSTEM, NOTE_SYSTEM
from provseg.report import render
from provseg.runner import Experiment
from provseg.schemas import Document, Section
from provseg.systems import SystemConfig

from .conftest import FakeLLM
from .test_systems import ScriptedVerifier


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def setup(tmp_path, paper, question_bleu, question_unanswerable):
    cfg_chunk = ChunkingConfig(min_tokens=1, overlap_tokens=0)
    other = [
        Document(doc_id=f"9000.{i:05d}", title=f"Other paper {i}", sections=[Section(idx=0, heading="Abstract", paragraphs=[f"Abstract {i}."]), Section(idx=1, heading="Method", paragraphs=[f"Unrelated method number {i} uses a parser."])])
        for i in range(6)
    ]
    proc = tmp_path / "processed"
    for split, docs, qs in (("test", [paper], [question_bleu, question_unanswerable]), ("train", other, [])):
        p = corpus_mod.paths(proc, split, cfg_chunk)
        _write(p["docs"], docs)
        _write(p["questions"], qs)
        _write(p["chunks"], [c for d in docs for c in chunk_document(d, cfg_chunk)])
    spec = ModelSpec(provider="openai_compat", model="fake", base_url="http://fake.invalid/v1")
    cfg = ExperimentConfig(
        name="t",
        data={"raw_dir": str(tmp_path / "raw"), "processed_dir": str(proc), "split": "test", "pool_split": "train"},
        sample={"n_answerable": 1, "n_unanswerable": 1, "seed": 0, "max_per_doc": 2},
        chunking=cfg_chunk,
        noise={"levels": [0, 3], "mode": "random", "seed": 1},
        models={"base": spec, "judge": spec.model_copy(update={"model": "fake-judge"})},
        systems=[SystemConfig(name="rag_bm25", kind="rag"), SystemConfig(name="provseg", kind="provseg")],
        runs_dir=str(tmp_path / "runs"),
        cache_dir=str(tmp_path / "cache"),
        question_workers=2,
        note_workers=2,
    )
    return cfg


def route(req):
    if req.system == NOTE_SYSTEM:
        rel = "high" if "34.1" in req.user else "none"
        quotes = ["Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian"] if rel == "high" else []
        return {"relevance": rel, "evidence_quotes": quotes, "facts": [], "reasoning_note": "", "uncertainty": ""}
    if req.system == ANSWER_SYSTEM:
        if "English-French" in req.user.rsplit("Question:", 1)[1]:
            return {"answer": "Unanswerable", "statements": []}
        handle = next((h for h in ("P01", "P02", "P03", "P04", "P05", "P06", "P07") if f"[{h}]" in req.user and "34.1" in req.user.split(f"[{h}]", 1)[1][:400]), "P01")
        return {"answer": "34.1 BLEU", "statements": [{"text": "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian.", "citations": [handle]}]}
    if req.system == JUDGE_SYSTEM:
        return {"label": "supported" if "34.1" in req.user.split("Claim:")[0] else "not_supported", "rationale": "scripted"}
    raise AssertionError(req.system[:50])


def test_full_pipeline(setup, monkeypatch, tmp_path):
    fake = FakeLLM(route=route)
    monkeypatch.setattr("provseg.runner.make_client", lambda spec, cache_dir, offline=False: fake)
    monkeypatch.setattr("provseg.runner.NLIVerifier", lambda *a, **k: ScriptedVerifier())
    exp = Experiment(setup)
    stats = exp.build()
    assert stats["questions"] == 2 and stats["instances"] == 4 and stats["evidence_mapping"] == {"matched": 2, "total": 2}
    inst = [json.loads(line) for line in (exp.dir / "instances.jsonl").read_text().splitlines()]
    noisy = next(i for i in inst if i["noise_level"] == 3)
    assert sum(c.startswith("noise") for c in noisy["chunk_ids"]) == 3

    r = exp.predict()
    assert r == {"tasks": 8, "skipped_done": 0, "errors": 0}
    assert exp.predict() == {"tasks": 0, "skipped_done": 8, "errors": 0}  # resumable

    j = exp.judge()
    assert j["judged"] == 8
    assert exp.judge()["judged"] == 0

    rows = per_question_rows(exp)
    assert len(rows) == 8
    bleu = [r for r in rows if r["qid"] == "q-bleu"]
    assert all(r["answer_f1"] == pytest.approx(1.0) for r in bleu)
    assert all(r["citation_precision"] == 1.0 and r["unsupported_claim_rate"] == 0.0 for r in bleu)
    assert all(r["evidence_recall"] == 1.0 for r in bleu)
    none = [r for r in rows if r["qid"] == "q-none"]
    assert all(r["abstained"] and r["answer_f1"] == 1.0 for r in none)

    agg = aggregate(rows)
    write_outputs(exp, rows, agg)
    failures.write(exp, failures.collect(exp, rows))
    report = render(exp.dir).read_text(encoding="utf-8")
    assert "## Noise level 0" in report and "## Noise level 3" in report and "provseg" in report
    with (exp.dir / "per_question.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 8
    manifest = json.loads((exp.dir / "manifest.json").read_text())
    assert manifest["config"]["name"] == "t" and manifest["prompt_version"]


def test_offline_replay_uses_cache_only(setup, tmp_path):
    from provseg.config import make_client
    from provseg.llm import LLMError, LLMRequest

    client = make_client(setup.models["base"], tmp_path / "empty-cache", offline=True)
    with pytest.raises(LLMError, match="offline replay"):
        client.complete(LLMRequest(system="s", user="u"))


def test_nli_judge_must_differ_from_verifier(setup, monkeypatch):
    setup.judge.kind = "nli"
    setup.judge.nli_model = setup.verifier.model
    exp = Experiment(setup)
    exp.build()
    with pytest.raises(ValueError, match="must differ"):
        exp.judge()
