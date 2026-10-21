"""The explorer API serves a built run (with and without predictions) from disk only."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from provseg.app.server import create_app
from provseg.runner import Experiment

from .conftest import FakeLLM
from .test_runner import route, setup  # noqa: F401  (fixture re-export)
from .test_systems import ScriptedVerifier


@pytest.fixture
def built(setup, monkeypatch):  # noqa: F811
    fake = FakeLLM(route=route)
    monkeypatch.setattr("provseg.runner.make_client", lambda spec, cache_dir, offline=False: fake)
    monkeypatch.setattr("provseg.runner.NLIVerifier", lambda *a, **k: ScriptedVerifier())
    exp = Experiment(setup)
    exp.build()
    return exp


def test_run_listing_and_questions_before_predictions(built):
    client = TestClient(create_app(built.cfg.runs_dir))
    runs = client.get("/api/runs").json()
    assert [r["name"] for r in runs] == ["t"] and runs[0]["has_predictions"] is False
    qs = client.get("/api/runs/t/questions").json()
    assert {q["qid"] for q in qs} == {"q-bleu", "q-none"}
    detail = client.get("/api/runs/t/questions/q-bleu?noise=3").json()
    assert detail["predictions"] == []
    assert sum(c["is_distractor"] for c in detail["chunks"]) == 3
    assert [c["handle"] for c in detail["chunks"]][:2] == ["P01", "P02"]
    assert any(c["is_gold"] for c in detail["chunks"])


def test_question_detail_with_predictions_and_judgments(built):
    built.predict()
    built.judge()
    client = TestClient(create_app(built.cfg.runs_dir))
    d = client.get("/api/runs/t/questions/q-bleu?noise=0").json()
    by = {p["system"]: p for p in d["predictions"]}
    assert set(by) == {"rag_bm25", "provseg"}
    prov = by["provseg"]
    assert prov["statements"][0]["verdict"] == "entailed"
    assert prov["judgments"][0]["label"] == "supported"
    assert "cache_keys" not in prov["usage"]
    cid = prov["statements"][0]["citations"][0]
    chunk = client.get(f"/api/runs/t/chunks/{cid}").json()
    assert "34.1 BLEU" in chunk["text"] and chunk["title"] == "Sparse Adapters for Low-Resource Translation"
    raw = client.get("/api/runs/t/predictions.jsonl")
    assert raw.status_code == 200 and len(raw.text.strip().splitlines()) == 8
    s = client.get("/api/runs/t/summary").json()
    assert s["prediction_counts"]["provseg@0"] == 2 and s["metrics"] is None


def test_errors_are_404(built):
    client = TestClient(create_app(built.cfg.runs_dir))
    assert client.get("/api/runs/nope/questions").status_code == 404
    assert client.get("/api/runs/t/questions/unknown").status_code == 404
    assert client.get("/api/runs/t/questions/q-bleu?noise=99").status_code == 404
    assert client.get("/api/runs/t/chunks/unknown").status_code == 404


def test_static_front_end_is_served(built):
    client = TestClient(create_app(built.cfg.runs_dir))
    page = client.get("/")
    assert page.status_code == 200 and "Provenance Explorer" in page.text
    assert client.get("/app.js").status_code == 200


def test_empty_runs_dir(tmp_path):
    client = TestClient(create_app(tmp_path / "missing"))
    assert client.get("/api/runs").json() == []
    assert json.loads(client.get("/api/health").text) == {"ok": True, "runs": 0}


def test_server_picks_up_new_predictions_without_restart(built):
    client = TestClient(create_app(built.cfg.runs_dir))
    assert client.get("/api/runs/t/questions/q-bleu?noise=0").json()["predictions"] == []
    built.predict()
    assert len(client.get("/api/runs/t/questions/q-bleu?noise=0").json()["predictions"]) == 2
