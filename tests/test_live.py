"""Model-free live checks of the explorer: budgeted baseline contexts, featured questions,
quote anchoring and the NLI claim check (cross-encoder faked at its boundary)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from provseg.app import live
from provseg.app.live import MAX_CLAIM_CHARS, ClaimChecker, LiveError, budget_contexts, featured
from provseg.app.server import create_app
from provseg.schemas import Chunk, GoldReference, Question
from provseg.systems import chunk_costs
from provseg.verification import NLIVerifier

from .test_app import built  # noqa: F401  (fixture re-export)
from .test_runner import setup  # noqa: F401


def chunk(cid: str, text: str, distractor: bool = False, heading: str = "Results") -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d" if not distractor else "x", section_idx=0, section_heading=heading, chunk_idx=0, text=text, spans=[(0, 0, 0)], is_distractor=distractor)


class FakeCrossEncoder:
    """Entailment logit is high when every word of the claim occurs in the premise window."""

    class model:  # noqa: N801  (mirrors CrossEncoder.model.config.id2label)
        class config:  # noqa: N801
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls.append(list(pairs))
        out = []
        for premise, claim in pairs:
            words = [w.strip(".,").lower() for w in claim.split()]
            ok = all(w in premise.lower() for w in words)
            out.append([-4.0, 4.0, 0.0] if ok else [-2.0, -3.0, 3.0])
        return np.array(out)


@pytest.fixture
def fake_ce(monkeypatch):
    ce = FakeCrossEncoder()
    monkeypatch.setattr("provseg.verification._cross_encoder", lambda name: ce)
    return ce


# ---------- featured ----------


def q(qid: str, unanswerable: bool = False) -> Question:
    ref = GoldReference(answer="Unanswerable", type="none") if unanswerable else GoldReference(answer="x", type="extractive")
    return Question(qid=qid, doc_id="d", question=f"question {qid}?", references=[ref])


def rows(qid: str, by_noise: dict[int, float], method: str = "bm25") -> list[dict]:
    return [{"qid": qid, "method": method, "noise_level": k, "all_gold_in_context": v} for k, v in by_noise.items()]


def test_featured_picks_noise_losses_then_unanswerable_in_stable_order():
    questions = {k: q(k) for k in ("a", "b", "c", "d", "e", "f")} | {"u2": q("u2", True), "u1": q("u1", True), "u3": q("u3", True)}
    r = (
        rows("f", {0: 1.0, 50: 0.0})
        + rows("b", {0: 1.0, 50: 0.0})
        + rows("a", {0: 1.0, 50: 1.0})  # never lost
        + rows("c", {0: 0.0, 50: 0.0})  # never had it
        + rows("d", {0: 1.0, 50: 0.0}, method="direct")  # other method does not count
        + rows("e", {0: 1.0})  # no noise-50 row
        + rows("gone", {0: 1.0, 50: 0.0})  # not in this run's questions
    )
    picks = featured(r, questions, n_answerable=4, n_unanswerable=2)
    assert [p["qid"] for p in picks] == ["b", "f", "u1", "u2"]
    assert "but not with 50 distractors" in picks[0]["reason"] and "abstain" in picks[2]["reason"]


def test_featured_respects_limits_and_empty_rows():
    questions = {"u1": q("u1", True)}
    assert [p["qid"] for p in featured([], questions)] == ["u1"]
    questions |= {k: q(k) for k in "abcdef"}
    r = [x for k in "abcdef" for x in rows(k, {0: 1.0, 50: 0.0})]
    assert [p["qid"] for p in featured(r, questions, n_answerable=3, n_unanswerable=0)] == ["a", "b", "c"]


# ---------- budgeted contexts ----------


def test_budget_contexts_match_direct_and_bm25_under_the_same_budget():
    chunks = [
        chunk("c0", "§ Intro\n\nWe study translation for many languages in general terms."),
        chunk("c1", "§ Noise\n\nA parser for chemistry patents.", distractor=True),
        chunk("c2", "§ Results\n\nSparse adapters reach 34.1 BLEU on WMT16 English-Romanian."),
    ]
    costs = chunk_costs(chunks)
    budget = costs[0] + costs[1]  # room for exactly the first two chunks in document order
    out = budget_contexts("What BLEU do sparse adapters reach on WMT16?", chunks, budget, {"c2"})
    assert out["budget"] == budget
    assert out["direct"]["chunk_ids"] == ["c0", "c1"]
    assert out["direct"]["tokens"] == costs[0] + costs[1] and out["direct"]["distractor_tokens"] == costs[1]
    assert out["direct"]["gold_in_context"] == 0 and out["direct"]["gold_total"] == 1
    bm = out["rag_bm25"]
    assert "c2" in bm["chunk_ids"] and bm["gold_in_context"] == 1
    assert bm["tokens"] <= budget
    assert bm["chunk_ids"] == sorted(bm["chunk_ids"], key=["c0", "c1", "c2"].index)  # shown in document order
    assert max(bm["scores"], key=bm["scores"].get) == "c2"


def test_budget_contexts_with_budget_too_small_for_anything():
    out = budget_contexts("q", [chunk("c0", "§ A\n\nSome text here.")], 1, set())
    assert out["direct"]["chunk_ids"] == [] and out["rag_bm25"]["tokens"] == 0 and out["direct"]["gold_total"] == 0


# ---------- claim checker ----------

SRC = chunk("c2", "§ Results\n\nWe train for three days. Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian. Full fine-tuning reaches 33.2 BLEU.")
NOISE = chunk("n1", "§ Method\n\nA parser for chemistry patents is trained on claims.", distractor=True)


def make_checker():
    return ClaimChecker("fake-nli", 0.5, 0.5, (1, 3))


def test_check_anchors_quotes_at_exact_offsets_without_nli(fake_ce):
    claim = "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian."
    r = make_checker().check(claim, [SRC, NOISE], {"c2": "P03", "n1": "P02"}, nli=False)
    assert r["nli"] is None and fake_ce.calls == []
    s, n = r["sources"]
    assert s["handle"] == "P03" and s["quote"]["found"] and not s["quote"]["elided"]
    assert SRC.text[s["quote"]["start"] : s["quote"]["end"]] == "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian."
    assert n["quote"] == {"found": False, "start": None, "end": None, "elided": False} and n["is_distractor"]


def test_check_runs_the_verifier_and_points_at_the_best_window(fake_ce):
    r = make_checker().check("Sparse adapters reach 34.1 BLEU", [SRC, NOISE], {})
    assert r["nli"]["label"] == "entailed" and r["nli"]["entail"] > 0.9
    assert r["nli"]["window_sentences"] == [1, 3] and r["nli"]["model"] == "fake-nli"
    best = r["sources"][0]["nli"]
    assert best["window"] == "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian."  # the 1-sentence window wins the argmax first
    assert SRC.text[best["window_start"] : best["window_end"]] == best["window"]
    assert r["sources"][1]["nli"]["entail"] < 0.1
    assert r["nli"]["joint"] is not None  # two sources: best windows are also scored jointly


def test_check_rejects_unsupported_claims(fake_ce):
    r = make_checker().check("Sparse adapters reach 40 BLEU", [SRC], {})
    assert r["nli"]["label"] == "not_enough_evidence" and r["sources"][0]["quote"]["found"] is False


@pytest.mark.parametrize(
    "claim,sources,msg",
    [("   ", [SRC], "empty"), ("x" * (MAX_CLAIM_CHARS + 1), [SRC], "longer"), ("a claim", [], "at least one"), ("a claim", [SRC] * 5, "at most")],
)
def test_check_validates_input(claim, sources, msg):
    with pytest.raises(LiveError, match=msg):
        make_checker().check(claim, sources, {})


def test_verifier_output_unchanged_unless_windows_requested(fake_ce):
    v = NLIVerifier("fake-nli", window_sentences=(1, 3))
    plain = v.verify("Sparse adapters reach 34.1 BLEU", [SRC.text])
    assert set(plain.detail["per_source"][0]) == {"entail", "contradict"}
    rich = v.verify("Sparse adapters reach 34.1 BLEU", [SRC.text], with_windows=True)
    assert rich.label == plain.label and rich.entail == plain.entail and "window" in rich.detail["per_source"][0]


# ---------- API ----------


def test_api_question_detail_has_computed_contexts(built):  # noqa: F811
    client = TestClient(create_app(built.cfg.runs_dir))
    d = client.get("/api/runs/t/questions/q-bleu?noise=3").json()
    cc = d["computed_contexts"]
    ids = [c["chunk_id"] for c in d["chunks"]]
    assert cc["budget"] == built.cfg.context_budget
    for name in ("direct", "rag_bm25"):
        assert set(cc[name]["chunk_ids"]) <= set(ids) and cc[name]["tokens"] <= cc["budget"]
    assert cc["rag_bm25"]["gold_total"] == len(d["gold_chunk_ids"])


def test_api_featured_without_diagnostics_lists_only_unanswerable(built):  # noqa: F811
    client = TestClient(create_app(built.cfg.runs_dir))
    f = client.get("/api/runs/t/featured").json()
    assert [x["qid"] for x in f] == ["q-none"] and f[0]["unanswerable"] is True and f[0]["title"]


def test_api_check_end_to_end(built, fake_ce, monkeypatch):  # noqa: F811
    monkeypatch.setattr(live, "ClaimChecker", ClaimChecker)
    monkeypatch.setattr("provseg.app.server.nli_available", lambda: True)  # CI installs no `nli` extra; the encoder is faked
    client = TestClient(create_app(built.cfg.runs_dir))
    d = client.get("/api/runs/t/questions/q-bleu?noise=3").json()
    gold = d["gold_chunk_ids"][0]
    body = {"qid": "q-bleu", "noise": 3, "claim": "Sparse adapters reach 34.1 BLEU", "chunk_ids": [gold]}
    r = client.post("/api/runs/t/check", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["nli"]["label"] == "entailed" and out["sources"][0]["handle"] == next(c["handle"] for c in d["chunks"] if c["chunk_id"] == gold)
    assert out["sources"][0]["quote"]["found"]

    assert client.post("/api/runs/t/check", json={**body, "chunk_ids": ["not-in-instance"]}).status_code == 400
    assert client.post("/api/runs/t/check", json={**body, "qid": "nope"}).status_code == 404
    assert client.post("/api/runs/t/check", json={**body, "claim": " "}).status_code == 400
    assert client.post("/api/runs/t/check", json={**body, "chunk_ids": []}).status_code == 400

    monkeypatch.setattr("provseg.app.server.nli_available", lambda: False)
    assert client.get("/api/health").json()["live_nli"] is False
    assert client.post("/api/runs/t/check", json=body).status_code == 503
    no_nli = client.post("/api/runs/t/check", json={**body, "nli": False})
    assert no_nli.status_code == 200 and no_nli.json()["nli"] is None


def test_nli_switch_reads_env_and_package(monkeypatch):
    from provseg.app import server

    monkeypatch.setattr(server.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("PROVSEG_LIVE_NLI", "0")
    assert server.nli_available() is False
    monkeypatch.delenv("PROVSEG_LIVE_NLI")
    assert server.nli_available() is True
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda name: None)
    assert server.nli_available() is False
