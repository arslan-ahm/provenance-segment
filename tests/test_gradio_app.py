"""Gradio Space: HTML renderers (escaping, empty states, highlights), the handlers behind the
UI, and the GPU-to-CPU fallback. The NLI call is a fake; no model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from provseg.app import gradio_app as g
from provseg.app.live import LiveError, with_cpu_fallback
from provseg.app.server import RunStore

from .test_app import built  # noqa: F401  (fixture re-export)
from .test_runner import setup  # noqa: F401

pytest.importorskip("gradio")

EVIL = '<script>alert("x")</script>'


def detail(**over) -> dict:
    d = {
        "qid": "q1",
        "question": f"What BLEU? {EVIL}",
        "doc_id": "2001.00001",
        "title": "Sparse <b>Adapters</b>",
        "unanswerable": False,
        "references": [
            {"answer": "34.1 BLEU", "type": "extractive", "evidence": [], "highlighted_evidence": ["Sparse adapters reach 34.1 BLEU on WMT16."]},
            {"answer": "34.1 BLEU", "type": "extractive", "evidence": [], "highlighted_evidence": ["short"]},
            {"answer": "Unanswerable", "type": "none", "evidence": [], "highlighted_evidence": []},
        ],
        "gold_chunk_ids": ["c2"],
        "noise_level": 1,
        "noise_levels": [0, 1],
        "chunks": [
            {"handle": "P01", "chunk_id": "c1", "heading": "Intro", "is_distractor": False, "is_gold": False, "n_tokens": 10},
            {"handle": "P02", "chunk_id": "n1", "heading": "Noise", "is_distractor": True, "is_gold": False, "n_tokens": 10},
            {"handle": "P03", "chunk_id": "c2", "heading": "Results", "is_distractor": False, "is_gold": True, "n_tokens": 10},
        ],
        "predictions": [],
        "computed_contexts": {
            "budget": 2048,
            "direct": {"chunk_ids": ["c1", "n1"], "tokens": 100, "distractor_tokens": 50, "gold_in_context": 0, "gold_total": 1},
            "rag_bm25": {"chunk_ids": ["c2", "n1"], "tokens": 100, "distractor_tokens": 25, "gold_in_context": 1, "gold_total": 1, "scores": {"c1": 0.1, "n1": 0.5, "c2": 2.0}},
        },
    }
    return d | over


def test_render_question_escapes_and_shows_computed_contexts():
    html = g.render_question(detail())
    assert EVIL not in html and "&lt;script&gt;" in html and "&lt;b&gt;Adapters" in html
    assert html.count('class="strip-row"') == 3  # document + BM25 + first chunks
    assert "BM25 fits 1 of 3" not in html and "BM25 fits 1 of 1 gold chunks (distractors take 25% of it)" in html
    assert "taking the first chunks fits 0" in html and "2,048-token" in html
    assert html.count("cell distractor") == 3 + 1  # one per strip row, plus the legend swatch
    assert "unanswerable" not in html


def test_render_question_hides_computed_rows_for_systems_that_ran():
    pred = {"system": "rag_bm25", "context": [{"chunk_ids": ["c2"]}]}
    html = g.render_question(detail(predictions=[pred], unanswerable=True))
    names = [x.split("</div>")[0] for x in html.split('<div class="name">')[1:]]
    assert names == ["Document", "rag_bm25", "First chunks"]
    assert "every annotator: unanswerable" in html


def test_mark_text_highlights_exact_offsets_and_escapes():
    text = "§ Results\n\nWe get <34.1> BLEU. Done."
    start = text.index("<34.1>")
    html = g.mark_text(text, [(start, start + 6)])
    assert '<mark>&lt;34.1&gt;</mark>' in html
    assert '<span class="sec">Results</span>' in html
    assert g.mark_text("plain", []) == '<div class="source">plain</div>'


def check_result(**over):
    r = {
        "claim": "x",
        "nli": {"label": "entailed", "entail": 0.951, "contradict": 0.0, "joint": None, "model": "nli-model", "entail_threshold": 0.5, "window_sentences": [1, 3]},
        "sources": [
            {"chunk_id": "c2", "handle": "P03", "is_distractor": False, "quote": {"found": True, "start": 0, "end": 6, "elided": False},
             "nli": {"entail": 0.951, "contradict": 0.0, "window": "Sparse adapters", "window_start": 0, "window_end": 15}},
            {"chunk_id": "n1", "handle": "P02", "is_distractor": True, "quote": {"found": False, "start": None, "end": None, "elided": False},
             "nli": {"entail": 0.01, "contradict": 0.9, "window": f"bad {EVIL}", "window_start": 4, "window_end": 8}},
        ],
    }
    return r | over


def test_render_check_verdict_quotes_windows_and_passages():
    html = g.render_check(check_result(), {"c2": "Sparse adapters reach 34.1 BLEU.", "n1": "Some noise text."})
    assert "Supported" in html and "entailment 95%" in html and "nli-model" in html
    assert "verbatim at characters 0 to 6" in html and "<mark>Sparse</mark>" in html
    assert "not verbatim" in html and "distractor" in html and "Some<mark> noi</mark>se text." in html
    assert EVIL not in html and "decided by the passages together" not in html and "Ran on CPU" not in html


def test_render_check_joint_decision_fallback_note_and_quote_only():
    r = check_result()
    r["nli"]["joint"] = {"entail": 0.97, "contradict": 0.0}
    r["fallback"] = "Ran on CPU because the GPU was not available (ZeroGPU quota exceeded: limit)."
    html = g.render_check(r, {})
    assert "decided by the passages together" in html and "Ran on CPU because" in html
    q = g.render_check(check_result(nli=None, sources=[{**check_result()["sources"][0], "nli": None}]), {"c2": "Sparse adapters"})
    assert "Quote search only" in q and "entailment" not in q


def test_render_answers_empty_and_full():
    assert "No model answers" in g.render_answers(detail())
    pred = {
        "system": "provseg", "kind": "provseg", "answer": f"34.1 {EVIL}", "abstained": False, "error": None, "context": [],
        "statements": [
            {"text": "Adapters reach 34.1 BLEU.", "citations": ["c2"], "invalid_citations": [], "verdict": "entailed"},
            {"text": "Uses noise.", "citations": ["n1"], "invalid_citations": ["P77"], "verdict": "not_enough_evidence"},
            {"text": "Uncited.", "citations": [], "invalid_citations": [], "verdict": "unverified"},
        ],
        "dropped_statements": [{"text": "Dropped claim.", "citations": ["c1"], "invalid_citations": [], "verdict": "contradicted"}],
        "judgments": [{"statement_idx": 0, "scope": "c2", "label": "supported"}, {"statement_idx": "d0", "scope": "c1", "label": "not_supported"}],
    }
    html = g.render_answers(detail(predictions=[pred]))
    assert EVIL not in html and "34.1 &lt;script&gt;" in html
    assert '<span class="cite">P03</span>' in html and '<span class="cite dis">P02</span>' in html and "P77" in html
    assert "verifier: supported" in html and "judge: supported" in html and "no citation" in html
    assert 'class="stmt dropped"' in html and "removed by verifier" in html and "judge: not supported" in html
    abst = g.render_answers(detail(predictions=[{**pred, "abstained": True, "statements": [], "dropped_statements": []}]))
    assert "Abstained" in abst


def test_render_results_with_and_without_metrics(tmp_path):
    (tmp_path / "diagnostics").mkdir()
    cell = {"mean": 0.776, "low": 0.692, "high": 0.852}
    (tmp_path / "diagnostics" / "retrieval.json").write_text(json.dumps({"context_budget": 2048, "table": [
        {"method": "bm25", "noise_level": 0, "gold_recall_at_budget": cell, "all_gold_in_context": cell, "distractor_token_share": cell}]}))
    (tmp_path / "diagnostics" / "verifier_sanity.json").write_text(json.dumps({"summary": {"n": 60, "gold_entailed_rate": 0.65, "distractor_entailed_rate": 0.0167}}))
    html = g.render_results(tmp_path)
    assert "77.6" in html and "69.2 to 85.2" in html and "65.0" in html and "not run yet" in html
    (tmp_path / "metrics.json").write_text(json.dumps({"table": [{"system": "provseg", "noise_level": 50, "answer_f1": cell}]}))
    assert "Main comparison</h3>" in g.render_results(tmp_path) and "not run yet" not in g.render_results(tmp_path)


def test_presets_and_default_passages():
    d = detail()
    assert g.claim_presets(d) == [("Evidence: Sparse adapters reach 34.1 BLEU on WMT16.", "Sparse adapters reach 34.1 BLEU on WMT16."), ("Answer: 34.1 BLEU", "34.1 BLEU")]
    assert g.default_passages(d) == ["c2"]
    assert g.default_passages(detail(gold_chunk_ids=[])) == ["c2"]  # top BM25 chunk when there is no gold
    assert g.passage_choices(d)[1] == ("P02  Noise  (distractor)", "n1")


# ---------- handlers on a real (fixture) run ----------


def test_handlers_load_and_check(built):  # noqa: F811
    calls = []

    def fake_check(claim, chunks, handles, nli):
        calls.append((claim, [c.chunk_id for c in chunks], nli))
        if not claim.strip():
            raise LiveError("claim is empty")
        return {"claim": claim, "nli": None, "sources": [{"chunk_id": c.chunk_id, "handle": handles[c.chunk_id], "is_distractor": c.is_distractor,
                                                          "quote": {"found": False, "start": None, "end": None, "elided": False}} for c in chunks]}

    exp = RunStore(Path(built.cfg.runs_dir)).get("t")  # loaded exactly as the Space loads it
    load, run_check = g.make_handlers(exp, fake_check)
    view, preset, claim, passages, result, answers = load("q-bleu", 3)
    assert "What BLEU score" in view and result == "" and "No model answers" in answers
    assert passages.value == built_gold(exp) and passages.value
    noisy = [c for c in exp.instances[("q-bleu", 3)] if exp.chunk_by_id[c].is_distractor]
    html = run_check("q-bleu", 3, "a claim", passages.value + noisy[:1] + ["not-in-instance"], True)
    assert calls[-1] == ("a claim", passages.value + noisy[:1], True)  # foreign ids are dropped before the check
    assert "Quote search only" in html
    assert "claim is empty" in run_check("q-bleu", 3, " ", passages.value, False)
    assert "Pick a question first" in run_check(None, 3, "x", [], True)
    bad = load("nope", 3)
    assert "unknown question" in bad[0]

    def broken(*a):
        raise RuntimeError("ZeroGPU worker crashed")

    _, run_broken = g.make_handlers(exp, broken)
    assert "The check could not run: ZeroGPU worker crashed" in run_broken("q-bleu", 3, "x", passages.value, True)


def built_gold(exp):
    return sorted(set().union(*exp.gold["q-bleu"]))[:4]


def test_build_demo_constructs(built):  # noqa: F811
    demo, kw = g.build_demo(built.cfg.runs_dir, "t", lambda *a: {})
    assert set(kw) == {"theme", "css"} and demo is not None


# ---------- GPU fallback ----------


class QuotaError(Exception):
    title = "ZeroGPU quota exceeded"


def test_fallback_runs_cpu_and_says_why():
    r = with_cpu_fallback(lambda: (_ for _ in ()).throw(QuotaError("You have exceeded your GPU quota")), lambda: {"nli": {"label": "entailed"}})
    assert r["nli"]["label"] == "entailed"
    assert r["fallback"] == "Ran on CPU because the GPU was not available (ZeroGPU quota exceeded: You have exceeded your GPU quota)."
    r2 = with_cpu_fallback(lambda: (_ for _ in ()).throw(RuntimeError("no device")), lambda: {})
    assert "(RuntimeError: no device)" in r2["fallback"]
    assert with_cpu_fallback(lambda: {"gpu": True}, lambda: {"cpu": True}) == {"gpu": True}
    with pytest.raises(LiveError):
        with_cpu_fallback(lambda: (_ for _ in ()).throw(LiveError("bad input")), lambda: {})
