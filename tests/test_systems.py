"""End-to-end behaviour of the four systems with a scripted model and a scripted verifier."""

from __future__ import annotations

import pytest

from provseg.chunking import ChunkingConfig, chunk_document
from provseg.llm import LLMError
from provseg.prompts import ABSTAIN, ANSWER_SYSTEM, MERGE_SYSTEM, NOTE_SYSTEM, REANSWER_SYSTEM, REVISE_SYSTEM
from provseg.schemas import Chunk
from provseg.systems import Resources, SystemConfig, canonical_handle, chunk_costs, run_system
from provseg.verification import Verdict

from .conftest import FakeLLM, handles_in

GOOD_QUOTE = "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian"
FAKE_QUOTE = "Sparse adapters achieve state-of-the-art BLEU everywhere"
DISTRACTOR_TEXT = "§ Results\n\nOur parser reaches 91.2 F1 on the Penn Treebank. Training takes four hours on TPUs."


@pytest.fixture
def chunks(paper) -> list[Chunk]:
    cs = chunk_document(paper, ChunkingConfig(min_tokens=1, overlap_tokens=0))
    distractor = Chunk(
        chunk_id="noise00__doc_9999_00001_sec_03_chunk_00",
        doc_id="9999.00001",
        section_idx=3,
        section_heading="Results",
        chunk_idx=0,
        text=DISTRACTOR_TEXT,
        spans=[(3, 0, 0)],
        n_tokens=30,
        is_distractor=True,
    )
    # P01 Abstract, P02 distractor, P03 Introduction, P04 Experiments, P05 Figures and Tables
    return [cs[0], distractor, cs[1], cs[2], cs[3]]


class ScriptedVerifier:
    name = "scripted"

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def verify(self, claim, sources, meter=None):
        self.calls.append((claim, sources))
        ok = "34.1" in claim and any("34.1" in s for s in sources)
        return Verdict(label="entailed" if ok else "not_enough_evidence", entail=0.9 if ok else 0.1, contradict=0.0, detail={})


def default_answer():
    return {
        "answer": "34.1 BLEU",
        "statements": [
            {"text": "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian.", "citations": ["[P04]"]},
            {"text": "They were trained on TPUs.", "citations": ["P4", "P02"]},
            {"text": "The method is new.", "citations": ["P77"]},
        ],
    }


def router(answer=None, revise=None, merge=None):
    def route(req):
        if req.system == NOTE_SYSTEM:
            if "34.1 BLEU" in req.user:
                return {"relevance": "high", "evidence_quotes": [GOOD_QUOTE, FAKE_QUOTE], "facts": ["34.1 BLEU vs 33.2"], "reasoning_note": "direct answer", "uncertainty": ""}
            if "Penn Treebank" in req.user:
                return {"relevance": "medium", "evidence_quotes": ["Our parser reaches 91.2 F1 on the Penn Treebank"], "facts": ["91.2 F1"], "reasoning_note": "a score", "uncertainty": ""}
            return {"relevance": "low", "evidence_quotes": [], "facts": [], "reasoning_note": "", "uncertainty": "not about BLEU"}
        if req.system.startswith(MERGE_SYSTEM.split("{")[0]):
            return merge
        if req.system == ANSWER_SYSTEM:
            return answer or default_answer()
        if req.system == REVISE_SYSTEM:
            return revise or {"drop": True, "statement": ""}
        if req.system == REANSWER_SYSTEM:
            return {"answer": "34.1 BLEU (verified)"}
        raise AssertionError(f"unexpected prompt: {req.system[:60]}")

    return route


def run(kind, question, chunks, fake, verifier=None, **cfg):
    sc = SystemConfig(name=kind, kind=kind, **cfg)
    return run_system(sc, question, chunks, noise_level=1, res=Resources(base=fake, verifier=verifier, note_workers=2))


def test_handles_hide_chunk_ids_and_distractor_labels(question_bleu, chunks):
    for kind in ("direct", "rag", "segment", "provseg"):
        fake = FakeLLM(route=router())
        run(kind, question_bleu, chunks, fake, verifier=ScriptedVerifier())
        for r in fake.requests:
            assert "noise" not in r.user and "doc_2001" not in r.user and "doc_9999" not in r.user


def test_direct_takes_document_order_within_budget(question_bleu, chunks):
    fake = FakeLLM(route=router())
    costs = chunk_costs(chunks)
    budget = costs[0] + costs[1] + costs[2]
    out = run("direct", question_bleu, chunks, fake, context_budget=budget)
    prompt = fake.stage_requests(ANSWER_SYSTEM)[0].user
    assert handles_in(prompt) == ["P01", "P02", "P03"]
    assert [c.handle for c in out.context] == ["P01", "P02", "P03"]
    assert out.context_tokens <= budget
    # P04 is cited but was never shown: it stays unresolved for baselines and is counted as out of context.
    st = out.statements[0]
    assert st.citations == [] and st.invalid_citations == ["[P04]"]


def test_rag_ranks_the_answer_chunk_first(question_bleu, chunks):
    fake = FakeLLM(route=router())
    out = run("rag", question_bleu, chunks, fake, context_budget=chunk_costs(chunks)[3])
    assert [c.handle for c in out.context] == ["P04"]
    assert out.trace["ranking"][0][0] == "P04"
    assert out.statements[0].citations == ["doc_2001_00001_sec_02_chunk_00"]
    # 'P4' is canonicalised to P04 and resolved; P02 was not in this context.
    assert out.statements[1].citations == ["doc_2001_00001_sec_02_chunk_00"] and out.statements[1].invalid_citations == ["P02"]
    assert out.statements[2].invalid_citations == ["P77"]


def test_segment_keeps_unanchored_quotes_but_provseg_removes_them(question_bleu, chunks):
    seg = run("segment", question_bleu, chunks, FakeLLM(route=router()))
    assert seg.trace["quote_stats"] == {"quotes": 3, "unanchored": 1}
    seg_ctx = "\n".join(i.text for i in seg.context)
    assert FAKE_QUOTE in seg_ctx and GOOD_QUOTE in seg_ctx

    prov = run("provseg", question_bleu, chunks, FakeLLM(route=router()), verifier=ScriptedVerifier())
    prov_ctx = "\n".join(i.text for i in prov.context)
    assert FAKE_QUOTE not in prov_ctx and GOOD_QUOTE in prov_ctx
    anchored = [q for n in prov.trace["notes"] for q in n["evidence"] if q["verified"] and q["handle"] == "P04"]
    assert len(anchored) == 1
    start, end = anchored[0]["start"], anchored[0]["end"]
    assert chunks[3].text[start:end] == GOOD_QUOTE


def test_filter_keeps_high_and_medium_only(question_bleu, chunks):
    out = run("segment", question_bleu, chunks, FakeLLM(route=router()))
    assert out.trace["filter"] == {"P01": "remove", "P02": "maybe", "P03": "remove", "P04": "keep", "P05": "remove"}
    assert [i.chunk_ids for i in out.context] == [["noise00__doc_9999_00001_sec_03_chunk_00"], ["doc_2001_00001_sec_02_chunk_00"]]


def test_provseg_verification_drops_unsupported_and_reanswers(question_bleu, chunks):
    fake = FakeLLM(route=router())
    verifier = ScriptedVerifier()
    out = run("provseg", question_bleu, chunks, fake, verifier=verifier)
    assert [s.text for s in out.statements] == ["Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian."]
    assert out.statements[0].verdict == "entailed" and out.statements[0].action == "kept"
    assert [s.text for s in out.dropped_statements] == ["They were trained on TPUs.", "The method is new."]
    # The uncited statement is dropped without a revise call; the TPU one is revised (and the model drops it).
    assert len(fake.stage_requests(REVISE_SYSTEM)) == 1
    assert out.answer == "34.1 BLEU (verified)" and not out.abstained
    # Verification reads the ORIGINAL chunk text of the cited chunks, never the notes.
    claim, sources = verifier.calls[0]
    assert sources == [chunks[3].text]
    tpu_sources = next(s for c, s in verifier.calls if "TPUs" in c)
    assert chunks[1].text in tpu_sources and chunks[3].text in tpu_sources
    # The raw (pre-stripping) statements are kept for the out-of-context metric.
    assert out.trace["raw_statements"][2]["invalid_citations"] == ["P77"]


def test_provseg_revision_that_verifies_is_kept(question_bleu, chunks):
    revise = {"drop": False, "statement": "Training details are not 34.1 related but the score is 34.1 BLEU."}
    out = run("provseg", question_bleu, chunks, FakeLLM(route=router(revise=revise)), verifier=ScriptedVerifier())
    revised = [s for s in out.statements if s.action == "revised"]
    assert len(revised) == 1 and revised[0].original_text == "They were trained on TPUs."


def test_provseg_abstains_when_nothing_survives(question_bleu, chunks):
    answer = {"answer": "33 BLEU", "statements": [{"text": "It gets 33 BLEU.", "citations": ["P04"]}]}
    fake = FakeLLM(route=router(answer=answer))
    out = run("provseg", question_bleu, chunks, fake, verifier=ScriptedVerifier())
    assert out.answer == ABSTAIN and out.abstained and out.statements == []
    assert fake.stage_requests(REANSWER_SYSTEM) == []


def test_provseg_abstains_on_answer_without_statements(question_bleu, chunks):
    out = run("provseg", question_bleu, chunks, FakeLLM(route=router(answer={"answer": "Yes", "statements": []})), verifier=ScriptedVerifier())
    assert out.abstained and out.trace["abstained_no_statements"]


def test_provseg_without_verification_keeps_statements_but_strips_bad_citations(question_bleu, chunks):
    out = run("provseg", question_bleu, chunks, FakeLLM(route=router()), verify=False)
    assert len(out.statements) == 3 and all(s.invalid_citations == [] for s in out.statements)
    assert out.statements[2].citations == []


def test_merge_is_triggered_and_provenance_checked(question_bleu, chunks):
    merge = {
        "notes": [
            {
                "evidence": [
                    {"handle": "P04", "quote": GOOD_QUOTE},
                    {"handle": "P04", "quote": "a quote the merger invented out of thin air"},
                    {"handle": "P03", "quote": "Adapters add small trainable modules to a frozen network"},
                ],
                "facts": [{"text": "34.1 beats 33.2", "handles": ["P04"]}, {"text": "made up", "handles": ["P09"]}],
                "reasoning_note": "merged",
            }
        ]
    }
    fake = FakeLLM(route=router(merge=merge))
    out = run("provseg", question_bleu, chunks, fake, verifier=ScriptedVerifier(), context_budget=60, merge_batch_tokens=10_000)
    rounds = out.trace["merge_rounds"]
    assert rounds and rounds[0]["batches"] == 1
    # P03 is a real quote but from a chunk that was not in the merged batch -> dropped; invented quote -> dropped.
    assert rounds[0]["dropped_quotes"] == 2 and rounds[0]["dropped_fact_handles"] == 1
    final = out.trace["final_notes"]
    if final:  # budget is tiny, the deterministic fallback may truncate everything
        assert [q["text"] for q in final[0]["evidence"]] == [GOOD_QUOTE]


def test_errors_become_error_outputs(question_bleu, chunks):
    def boom(req):
        raise LLMError("provider down")

    out = run("rag", question_bleu, chunks, FakeLLM(route=boom))
    assert out.error == "provider down" and out.answer == "" and not out.abstained


@pytest.mark.parametrize("raw,expected", [("P04", "P04"), ("[P4]", "P04"), ("p 12", "P12"), ("P004", "P04"), ("chunk", None), ("", None)])
def test_canonical_handle(raw, expected):
    assert canonical_handle(raw) == expected


def test_usage_is_accounted_per_stage(question_bleu, chunks):
    out = run("provseg", question_bleu, chunks, FakeLLM(route=router()), verifier=ScriptedVerifier())
    stages = out.usage["by_stage"]
    assert stages["note"]["calls"] == len(chunks)
    assert stages["answer"]["calls"] == 1 and stages["revise"]["calls"] == 1 and stages["reanswer"]["calls"] == 1
    assert out.usage["calls"] == len(chunks) + 3
