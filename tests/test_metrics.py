import math

import pytest

from provseg.evaluation.aggregate import aggregate
from provseg.evaluation.judge import judgment_tasks
from provseg.evaluation.metrics import abstention_scores, answer_f1, evidence_scores, question_metrics
from provseg.evaluation.qasper_official import normalize_answer, set_f1, token_f1
from provseg.evaluation.stats import bootstrap_ci, paired_test
from provseg.schemas import Statement, SystemOutput

GOLD_SETS = [{"doc_2001_00001_sec_02_chunk_00"}, {"doc_2001_00001_sec_03_chunk_00"}]


def test_normalize_and_token_f1_follow_squad_rules():
    assert normalize_answer("The  BLEU, score!") == "bleu score"
    assert token_f1("34.1 BLEU", "34.1") == pytest.approx(2 / 3)
    assert token_f1("the", "a") == 0.0
    assert token_f1("Unanswerable", "Unanswerable") == 1.0


def test_set_f1_edge_cases():
    assert set_f1(set(), set()) == (1.0, 1.0, 1.0)
    assert set_f1({"a"}, set()) == (0.0, 0.0, 0.0)
    assert set_f1({"a", "b"}, {"a"}) == (0.5, 1.0, pytest.approx(2 / 3))


def test_answer_f1_takes_best_reference(question_bleu):
    assert answer_f1("34.1", question_bleu) == (1.0, "extractive")


def test_evidence_scores_max_over_references_and_nan_when_unmappable(question_bleu, question_unanswerable):
    assert evidence_scores({"doc_2001_00001_sec_03_chunk_00"}, GOLD_SETS, question_bleu)[2] == 1.0
    assert evidence_scores(set(), [set(), set()], question_unanswerable) == (1.0, 1.0, 1.0)
    p, r, f = evidence_scores({"x"}, [set(), set()], question_bleu)
    assert math.isnan(f)


def _out(statements, answer="34.1 BLEU", dropped=(), raw=None, system="provseg", noise=0, qid="q-bleu"):
    return SystemOutput(
        qid=qid,
        system=system,
        noise_level=noise,
        answer=answer,
        abstained=answer == "Unanswerable",
        statements=statements,
        dropped_statements=list(dropped),
        context=[],
        context_tokens=100,
        trace={"raw_statements": raw or [s.model_dump() for s in statements]},
        usage={"calls": 3, "input_tokens": 300, "output_tokens": 60, "cost_usd": 0.01, "llm_latency_s": 1.5},
    )


def j(si, scope, label):
    return {"statement_idx": si, "scope": scope, "label": label}


def test_citation_precision_and_support_alce_style(question_bleu):
    s0 = Statement(text="a", citations=["c1"])
    s1 = Statement(text="b", citations=["c1", "c2"])
    s2 = Statement(text="c", citations=[])
    judg = [j(0, "c1", "supported"), j(1, "c1", "partially_supported"), j(1, "c2", "not_supported"), j(1, "joint", "supported")]
    m = question_metrics(_out([s0, s1, s2]), question_bleu, GOLD_SETS, judg)
    # pairs: (s0,c1) precise; (s1,c1) partial + jointly supported -> precise; (s1,c2) not -> imprecise
    assert m["citation_precision"] == pytest.approx(2 / 3)
    # s0 supported, s1 jointly supported, s2 uncited -> unsupported
    assert m["statement_support"] == pytest.approx(2 / 3)
    assert m["unsupported_claim_rate"] == pytest.approx(1 / 3)
    assert m["fully_supported_answer"] == 0.0


def test_partial_without_joint_support_is_imprecise(question_bleu):
    s = Statement(text="b", citations=["c1", "c2"])
    m = question_metrics(_out([s]), question_bleu, GOLD_SETS, [j(0, "c1", "partially_supported"), j(0, "c2", "supported"), j(0, "joint", "partially_supported")])
    assert m["citation_precision"] == 0.5 and m["statement_support"] == 0.0


def test_unjudged_prediction_has_nan_support_metrics(question_bleu):
    m = question_metrics(_out([Statement(text="a", citations=["c1"])]), question_bleu, GOLD_SETS, None)
    assert math.isnan(m["citation_precision"]) and math.isnan(m["unsupported_claim_rate"])


def test_uncited_statements_need_no_judge(question_bleu):
    m = question_metrics(_out([Statement(text="a")]), question_bleu, GOLD_SETS, None)
    assert m["unsupported_claim_rate"] == 1.0 and math.isnan(m["citation_precision"])


def test_out_of_context_distractor_and_false_rejection(question_bleu):
    st = Statement(text="a", citations=["noise00__x", "doc_2001_00001_sec_02_chunk_00"])
    raw = [{"raw_citations": ["P1", "P2", "P9"], "invalid_citations": ["P9"]}]
    dropped = Statement(text="revised", original_text="orig", citations=["c9"])
    judg = [j(0, "noise00__x", "not_supported"), j(0, "doc_2001_00001_sec_02_chunk_00", "supported"), j(0, "joint", "supported"), j("d0", "c9", "supported")]
    m = question_metrics(_out([st], dropped=[dropped], raw=raw), question_bleu, GOLD_SETS, judg)
    assert m["out_of_context_rate"] == pytest.approx(1 / 3)
    assert m["distractor_citation_rate"] == 0.5
    assert m["false_rejections"] == 1
    assert m["evidence_recall"] == 1.0 and m["evidence_precision"] == 0.5


def test_judgment_tasks_cover_pairs_joint_and_dropped():
    out = _out(
        [Statement(text="a", citations=["c1", "c2"]), Statement(text="b", citations=["missing"])],
        dropped=[Statement(text="new", original_text="old", citations=["c1"])],
    )
    tasks = judgment_tasks(out, {"c1": "§ H\n\nOne.", "c2": "Two."})
    assert [(t["statement_idx"], t["scope"]) for t in tasks] == [(0, "c1"), (0, "c2"), (0, "joint"), ("d0", "c1")]
    assert tasks[0]["source"] == "One." and tasks[2]["source"] == "One.\n\nTwo."
    assert tasks[3]["claim"] == "old"


def test_abstention_scores():
    rows = [
        {"abstained": True, "gold_unanswerable": True},
        {"abstained": True, "gold_unanswerable": False},
        {"abstained": False, "gold_unanswerable": True},
        {"abstained": False, "gold_unanswerable": False},
    ]
    s = abstention_scores(rows)
    assert s["abstain_precision"] == 0.5 and s["abstain_recall"] == 0.5 and s["abstain_f1"] == 0.5
    assert s["false_abstention_rate"] == 0.5
    assert math.isnan(abstention_scores([{"abstained": False, "gold_unanswerable": False}])["abstain_precision"])


def test_bootstrap_ci_and_paired_test():
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 0.0, 1.0, float("nan")])
    assert mean == 0.75 and lo <= mean <= hi
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    res = paired_test([1.0] * 30, [0.0] * 30)
    assert res["diff"] == 1.0 and res["p_value"] < 0.01 and res["n"] == 30
    same = paired_test([0.5] * 10, [0.5] * 10)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0
    assert paired_test([], [])["n"] == 0


def test_aggregate_shapes_and_pairing(question_bleu):
    rows = []
    for system, f1 in (("provseg", 1.0), ("rag_bm25", 0.0)):
        for k in (0, 10):
            for i in range(5):
                rows.append(
                    {
                        "qid": f"q{i}", "system": system, "noise_level": k, "error": None, "gold_unanswerable": False, "abstained": False,
                        "answer_f1": f1, "evidence_precision": 1.0, "evidence_recall": 1.0, "evidence_f1": 1.0, "citation_precision": f1,
                        "statement_support": f1, "unsupported_claim_rate": 1 - f1, "fully_supported_answer": f1, "out_of_context_rate": 0.0,
                        "distractor_citation_rate": 0.0, "n_statements": 2, "n_dropped": 0, "false_rejections": 0, "context_tokens": 100,
                        "calls": 1, "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "llm_latency_s": 0.0,
                    }
                )
    agg = aggregate(rows)
    assert len(agg["table"]) == 4
    p = next(x for x in agg["paired"] if x["metric"] == "answer_f1" and x["noise_level"] == 0)
    assert p["diff"] == 1.0 and p["n"] == 5
    rob = next(x for x in agg["robustness"] if x["system"] == "provseg" and x["metric"] == "answer_f1")
    assert rob["delta_vs_clean"] == 0.0
    assert agg["table"][0]["abstain_precision"] is None  # NaN serialised as null
