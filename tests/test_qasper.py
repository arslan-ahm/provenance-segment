import pytest

from provseg.data.qasper import normalize_paragraph, parse_row, sample_questions
from provseg.schemas import GoldReference, Question


def _ans(**kw):
    base = {"unanswerable": False, "extractive_spans": [], "yes_no": None, "free_form_answer": "", "evidence": [], "highlighted_evidence": []}
    return base | kw


ROW = {
    "id": "1234.5678",
    "title": "  A   Paper ",
    "abstract": "We study things.",
    "full_text": {"section_name": ["Intro", "Empty", None], "paragraphs": [["P one.", "  P   two. "], [], ["Tail para."]]},
    "figures_and_tables": {"caption": ["Table 1: Scores."], "file": ["t1.png"]},
    "qas": {
        "question": ["Which data?", "Is it good?", "Odd one", "Is it bad?"],
        "question_id": ["q1", "q2", "q3", "q4"],
        "answers": [
            {"answer": [_ans(extractive_spans=["SQuAD", "GLUE"], evidence=["P one."]), _ans(unanswerable=True)]},
            {"answer": [_ans(yes_no=True), _ans(free_form_answer="mostly")]},
            {"answer": [_ans()]},
            {"answer": [_ans(yes_no=False)]},
        ],
    },
}


def test_parse_row_sections_and_captions():
    doc, qs = parse_row(ROW)
    assert doc.title == "A Paper"
    assert [(s.idx, s.heading) for s in doc.sections] == [(0, "Abstract"), (1, "Intro"), (2, "Untitled"), (3, "Figures and Tables")]
    assert doc.sections[1].paragraphs == ["P one.", "P two."]


def test_parse_row_answer_types_follow_official_evaluator():
    _, qs = parse_row(ROW)
    by = {q.qid: q for q in qs}
    assert [(r.answer, r.type) for r in by["q1"].references] == [("SQuAD, GLUE", "extractive"), ("Unanswerable", "none")]
    assert [(r.answer, r.type) for r in by["q2"].references] == [("Yes", "boolean"), ("mostly", "abstractive")]
    assert [(r.answer, r.type) for r in by["q4"].references] == [("No", "boolean")]
    assert "q3" not in by  # malformed annotation skipped
    assert by["q1"].unanswerable_any and not by["q1"].unanswerable_all


def test_normalize_paragraph_strips_float_prefix():
    assert normalize_paragraph("FLOAT SELECTED: Table 1:  Scores.") == "table 1: scores."


def _q(i, doc, kind):
    ref = GoldReference(answer="Unanswerable", type="none") if kind == "u" else GoldReference(answer="x", type="extractive")
    return Question(qid=f"{kind}{i}", doc_id=doc, question="?", references=[ref])


def test_sample_questions_is_seeded_balanced_and_spread():
    qs = [_q(i, f"d{i % 7}", "a") for i in range(40)] + [_q(i, f"d{i % 7}", "u") for i in range(10)]
    s1 = sample_questions(qs, 6, 3, seed=1, max_per_doc=2)
    assert s1 == sample_questions(qs, 6, 3, seed=1, max_per_doc=2)
    assert sum(q.unanswerable_all for q in s1) == 3 and len(s1) == 9
    per_doc = {}
    for q in s1:
        per_doc[q.doc_id] = per_doc.get(q.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2
    assert sample_questions(list(reversed(qs)), 6, 3, seed=1) == s1  # input order does not matter


def test_sample_questions_raises_when_pool_too_small():
    with pytest.raises(ValueError):
        sample_questions([_q(0, "d", "a")], 2, 0, seed=0)
