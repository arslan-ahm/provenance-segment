"""QASPER (Dasigi et al., NAACL 2021; CC-BY-4.0) loader.

Reads the Hugging Face parquet export, keeps every paragraph with its section, adds
figure/table captions as a final section (QASPER marks float evidence with
``FLOAT SELECTED``) and converts answers into the reference format used by the
official QASPER evaluator.
"""

from __future__ import annotations

import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ..schemas import Document, GoldReference, Question, Section

PARQUET_URL = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"
SPLITS = ("train", "validation", "test")
FLOAT_PREFIX = "FLOAT SELECTED: "
FLOAT_SECTION = "Figures and Tables"


def download(raw_dir: str | Path, splits=SPLITS) -> list[Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in splits:
        p = raw_dir / f"qasper_{split}.parquet"
        if not p.exists():
            urllib.request.urlretrieve(PARQUET_URL.format(split=split), p)
        paths.append(p)
    return paths


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _as_list(x) -> list:
    return [] if x is None else list(x)


def _reference(ans: dict) -> GoldReference | None:
    evidence = [_clean(e) for e in _as_list(ans.get("evidence")) if _clean(e)]
    highlighted = [_clean(e) for e in _as_list(ans.get("highlighted_evidence")) if _clean(e)]
    if ans.get("unanswerable"):
        return GoldReference(answer="Unanswerable", type="none", evidence=[], highlighted_evidence=[])
    spans = [s for s in _as_list(ans.get("extractive_spans")) if s]
    if spans:
        return GoldReference(answer=", ".join(spans), type="extractive", evidence=evidence, highlighted_evidence=highlighted)
    if ans.get("free_form_answer"):
        return GoldReference(answer=ans["free_form_answer"], type="abstractive", evidence=evidence, highlighted_evidence=highlighted)
    if ans.get("yes_no") is True:
        return GoldReference(answer="Yes", type="boolean", evidence=evidence, highlighted_evidence=highlighted)
    if ans.get("yes_no") is False:
        return GoldReference(answer="No", type="boolean", evidence=evidence, highlighted_evidence=highlighted)
    return None  # malformed annotation; the official evaluator raises here, we skip it


def parse_row(row) -> tuple[Document, list[Question]]:
    doc_id = str(row["id"])
    sections = [Section(idx=0, heading="Abstract", paragraphs=[_clean(row["abstract"])])]
    ft = row["full_text"]
    for name, paras in zip(_as_list(ft["section_name"]), _as_list(ft["paragraphs"])):
        cleaned = [_clean(p) for p in _as_list(paras) if _clean(p)]
        if cleaned:
            sections.append(Section(idx=len(sections), heading=_clean(name) or "Untitled", paragraphs=cleaned))
    captions = [_clean(c) for c in _as_list((row.get("figures_and_tables") or {}).get("caption")) if _clean(c)]
    if captions:
        sections.append(Section(idx=len(sections), heading=FLOAT_SECTION, paragraphs=captions))
    doc = Document(doc_id=doc_id, title=_clean(row["title"]), sections=sections)

    questions = []
    qas = row["qas"]
    for q, qid, answers in zip(_as_list(qas["question"]), _as_list(qas["question_id"]), _as_list(qas["answers"])):
        refs = [r for a in _as_list(answers["answer"]) if (r := _reference(a)) is not None]
        if refs:
            questions.append(Question(qid=str(qid), doc_id=doc_id, question=_clean(q), references=refs))
    return doc, questions


def load_split(raw_dir: str | Path, split: str) -> tuple[dict[str, Document], list[Question]]:
    df = pd.read_parquet(Path(raw_dir) / f"qasper_{split}.parquet")
    docs: dict[str, Document] = {}
    questions: list[Question] = []
    for _, row in df.iterrows():
        doc, qs = parse_row(row)
        docs[doc.doc_id] = doc
        questions.extend(qs)
    return docs, questions


def sample_questions(
    questions: list[Question],
    n_answerable: int,
    n_unanswerable: int,
    seed: int,
    max_per_doc: int = 2,
) -> list[Question]:
    """Seeded sample that spreads questions across papers.

    Answerable = no annotator marked it unanswerable; unanswerable = every annotator did.
    Questions where annotators disagree are excluded from both pools so that the
    abstention labels are unambiguous.
    """
    rng = random.Random(seed)
    pools = {"ans": [q for q in questions if not q.unanswerable_any], "unans": [q for q in questions if q.unanswerable_all]}
    chosen: list[Question] = []
    per_doc: dict[str, int] = defaultdict(int)
    for key, n in (("ans", n_answerable), ("unans", n_unanswerable)):
        pool = sorted(pools[key], key=lambda q: q.qid)
        rng.shuffle(pool)
        taken = 0
        for q in pool:
            if taken >= n:
                break
            if per_doc[q.doc_id] >= max_per_doc:
                continue
            chosen.append(q)
            per_doc[q.doc_id] += 1
            taken += 1
        if taken < n:
            raise ValueError(f"only {taken} {key} questions available with max_per_doc={max_per_doc}, wanted {n}")
    return chosen


def normalize_paragraph(text: str) -> str:
    text = _clean(text)
    if text.startswith(FLOAT_PREFIX):
        text = text[len(FLOAT_PREFIX):]
    return text.lower()
