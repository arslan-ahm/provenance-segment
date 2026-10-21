from provseg.chunking import ChunkingConfig, chunk_document, gold_chunk_ids, split_sentences
from provseg.schemas import Document, Section


def test_small_sections_are_packed_and_ids_are_stable(paper):
    cfg = ChunkingConfig(target_tokens=1000, max_tokens=1200, min_tokens=500, overlap_tokens=0)
    chunks = chunk_document(paper, cfg)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "doc_2001_00001_sec_00_chunk_00"
    assert c.spans == [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 0)]
    assert "§ Experiments" in c.text and "§ Abstract" in c.text
    assert chunk_document(paper, cfg) == chunks  # deterministic


def test_section_boundary_breaks_once_min_tokens_reached(paper):
    cfg = ChunkingConfig(target_tokens=1000, max_tokens=1200, min_tokens=1, overlap_tokens=0)
    chunks = chunk_document(paper, cfg)
    assert [c.chunk_id for c in chunks] == [
        "doc_2001_00001_sec_00_chunk_00",
        "doc_2001_00001_sec_01_chunk_00",
        "doc_2001_00001_sec_02_chunk_00",
        "doc_2001_00001_sec_03_chunk_00",
    ]
    assert [c.section_heading for c in chunks] == ["Abstract", "Introduction", "Experiments", "Figures and Tables"]


def test_long_section_is_split_with_overlap_only_mid_section():
    paras = [f"Paragraph {i} says that fact number {i} holds. It also says detail {i} matters." for i in range(12)]
    doc = Document(doc_id="x", title="t", sections=[Section(idx=0, heading="Body", paragraphs=paras), Section(idx=1, heading="Next", paragraphs=["New section text here."])])
    cfg = ChunkingConfig(target_tokens=60, max_tokens=80, min_tokens=10, overlap_tokens=15)
    chunks = chunk_document(doc, cfg)
    body = [c for c in chunks if c.section_idx == 0]
    assert len(body) > 1
    assert [c.chunk_idx for c in body] == list(range(len(body)))
    assert body[0].overlap_chars == 0
    for prev, cur in zip(body, body[1:]):
        assert cur.overlap_chars > 0
        overlap = cur.text.split("\n\n")[1]
        assert overlap in prev.text  # carried-over sentences come from the previous chunk
        assert cur.spans[0][1] == prev.spans[-1][2] + 1  # ownership is contiguous, never shared
    nxt = [c for c in chunks if c.section_idx == 1][0]
    assert nxt.overlap_chars == 0 and nxt.chunk_id.endswith("sec_01_chunk_00")


def test_overlong_paragraph_is_split_on_sentences_and_keeps_paragraph_index():
    para = " ".join(f"Sentence {i} reports a measured value of {i}.{i} units." for i in range(40))
    doc = Document(doc_id="y", title="t", sections=[Section(idx=0, heading="Body", paragraphs=[para])])
    chunks = chunk_document(doc, ChunkingConfig(target_tokens=80, max_tokens=100, min_tokens=10, overlap_tokens=0))
    assert len(chunks) > 1
    assert all(c.spans == [(0, 0, 0)] for c in chunks)
    assert all(c.n_tokens <= 120 for c in chunks)
    joined = " ".join(c.text.replace("§ Body\n\n", "") for c in chunks)
    assert joined == para


def test_gold_evidence_maps_to_owning_chunk_including_float_captions(paper, question_bleu):
    cfg = ChunkingConfig(target_tokens=1000, max_tokens=1200, min_tokens=1, overlap_tokens=0)
    chunks = chunk_document(paper, cfg)
    gold, matched, total = gold_chunk_ids(question_bleu, paper, chunks)
    assert gold == [{"doc_2001_00001_sec_02_chunk_00"}, {"doc_2001_00001_sec_03_chunk_00"}]
    assert (matched, total) == (2, 2)


def test_gold_mapping_ignores_distractors_and_counts_misses(paper, question_bleu):
    chunks = chunk_document(paper, ChunkingConfig(min_tokens=1))
    fake = chunks[2].model_copy(update={"chunk_id": "noise00__other", "is_distractor": True})
    q = question_bleu.model_copy(deep=True)
    q.references[0].evidence.append("A sentence that is nowhere in the paper at all, long enough to be considered.")
    gold, matched, total = gold_chunk_ids(q, paper, [fake] + chunks)
    assert "noise00__other" not in gold[0]
    assert (matched, total) == (2, 3)


def test_split_sentences():
    assert split_sentences("One. Two! Three? [4] four. lower case stays.") == ["One.", "Two!", "Three?", "[4] four. lower case stays."]
