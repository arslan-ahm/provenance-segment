from provseg.noise import DistractorPool, inject
from provseg.schemas import Chunk, Document, Section


def _chunk(doc, i, text=None, heading="Body"):
    return Chunk(chunk_id=f"{doc}_c{i}", doc_id=doc, section_idx=1, section_heading=heading, chunk_idx=i, text=text or f"text {doc} {i}", spans=[(1, i, i)], n_tokens=10)


def _doc(doc_id):
    return Document(doc_id=doc_id, title=f"Title {doc_id}", sections=[Section(idx=0, heading="Abstract", paragraphs=[f"abstract of {doc_id}"])])


POOL = [_chunk(f"p{d}", i) for d in range(6) for i in range(10)] + [_chunk("target", 99)]


def test_draw_excludes_target_doc_and_marks_distractors():
    ds = DistractorPool(POOL).draw(_doc("target"), 20, seed=3)
    assert len(ds) == 20
    assert all(c.is_distractor and c.doc_id != "target" for c in ds)
    assert [c.chunk_id[:7] for c in ds] == [f"noise{i:02d}" for i in range(20)]
    assert len({c.chunk_id for c in ds}) == 20


def test_draw_is_deterministic_per_seed_and_doc():
    pool = DistractorPool(POOL)
    a = pool.draw(_doc("target"), 10, seed=3)
    b = pool.draw(_doc("target"), 10, seed=3)
    c = pool.draw(_doc("target"), 10, seed=4)
    assert a == b
    assert [x.chunk_id for x in a] != [x.chunk_id for x in c]


def test_abstract_chunks_never_used_as_distractors():
    pool = DistractorPool(POOL + [_chunk("p9", 0, heading="Abstract")])
    assert all(c.section_heading != "Abstract" for c in pool.draw(_doc("target"), 60, seed=1))


def test_levels_are_nested_and_preserve_original_order():
    doc_chunks = [_chunk("target", i) for i in range(8)]
    ds = DistractorPool(POOL).draw(_doc("target"), 12, seed=5)
    levels = {k: inject(doc_chunks, ds, k, seed=5, doc_id="target") for k in (0, 3, 7, 12)}
    assert levels[0] == doc_chunks
    for k, seq in levels.items():
        assert len(seq) == 8 + k
        assert [c for c in seq if not c.is_distractor] == doc_chunks
    # The distractors at a lower level are a subset, in the same relative positions, of the higher level.
    for lo, hi in ((3, 7), (7, 12)):
        hi_ids = [c.chunk_id for c in levels[hi]]
        lo_ids = [c.chunk_id for c in levels[lo]]
        assert [c for c in hi_ids if c in set(lo_ids)] == lo_ids


def test_hard_mode_prefers_topically_similar_chunks():
    similar = [_chunk("p7", i, text="sparse adapters translation bleu low resource") for i in range(5)]
    pool = DistractorPool(POOL + similar)
    doc = Document(doc_id="target", title="Sparse adapters for translation", sections=[Section(idx=0, heading="Abstract", paragraphs=["low resource translation bleu adapters"])])
    ds = pool.draw(doc, 5, seed=1, mode="hard")
    # Hard distractors are a seeded sample from the 3*k most similar chunks of other papers.
    from provseg.retrieval import tokenize

    scores = pool._index().get_scores(tokenize(doc.title + " " + doc.sections[0].paragraphs[0]))
    ranked = sorted(range(len(pool.chunks)), key=lambda i: -scores[i])
    top = {pool.chunks[i].chunk_id for i in ranked[:15]}
    assert all(c.chunk_id.split("__", 1)[1] in top for c in ds)
    assert sum(c.doc_id == "p7" for c in ds) >= 1
