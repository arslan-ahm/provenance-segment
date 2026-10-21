from provseg.quotes import locate, normalise

SOURCE = "Sparse adapters reach 34.1 BLEU on WMT16  English–Romanian,\ncompared with 33.2 BLEU for “full” fine-tuning."


def test_exact_quote_returns_original_offsets():
    q = "reach 34.1 BLEU on WMT16"
    start, end, elided = locate(q, SOURCE)
    assert SOURCE[start:end] == q
    assert elided is False


def test_whitespace_and_typography_are_folded_but_offsets_point_at_original():
    q = 'WMT16 English-Romanian, compared with 33.2 BLEU for "full" fine-tuning'
    start, end, _ = locate(q, SOURCE)
    assert SOURCE[start:end] == "WMT16  English–Romanian,\ncompared with 33.2 BLEU for “full” fine-tuning"


def test_surrounding_quote_marks_are_ignored():
    assert locate('"Sparse adapters reach 34.1 BLEU"', SOURCE) is not None


def test_paraphrase_is_rejected():
    assert locate("Sparse adapters achieve 34.1 BLEU on WMT16", SOURCE) is None


def test_changed_number_is_rejected():
    assert locate("reach 34.2 BLEU on WMT16", SOURCE) is None


def test_case_is_not_folded():
    assert locate("sparse Adapters reach 34.1 BLEU", SOURCE) is None


def test_too_short_quote_is_rejected():
    assert locate("BLEU", SOURCE) is None


def test_ellipsis_fragments_in_order_are_accepted_and_flagged():
    start, end, elided = locate("Sparse adapters reach 34.1 ... compared with 33.2 BLEU", SOURCE)
    assert elided is True
    assert SOURCE[start:].startswith("Sparse adapters")
    assert SOURCE[:end].endswith("33.2 BLEU")


def test_ellipsis_fragments_out_of_order_are_rejected():
    assert locate("compared with 33.2 BLEU ... Sparse adapters reach 34.1", SOURCE) is None


def test_ellipsis_with_tiny_fragment_is_rejected():
    assert locate("Sparse adapters reach ... BLEU", SOURCE) is None


def test_normalise_collapses_whitespace_and_trims():
    assert normalise("  a \n\n b\t c  ") == "a b c"
