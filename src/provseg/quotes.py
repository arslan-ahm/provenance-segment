"""Anchoring model-produced quotes to exact character offsets in source chunks.

A quote is accepted only if, after light normalisation (Unicode NFKC, whitespace
collapse, typographic quotes/dashes folded, case kept), it occurs verbatim in the
chunk. Quotes containing an ellipsis are accepted when every fragment of at least
three words occurs in order; they are flagged ``elided``. There is no fuzzy matching:
a paraphrase is a fabricated quote.
"""

from __future__ import annotations

import re
import unicodedata

_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Normalised text plus, for each normalised char, its index in the original."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).translate(_FOLD):
            if c.isspace():
                if prev_space:
                    continue
                c = " "
                prev_space = True
            else:
                prev_space = False
            out.append(c)
            idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalise(text: str) -> str:
    return _normalise_with_map(text)[0]


def locate(quote: str, source: str) -> tuple[int, int, bool] | None:
    """Return (start, end, elided) offsets of ``quote`` in ``source`` or None."""
    q = normalise(quote).strip(" \"'")
    if len(q) < 8:
        return None
    norm, back = _normalise_with_map(source)
    pos = norm.find(q)
    if pos >= 0:
        return back[pos], back[pos + len(q) - 1] + 1, False
    fragments = [f.strip(" \"'") for f in _ELLIPSIS.split(q) if f.strip(" \"'")]
    if len(fragments) < 2 or any(len(f.split()) < 3 for f in fragments):
        return None
    cursor, start = 0, None
    for f in fragments:
        p = norm.find(f, cursor)
        if p < 0:
            return None
        if start is None:
            start = p
        cursor = p + len(f)
    return back[start], back[cursor - 1] + 1, True
