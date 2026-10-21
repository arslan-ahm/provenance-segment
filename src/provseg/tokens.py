"""Model-agnostic token counting used for every budget in the pipeline.

Budgets are measured with tiktoken's ``cl100k_base`` so that all systems share one
yardstick regardless of which provider serves the model. Provider-reported usage is
recorded separately for cost accounting.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline first run: fall back to a character heuristic
        return None


@lru_cache(maxsize=500_000)
def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc is None:
        return max(1, len(text) // 4) if text else 0
    return len(enc.encode(text, disallowed_special=()))


def tokenizer_name() -> str:
    return "tiktoken/cl100k_base" if _encoder() is not None else "chars/4-heuristic"
