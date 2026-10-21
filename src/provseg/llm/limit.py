"""Bounds concurrent in-flight requests per model across all threads of a run."""

from __future__ import annotations

import threading

from .base import LLMClient, LLMRequest, LLMResponse


class ConcurrencyLimited:
    def __init__(self, inner: LLMClient, limit: int):
        self.inner = inner
        self.spec = inner.spec
        self._sem = threading.BoundedSemaphore(max(1, limit))

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._sem:
            return self.inner.complete(request)
