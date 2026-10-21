"""Content-addressed on-disk cache of raw model calls.

Every request/response pair is stored as one JSON file keyed by the SHA-256 of the
canonical request. The cache doubles as the raw record of every model call an
experiment made, so re-running an experiment replays identical outputs and a third
party can audit exactly what the model saw and said.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from .base import LLMClient, LLMError, LLMRequest, LLMResponse, ModelSpec


def request_key(spec: ModelSpec, request: LLMRequest) -> str:
    canonical = json.dumps(request.cache_payload(spec), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CachedClient:
    """Wraps a client; ``offline=True`` turns any cache miss into an error (pure replay)."""

    def __init__(self, inner: LLMClient, cache_dir: str | Path, offline: bool = False):
        self.inner = inner
        self.spec = inner.spec
        self.root = Path(cache_dir)
        self.offline = offline
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        p = self._path(key)
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        r = rec["response"]
        return LLMResponse(
            text=r["text"],
            input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"],
            latency_s=r["latency_s"],
            model=r["model"],
            stop_reason=r.get("stop_reason"),
            cached=True,
            cache_key=key,
        )

    def put(self, key: str, request: LLMRequest, resp: LLMResponse) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "key": key,
            "created_utc": datetime.now(UTC).isoformat(),
            "request": request.cache_payload(self.spec),
            "response": {
                "text": resp.text,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "latency_s": resp.latency_s,
                "model": resp.model,
                "stop_reason": resp.stop_reason,
            },
        }
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        with self._lock:
            os.replace(tmp, p)

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request_key(self.spec, request)
        hit = self.get(key)
        if hit is not None:
            return hit
        if self.offline:
            raise LLMError(f"offline replay: no cached response for key {key[:12]} (model {self.spec.model})")
        resp = self.inner.complete(request)
        self.put(key, request, resp)
        resp.cache_key = key
        return resp
