"""Concrete provider clients.

* ``AnthropicClient`` uses the official ``anthropic`` SDK (optional extra ``anthropic``).
* ``OpenAICompatClient`` speaks the ``/chat/completions`` wire format over httpx, which
  covers OpenAI, OpenRouter, Groq, vLLM, LM Studio and Ollama (``http://localhost:11434/v1``).
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

from .base import LLMError, LLMRequest, LLMResponse, ModelSpec

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _api_key(spec: ModelSpec, required: bool) -> str | None:
    if not spec.api_key_env:
        return None
    key = os.environ.get(spec.api_key_env)
    if required and not key:
        raise LLMError(f"environment variable {spec.api_key_env} is not set (needed for model {spec.model})")
    return key


class AnthropicClient:
    def __init__(self, spec: ModelSpec, max_retries: int = 6):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise LLMError("install the 'anthropic' extra: uv sync --extra anthropic") from e
        self.spec = spec
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": spec.timeout_s}
        key = _api_key(spec, required=False)
        if key:
            kwargs["api_key"] = key
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        spec = self.spec
        params: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": request.max_tokens or spec.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        output_config: dict[str, Any] = {}
        if spec.effort:
            output_config["effort"] = spec.effort
        if request.json_schema is not None and spec.json_mode == "schema":
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            params["output_config"] = output_config
        if spec.temperature is not None:
            params["temperature"] = spec.temperature
        params.update(spec.extra)
        t0 = time.perf_counter()
        try:
            resp = self._client.messages.create(**params)
        except self._anthropic.APIStatusError as e:
            raise LLMError(f"anthropic API error {e.status_code}: {e.message}") from e
        except self._anthropic.APIConnectionError as e:
            raise LLMError(f"anthropic connection error: {e}") from e
        latency = time.perf_counter() - t0
        if resp.stop_reason == "refusal":
            raise LLMError("model refused the request (stop_reason=refusal)")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
            model=resp.model,
            stop_reason=resp.stop_reason,
        )


class OpenAICompatClient:
    def __init__(self, spec: ModelSpec, max_retries: int = 6, transport: httpx.BaseTransport | None = None):
        if not spec.base_url:
            raise LLMError(f"openai_compat model {spec.model} needs base_url")
        self.spec = spec
        self.max_retries = max_retries
        headers = {"Content-Type": "application/json"}
        key = _api_key(spec, required=bool(spec.api_key_env))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=spec.base_url.rstrip("/"), headers=headers, timeout=spec.timeout_s, transport=transport
        )

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        spec = self.spec
        body: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": request.max_tokens or spec.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if request.json_schema is not None:
            if spec.json_mode == "schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": request.schema_name, "schema": request.json_schema, "strict": True},
                }
            elif spec.json_mode == "object":
                body["response_format"] = {"type": "json_object"}
        body.update(spec.extra)
        return body

    def complete(self, request: LLMRequest) -> LLMResponse:
        body = self._body(request)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as e:
                last = e
            else:
                if r.status_code == 200:
                    data = r.json()
                    choice = data["choices"][0]
                    usage = data.get("usage") or {}
                    return LLMResponse(
                        text=choice["message"].get("content") or "",
                        input_tokens=int(usage.get("prompt_tokens", 0)),
                        output_tokens=int(usage.get("completion_tokens", 0)),
                        latency_s=time.perf_counter() - t0,
                        model=data.get("model", self.spec.model),
                        stop_reason=choice.get("finish_reason"),
                    )
                if r.status_code not in RETRYABLE_STATUS:
                    raise LLMError(f"{self.spec.base_url} returned {r.status_code}: {r.text[:500]}")
                last = LLMError(f"status {r.status_code}: {r.text[:200]}")
            if attempt < self.max_retries:
                time.sleep(min(60.0, 2**attempt + random.random()))
        raise LLMError(f"request failed after {self.max_retries + 1} attempts: {last}")


def build_client(spec: ModelSpec) -> AnthropicClient | OpenAICompatClient:
    if spec.provider == "anthropic":
        return AnthropicClient(spec)
    return OpenAICompatClient(spec)
