"""Provider-agnostic LLM request/response types and the model spec read from config."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """One named model endpoint. Prices are USD per 1M tokens and only used for reporting."""

    provider: Literal["anthropic", "openai_compat"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = 4096
    # None means "do not send": several current Claude models reject sampling params.
    temperature: float | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    json_mode: Literal["schema", "object", "none"] = "schema"
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    max_concurrency: int = 4
    timeout_s: float = 300.0
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LLMRequest:
    system: str
    user: str
    json_schema: dict[str, Any] | None = None
    schema_name: str = "output"
    max_tokens: int | None = None

    def cache_payload(self, spec: ModelSpec) -> dict[str, Any]:
        return {
            "provider": spec.provider,
            "model": spec.model,
            "base_url": spec.base_url,
            "temperature": spec.temperature,
            "effort": spec.effort,
            "json_mode": spec.json_mode,
            "max_tokens": self.max_tokens or spec.max_tokens,
            "system": self.system,
            "user": self.user,
            "json_schema": self.json_schema,
        }


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    stop_reason: str | None = None
    cached: bool = False
    cache_key: str | None = None


class LLMClient(Protocol):
    spec: ModelSpec

    def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries or returns unusable output."""


@dataclass
class CallRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    cache_key: str | None
    cost_usd: float


@dataclass
class UsageMeter:
    """Collects every model call made while answering one question."""

    calls: list[CallRecord] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    def record(self, stage: str, spec: ModelSpec, resp: LLMResponse) -> None:
        cost = (
            resp.input_tokens * spec.price_in_per_mtok + resp.output_tokens * spec.price_out_per_mtok
        ) / 1_000_000
        self.calls.append(
            CallRecord(
                stage=stage,
                model=spec.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_s=resp.latency_s,
                cached=resp.cached,
                cache_key=resp.cache_key,
                cost_usd=cost,
            )
        )

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, float]] = {}
        for c in self.calls:
            s = by_stage.setdefault(c.stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["input_tokens"] += c.input_tokens
            s["output_tokens"] += c.output_tokens
            s["cost_usd"] += c.cost_usd
        return {
            "calls": len(self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            # Latency is the provider latency summed over calls (sequential-equivalent);
            # cached calls report the latency of the original call.
            "llm_latency_s": round(sum(c.latency_s for c in self.calls), 3),
            "cost_usd": round(sum(c.cost_usd for c in self.calls), 6),
            "cache_hits": sum(1 for c in self.calls if c.cached),
            "by_stage": by_stage,
            "cache_keys": [c.cache_key for c in self.calls],
        }
