"""Structured (JSON) calls: strict schema generation, tolerant parsing, one repair retry."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from .base import LLMClient, LLMError, LLMRequest, UsageMeter


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema rewritten for strict structured-output modes.

    Inlines ``$defs``, marks every property required and forbids extra properties,
    which both Anthropic and OpenAI strict modes need.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            out = {k: resolve(v) for k, v in node.items() if k not in ("title", "default")}
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"].keys())
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may wrap it in prose or code fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON object found in model output")


def call_structured[T: BaseModel](
    client: LLMClient,
    meter: UsageMeter | None,
    stage: str,
    system: str,
    user: str,
    out_model: type[T],
    max_tokens: int | None = None,
) -> T:
    # Stage caps are floors, never below the model's configured max_tokens: on models that
    # think before answering, thinking tokens count against this limit and a low cap would
    # truncate the JSON.
    max_tokens = max(max_tokens or 0, client.spec.max_tokens)
    schema = strict_schema(out_model)
    hint = (
        "\n\nRespond with a single JSON object that matches this JSON schema and nothing else:\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    sys_prompt = system if client.spec.json_mode == "schema" else system + hint
    request = LLMRequest(system=sys_prompt, user=user, json_schema=schema, schema_name=out_model.__name__, max_tokens=max_tokens)
    resp = client.complete(request)
    if meter is not None:
        meter.record(stage, client.spec, resp)
    try:
        return out_model.model_validate(extract_json(resp.text))
    except (ValueError, ValidationError) as first_err:
        repair = LLMRequest(
            system=sys_prompt,
            user=(
                f"{user}\n\nYour previous reply could not be parsed ({str(first_err)[:300]}). "
                "Reply again with only the JSON object."
            ),
            json_schema=schema,
            schema_name=out_model.__name__,
            max_tokens=max_tokens,
        )
        resp2 = client.complete(repair)
        if meter is not None:
            meter.record(stage + ":repair", client.spec, resp2)
        try:
            return out_model.model_validate(extract_json(resp2.text))
        except (ValueError, ValidationError) as e:
            # Do not keep unusable replies in the cache, or a resumed run would replay the same failure.
            invalidate = getattr(client, "invalidate", None)
            if invalidate is not None:
                for key in (resp.cache_key, resp2.cache_key):
                    if key:
                        invalidate(key)
            truncated = resp2.stop_reason in ("max_tokens", "length")
            why = f" (output truncated at max_tokens={max_tokens}; raise it in configs/models.yaml)" if truncated else ""
            raise LLMError(f"{stage}: unparseable structured output after repair{why}: {e}") from e
