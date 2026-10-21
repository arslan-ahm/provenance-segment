import json

import httpx
import pytest
from pydantic import BaseModel

from provseg.llm import CachedClient, LLMError, LLMRequest, ModelSpec, UsageMeter, call_structured, extract_json, request_key, strict_schema
from provseg.llm.providers import OpenAICompatClient
from provseg.prompts import AnswerOut, NoteOut

from .conftest import FAKE_SPEC, FakeLLM


def test_strict_schema_inlines_refs_and_requires_everything():
    s = strict_schema(AnswerOut)
    assert "$defs" not in json.dumps(s)
    assert s["required"] == ["answer", "statements"] and s["additionalProperties"] is False
    item = s["properties"]["statements"]["items"]
    assert item["required"] == ["text", "citations"] and item["additionalProperties"] is False
    assert strict_schema(NoteOut)["properties"]["relevance"]["enum"] == ["high", "medium", "low", "none"]


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go: {"a": 1} hope it helps', '  {"a": 1}  '],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


class Out(BaseModel):
    x: int


def test_call_structured_repairs_once_and_records_both_calls():
    replies = iter(["not json", '{"x": 3}'])
    fake = FakeLLM(route=lambda r: next(replies))
    meter = UsageMeter()
    assert call_structured(fake, meter, "stage", "sys", "user", Out).x == 3
    assert [c.stage for c in meter.calls] == ["stage", "stage:repair"]
    assert "could not be parsed" in fake.requests[1].user
    assert meter.summary()["cost_usd"] == pytest.approx(2 * (100 * 1.0 + 20 * 2.0) / 1e6)


def test_call_structured_gives_up_after_repair():
    fake = FakeLLM(route=lambda r: '{"x": "nope"}')
    with pytest.raises(LLMError):
        call_structured(fake, None, "stage", "sys", "user", Out)


def test_schema_hint_added_only_when_provider_lacks_schema_mode():
    fake = FakeLLM(route=lambda r: '{"x": 1}')
    call_structured(fake, None, "s", "SYS", "u", Out)
    assert fake.requests[0].system == "SYS"
    fake2 = FakeLLM(route=lambda r: '{"x": 1}', spec=FAKE_SPEC.model_copy(update={"json_mode": "object"}))
    call_structured(fake2, None, "s", "SYS", "u", Out)
    assert fake2.requests[0].system.startswith("SYS\n\nRespond with a single JSON object")


def test_cache_replays_and_offline_miss_raises(tmp_path):
    fake = FakeLLM(route=lambda r: '{"x": 1}')
    cached = CachedClient(fake, tmp_path)
    req = LLMRequest(system="s", user="u")
    first = cached.complete(req)
    second = cached.complete(req)
    assert len(fake.requests) == 1
    assert not first.cached and second.cached and second.text == first.text
    assert second.cache_key == request_key(FAKE_SPEC, req)
    stored = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert stored["request"]["user"] == "u" and stored["response"]["text"] == '{"x": 1}'
    offline = CachedClient(fake, tmp_path, offline=True)
    assert offline.complete(req).cached
    with pytest.raises(LLMError, match="offline replay"):
        offline.complete(LLMRequest(system="s", user="different"))


def test_cache_key_depends_on_model_and_prompt():
    a = request_key(FAKE_SPEC, LLMRequest(system="s", user="u"))
    assert a == request_key(FAKE_SPEC, LLMRequest(system="s", user="u"))
    assert a != request_key(FAKE_SPEC.model_copy(update={"model": "other"}), LLMRequest(system="s", user="u"))
    assert a != request_key(FAKE_SPEC, LLMRequest(system="s", user="u2"))


def _openai_client(handler, **spec_kw):
    spec = ModelSpec(provider="openai_compat", model="m", base_url="http://api.test/v1", temperature=0.0, **spec_kw)
    return OpenAICompatClient(spec, max_retries=2, transport=httpx.MockTransport(handler))


def test_openai_compat_payload_and_parsing(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "m-1", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})

    client = _openai_client(handler)
    resp = client.complete(LLMRequest(system="S", user="U", json_schema={"type": "object"}, schema_name="Out", max_tokens=99))
    assert seen["url"] == "http://api.test/v1/chat/completions"
    assert seen["body"] == {
        "model": "m",
        "max_tokens": 99,
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "Out", "schema": {"type": "object"}, "strict": True}},
    }
    assert (resp.text, resp.input_tokens, resp.output_tokens, resp.model) == ("hi", 7, 2, "m-1")


def test_openai_compat_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("provseg.llm.providers.time.sleep", lambda s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    assert _openai_client(handler).complete(LLMRequest(system="s", user="u")).text == "ok"
    assert len(calls) == 3


def test_openai_compat_does_not_retry_400():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, text="bad schema")

    with pytest.raises(LLMError, match="400"):
        _openai_client(handler).complete(LLMRequest(system="s", user="u"))
    assert len(calls) == 1


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("PROVSEG_TEST_KEY", raising=False)
    with pytest.raises(LLMError, match="PROVSEG_TEST_KEY"):
        _openai_client(lambda r: httpx.Response(200), api_key_env="PROVSEG_TEST_KEY")
