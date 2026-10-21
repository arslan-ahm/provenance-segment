"""Shared hand-written fixtures. No network, no model downloads, no data files."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from provseg.llm import LLMRequest, LLMResponse, ModelSpec
from provseg.schemas import Document, GoldReference, Question, Section

FAKE_SPEC = ModelSpec(provider="openai_compat", model="fake-model", base_url="http://fake.invalid/v1", price_in_per_mtok=1.0, price_out_per_mtok=2.0)


@dataclass
class FakeLLM:
    """Scripted client. ``route`` maps a request to a reply (dict -> JSON, or raw str)."""

    route: Callable[[LLMRequest], dict | str]
    spec: ModelSpec = field(default_factory=lambda: FAKE_SPEC)
    requests: list[LLMRequest] = field(default_factory=list)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        reply = self.route(request)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, input_tokens=100, output_tokens=20, latency_s=0.01, model=self.spec.model)

    def stage_requests(self, marker: str) -> list[LLMRequest]:
        return [r for r in self.requests if marker in r.system]


@pytest.fixture
def paper() -> Document:
    return Document(
        doc_id="2001.00001",
        title="Sparse Adapters for Low-Resource Translation",
        sections=[
            Section(idx=0, heading="Abstract", paragraphs=["We propose sparse adapters for low-resource machine translation. Our model improves BLEU on WMT16 English-Romanian."]),
            Section(
                idx=1,
                heading="Introduction",
                paragraphs=[
                    "Neural machine translation needs large parallel corpora. Low-resource pairs lack such data.",
                    "Adapters add small trainable modules to a frozen network. We make them sparse to save memory.",
                ],
            ),
            Section(
                idx=2,
                heading="Experiments",
                paragraphs=[
                    "We evaluate on WMT16 English-Romanian and IWSLT14 German-English. The baseline is a full fine-tuned Transformer.",
                    "Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian, compared with 33.2 BLEU for full fine-tuning.",
                    "Training uses 8 GPUs for 2 days. We do not report results on English-French.",
                ],
            ),
            Section(idx=3, heading="Figures and Tables", paragraphs=["Table 1: BLEU scores on WMT16 En-Ro and IWSLT14 De-En."]),
        ],
    )


@pytest.fixture
def question_bleu() -> Question:
    return Question(
        qid="q-bleu",
        doc_id="2001.00001",
        question="What BLEU score do sparse adapters reach on WMT16 English-Romanian?",
        references=[
            GoldReference(
                answer="34.1 BLEU",
                type="extractive",
                evidence=["Sparse adapters reach 34.1 BLEU on WMT16 English-Romanian, compared with 33.2 BLEU for full fine-tuning."],
            ),
            GoldReference(answer="34.1", type="extractive", evidence=["FLOAT SELECTED: Table 1: BLEU scores on WMT16 En-Ro and IWSLT14 De-En."]),
        ],
    )


@pytest.fixture
def question_unanswerable() -> Question:
    return Question(
        qid="q-none",
        doc_id="2001.00001",
        question="What BLEU score do they get on English-French?",
        references=[GoldReference(answer="Unanswerable", type="none"), GoldReference(answer="Unanswerable", type="none")],
    )


def handles_in(text: str) -> list[str]:
    return re.findall(r"\[(P\d+)\]", text)
