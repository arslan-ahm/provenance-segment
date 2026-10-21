"""Experiment configuration (YAML) and construction of model clients from it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from .chunking import ChunkingConfig
from .llm import CachedClient, ModelSpec, build_client
from .llm.limit import ConcurrencyLimited
from .systems import SystemConfig


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    split: Literal["validation", "test"] = "test"
    pool_split: Literal["train", "validation", "test"] = "train"


class SampleConfig(BaseModel):
    n_answerable: int = 100
    n_unanswerable: int = 30
    seed: int = 13
    max_per_doc: int = 2


class NoiseConfig(BaseModel):
    levels: list[int] = Field(default_factory=lambda: [0, 10, 25, 50])
    mode: Literal["random", "hard"] = "random"
    seed: int = 7


class VerifierConfig(BaseModel):
    kind: Literal["nli", "llm"] = "nli"
    model: str = "cross-encoder/nli-deberta-v3-small"
    model_ref: str | None = None  # for kind=llm: key into `models`
    entail_threshold: float = 0.5
    contradict_threshold: float = 0.5


class JudgeConfig(BaseModel):
    kind: Literal["llm", "nli"] = "llm"
    model_ref: str | None = "judge"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    audit_sample: int = 100


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    context_budget: int = 2048
    base_model: str = "base"
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    systems: list[SystemConfig]
    runs_dir: str = "runs"
    cache_dir: str = "cache/llm"
    question_workers: int = 4
    note_workers: int = 8

    def system(self, name: str) -> SystemConfig:
        for s in self.systems:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.name


def load_experiment(path: str | Path, models_path: str | Path | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models_file = raw.pop("models_file", None)
    if models_path or models_file:
        mp = Path(models_path or Path(path).parent / models_file)
        models = yaml.safe_load(mp.read_text(encoding="utf-8"))["models"]
        raw["models"] = {**models, **raw.get("models", {})}
    cfg = ExperimentConfig.model_validate(raw)
    # The context budget is an experiment-level control shared by every system.
    for s in cfg.systems:
        s.context_budget = cfg.context_budget
    return cfg


def make_client(spec: ModelSpec, cache_dir: str | Path, offline: bool = False):
    if offline:
        # Replay never touches the network, so no provider client (or API key) is needed.
        class _Offline:
            def __init__(self, spec):
                self.spec = spec

            def complete(self, request):  # pragma: no cover - CachedClient raises before this
                raise RuntimeError("offline")

        return CachedClient(_Offline(spec), cache_dir, offline=True)
    return CachedClient(ConcurrencyLimited(build_client(spec), spec.max_concurrency), cache_dir)
