"""Read-only JSON API and static front end for exploring experiment runs.

The app never calls a model provider: it serves what is on disk under ``runs/<name>/``
plus model-free checks (budgeted baseline contexts, quote anchoring and the local NLI
verifier, see ``live.py``), so it can be hosted publicly (for example on a Hugging Face
Space) without API keys. Set ``PROVSEG_LIVE_NLI=0`` to switch the NLI check off.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import ExperimentConfig
from ..runner import Experiment, read_jsonl
from . import views
from .live import MAX_CLAIM_CHARS, MAX_SOURCES, ClaimChecker, LiveError, handles_for

STATIC = Path(__file__).parent / "static"


class RunStore:
    """Loads each run once and keeps it in memory (runs are small and immutable once served)."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self._cache: dict[str, Experiment] = {}
        self._stamps: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if (p / "manifest.json").exists() and (p / "questions.jsonl").exists())

    def get(self, name: str) -> Experiment:
        if name not in self.names():
            raise HTTPException(404, f"unknown run {name!r}")
        d = self.runs_dir / name
        # Reload when predictions or judgments change on disk (e.g. a run still in progress).
        stamp = tuple(p.stat().st_mtime_ns for p in sorted(d.glob("*.jsonl")))
        with self._lock:
            if name not in self._cache or self._stamps.get(name) != stamp:
                self._stamps[name] = stamp
                manifest = json.loads((self.runs_dir / name / "manifest.json").read_text(encoding="utf-8"))
                cfg = ExperimentConfig.model_validate({**manifest["config"], "runs_dir": str(self.runs_dir)})
                exp = Experiment(cfg)
                exp.load_built()
                exp._preds = {(o.system, o.noise_level, o.qid): o for o in exp.latest_predictions()}
                exp._judg = exp.judgments()
                self._cache[name] = exp
            return self._cache[name]


class CheckIn(BaseModel):
    qid: str
    noise: int = 0
    claim: str = Field(max_length=MAX_CLAIM_CHARS * 2)
    chunk_ids: list[str] = Field(max_length=MAX_SOURCES * 2)
    nli: bool = True


def nli_available() -> bool:
    return os.environ.get("PROVSEG_LIVE_NLI", "1") != "0" and importlib.util.find_spec("sentence_transformers") is not None


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_app(runs_dir: str | Path = "runs") -> FastAPI:
    store = RunStore(Path(runs_dir))
    app = FastAPI(title="Provenance-Segment explorer", docs_url="/api/docs", redoc_url=None)
    checkers: dict[str, ClaimChecker] = {}
    checkers_lock = threading.Lock()

    def checker(exp: Experiment) -> ClaimChecker:
        v = exp.cfg.verifier
        key = f"{v.model}|{v.entail_threshold}|{v.contradict_threshold}|{v.window_sentences}"
        with checkers_lock:
            if key not in checkers:
                checkers[key] = ClaimChecker(v.model, v.entail_threshold, v.contradict_threshold, tuple(v.window_sentences))
            return checkers[key]

    if os.environ.get("PROVSEG_WARM_NLI") == "1" and nli_available():
        # Load the verifier in the background so the first visitor's check does not pay for it.
        def warm():
            for name in store.names():
                checker(store.get(name)).verifier.score_pairs([("A warm-up premise.", "A warm-up claim.")])

        threading.Thread(target=warm, daemon=True).start()

    @app.get("/api/health")
    def health():
        return {"ok": True, "runs": len(store.names()), "live_nli": nli_available()}

    @app.get("/api/runs")
    def runs():
        out = []
        for name in store.names():
            d = store.runs_dir / name
            m = _json(d / "manifest.json")
            cfg = m["config"]
            out.append(
                {
                    "name": name,
                    "description": cfg.get("description", ""),
                    "created_utc": m.get("created_utc"),
                    "base_model": cfg["models"].get(cfg["base_model"], {}).get("model"),
                    "systems": [s["name"] for s in cfg["systems"]],
                    "noise_levels": cfg["noise"]["levels"],
                    "context_budget": cfg["context_budget"],
                    "sample_stats": m.get("sample_stats"),
                    "has_predictions": (d / "predictions.jsonl").exists(),
                    "has_metrics": (d / "metrics.json").exists(),
                    "has_retrieval": (d / "diagnostics" / "retrieval.json").exists(),
                }
            )
        return out

    @app.get("/api/runs/{name}/summary")
    def summary(name: str):
        exp = store.get(name)
        d = exp.dir
        counts: dict[str, int] = {}
        for system, k, _ in exp._preds:
            key = f"{system}@{k}"
            counts[key] = counts.get(key, 0) + 1
        return {
            "manifest": _json(d / "manifest.json"),
            "metrics": _json(d / "metrics.json"),
            "retrieval": _json(d / "diagnostics" / "retrieval.json"),
            "failures": _json(d / "failures.json"),
            "prediction_counts": counts,
        }

    @app.get("/api/runs/{name}/questions")
    def questions(name: str):
        return views.question_list(store.get(name))

    @app.get("/api/runs/{name}/questions/{qid}")
    def question(name: str, qid: str, noise: int = Query(0)):
        try:
            return views.question_detail(store.get(name), qid, noise)
        except views.NotFound as e:
            raise HTTPException(404, e.args[0]) from e

    @app.get("/api/runs/{name}/featured")
    def featured_questions(name: str):
        return views.featured_questions(store.get(name))

    @app.post("/api/runs/{name}/check")
    def check(name: str, body: CheckIn):
        exp = store.get(name)
        ids = exp.instances.get((body.qid, body.noise))
        if ids is None:
            raise HTTPException(404, "unknown question or noise level")
        allowed = set(ids)
        if not set(body.chunk_ids) <= allowed:
            raise HTTPException(400, "every passage must belong to this question's instance")
        use_nli = body.nli and nli_available()
        if body.nli and not use_nli:
            raise HTTPException(503, "the NLI verifier is not available on this host (install the `nli` extra)")
        try:
            return checker(exp).check(body.claim, [exp.chunk_by_id[c] for c in dict.fromkeys(body.chunk_ids)], handles_for(ids), nli=use_nli)
        except LiveError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/runs/{name}/chunks/{chunk_id}")
    def chunk(name: str, chunk_id: str):
        exp = store.get(name)
        c = exp.chunk_by_id.get(chunk_id)
        if c is None:
            raise HTTPException(404, "unknown chunk")
        origin = c.doc_id
        return {**c.model_dump(), "title": exp.documents.get(origin, {}).get("title", origin)}

    @app.get("/api/runs/{name}/predictions.jsonl")
    def raw_predictions(name: str):
        p = store.get(name).dir / "predictions.jsonl"
        if not p.exists():
            raise HTTPException(404, "no predictions")
        return FileResponse(p, media_type="application/x-ndjson", filename=f"{name}_predictions.jsonl")

    @app.get("/api/runs/{name}/report")
    def report(name: str):
        p = store.get(name).dir / "REPORT.md"
        return {"markdown": p.read_text(encoding="utf-8") if p.exists() else None}

    if STATIC.exists():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def read_rows(path: Path) -> list[dict]:
    return read_jsonl(path)
