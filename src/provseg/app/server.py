"""Read-only JSON API and static front end for exploring experiment runs.

The app never calls a model: it serves what is on disk under ``runs/<name>/``, so it
can be hosted publicly (for example on a Hugging Face Space) without API keys.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import ExperimentConfig
from ..runner import Experiment, read_jsonl
from ..systems import make_handle

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


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def create_app(runs_dir: str | Path = "runs") -> FastAPI:
    store = RunStore(Path(runs_dir))
    app = FastAPI(title="Provenance-Segment explorer", docs_url="/api/docs", redoc_url=None)

    @app.get("/api/health")
    def health():
        return {"ok": True, "runs": len(store.names())}

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
        exp = store.get(name)
        out = []
        for q in exp.questions.values():
            done = sorted({(s, k) for (s, k, qid) in exp._preds if qid == q.qid})
            out.append(
                {
                    "qid": q.qid,
                    "question": q.question,
                    "doc_id": q.doc_id,
                    "title": exp.documents.get(q.doc_id, {}).get("title", q.doc_id),
                    "unanswerable": q.unanswerable_all,
                    "answers": sorted({r.answer for r in q.references}),
                    "predictions": [{"system": s, "noise_level": k} for s, k in done],
                }
            )
        return out

    @app.get("/api/runs/{name}/questions/{qid}")
    def question(name: str, qid: str, noise: int = Query(0)):
        exp = store.get(name)
        q = exp.questions.get(qid)
        if q is None:
            raise HTTPException(404, "unknown question")
        ids = exp.instances.get((qid, noise))
        if ids is None:
            raise HTTPException(404, f"no instance at noise level {noise}")
        gold = sorted(set().union(*exp.gold[qid])) if exp.gold[qid] else []
        chunks = []
        for i, cid in enumerate(ids):
            c = exp.chunk_by_id[cid]
            chunks.append({"handle": make_handle(i), "chunk_id": cid, "heading": c.section_heading, "is_distractor": c.is_distractor, "is_gold": cid in gold, "n_tokens": c.n_tokens})
        preds = []
        for s in exp.cfg.systems:
            o = exp._preds.get((s.name, noise, qid))
            if o is None:
                continue
            judgments = exp._judg.get((s.name, noise, qid))
            preds.append(
                {
                    "system": s.name,
                    "kind": s.kind,
                    "answer": o.answer,
                    "abstained": o.abstained,
                    "error": o.error,
                    "statements": [st.model_dump() for st in o.statements],
                    "dropped_statements": [st.model_dump() for st in o.dropped_statements],
                    "context": [c.model_dump() for c in o.context],
                    "context_tokens": o.context_tokens,
                    "usage": {k: v for k, v in o.usage.items() if k != "cache_keys"},
                    "judgments": judgments,
                    "trace": {k: v for k, v in o.trace.items() if k in ("filter", "quote_stats", "merge_rounds", "truncated_notes", "ranking", "notes", "final_notes", "abstained_no_statements")},
                }
            )
        return {
            "qid": qid,
            "question": q.question,
            "doc_id": q.doc_id,
            "title": exp.documents.get(q.doc_id, {}).get("title", q.doc_id),
            "unanswerable": q.unanswerable_all,
            "references": [r.model_dump() for r in q.references],
            "gold_chunk_ids": gold,
            "noise_level": noise,
            "noise_levels": sorted({k for (qq, k) in exp.instances if qq == qid}),
            "chunks": chunks,
            "predictions": preds,
        }

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
