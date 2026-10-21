"""Experiment orchestration: sample -> instances -> predictions -> judgments -> metrics.

Everything a run produces lives under ``runs/<name>/``:

manifest.json        config snapshot, prompt version, code version, package versions, tokenizer
questions.jsonl      sampled questions with references and gold evidence chunk sets
distractors.jsonl    every distractor chunk used (with its true origin in the ID)
instances.jsonl      (qid, noise level) -> exact ordered chunk IDs the systems received
predictions.jsonl    one SystemOutput per (system, noise level, question), incl. full trace
judgments.jsonl      evaluation-judge labels per (statement, citation) and joint
per_question.csv     every metric for every prediction
metrics.json         aggregates with bootstrap CIs and paired tests

Prediction and judgment steps are resumable: finished rows are skipped on re-run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from rich.progress import Progress

from . import corpus as corpus_mod
from .chunking import gold_chunk_ids
from .config import ExperimentConfig, make_client
from .data.qasper import sample_questions
from .noise import DistractorPool, inject
from .prompts import PROMPT_VERSION
from .retrieval import DenseRanker
from .schemas import Chunk, Question, SystemOutput
from .systems import Resources, run_system
from .tokens import tokenizer_name
from .verification import LLMVerifier, NLIVerifier


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict | str) -> None:
        line = row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut short by an interrupted run; it will be redone
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _versions() -> dict[str, str]:
    out = {}
    for pkg in ("provseg", "anthropic", "httpx", "tiktoken", "rank-bm25", "sentence-transformers", "transformers", "torch", "numpy"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.dir = cfg.run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._corpus: corpus_mod.Corpus | None = None
        self.questions: dict[str, Question] = {}
        self.gold: dict[str, list[set[str]]] = {}
        self.instances: dict[tuple[str, int], list[str]] = {}
        self.chunk_by_id: dict[str, Chunk] = {}
        self.documents: dict[str, dict] = {}

    # ---- setup ------------------------------------------------------------------

    @property
    def corpus(self) -> corpus_mod.Corpus:
        if self._corpus is None:
            d = self.cfg.data
            self._corpus = corpus_mod.load(d.processed_dir, d.split, self.cfg.chunking, raw_dir=d.raw_dir)
        return self._corpus

    def write_manifest(self, extra: dict | None = None) -> None:
        manifest = {
            "name": self.cfg.name,
            "created_utc": datetime.now(UTC).isoformat(),
            "prompt_version": PROMPT_VERSION,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "tokenizer": tokenizer_name(),
            "chunk_tag": corpus_mod.chunk_tag(self.cfg.chunking),
            "config": json.loads(self.cfg.model_dump_json()),
            **(extra or {}),
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def build(self) -> dict:
        """Sample questions and construct the matched noisy instances (deterministic)."""
        cfg = self.cfg
        corpus = self.corpus
        s = cfg.sample
        sample = sample_questions(corpus.questions, s.n_answerable, s.n_unanswerable, s.seed, s.max_per_doc)
        k_max = max(cfg.noise.levels)
        distractors: dict[str, list[Chunk]] = {}
        if k_max > 0:
            d = cfg.data
            pool_corpus = corpus_mod.load(d.processed_dir, d.pool_split, cfg.chunking, raw_dir=d.raw_dir)
            pool = DistractorPool(pool_corpus.all_chunks)
            for doc_id in sorted({q.doc_id for q in sample}):
                distractors[doc_id] = pool.draw(corpus.docs[doc_id], k_max, cfg.noise.seed, cfg.noise.mode)

        q_rows, inst_rows, dis_rows = [], [], []
        mapped = total = 0
        for q in sample:
            doc_chunks = corpus.chunks[q.doc_id]
            gold_sets, m, t = gold_chunk_ids(q, corpus.docs[q.doc_id], doc_chunks)
            mapped, total = mapped + m, total + t
            q_rows.append({**q.model_dump(), "gold_chunk_sets": [sorted(g) for g in gold_sets]})
            for k in cfg.noise.levels:
                seq = inject(doc_chunks, distractors.get(q.doc_id, []), k, cfg.noise.seed, q.doc_id)
                inst_rows.append({"qid": q.qid, "noise_level": k, "chunk_ids": [c.chunk_id for c in seq]})
        for doc_id, ds in distractors.items():
            for c in ds:
                dis_rows.append({"target_doc_id": doc_id, **c.model_dump()})

        # The run directory is self-contained: it carries the chunks and titles of every sampled
        # paper, so it can be audited or served without the processed corpus.
        doc_ids = sorted({q.doc_id for q in sample})
        chunk_rows = [c.model_dump() for d in doc_ids for c in corpus.chunks[d]]
        doc_rows = [{"doc_id": d, "title": corpus.docs[d].title, "license": corpus.docs[d].license, "source": corpus.docs[d].source} for d in doc_ids]
        for name, rows in (("questions", q_rows), ("instances", inst_rows), ("distractors", dis_rows), ("doc_chunks", chunk_rows), ("documents", doc_rows)):
            with (self.dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = {
            "questions": len(q_rows),
            "answerable": sum(1 for q in sample if not q.unanswerable_any),
            "unanswerable": sum(1 for q in sample if q.unanswerable_all),
            "papers": len({q.doc_id for q in sample}),
            "instances": len(inst_rows),
            "evidence_mapping": {"matched": mapped, "total": total},
        }
        self.write_manifest({"sample_stats": stats})
        return stats

    def load_built(self) -> None:
        q_rows = read_jsonl(self.dir / "questions.jsonl")
        if not q_rows:
            raise FileNotFoundError(f"{self.dir}/questions.jsonl missing; run `provseg build` first")
        for r in q_rows:
            gold = r.pop("gold_chunk_sets")
            q = Question.model_validate(r)
            self.questions[q.qid] = q
            self.gold[q.qid] = [set(g) for g in gold]
        local = read_jsonl(self.dir / "doc_chunks.jsonl")
        for c in (Chunk.model_validate(r) for r in local) if local else self.corpus.all_chunks:
            self.chunk_by_id[c.chunk_id] = c
        self.documents = {r["doc_id"]: r for r in read_jsonl(self.dir / "documents.jsonl")}
        for r in read_jsonl(self.dir / "distractors.jsonl"):
            r.pop("target_doc_id", None)
            c = Chunk.model_validate(r)
            self.chunk_by_id[c.chunk_id] = c
        for r in read_jsonl(self.dir / "instances.jsonl"):
            self.instances[(r["qid"], r["noise_level"])] = r["chunk_ids"]

    # ---- predictions ------------------------------------------------------------

    def resources(self, offline: bool = False, need_verifier: bool = True, need_dense: bool = False) -> Resources:
        cfg = self.cfg
        base = make_client(cfg.models[cfg.base_model], cfg.cache_dir, offline)
        verifier = None
        if need_verifier:
            v = cfg.verifier
            if v.kind == "nli":
                verifier = NLIVerifier(v.model, v.entail_threshold, v.contradict_threshold, tuple(v.window_sentences))
            else:
                verifier = LLMVerifier(make_client(cfg.models[v.model_ref or cfg.base_model], cfg.cache_dir, offline))
        return Resources(base=base, verifier=verifier, dense=DenseRanker() if need_dense else None, note_workers=cfg.note_workers)

    def predict(self, systems: list[str] | None = None, limit: int | None = None, offline: bool = False, noise_levels: list[int] | None = None) -> dict:
        self.load_built()
        cfg = self.cfg
        chosen = [s for s in cfg.systems if systems is None or s.name in systems]
        levels = noise_levels or cfg.noise.levels
        qids = list(self.questions)[: limit or None]
        pred_path = self.dir / "predictions.jsonl"
        done = {(r["system"], r["noise_level"], r["qid"]) for r in read_jsonl(pred_path) if not r.get("error")}
        tasks = [(s, k, qid) for s in chosen for k in levels for qid in qids if (s.name, k, qid) not in done]
        need_verifier = any(s.kind == "provseg" and s.verify for s in chosen)
        need_dense = any(s.kind == "rag" and s.retriever == "dense" for s in chosen)
        res = self.resources(offline, need_verifier, need_dense)
        writer = JsonlWriter(pred_path)
        errors = 0

        def work(task) -> SystemOutput:
            s, k, qid = task
            chunks = [self.chunk_by_id[c] for c in self.instances[(qid, k)]]
            return run_system(s, self.questions[qid], chunks, k, res)

        with Progress() as progress, ThreadPoolExecutor(max_workers=cfg.question_workers) as pool:
            bar = progress.add_task("predict", total=len(tasks))
            futures = [pool.submit(work, t) for t in tasks]
            for fut in as_completed(futures):
                out = fut.result()
                errors += out.error is not None
                writer.write(out.model_dump_json())
                progress.advance(bar)
        return {"tasks": len(tasks), "skipped_done": len(done), "errors": errors}

    def latest_predictions(self) -> list[SystemOutput]:
        """Last successful prediction per (system, noise, qid); errors kept only if nothing succeeded."""
        best: dict[tuple, SystemOutput] = {}
        for r in read_jsonl(self.dir / "predictions.jsonl"):
            out = SystemOutput.model_validate(r)
            key = (out.system, out.noise_level, out.qid)
            if key not in best or best[key].error is not None:
                best[key] = out
        return list(best.values())

    # ---- judging ----------------------------------------------------------------

    def judge(self, offline: bool = False, workers: int = 8) -> dict:
        from .evaluation.judge import LLMJudge, NLIJudge, judge_output

        self.load_built()
        jc = self.cfg.judge
        if jc.kind == "llm":
            judge = LLMJudge(make_client(self.cfg.models[jc.model_ref or "judge"], self.cfg.cache_dir, offline))
        else:
            if self.cfg.verifier.kind == "nli" and jc.nli_model == self.cfg.verifier.model:
                raise ValueError("the evaluation NLI judge must differ from the in-system verifier")
            judge = NLIJudge(jc.nli_model)
        path = self.dir / f"judgments_{_slug(judge.name)}.jsonl"
        done = {(r["system"], r["noise_level"], r["qid"]) for r in read_jsonl(path)}
        writer = JsonlWriter(path)
        chunk_text = {cid: c.text for cid, c in self.chunk_by_id.items()}
        n = 0
        outs = [o for o in self.latest_predictions() if (o.system, o.noise_level, o.qid) not in done and not o.error]
        with Progress() as progress:
            bar = progress.add_task("judge", total=len(outs))
            for out in outs:
                rows = judge_output(out, self.questions[out.qid].question, chunk_text, judge, workers)
                # A marker row records that this prediction was judged even if it had no citations.
                rows.append({"qid": out.qid, "system": out.system, "noise_level": out.noise_level, "scope": "__done__", "judge": judge.name})
                for r in rows:
                    writer.write(r)
                n += 1
                progress.advance(bar)
        return {"judged": n, "file": str(path)}

    def judgments(self) -> dict[tuple, list[dict]]:
        files = sorted(self.dir.glob("judgments_*.jsonl"))
        if not files:
            return {}
        preferred = [f for f in files if f.name.startswith("judgments_llm")] or files
        out: dict[tuple, list[dict]] = {}
        for r in read_jsonl(preferred[0]):
            key = (r["system"], r["noise_level"], r["qid"])
            out.setdefault(key, [])
            if r["scope"] != "__done__":
                out[key].append(r)
        return out


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
