"""Processed corpus on disk: documents, questions and chunks per split as JSONL.

``prepare`` is deterministic; the chunking config is hashed into the file name so a
changed chunker can never silently reuse stale chunks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import ChunkingConfig, chunk_document
from .data.qasper import load_split
from .schemas import Chunk, Document, Question


def chunk_tag(cfg: ChunkingConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:8]


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


def _read_jsonl(path: Path, model):
    with path.open(encoding="utf-8") as f:
        return [model.model_validate_json(line) for line in f if line.strip()]


@dataclass
class Corpus:
    split: str
    docs: dict[str, Document]
    questions: list[Question]
    chunks: dict[str, list[Chunk]]

    @property
    def all_chunks(self) -> list[Chunk]:
        return [c for cs in self.chunks.values() for c in cs]


def paths(processed_dir: str | Path, split: str, cfg: ChunkingConfig) -> dict[str, Path]:
    d = Path(processed_dir)
    return {
        "docs": d / f"{split}_docs.jsonl",
        "questions": d / f"{split}_questions.jsonl",
        "chunks": d / f"{split}_chunks_{chunk_tag(cfg)}.jsonl",
    }


def prepare(raw_dir: str | Path, processed_dir: str | Path, split: str, cfg: ChunkingConfig) -> dict[str, int]:
    p = paths(processed_dir, split, cfg)
    docs, questions = load_split(raw_dir, split)
    _write_jsonl(p["docs"], docs.values())
    _write_jsonl(p["questions"], questions)
    chunks = [c for d in docs.values() for c in chunk_document(d, cfg)]
    _write_jsonl(p["chunks"], chunks)
    meta = {"split": split, "docs": len(docs), "questions": len(questions), "chunks": len(chunks), "chunking": cfg.model_dump()}
    (Path(processed_dir) / f"{split}_chunks_{chunk_tag(cfg)}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load(processed_dir: str | Path, split: str, cfg: ChunkingConfig, raw_dir: str | Path | None = None) -> Corpus:
    p = paths(processed_dir, split, cfg)
    if not all(x.exists() for x in p.values()):
        if raw_dir is None:
            raise FileNotFoundError(f"processed corpus missing for split={split}; run `provseg prepare`")
        prepare(raw_dir, processed_dir, split, cfg)
    docs = {d.doc_id: d for d in _read_jsonl(p["docs"], Document)}
    questions = _read_jsonl(p["questions"], Question)
    chunks: dict[str, list[Chunk]] = {}
    for c in _read_jsonl(p["chunks"], Chunk):
        chunks.setdefault(c.doc_id, []).append(c)
    return Corpus(split=split, docs=docs, questions=questions, chunks=chunks)
