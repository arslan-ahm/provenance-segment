"""Command-line entry point: ``uv run provseg --help``."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_experiment

app = typer.Typer(add_completion=False, help="Provenance-Segment: verifiable long-document QA under a fixed context budget.")
console = Console()

ConfigOpt = typer.Option(..., "--config", "-c", help="Experiment YAML, e.g. configs/experiments/main.yaml")


def _exp(config: Path):
    from .runner import Experiment

    return Experiment(load_experiment(config))


@app.command()
def download(raw_dir: Path = typer.Option(Path("data/raw"))):
    """Download the QASPER parquet files (CC-BY-4.0) from Hugging Face."""
    from .data.qasper import download as dl

    for p in dl(raw_dir):
        console.print(f"[green]ok[/green] {p}")


@app.command()
def prepare(config: Path = ConfigOpt):
    """Parse and chunk the evaluation split and the distractor-pool split."""
    from . import corpus

    cfg = load_experiment(config)
    for split in dict.fromkeys([cfg.data.split, cfg.data.pool_split]):
        meta = corpus.prepare(cfg.data.raw_dir, cfg.data.processed_dir, split, cfg.chunking)
        console.print(meta)


@app.command()
def build(config: Path = ConfigOpt):
    """Sample questions and build the matched noisy instances (deterministic)."""
    stats = _exp(config).build()
    console.print_json(json.dumps(stats))


@app.command()
def estimate(config: Path = ConfigOpt):
    """Estimate calls, tokens and cost for a run without calling any model."""
    from .estimate import estimate as est

    r = est(_exp(config))
    t = Table(title=f"Estimate for base model {r['base_model']} ({r['instances']} instances, {r['unique_note_calls']} unique note calls)")
    for col in ("system", "calls", "input_tokens", "output_tokens", "cost_usd"):
        t.add_column(col)
    for row in r["systems"]:
        t.add_row(*(str(row[c]) for c in ("system", "calls", "input_tokens", "output_tokens", "cost_usd")))
    if r["judge"]:
        j = r["judge"]
        t.add_row(f"judge ({j['model']})", str(j["calls"]), str(j["input_tokens"]), str(j["output_tokens"]), str(j["cost_usd"]))
    console.print(t)
    console.print(f"Total ~ ${r['total_usd']} (rough; assumptions: {r['assumptions']})")


@app.command("retrieval-eval")
def retrieval_eval(config: Path = ConfigOpt, dense: bool = typer.Option(False, help="Also rank with a MiniLM bi-encoder (extra 'nli')")):
    """LLM-free diagnostic: gold evidence recall within the budget as noise grows."""
    from .diagnostics import run

    exp = _exp(config)
    run(exp, use_dense=dense)
    console.print((exp.dir / "diagnostics" / "RETRIEVAL.md").read_text(encoding="utf-8"))


@app.command()
def predict(
    config: Path = ConfigOpt,
    systems: str = typer.Option(None, help="Comma-separated system names (default: all)"),
    limit: int = typer.Option(None, help="Only the first N sampled questions (smoke runs)"),
    noise: str = typer.Option(None, help="Comma-separated noise levels (default: all)"),
    offline: bool = typer.Option(False, help="Replay from the cache only; a miss is an error"),
):
    """Run systems and append predictions (resumable)."""
    exp = _exp(config)
    r = exp.predict(
        systems=systems.split(",") if systems else None,
        limit=limit,
        offline=offline,
        noise_levels=[int(x) for x in noise.split(",")] if noise else None,
    )
    console.print(r)


@app.command()
def judge(config: Path = ConfigOpt, offline: bool = typer.Option(False), workers: int = typer.Option(8)):
    """Judge every (statement, citation) pair with the evaluation judge (resumable)."""
    console.print(_exp(config).judge(offline=offline, workers=workers))


@app.command()
def evaluate(config: Path = ConfigOpt, reference: str = typer.Option("rag_bm25"), focus: str = typer.Option("provseg")):
    """Compute metrics, CIs, paired tests, failure examples and REPORT.md."""
    from .evaluation import failures
    from .evaluation.aggregate import aggregate, per_question_rows, write_outputs
    from .report import render

    exp = _exp(config)
    exp.load_built()
    rows = per_question_rows(exp)
    if not rows:
        raise typer.BadParameter("no predictions yet; run `provseg predict` first")
    agg = aggregate(rows, reference_system=reference, focus_system=focus)
    write_outputs(exp, rows, agg)
    failures.write(exp, failures.collect(exp, rows))
    console.print(f"report: {render(exp.dir)}")


@app.command("audit-export")
def audit_export(config: Path = ConfigOpt, n: int = typer.Option(100), seed: int = typer.Option(0)):
    """Export a random sample of judged citation pairs as CSV for human audit of the judge."""
    exp = _exp(config)
    exp.load_built()
    rows = [j for js in exp.judgments().values() for j in js if j["scope"] != "joint"]
    rows.sort(key=lambda j: hashlib.sha256(f"{seed}|{j['system']}|{j['noise_level']}|{j['qid']}|{j['statement_idx']}|{j['scope']}".encode()).hexdigest())
    rows = rows[:n]
    out = exp.dir / "audit_sample.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "noise_level", "qid", "statement_idx", "chunk_id", "question", "claim", "source_text", "judge_label", "human_label", "human_notes"])
        for j in rows:
            w.writerow(
                [j["system"], j["noise_level"], j["qid"], j["statement_idx"], j["scope"], exp.questions[j["qid"]].question, j["claim"], exp.chunk_by_id[j["scope"]].text, j["label"], "", ""]
            )
    console.print(f"wrote {len(rows)} rows to {out} (fill human_label with supported / partially_supported / not_supported / contradicted)")


@app.command("audit-score")
def audit_score(csv_path: Path):
    """Agreement between the judge and the human labels in a filled audit CSV."""
    with csv_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip()]
    if not rows:
        raise typer.BadParameter("no human labels filled in yet")
    agree = sum(r["human_label"].strip() == r["judge_label"] for r in rows)
    binary = sum((r["human_label"].strip() == "supported") == (r["judge_label"] == "supported") for r in rows)
    console.print({"labelled": len(rows), "exact_agreement": agree / len(rows), "supported_vs_not_agreement": binary / len(rows)})


@app.command()
def serve(runs_dir: Path = typer.Option(Path("runs")), host: str = "127.0.0.1", port: int = 8000):
    """Start the provenance explorer web app."""
    import uvicorn

    from .app.server import create_app

    uvicorn.run(create_app(runs_dir), host=host, port=port)


if __name__ == "__main__":
    app()
