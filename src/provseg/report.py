"""Markdown report and figures generated from ``metrics.json`` and ``failures.json``."""

from __future__ import annotations

import json
from pathlib import Path

HEADLINE = [
    ("answer_f1", "Answer F1", True),
    ("citation_precision", "Citation prec.", True),
    ("unsupported_claim_rate", "Unsupported claims", False),
    ("evidence_f1", "Evidence F1", True),
    ("distractor_citation_rate", "Distractor cites", False),
]


def _fmt(cell: dict | None, pct: bool = True) -> str:
    if not cell or cell.get("mean") is None:
        return "n/a"
    m, lo, hi = cell["mean"], cell.get("low"), cell.get("high")
    if pct:
        return f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]" if lo is not None else f"{100 * m:.1f}"
    return f"{m:.2f}"


def _num(x, pct=True) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}" if pct else f"{x:.3g}"


def figures(metrics: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    table = metrics["table"]
    systems = sorted({r["system"] for r in table})
    paths = []
    for key, label, _ in HEADLINE[:3]:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
        for s in systems:
            pts = sorted((r["noise_level"], r[key]) for r in table if r["system"] == s and r[key]["mean"] is not None)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [100 * p[1]["mean"] for p in pts]
            lo = [100 * (p[1]["mean"] - p[1]["low"]) for p in pts]
            hi = [100 * (p[1]["high"] - p[1]["mean"]) for p in pts]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, linewidth=1.5, label=s)
        ax.set_xlabel("Distractor chunks injected")
        ax.set_ylabel(f"{label} (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        p = out_dir / f"{key}_vs_noise.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def render(run_dir: Path) -> Path:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    failures_path = run_dir / "failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else {}
    figs = figures(metrics, run_dir / "figures")
    cfg = manifest["config"]
    L: list[str] = []
    L.append(f"# Results: {manifest['name']}\n")
    L.append(
        f"Generated from `{run_dir.as_posix()}`. Prompt version `{manifest['prompt_version']}`, code `{manifest.get('git_commit') or 'uncommitted'}`, "
        f"tokenizer `{manifest['tokenizer']}`, base model `{cfg['models'].get(cfg['base_model'], {}).get('model', cfg['base_model'])}`, "
        f"context budget {cfg['context_budget']} tokens, noise mode `{cfg['noise']['mode']}`.\n"
    )
    L.append("Values are means over questions with 95% bootstrap CIs, in percent. Answer F1 and evidence F1 are over answerable questions.\n")
    for k in sorted({r["noise_level"] for r in metrics["table"]}):
        L.append(f"\n## Noise level {k}\n")
        L.append("| System | n | " + " | ".join(h[1] for h in HEADLINE) + " | Abstain F1 | False abstain | Calls | Cost $ |")
        L.append("|" + "---|" * (len(HEADLINE) + 6))
        for r in [r for r in metrics["table"] if r["noise_level"] == k]:
            L.append(
                f"| {r['system']} | {r['n']} | "
                + " | ".join(_fmt(r[key]) for key, _, _ in HEADLINE)
                + f" | {_num(r.get('abstain_f1'))} | {_num(r.get('false_abstention_rate'))} | {_fmt(r['calls'], pct=False)} | {_fmt(r['cost_usd'], pct=False)} |"
            )
    if metrics.get("paired"):
        L.append("\n## Paired comparisons (focus - reference)\n")
        L.append("| Noise | Metric | A | B | n | Diff (pp) | 95% CI | p (sign-flip) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in metrics["paired"]:
            if p["diff"] is None:
                continue
            L.append(f"| {p['noise_level']} | {p['metric']} | {p['a']} | {p['b']} | {p['n']} | {100 * p['diff']:+.1f} | [{100 * p['low']:+.1f}, {100 * p['high']:+.1f}] | {p['p_value']:.4f} |")
    if metrics.get("robustness"):
        L.append("\n## Change relative to the clean document (same questions)\n")
        L.append("| System | Metric | Noise | Delta (pp) | 95% CI | n |")
        L.append("|---|---|---|---|---|---|")
        for r in metrics["robustness"]:
            if r["delta_vs_clean"] is None:
                continue
            L.append(f"| {r['system']} | {r['metric']} | {r['noise_level']} | {100 * r['delta_vs_clean']:+.1f} | [{100 * r['low']:+.1f}, {100 * r['high']:+.1f}] | {r['n']} |")
    for f in figs:
        L.append(f"\n![{f.stem}](figures/{f.name})")
    if failures:
        L.append("\n## Failure examples\n")
        L.append("Counts per category (all systems and noise levels); up to 8 deterministic examples each are in `failures.json`.\n")
        for key in sorted(k for k in failures if k.endswith("__count")):
            L.append(f"- `{key[:-7]}`: {failures[key]}")
        for cat in ("unsupported_kept", "false_rejection", "distractor_cited", "fabricated_quote"):
            for ex in failures.get(cat, [])[:2]:
                L.append(f"\n**{cat}** ({ex['system']}, noise {ex['noise_level']}, `{ex['qid'][:10]}`): {ex['question']}")
                if "statement" in ex:
                    L.append(f"> {ex['statement']}  \n> cites: {', '.join(ex.get('citations', []))}")
                if "quote" in ex:
                    L.append(f"> quote not found in [{ex['handle']}]: \"{ex['quote'][:300]}\"")
                if ex.get("judge_rationale"):
                    L.append(f"\nJudge: {ex['judge_rationale']}")
    out = run_dir / "REPORT.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
