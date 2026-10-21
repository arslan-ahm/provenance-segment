"""Gradio front end for the Hugging Face Space (ZeroGPU hardware).

Everything here renders what is on disk plus the model-free checks of ``live.py``. The NLI
call is injected as ``check_fn`` so the Space entry point can run it inside ``@spaces.GPU``
while tests pass a fake. Every string from data or from the visitor is HTML-escaped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from html import escape
from pathlib import Path

from .live import MAX_SOURCES, LiveError, handles_for
from .server import RunStore
from .views import NotFound, featured_questions, question_detail, question_list

REPO = "https://github.com/arslan-ahm/provenance-segment"
PAGE = "https://arslan-ahm.github.io/provenance-segment/"

CheckFn = Callable[..., dict]


def e(x) -> str:
    return escape("" if x is None else str(x))


def pct(v, digits: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{digits}f}"


# ---------------------------------------------------------------- renderers


def strip_row(label: str, chunks: list[dict], in_ctx: set[str]) -> str:
    cells = []
    for c in chunks:
        cls = "cell" + (" distractor" if c["is_distractor"] else "") + (" gold" if c["is_gold"] else "") + (" in" if c["chunk_id"] in in_ctx else "")
        tip = f'{c["handle"]} {c["heading"]}' + (" (distractor)" if c["is_distractor"] else "") + (" (gold)" if c["is_gold"] else "")
        cells.append(f'<span class="{cls}" title="{e(tip)}"></span>')
    return f'<div class="strip-row"><div class="name">{e(label)}</div><div class="cells">{"".join(cells)}</div></div>'


def render_question(d: dict) -> str:
    answers = sorted({r["answer"] for r in d["references"]})
    chips = "".join(f'<span class="chip">{e(a)}</span>' for a in answers)
    cc = d.get("computed_contexts") or {}
    ran = {p["system"] for p in d["predictions"]}
    rows = [strip_row("Document", d["chunks"], set())]
    for p in d["predictions"]:
        rows.append(strip_row(p["system"], d["chunks"], {cid for item in p["context"] for cid in item["chunk_ids"]}))
    computed = [(k, label) for k, label in (("rag_bm25", "BM25 top-k"), ("direct", "First chunks")) if cc and k not in ran]
    for k, label in computed:
        rows.append(strip_row(label, d["chunks"], set(cc[k]["chunk_ids"])))
    fit = ""
    if computed and cc["rag_bm25"]["gold_total"]:
        bm, di = cc["rag_bm25"], cc["direct"]
        share = round(100 * bm["distractor_tokens"] / bm["tokens"]) if bm["tokens"] else 0
        fit = (
            f'<p class="fit">Within the {cc["budget"]:,}-token budget, BM25 fits {bm["gold_in_context"]} of {bm["gold_total"]} gold chunks '
            f"(distractors take {share}% of it); taking the first chunks fits {di['gold_in_context']}. "
            "These rows use each system's own ranking and budget, with no model call.</p>"
        )
    n_dis = sum(c["is_distractor"] for c in d["chunks"])
    kind = '<span class="tag">every annotator: unanswerable</span>' if d["unanswerable"] else ""
    return (
        '<div class="pv">'
        f'<h2 class="q">{e(d["question"])}</h2>'
        f'<div class="paper">{e(d["title"])} <span class="mono">({e(d["doc_id"])})</span> {kind}</div>'
        f'<div class="refs"><span class="k">Annotator answers</span>{chips}</div>'
        f'<div class="panel strip">{"".join(rows)}'
        '<div class="legend"><span><span class="cell gold"></span>gold evidence</span><span><span class="cell distractor"></span>injected distractor</span>'
        '<span><span class="cell in"></span>in that context</span></div>'
        f'<div class="count">{len(d["chunks"])} chunks, {n_dis} of them injected from other papers.</div>{fit}</div>'
        "</div>"
    )


def mark_text(text: str, ranges: list[tuple[int, int]]) -> str:
    """Chunk text as HTML with ``ranges`` highlighted and section lines styled."""
    cuts = sorted({0, len(text), *[x for r in ranges for x in r if 0 <= x <= len(text)]})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        seg = e(text[a:b])
        if any(x <= a and b <= y for x, y in ranges):
            seg = f"<mark>{seg}</mark>"
        out.append(seg)
    html = "".join(out)
    lines = [f'<span class="sec">{ln[2:]}</span>' if ln.startswith("§ ") else ln for ln in html.split("\n")]
    return '<div class="source">' + "\n".join(lines) + "</div>"


VERDICT = {"entailed": ("good", "Supported"), "not_enough_evidence": ("warn", "Not enough evidence"), "contradicted": ("bad", "Contradicted")}


def render_check(r: dict, texts: dict[str, str]) -> str:
    if r.get("nli"):
        cls, label = VERDICT.get(r["nli"]["label"], ("neutral", r["nli"]["label"]))
        joint = r["nli"].get("joint")
        decided = ' <span class="sub">decided by the passages together</span>' if joint and joint["entail"] >= r["nli"]["entail"] else ""
        head = (
            f'<div class="rhead"><span class="badge {cls}">{e(label)}</span> <span class="score">entailment {round(100 * r["nli"]["entail"])}%</span>{decided}'
            f'<span class="model mono">{e(r["nli"]["model"])}</span></div>'
        )
    else:
        head = '<div class="rhead"><span class="badge neutral">Quote search only</span></div>'
    rows = []
    for s in r["sources"]:
        q, n = s["quote"], s.get("nli")
        ranges, note = [], ""
        if q["found"]:
            ranges, note = [(q["start"], q["end"])], "The claim occurs verbatim at the highlighted offsets."
        elif n and n.get("window_start") is not None:
            ranges, note = [(n["window_start"], n["window_end"])], "Highlighted: the window the verifier scored highest."
        quote = (
            f'<span class="badge good">verbatim at characters {q["start"]} to {q["end"]}{", with an elision" if q["elided"] else ""}</span>'
            if q["found"]
            else '<span class="badge neutral">not verbatim</span>'
        )
        score = f'<span class="score">entailment {round(100 * n["entail"])}%</span>' if n else ""
        dis = '<span class="badge neutral">distractor</span>' if s["is_distractor"] else ""
        window = f"<blockquote>{e(n['window'])}</blockquote>" if n and n.get("window") else ""
        passage = f"<details><summary>Show passage {e(s['handle'])}</summary><p class='sub'>{e(note)}</p>{mark_text(texts.get(s['chunk_id'], ''), ranges)}</details>"
        rows.append(f'<div class="src"><div class="tags"><span class="handle mono">{e(s["handle"])}</span>{quote}{score}{dis}</div>{window}{passage}</div>')
    note = f'<p class="sub">{e(r["fallback"])}</p>' if r.get("fallback") else ""
    return f'<div class="check">{head}{note}{"".join(rows)}</div>'


JUDGE = {"supported": ("good", "judge: supported"), "partially_supported": ("warn", "judge: partial"), "not_supported": ("bad", "judge: not supported"), "contradicted": ("bad", "judge: contradicted")}


def render_answers(d: dict) -> str:
    if not d["predictions"]:
        return (
            '<div class="empty"><b>No model answers for this question at this noise level yet.</b>'
            "<span>System answers, their cited statements, verifier decisions and judge labels appear here once the experiment has been run "
            "with a model provider. Everything above is computed without a model.</span></div>"
        )
    handle = {c["chunk_id"]: c["handle"] for c in d["chunks"]}
    distractor = {c["chunk_id"] for c in d["chunks"] if c["is_distractor"]}
    cards = []
    for p in d["predictions"]:
        judg = p.get("judgments") or []

        def stmt(st, idx, dropped, judg=judg, p=p):
            scope = st["citations"][0] if len(st["citations"]) == 1 else "joint"
            jl = next((j["label"] for j in judg if j["statement_idx"] == idx and j["scope"] == scope), None)
            cites = "".join(f'<span class="cite{" dis" if c in distractor else ""}">{e(handle.get(c, c))}</span>' for c in st["citations"])
            cites += "".join(f'<span class="cite bad" title="not in the final context">{e(raw)}</span>' for raw in st.get("invalid_citations", []))
            if not st["citations"] and not st.get("invalid_citations"):
                cites += '<span class="badge bad">no citation</span>'
            badges = ""
            if p["kind"] == "provseg" and st.get("verdict") in VERDICT:
                cls, label = VERDICT[st["verdict"]]
                badges += f'<span class="badge {cls}">verifier: {e(label.lower())}</span>'
            if jl in JUDGE:
                badges += f'<span class="badge {JUDGE[jl][0]}">{JUDGE[jl][1]}</span>'
            if dropped:
                badges += '<span class="badge neutral">removed by verifier</span>'
            return f'<div class="stmt{" dropped" if dropped else ""}"><p>{e(st["text"])}</p><div class="tags">{cites}{badges}</div></div>'

        answer = "Abstained: the context does not establish an answer." if p["abstained"] else (p["answer"] or "(empty)")
        body = "".join(stmt(st, i, False) for i, st in enumerate(p["statements"]))
        body += "".join(stmt(st, f"d{i}", True) for i, st in enumerate(p["dropped_statements"]))
        err = f'<div class="empty">Run error: {e(p["error"])}</div>' if p.get("error") else ""
        cards.append(f'<div class="card"><div class="sys mono">{e(p["system"])}</div>{err}<div class="ans">{e(answer)}</div>{body}</div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def _ci(c) -> str:
    if not c or c.get("mean") is None:
        return "<td>n/a</td>"
    return f'<td>{pct(c["mean"])}<span class="ci">{pct(c["low"])} to {pct(c["high"])}</span></td>'


def render_results(run_dir: Path, extra_dirs: list[tuple[str, Path]] = ()) -> str:
    parts = []
    retr = _load(run_dir / "diagnostics" / "retrieval.json")
    if retr:
        rows = "".join(
            f'<tr><td>{e(r["method"])}</td><td>{r["noise_level"]}</td>{_ci(r["gold_recall_at_budget"])}{_ci(r["all_gold_in_context"])}{_ci(r["distractor_token_share"])}</tr>'
            for r in retr["table"]
        )
        parts.append(
            f"<h3>Gold evidence that fits into the {retr['context_budget']:,}-token budget</h3>"
            "<p class='sub'>Retrieval only, no model calls; answerable questions, means with 95% bootstrap CIs (%).</p>"
            '<div class="tw"><table><thead><tr><th>Method</th><th>Noise</th><th>Gold recall</th><th>All gold in context</th><th>Distractor share</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    sanity = []
    for label, path in [("3-sentence windows, test sample", run_dir / "diagnostics" / "verifier_sanity.json")] + [
        (lbl, p) for lbl, p in extra_dirs
    ]:
        s = (_load(path) or {}).get("summary")
        if s:
            sanity.append(f"<tr><td>{e(label)}</td><td>{s['n']}</td><td>{pct(s['gold_entailed_rate'])}</td><td>{pct(s['distractor_entailed_rate'])}</td></tr>")
    if sanity:
        parts.append(
            "<h3>Verifier sanity check</h3><p class='sub'>Verbatim annotator evidence used as the claim: share entailed by its own chunk and by a random distractor chunk (%).</p>"
            '<div class="tw"><table><thead><tr><th>Setting</th><th>n</th><th>Own chunk</th><th>Distractor</th></tr></thead>'
            f"<tbody>{''.join(sanity)}</tbody></table></div>"
        )
    metrics = _load(run_dir / "metrics.json")
    if metrics:
        rows = "".join(
            f'<tr><td>{e(r["system"])}</td><td>{r["noise_level"]}</td>{_ci(r.get("answer_f1"))}{_ci(r.get("citation_precision"))}{_ci(r.get("unsupported_claim_rate"))}{_ci(r.get("distractor_citation_rate"))}</tr>'
            for r in metrics["table"]
        )
        parts.append(
            "<h3>Main comparison</h3>"
            '<div class="tw"><table><thead><tr><th>System</th><th>Noise</th><th>Answer F1</th><th>Citation precision</th><th>Unsupported claims</th><th>Distractor citations</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    else:
        parts.append(
            '<div class="empty"><b>Main comparison: not run yet.</b><span>Answer F1, citation precision, unsupported-claim rate, abstention and cost '
            "need the base model. No model result is shown until it has been produced by the published commands.</span></div>"
        )
    return '<div class="results">' + "".join(parts) + "</div>"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def claim_presets(d: dict) -> list[tuple[str, str]]:
    answers = [a for a in dict.fromkeys(r["answer"] for r in d["references"]) if a and a != "Unanswerable" and len(a) <= 600]
    evidence = [x for x in dict.fromkeys(x for r in d["references"] for x in r["highlighted_evidence"]) if 12 <= len(x) <= 600]
    return [(f"Evidence: {t}", t) for t in evidence[:3]] + [(f"Answer: {t}", t) for t in answers[:3]]


def passage_choices(d: dict) -> list[tuple[str, str]]:
    def label(c):
        return f'{c["handle"]}  {c["heading"]}' + ("  (gold)" if c["is_gold"] else "") + ("  (distractor)" if c["is_distractor"] else "")

    return [(label(c), c["chunk_id"]) for c in d["chunks"]]


def default_passages(d: dict) -> list[str]:
    if d["gold_chunk_ids"]:
        return [c for c in d["gold_chunk_ids"] if any(x["chunk_id"] == c for x in d["chunks"])][:MAX_SOURCES]
    scores = (d.get("computed_contexts") or {}).get("rag_bm25", {}).get("scores", {})
    return sorted(scores, key=lambda k: -scores[k])[:1]


def question_choices(exp) -> list[tuple[str, str]]:
    feats = featured_questions(exp)
    seen = {f["qid"] for f in feats}
    out = [(f"Featured: {f['question']}", f["qid"]) for f in feats]
    out += [(q["question"] + ("  (unanswerable)" if q["unanswerable"] else ""), q["qid"]) for q in question_list(exp) if q["qid"] not in seen]
    return out


# ---------------------------------------------------------------- app


CSS = """
.gradio-container { max-width: 1200px !important; }
.pv .q { font-size: 1.35rem; margin: 0 0 4px; }
.pv .paper, .sub, .count, .fit { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.pv .refs { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 10px 0 14px; }
.pv .refs .k { color: var(--body-text-color-subdued); font-size: 0.85rem; margin-right: 4px; }
.chip { border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 1px 9px; font-size: 0.82rem; }
.tag { font-size: 0.78rem; border: 1px solid var(--border-color-primary); border-radius: 4px; padding: 0 6px; margin-left: 6px; }
.panel { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px 14px; background: var(--background-fill-secondary); }
.strip-row { display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 10px; align-items: center; margin-bottom: 6px; }
.strip-row .name { font-size: 0.8rem; color: var(--body-text-color-subdued); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cells { display: flex; gap: 2px; }
.cell { flex: 1 1 0; min-width: 3px; height: 16px; border-radius: 2px; border: 1px solid var(--border-color-primary); background: var(--background-fill-primary); display: inline-block; }
.cell.distractor { background: repeating-linear-gradient(135deg, var(--background-fill-primary) 0 3px, var(--border-color-primary) 3px 4px); }
.cell.gold { background: #f2c94c; border-color: #b8860b; }
.cell.in { box-shadow: inset 0 -4px 0 var(--color-accent); border-color: var(--color-accent); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 0.8rem; color: var(--body-text-color-subdued); margin-top: 6px; }
.legend > span { display: inline-flex; align-items: center; gap: 6px; }
.legend .cell { flex: none; width: 14px; height: 11px; }
.count { margin-top: 6px; }
.fit { margin-top: 6px; max-width: 100ch; }
.mono { font-family: var(--font-mono); font-size: 0.86em; }
.badge { display: inline-block; font-size: 0.78rem; font-weight: 500; padding: 1px 8px; border-radius: 999px; background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); }
.badge.good { color: #0a7d0a; background: rgba(10, 125, 10, 0.1); border-color: rgba(10, 125, 10, 0.3); }
.badge.warn { color: #8a5a00; background: rgba(200, 140, 0, 0.12); border-color: rgba(200, 140, 0, 0.35); }
.badge.bad { color: #b42e2e; background: rgba(180, 46, 46, 0.1); border-color: rgba(180, 46, 46, 0.3); }
.dark .badge.good { color: #6fd36f; } .dark .badge.warn { color: #e8b44a; } .dark .badge.bad { color: #ef8a8a; }
.check { display: grid; gap: 12px; }
.rhead { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.rhead .model { margin-left: auto; color: var(--body-text-color-subdued); }
.score { font-family: var(--font-mono); font-size: 0.85rem; }
.src { border-top: 1px solid var(--border-color-primary); padding-top: 10px; display: grid; gap: 8px; min-width: 0; }
.tags { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.handle, .cite { border: 1px solid var(--border-color-primary); border-radius: 4px; padding: 0 6px; font-family: var(--font-mono); font-size: 0.8rem; }
.cite.dis { border-style: dashed; } .cite.bad { color: #b42e2e; }
blockquote { margin: 0; padding: 6px 10px; border-left: 3px solid #f2c94c; background: var(--background-fill-secondary); overflow-wrap: anywhere; }
details summary { cursor: pointer; color: var(--body-text-color-subdued); font-size: 0.88rem; }
.source { white-space: pre-wrap; line-height: 1.65; font-size: 0.92rem; overflow-wrap: anywhere; margin-top: 6px; }
.source .sec { font-weight: 600; color: var(--body-text-color-subdued); }
.source mark { background: rgba(242, 201, 76, 0.45); color: inherit; border-radius: 2px; }
.empty { border: 1px dashed var(--border-color-primary); border-radius: 8px; padding: 16px; display: grid; gap: 6px; color: var(--body-text-color-subdued); }
.empty b { color: var(--body-text-color); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap: 14px; }
.card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px 14px; display: grid; gap: 8px; align-content: start; }
.card .ans { font-weight: 600; }
.stmt { border-top: 1px solid var(--border-color-primary); padding-top: 8px; display: grid; gap: 6px; }
.stmt.dropped p { text-decoration: line-through; color: var(--body-text-color-subdued); }
.results h3 { margin: 18px 0 2px; }
.tw { overflow-x: auto; border: 1px solid var(--border-color-primary); border-radius: 8px; margin-top: 8px; }
.tw table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.tw th, .tw td { padding: 7px 12px; text-align: right; white-space: nowrap; border: 0 !important; }
.tw th:first-child, .tw td:first-child { text-align: left; }
.tw tbody tr { border-top: 1px solid var(--border-color-primary); }
.tw td { font-family: var(--font-mono); } .tw td:first-child { font-family: inherit; }
.ci { display: block; font-size: 0.75rem; color: var(--body-text-color-subdued); }
.intro { max-width: 80ch; }
.intro h1 { font-size: 1.7rem; margin: 6px 0 4px; letter-spacing: -0.02em; }
.intro p { color: var(--body-text-color-subdued); margin: 0; }
.intro .links { display: flex; flex-wrap: wrap; gap: 6px 16px; margin-top: 8px; font-size: 0.92rem; }
@media (max-width: 640px) { .strip-row { grid-template-columns: minmax(0, 1fr); gap: 2px; } }
"""

INTRO = f"""
<div class="intro">
<h1>Provenance-Segment</h1>
<p>Long-document question answering where every cited statement is checked against the exact source passage.
Explore QASPER questions with injected distractor passages, and test claims with the project's verifier.</p>
<div class="links"><a href="{REPO}" target="_blank" rel="noopener">Code</a><a href="{PAGE}" target="_blank" rel="noopener">Project page</a>
<a href="{REPO}/blob/main/docs/TECHNICAL_REPORT.md" target="_blank" rel="noopener">Technical report</a>
<a href="{REPO}/blob/main/docs/PROTOCOL.md" target="_blank" rel="noopener">Protocol</a></div>
</div>
"""

ABOUT = """
### What this demo runs

- **Question views** come from the frozen sample in `runs/main_v1` (130 QASPER test questions, each at 0, 10, 25 and 50 injected distractor chunks).
  The BM25 and first-chunks rows are built with the systems' own ranking and the same 2,048-token budget, without a model call.
- **Check a claim** runs Provenance-Segment's verification step: exact quote anchoring at character offsets, then the NLI verifier
  (`cross-encoder/nli-deberta-v3-small`, best of 1- and 3-sentence windows, supported at entailment 0.5 or above) on the original passage text.
  On this Space it runs on a ZeroGPU slice, which counts against the visitor's daily ZeroGPU quota (a second or two per check);
  when no GPU is available the same check runs on CPU and the result says so.
- **Model answers** (all systems, cited statements, verifier and judge labels) appear here once the main experiment has been run.
  Nothing on this page is model output until then.

QASPER: Dasigi et al., NAACL 2021, CC-BY-4.0. Segment+: Shi et al., EMNLP 2024.
"""

QUOTA_HINT = "The check could not run: {err}. On this Space the verifier uses a small daily ZeroGPU quota per visitor; signing in to Hugging Face raises it. Quote search still works."


def make_handlers(exp, check_fn: CheckFn):
    """The two event handlers, separate from the layout so they can be tested directly."""
    import gradio as gr

    def load(qid: str, noise: int):
        try:
            d = question_detail(exp, qid, int(noise))
        except (NotFound, TypeError, ValueError) as err:
            msg = f'<div class="empty"><b>{e(err.args[0] if err.args else "Pick a question first.")}</b></div>'
            return msg, gr.Dropdown(choices=[], value=None), "", gr.Dropdown(choices=[], value=[], multiselect=True), "", msg
        presets = claim_presets(d)
        return (
            render_question(d),
            gr.Dropdown(choices=presets, value=presets[0][1] if presets else None),
            presets[0][1] if presets else "",
            gr.Dropdown(choices=passage_choices(d), value=default_passages(d), multiselect=True),
            "",
            render_answers(d),
        )

    def run_check(qid: str, noise: int, claim: str, chunk_ids: list[str], nli: bool):
        ids = exp.instances.get((qid, int(noise))) if qid is not None and noise is not None else None
        if ids is None:
            return '<div class="empty"><b>Pick a question first.</b></div>'
        allowed = set(ids)
        chunks = [exp.chunk_by_id[c] for c in dict.fromkeys(chunk_ids or []) if c in allowed]
        try:
            r = check_fn(claim, chunks, handles_for(ids), nli)
        except LiveError as err:
            return f'<div class="empty"><b>{e(err)}</b></div>'
        except Exception as err:  # anything the GPU path could not recover from reaches the visitor with its reason
            return f'<div class="empty"><b>{e(QUOTA_HINT.format(err=err))}</b></div>'
        return render_check(r, {c.chunk_id: c.text for c in chunks})

    return load, run_check


def build_demo(runs_dir: str | Path, run_name: str, check_fn: CheckFn, extra_sanity: list[tuple[str, Path]] = ()):
    """Return (demo, launch_kwargs). ``check_fn(claim, chunks, handles, nli)`` returns a ClaimChecker result."""
    import gradio as gr

    exp = RunStore(Path(runs_dir)).get(run_name)
    load, run_check = make_handlers(exp, check_fn)
    choices = question_choices(exp)
    levels = sorted({k for (_, k) in exp.instances})
    first_noise = 50 if 50 in levels else levels[-1]

    theme = gr.themes.Base(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.stone, font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
                           font_mono=[gr.themes.GoogleFont("Geist Mono"), "ui-monospace", "monospace"], radius_size=gr.themes.sizes.radius_sm)

    with gr.Blocks(title="Provenance-Segment") as demo:
        gr.HTML(INTRO)
        with gr.Tabs():
            with gr.Tab("Explore"):
                with gr.Row(equal_height=True):
                    q = gr.Dropdown(choices=choices, value=choices[0][1], label="Question", scale=4, min_width=300, filterable=True)
                    noise = gr.Radio(choices=levels, value=first_noise, label="Distractor chunks injected", scale=2, min_width=260)
                view = gr.HTML()
                gr.Markdown("### Check a claim against the source\nExact quote anchoring, then the NLI verifier on the original text of the passages you choose. "
                            "Start from an annotator's evidence or answer, then edit it: change a number, negate it, or point it at a distractor.")
                preset = gr.Dropdown(label="Start from", choices=[], interactive=True)
                claim = gr.Textbox(label="Claim", lines=3, max_length=600)
                passages = gr.Dropdown(label=f"Source passages (up to {MAX_SOURCES})", choices=[], multiselect=True, max_choices=MAX_SOURCES)
                with gr.Row():
                    verify = gr.Button("Verify", variant="primary")
                    quote = gr.Button("Find as quote", variant="secondary")
                result = gr.HTML()
                gr.Markdown("### System answers")
                answers = gr.HTML()
            with gr.Tab("Results so far"):
                gr.HTML(render_results(exp.dir, list(extra_sanity)))
            with gr.Tab("About"):
                gr.Markdown(ABOUT)

        outs = [view, preset, claim, passages, result, answers]
        demo.load(load, [q, noise], outs)
        q.change(load, [q, noise], outs)
        noise.change(load, [q, noise], outs)
        preset.input(lambda v: v or "", preset, claim)
        verify.click(lambda a, b, c, d: run_check(a, b, c, d, True), [q, noise, claim, passages], result, concurrency_limit=4)
        quote.click(lambda a, b, c, d: run_check(a, b, c, d, False), [q, noise, claim, passages], result)

    return demo, {"theme": theme, "css": CSS}
