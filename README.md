# Provenance-Segment

**Source-verifiable long-document question answering with short-context LLMs.**

Provenance-Segment answers a question about a long document and shows, for every claim in
the answer, the exact passage of the original document that supports it. It extends the
Segment+ idea (Shi et al., EMNLP 2024: per-segment notes that separate *evidence* from
*reasoning*, filtered and merged until they fit a short context window) with three
provenance guarantees:

1. **Quotes are anchored, not trusted.** Every evidence quote a model writes into a note is
   located at exact character offsets in its source chunk. A quote that does not occur
   verbatim is removed, so a summary can never become the only evidence.
2. **Provenance survives merging.** When notes are merged to fit the budget, each quote and fact
   keeps the chunk it came from; merged quotes are re-anchored, invented links are dropped.
3. **Every statement is checked against the original source.** The answer's statements are
   verified by NLI against the *original text* of the chunks they cite (never against the
   notes). Unsupported statements are revised once or removed; if nothing survives, the system
   abstains.

The research question: *can chunk-level evidence tracking and claim verification reduce
unsupported answers in long-document QA, without a long-context model?*
The pre-registered hypothesis, data, settings and metrics are in
[docs/PROTOCOL.md](docs/PROTOCOL.md).

| | |
|---|---|
| Interactive demo | https://huggingface.co/spaces/ARslan-Ahamd/provenance-segment |
| Project page | https://arslan-ahm.github.io/provenance-segment/ |
| Technical report | [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) |
| Companion project | [GraphTrace-Bench](https://github.com/arslan-ahm/graphtrace-bench) |

## Status

| Part | State |
|---|---|
| Pipeline, four systems, ablations, evaluation, statistics, report generator | built, 104 tests passing |
| Frozen sample and noisy instances (`runs/main_v1/`) | **built from real QASPER data and committed** |
| Retrieval diagnostic (no LLM) | **run on real data**, results below |
| NLI verifier sanity check (no LLM) | **run on real data**: 3-sentence windows entailed only 65% of verbatim evidence on test, so window sizes were re-chosen on validation (100% / 5% false entailment); see `docs/PROTOCOL.md` |
| Main experiment with the base model | **not run yet**: needs an API key and a budget decision (`provseg estimate`) |
| Human audit of the judge | pending the main run |

No model-generated result is reported anywhere in this repository until it has been produced
by the commands below; the explorer shows empty states until then.

## First result: evidence that reaches a fixed budget (no LLM)

Answerable questions from the frozen sample (n = 100), 2,048-token context budget, random
distractor chunks from other papers. Share of gold evidence chunks that fit into the budget
(mean, 95% bootstrap CI):

| Distractor chunks | 0 | 10 | 25 | 50 |
|---|---|---|---|---|
| BM25 top-k | 77.6 [69.2, 85.2] | 67.0 [57.9, 75.7] | 61.5 [52.5, 71.0] | 52.0 [42.4, 61.8] |
| First chunks (direct) | 42.8 [33.5, 52.0] | 19.2 [12.2, 27.5] | 8.5 [4.0, 14.5] | 3.0 [0.5, 6.5] |

Even good retrieval loses a third of the gold evidence by 50 distractors, and distractors take
39.5% of its budget. This is the input-side motivation for reading every segment (Segment+)
and for checking what the answer finally cites. Full table: `runs/main_v1/diagnostics/RETRIEVAL.md`.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra anthropic --extra nli --extra report --extra app
uv run provseg download                                   # QASPER parquet (CC-BY-4.0)
uv run provseg prepare -c configs/experiments/main.yaml   # parse + chunk test and train splits
uv run provseg build   -c configs/experiments/main.yaml   # frozen sample + noisy instances (deterministic)
uv run provseg retrieval-eval -c configs/experiments/main.yaml
uv run provseg estimate -c configs/experiments/main.yaml  # calls, tokens and USD before spending anything
```

Run the experiment (calls the model provider; set `ANTHROPIC_API_KEY`, or point
`configs/models.yaml` at any OpenAI-compatible endpoint or a local Ollama model):

```bash
uv run provseg predict  -c configs/experiments/smoke.yaml          # tiny dev run first
uv run provseg predict  -c configs/experiments/main.yaml           # resumable
uv run provseg judge    -c configs/experiments/main.yaml
uv run provseg evaluate -c configs/experiments/main.yaml           # metrics.json, per_question.csv, failures.json, REPORT.md
uv run provseg audit-export -c configs/experiments/main.yaml       # 100 pairs for human labelling
uv run provseg audit-score runs/main_v1/audit_sample.csv
```

Explore any run in the browser:

```bash
uv run provseg serve      # http://127.0.0.1:8000
```

`provseg estimate` for the main run on `claude-opus-5-5` gives roughly 7,971 unique note
calls and about $320 in total including the judge (rough; assumptions are printed). The
same run with the `small` model in `configs/models.yaml` costs a small fraction of that and is
a sensible first full pass.

## How it works

```
document ──chunk (stable IDs, owned paragraph spans)──► chunks ──(+ injected distractors)──► instance
                                                                                               │
  direct:   first chunks until the budget is full ──────────────────────────────────────────┐ │
  rag:      BM25 top chunks until the budget is full ───────────────────────────────────────┤ │
  segment:  note per chunk → keep/maybe/remove → merge batches until notes fit ─────────────┤ │
  provseg:  segment + quote anchoring + provenance-checked merge ───────────────────────────┤ │
                                                                                            ▼
                         same answer prompt, same budget, neutral handles P01..Pnn → answer + cited statements
                                                                                            │
  provseg only:  strip out-of-context citations → NLI(statement, ORIGINAL cited chunks) → revise / drop / abstain
                                                                                            │
  evaluation:    QASPER F1 · evidence P/R · separate LLM judge per (statement, citation) · bootstrap CIs · paired tests
```

Every run writes `runs/<name>/` with the config snapshot, prompt version, package versions,
the exact chunk sequence each system saw, every prediction with its full trace, judge labels,
per-question metrics and failure examples. Every raw model call is stored in a
content-addressed cache (`cache/llm/`), so `--offline` replays a run byte for byte.

## Repository map

| Path | What |
|---|---|
| `src/provseg/chunking.py`, `quotes.py`, `noise.py` | provenance-preserving chunking, exact quote anchoring, matched noise |
| `src/provseg/systems.py` | the four systems behind one interface |
| `src/provseg/verification.py` | NLI and LLM verifiers (claims vs original source) |
| `src/provseg/evaluation/` | official QASPER scoring, judge, metrics, bootstrap/permutation statistics, failure mining |
| `src/provseg/llm/` | Anthropic (official SDK) and OpenAI-compatible clients, strict JSON, raw-call cache |
| `src/provseg/app/` | explorer (FastAPI + static front end) and its model-free live checks (`live.py`) |
| `site/`, `space/`, `scripts/deploy_space.py` | project page, Space card, Space deployment |
| `configs/` | models and pre-registered experiments |
| `runs/main_v1/` | frozen sample, instances, distractors, diagnostics |
| `docs/` | protocol, technical report, 5-minute demo script |

## Hosting

**Explorer (Hugging Face Space).** The explorer never calls a model provider, so it can be
public. Besides the committed runs it offers model-free checks: the contexts the `direct` and
`rag_bm25` systems would build under the budget, exact quote anchoring, and the project's NLI
verifier run on CPU against the original passages (`PROVSEG_LIVE_NLI=0` turns that off). The
`Dockerfile` bakes the NLI model into the image. Publish with the `hf` CLI (`hf auth login` first):

```bash
uv run python scripts/deploy_space.py --space ARslan-Ahamd/provenance-segment --dry-run
uv run python scripts/deploy_space.py --space ARslan-Ahamd/provenance-segment
```

The script uploads the committed `Dockerfile`, project files, `src/` and `runs/`, with
`space/README.md` as the Space card. Any container host works too:
`docker build -t provseg-explorer . && docker run -p 7860:7860 provseg-explorer`.

**Project page (GitHub Pages).** `site/` is a static page deployed by
`.github/workflows/pages.yml`. Enable it once under *Settings > Pages > Source: GitHub Actions*.

**Automation.** `.github/workflows/ci.yml` runs lint and tests on every push.
`.github/workflows/space.yml` re-deploys the Space when `src/`, `runs/` or `space/` change on
`main`, if the repository has an `HF_TOKEN` secret (a write token); without it the job is skipped.

## What is in git

Code, configs, docs and frozen run directories (sample, instances, diagnostics, and later
predictions/metrics). Raw QASPER files, the processed corpus and the raw model-call cache are
not; the first two are rebuilt by `download` + `prepare`, the cache is published as a release
asset next to each results run.

## Citation and licences

Code: MIT. QASPER: Dasigi et al., *A Dataset of Information-Seeking Questions and Answers
Anchored in Research Papers*, NAACL 2021, CC-BY-4.0. Segment+: Shi et al., *SEGMENT+: Long Text
Processing with Short-Context Language Models*, EMNLP 2024.
