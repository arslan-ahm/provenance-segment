---
title: Provenance-Segment
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
license: mit
short_description: Check long-document QA claims against their exact sources
models:
  - cross-encoder/nli-deberta-v3-small
datasets:
  - allenai/qasper
---

# Provenance-Segment

Interactive companion to [Provenance-Segment](https://github.com/arslan-ahm/provenance-segment):
source-verifiable long-document question answering with short-context LLMs, evaluated on
QASPER (CC-BY-4.0) with injected distractor passages.

What you can do here today, with no model provider and no API key:

- **Explore** the 130 frozen questions at 0, 10, 25 or 50 injected distractor chunks: where the
  gold evidence sits, and which chunks BM25 top-k and a first-chunks baseline fit into the same
  2,048-token budget. The first entries are chosen by a fixed rule from the retrieval diagnostic
  (BM25 fits all of their gold evidence into the budget with no noise, but not with 50
  distractors), followed by unanswerable ones.
- **Check a claim against the source**: exact quote anchoring (character offsets) and the
  project's NLI verifier (`cross-encoder/nli-deberta-v3-small`, 1- and 3-sentence windows) on
  the original text of the passages you choose. It runs on a ZeroGPU slice, which counts against
  your daily ZeroGPU quota (signing in raises it); when no GPU is available the same check runs on
  CPU and the result says so.
- **Results so far**: the retrieval diagnostic and the verifier sanity checks, with 95% CIs.

System answers, cited statements, verifier decisions and judge labels appear once the main
experiment has been run (see the repository). Nothing on this Space is model output until then.

- Code, protocol, results: https://github.com/arslan-ahm/provenance-segment
- Project page: https://arslan-ahm.github.io/provenance-segment/
- Technical report: https://github.com/arslan-ahm/provenance-segment/blob/main/docs/TECHNICAL_REPORT.md
