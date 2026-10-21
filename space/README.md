---
title: Provenance-Segment Explorer
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Check long-document QA claims against their exact sources
---

# Provenance-Segment Explorer

Interactive companion to [Provenance-Segment](https://github.com/arslan-ahm/provenance-segment):
source-verifiable long-document question answering with short-context LLMs, evaluated on
QASPER (CC-BY-4.0) with injected distractor passages.

What you can do here today, with no model provider and no API key:

- **Start here** on the overview: questions chosen by a fixed rule from the retrieval diagnostic
  (BM25 keeps all gold evidence with no noise but loses it with 50 distractors), plus
  unanswerable ones.
- **See the evidence problem** for any of the 130 frozen questions at 0, 10, 25 or 50 injected
  distractor chunks: where the gold evidence sits, and which chunks BM25 top-k and a
  first-chunks baseline fit into the same 2,048-token budget.
- **Check a claim against the source**: exact quote anchoring (character offsets) and the
  project's NLI verifier (`cross-encoder/nli-deberta-v3-small`, 1- and 3-sentence windows),
  run live on CPU against the original text of the passages you choose. Start from an
  annotator's answer or evidence, then edit it or point it at a distractor.

System answers, cited statements, verifier decisions and judge labels appear in the same
views once the main experiment has been run (see the repository). Nothing on this page is
model output until then.

- Code, protocol, results: https://github.com/arslan-ahm/provenance-segment
- Project page: https://arslan-ahm.github.io/provenance-segment/
- Technical report: https://github.com/arslan-ahm/provenance-segment/blob/main/docs/TECHNICAL_REPORT.md
