# Provenance-Segment: Verifiable Long-Document QA under a Fixed Context Budget

*Technical report, working draft. Arslan Ahmad. Version of 2026-10-21.*

> **Status of results.** Sections 5.1 and 5.2 report experiments that were actually run
> (no LLM involved). Sections 5.3 to 5.6 describe experiments whose settings are frozen in
> `docs/PROTOCOL.md` but which have **not been run yet**; they contain no numbers by design.
> They are filled by `provseg evaluate`, which writes `runs/main_v1/REPORT.md`.

## Abstract

Long-document question answering with short-context models has to compress: retrieve a few
chunks, or summarise every segment and merge the summaries (Segment+, Shi et al., 2024).
Compression is where evidence is lost or fabricated, and a final answer that cites a source
does not show that the source still says what the answer claims. We present
Provenance-Segment, which keeps chunk-level provenance through every compression step and
verifies each answer statement against the original text of the chunks it cites. Quotes in
intermediate notes are anchored to exact character offsets or removed; merged notes keep the
chunk of every quote and fact; statements that fail natural-language-inference verification
are revised once or removed, and the system abstains when nothing survives. We evaluate on
QASPER with matched distractor injection (0/10/25/50 chunks) under a 2,048-token final context
budget against direct-prefix, BM25-RAG and Segment+-style baselines, scoring citations with a
judge that is separate from the in-system verifier. A retrieval-only diagnostic already shows
that BM25 keeps 77.6% of gold evidence within the budget on clean papers and 52.0% with 50
distractors, motivating approaches that read every segment and check what they finally cite.

## 1. Introduction

Two families of methods let short-context models answer questions about long documents.
Retrieval-augmented generation (RAG) selects a few chunks; segment-and-merge methods such as
Segment+ read every segment, write structured notes that separate *evidence* (verbatim
sentences, for precision) from *reasoning* (compressed context, for recall), filter unhelpful
notes and merge the rest until they fit the window. Both produce answers whose grounding is hard
to audit: RAG may never retrieve the needed passage, and segment notes are model-written text
in which quotes can be paraphrased or invented, and merges can attach a fact to the wrong source.

We ask: **can chunk-level evidence tracking and claim verification reduce unsupported answers in
long-document QA without a long-context model?** Our contributions:

1. A provenance-preserving variant of segment-and-merge QA in which every quote is anchored to
   character offsets in its source chunk and every merged fact keeps its chunk handles.
2. Claim verification against original source text (never against notes), with revise, drop and
   abstain policies.
3. A controlled evaluation protocol: identical base model, answer prompt, citation format and
   context budget across systems; matched, nested distractor injection; neutral citation handles
   that do not reveal which chunks are distractors; an evaluation judge separate from the verifier;
   bootstrap CIs and paired permutation tests; a human audit of the judge.

## 2. Related work

- **Segment+** (Shi et al., EMNLP 2024) introduced structured evidence/reasoning notes with
  filtering and batch merging for short-context models, evaluated on long-document QA and
  needle-in-a-haystack tasks. We reuse its note structure and add provenance and verification.
- **Attributed QA and citation evaluation.** ALCE (Gao et al., 2023) defines citation recall and
  precision with an NLI judge; we adopt its precision definition at the (statement, citation) level.
- **Long-document QA benchmarks.** QASPER (Dasigi et al., 2021) provides questions over NLP papers
  with paragraph-level evidence and unanswerable questions, which makes evidence and abstention
  measurable.
- **Post-hoc verification and revision** of generated text against sources (e.g., RARR, Gao et al.,
  2023) motivates the revise-or-drop step; our variant constrains revision to the cited chunks only.

## 3. Method

### 3.1 Chunking with provenance

Paragraphs are packed in document order up to 550 tokens (cap 700). Section boundaries are
preferred breaks once a chunk holds 300 tokens, so short adjacent sections share a chunk and long
ones are split; overlong paragraphs are split at sentence boundaries; chunks that start mid-section
carry a 50-token overlap marked as not owned. Each chunk records the exact paragraphs it owns,
`(section, first paragraph, last paragraph)`, and gets a stable ID such as
`doc_1912_01214_sec_04_chunk_01`. On the QASPER test split this gives a median of 11 chunks per
paper (median 449 tokens per chunk) and maps 96.1% of annotator evidence paragraphs to chunks.

### 3.2 Notes, filtering and merging

For every chunk, the model writes a note with `relevance`, up to four verbatim
`evidence_quotes`, `facts`, a `reasoning_note` and what the chunk does not establish. Quotes are
located in the chunk after light normalisation (Unicode, whitespace, typographic quotes and
dashes; never case or wording). Unlocatable quotes are counted and, in Provenance-Segment, removed.
Notes rated high are kept, medium are kept as "maybe", low/none are removed. If the rendered notes
exceed the budget they are merged in batches; the merge prompt requires quotes to be copied
unchanged with their single source handle, and we re-anchor every merged quote and keep only
handles from the merged batch. If merging stops shrinking the notes, the least relevant notes are
dropped deterministically.

### 3.3 Answering with neutral citation handles

All systems use one answer prompt that returns a concise answer and a list of statements, each
citing context items. The model sees handles `P01..Pnn` assigned in document order, never chunk
IDs, so distractor chunks cannot be recognised from their labels. Handles are resolved to stable
chunk IDs deterministically; handles that never entered the final context are recorded and, in
Provenance-Segment, stripped.

### 3.4 Verification

Each statement is checked against the original text of its cited chunks with an NLI cross-encoder
(`nli-deberta-v3-small`). Because a chunk exceeds the model's input, the premise is scored as
three-sentence windows and the best window counts; for multi-citation statements, the best windows
of all cited chunks are also scored jointly. A statement is supported if entailment probability is
at least 0.5. Unsupported statements are revised once using only the cited passages and re-verified;
otherwise they are dropped. If any statement changed, the final answer is regenerated from the
surviving statements; if none survive, the system answers "Unanswerable".

## 4. Experimental setup

Fully specified in `docs/PROTOCOL.md`. In brief: QASPER test, 100 answerable and 30 unanswerable
questions over 113 papers (seed 13); distractors from QASPER train papers injected at 10/25/50
chunks with nested positions; final context budget 2,048 tokens measured on the exact rendered
context; base model `claude-opus-5-5`; systems `direct`, `rag_bm25`, `segment`, `provseg`,
`provseg_noverify`; evaluation judge `claude-sonnet-5-5`, separate from the verifier; 95% bootstrap
CIs and paired sign-flip permutation tests.

## 5. Results

### 5.1 How much gold evidence fits into the budget (run; no LLM)

For each answerable question with mapped evidence (n = 100) and each noise level, we rank chunks
and fill the 2,048-token budget, then measure the share of gold evidence chunks inside it.

| Method | Noise | Gold recall @ budget | All gold in context | Distractor share of budget | MRR of first gold |
|---|---|---|---|---|---|
| BM25 | 0 | 77.6 [69.2, 85.2] | 75.0 [66.0, 83.0] | 0.0 | 52.3 [45.9, 59.5] |
| BM25 | 10 | 67.0 [57.9, 75.7] | 65.0 [56.0, 74.0] | 22.6 [18.2, 27.5] | 47.4 [41.0, 54.0] |
| BM25 | 25 | 61.5 [52.5, 71.0] | 60.0 [51.0, 70.0] | 34.7 [29.2, 40.6] | 42.7 [36.3, 49.8] |
| BM25 | 50 | 52.0 [42.4, 61.8] | 50.0 [40.0, 60.0] | 39.5 [33.0, 45.8] | 42.3 [35.7, 49.9] |
| Direct | 0 | 42.8 [33.5, 52.0] | 41.0 [32.0, 50.0] | 0.0 | 28.7 [24.3, 33.6] |
| Direct | 10 | 19.2 [12.2, 27.5] | 18.0 [11.0, 26.0] | 50.1 [45.9, 54.3] | 15.6 [12.9, 18.6] |
| Direct | 25 | 8.5 [4.0, 14.5] | 7.0 [3.0, 13.0] | 73.1 [69.4, 76.5] | 9.1 [7.7, 10.6] |
| Direct | 50 | 3.0 [0.5, 6.5] | 2.0 [0.0, 5.0] | 87.7 [84.7, 90.4] | 5.5 [4.7, 6.5] |

Even at zero noise, a quarter of questions lose part of their gold evidence under BM25 at this
budget; at 50 distractors half do, and distractors occupy about 40% of the context. The
segment-based systems read every chunk, so their ceiling is 1.0 and their losses move to the
filter, merge and answer stages, which the main experiment measures.

### 5.2 Verifier sanity check (run; no LLM)

VERIFIER_SANITY_PLACEHOLDER

### 5.3 Main comparison (pending)

Answer F1, evidence P/R/F1, citation precision, unsupported-claim rate and distractor-citation
rate for the five systems at each noise level, with CIs. Generated into `runs/main_v1/REPORT.md`.

### 5.4 Hypothesis test (pending)

Paired differences provseg minus rag_bm25 at noise 50 for unsupported-claim rate and answer F1,
with the decision rule of the protocol.

### 5.5 Abstention and false rejection (pending)

Abstention precision/recall/F1 on the 30 unanswerable questions; false abstention rate on the
100 answerable ones; statements removed by the verifier that the judge considers supported.

### 5.6 Cost (pending)

Calls, tokens, latency and USD per answered question for every system.

## 6. Failure analysis (pending)

`provseg evaluate` extracts deterministic examples for: unsupported statements that were kept
(verifier misses), supported statements that were removed (false rejections), citations of
distractors, answers that flip under noise, wrong abstentions, hallucinated answers to unanswerable
questions, and fabricated note quotes. These are reported in full, including those unfavourable to
the method.

## 7. Limitations

Single dataset (QASPER); synthetic distractors; LLM-judged support (audited on 100 pairs); base
model and judge from the same family unless the cross-family judge is used; token F1 penalises
verbose but correct answers; NLI verification of paraphrased claims is imperfect, so the
verification step can both miss subtle hallucinations and reject valid multi-hop reasoning, both
of which are measured rather than assumed away.

## 8. Reproducibility

`uv run provseg build` reproduces the sample and instances byte for byte from the public QASPER
files. Every raw model call is stored in a content-addressed cache and published with the results;
`provseg predict --offline` replays a run without network access. The run manifest records the
config, prompt version, code commit, package versions and tokenizer.

## References

- Dasigi, P., Lo, K., Beltagy, I., Cohan, A., Smith, N. A., Gardner, M. (2021). A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. NAACL.
- Gao, T., Yen, H., Yu, J., Chen, D. (2023). Enabling Large Language Models to Generate Text with Citations (ALCE). EMNLP.
- Gao, L., Dai, Z., Pasupat, P., et al. (2023). RARR: Researching and Revising What Language Models Say, Using Language Models. ACL.
- Shi, W., Li, S., Yu, K., et al., Xiao, Y. (2024). SEGMENT+: Long Text Processing with Short-Context Language Models. EMNLP.
