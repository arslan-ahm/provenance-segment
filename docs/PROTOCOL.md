# Pre-registered protocol: Provenance-Segment

Status: **frozen on 2026-10-21, before any model prediction was generated.** The sample,
noise instances, prompts and metrics below are fixed by `configs/experiments/main.yaml`,
prompt version `2026-10-21.1`, and the committed files in `runs/main_v1/`. Changes after
the first prediction go into a new config with a new run name, and are reported as such.

## Research question

Can chunk-level evidence tracking and claim verification reduce unsupported answers in
long-document question answering, without a long-context model?

## Primary hypothesis

> Compared with **standard BM25 top-k RAG** under the same 2,048-token final context budget
> and the same base model, **Provenance-Segment** will reduce the **unsupported-claim rate**
> on **QASPER test questions with 50 injected distractor chunks**, while keeping **QASPER
> answer F1 on answerable questions** no worse than RAG's by more than **2.0 points**.

Decision rule. The hypothesis is supported if, at noise level 50:

1. the paired difference in unsupported-claim rate (provseg minus rag_bm25) has a 95% bootstrap
   CI entirely below 0, and the two-sided sign-flip permutation test gives p < 0.05; and
2. the lower bound of the 95% bootstrap CI of the paired answer-F1 difference is at least -2.0 points.

Both conditions are also reported at noise levels 0, 10 and 25 (descriptive, not confirmatory).

## Secondary questions (descriptive)

- S1. Does citation precision degrade more slowly with noise for provseg than for RAG
  (difference of the noise-50 minus noise-0 deltas)?
- S2. Does verification change abstention quality on unanswerable questions (abstention F1)
  and how often does it wrongly abstain on answerable ones (false abstention rate)?
- S3. Ablations (`configs/experiments/ablations.yaml`): drop vs revise, atomic vs statement
  claims, no quote anchoring, no verification.
- S4. Cost: model calls, tokens, latency and USD per answered question.

## Data

- QASPER (Dasigi et al., 2021; CC-BY-4.0), Hugging Face parquet export.
- Evaluation split: **test** (416 papers, 1,451 questions). The validation split is used only
  for development (`configs/experiments/smoke.yaml`) and never reported.
- Sample: 100 answerable questions (no annotator marked them unanswerable) and 30 unanswerable
  questions (every annotator marked them unanswerable), seed 13, at most 2 questions per paper:
  130 questions over 113 papers. Questions on which annotators disagree about answerability
  are excluded so abstention labels are unambiguous.
- Distractor pool: QASPER **train** papers, so no distractor comes from an evaluation paper.

## Fixed settings

| Setting | Value |
|---|---|
| Chunking | section-aware, target 550 / max 700 / min 300 tokens, 50-token overlap mid-section, `tiktoken cl100k_base` |
| Chunks per paper | median 11 (test split) |
| Gold evidence | annotator evidence paragraphs mapped to owning chunks; 423 of 443 evidence strings mapped (95.5%) in the sample |
| Noise | 0 / 10 / 25 / 50 random distractor chunks, nested (each level contains the previous one at the same positions), seed 7 |
| Context budget | 2,048 tokens of final context, measured on the exact rendered text, for every system |
| Base model | `claude-opus-5-5`, effort `medium`, thinking managed by the API (no sampling parameters accepted) |
| Answer prompt | identical for all systems; JSON with a concise answer and cited statements |
| Citation handles | neutral `P01..Pnn` in document order; distractors indistinguishable by label |
| In-system verifier | `cross-encoder/nli-deberta-v3-small`, 1- and 3-sentence windows, entail threshold 0.5 (window setting chosen on the validation split, see below) |
| Evaluation judge | `claude-sonnet-5-5` rubric judge (separate model, never used inside a system) |
| Human audit | 100 randomly sampled judged citation pairs, labelled blind to system (`provseg audit-export`) |

## Verifier setting (decided before any prediction, on validation data only)

A sanity check on the frozen test sample (verbatim gold highlighted-evidence sentences as claims, n = 60)
showed that 3-sentence windows entail only 65.0% of verbatim claims against their own chunk (1.7% false
entailment against a distractor chunk). To avoid tuning on test data, window settings were compared on a
separate validation-split sample (`configs/experiments/verifier_dev.yaml`, n = 60):

| Windows | Verbatim claim entailed by its chunk | Entailed by a distractor chunk | CPU s / pair |
|---|---|---|---|
| 3 sentences | 71.7% | 3.3% | 6.0 |
| 1 and 3 sentences | 100.0% | 5.0% | 8.0 |

`[1, 3]` was adopted. This check uses verbatim claims, the easy case; how the verifier handles paraphrased
model statements is measured in the main run through false rejections and the human audit.

## Systems

| Name | Context construction | Verification |
|---|---|---|
| `direct` | first chunks in document order until the budget is full | none |
| `rag_bm25` | BM25 top-ranked chunks until the budget is full | none |
| `segment` | Segment+-style notes for every chunk, filter (keep high/medium), batch merge until the notes fit | none |
| `provseg` | `segment` + quotes anchored to exact offsets (unanchored quotes removed), provenance-checked merges, out-of-context citations stripped | NLI check of every statement against the original text of its cited chunks; revise once, else drop; re-answer; abstain if nothing survives |
| `provseg_noverify` | as `provseg` | none (ablation) |

## Metrics

Per question, then averaged with 95% percentile bootstrap CIs (2,000 resamples):

- **Answer F1**: official QASPER token F1, max over annotators; abstention is the string `Unanswerable`.
- **Evidence P/R/F1**: cited chunk IDs vs gold evidence chunks, max over annotators.
- **Citation precision** (ALCE-style): a (statement, citation) pair is precise if that chunk alone supports the statement, or partially supports it while the statement is supported jointly.
- **Unsupported-claim rate**: share of answer statements not supported by their citations; uncited statements count as unsupported.
  It is defined only for answers that contain statements, so a system could lower it by abstaining more.
  That is why the primary hypothesis pairs it with a non-inferiority bound on answer F1 over answerable
  questions (abstaining on an answerable question scores F1 = 0), and why the false abstention rate and the
  number of statements per answer are reported next to it.
- **Out-of-context citation rate**: emitted citations naming no item of the final context.
- **Distractor citation rate**: citations pointing to injected distractor chunks.
- **Abstention P/R/F1** over all 130 questions and **false abstention rate** on answerable questions.
- **False rejections**: statements provseg removed that the judge considers supported.
- **Efficiency**: model calls, input/output tokens, provider latency, USD.

Paired comparisons use the same questions; p-values from a two-sided sign-flip permutation test.

## Threats to validity

- **Judge validity.** Support is decided by an LLM judge. Mitigations: the judge is a different
  model from the base model and from the in-system verifier; a 100-pair human audit reports judge
  agreement; a cross-family judge (`judge_openai` in `configs/models.yaml`) can be swapped in.
- **Same-family bias.** Base model and judge are both Claude models. Reported as a limitation;
  the cross-family judge rerun is the remedy.
- **Token F1 penalises verbose answers.** The same prompt asks every system for a concise answer;
  an optional correctness judge is available for a second view.
- **Evidence mapping.** 4.5% of evidence strings (mostly table/figure references) do not map to a
  chunk; those references are excluded from evidence metrics, not counted as misses.
- **Synthetic noise.** Random distractors from other papers are easier to ignore than in-document
  noise. `configs/experiments/hard_noise.yaml` (BM25-similar distractors) is the harder variant.
- **Nondeterminism.** The base model takes no temperature; every raw call is cached and
  committed as a release asset so the exact outputs can be replayed (`--offline`).
- **Single dataset.** QASPER only. Generalisation to other long-document sets is not claimed.
