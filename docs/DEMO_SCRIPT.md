# Five-minute demo script

Audience: a professor or reviewer who knows long-context LLM work. Show the explorer
(`uv run provseg serve`) next to the terminal. Timings are targets.

## 0:00 - 0:40  The problem

"Long-document QA systems compress: they retrieve a few chunks or summarise segments. The answer
then cites something, but nobody checks that what is cited still says what the answer claims.
This project asks whether tracking evidence at the chunk level and verifying each claim against
the original text reduces unsupported answers, without a long-context model."

## 0:40 - 1:40  The setup is fixed before any model runs

Open **Overview**.
- Point at the run description and the stat strip: 130 QASPER test questions, 113 papers,
  520 matched instances at 0/10/25/50 distractor chunks.
- "Every number on this page so far was produced without an LLM." Show the retrieval chart:
  BM25 keeps 77.6% of gold evidence in a 2,048-token budget on clean papers and 52.0% with 50
  distractors; reading chunks in order keeps 3.0%. "That is why Segment+ reads every segment,
  and why I check what is finally cited."
- Mention the protocol is pre-registered in `docs/PROTOCOL.md`, including the exact decision rule.

## 1:40 - 3:20  One question, end to end

Open **Questions**, pick a question, switch the distractor control from 0 to 50.
- The document strip: yellow cells are gold evidence, hatched cells are injected distractors,
  blue underline marks what each system put into its final context.
- Click a citation chip on the provseg card: the source panel opens on the original chunk, with the
  note's quotes highlighted at their exact character offsets.
- Show a removed statement under "removed by verification" and the verifier and judge badges
  side by side: "the verifier is inside the system; the judge is a different model used only
  for scoring, so the method is not graded by its own checker."
- Open the notes list: underlined quotes are ones the model wrote that do not exist in the source.

## 3:20 - 4:20  Results and failures

(After the main run.) Show the three charts: answer F1, citation precision, unsupported claims vs
noise, with CIs. Then open `runs/main_v1/REPORT.md` at the failure examples:
verifier misses, false rejections, distractor citations. "These are reported, not hidden."

## 4:20 - 5:00  Reproducibility and next step

- `provseg predict --offline` replays every model call from the content-addressed cache.
- Next: cross-family judge, the hard-noise variant, and the knowledge-graph extension
  (GraphTrace-Bench) that asks whether the evidence was *used* through a valid reasoning path.
