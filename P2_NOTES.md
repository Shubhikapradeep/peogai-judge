# P2 notes: data, metrics, tables and figures

## Data check (resolved 15 Nov)

The first batch of results mixed two samples: `exp1_3b` and `exp1_7b` ran on the pre-fix sample, and `exp3_3b` was partly on it. P1 re-ran those three on the current `data/pairs_150.jsonl`. All eight runs now match it on every row (0 mismatches), and none match the old sample, which differs on 196 rows. `metrics.py` repeats this check on every run and records it in `metrics.json` under `data_check`. Those runs predate `question_id` logging, so the check uses difficulty labels. Future runs record `question_id` and are checked exactly.

## Status

| Plan task | Status |
|---|---|
| 1. Check the dataset | Done. `mattymchen/codejudgebench`, config `codegen`, 7 splits (named by generator model), 2,103 pairs, 676 distinct problems |
| 2. `sample_data.py` → `data/pairs_150.jsonl` | Done (script by P1, one fix by P2, see below). 150 pairs, 50 easy / 50 medium / 50 hard, seed 42 |
| 3. Data stats | Done: `python -m src.data_stats` → `results/tables/t0_data.*` (data slide) and `results/data_stats.json` |
| 4. Prompts | Done by P1 in `prompts/`. Reviewed, not changed |
| 5. `perturb.py` → `data/pairs_150_verbose.jsonl` | Done (script by P1). 144 of 150 answers got step comments, 6 got only the header comment |
| 6. `src/metrics.py` | Done, tested on synthetic results |
| 7. Tables and charts | **Done** on the final results: `results/tables/`, `results/figures/` |
| 8. 3–5 main findings | **Done**: `results/findings.md` (every claim backed by a test, p < 0.05) |
| 9. Results section | **Done**: `report/results_auto.tex` (about 2 pages: 4 tables, the position-bias and fixes figures, one-line summary). `python -m src.write_results --full` adds the other figures and the difficulty table. `report/results.tex` is the old hand-fill template, now superseded |
| Methodology text (for P4) | `report/methodology_p2.tex`: Dataset, Judge prompts and Metrics subsections, ready to paste |

## Run it

```bash
pip install matplotlib                       # metrics.py itself needs only the standard library
python -m src.metrics                        # results/raw/*.jsonl -> results/metrics.json, results/key_numbers.md, results/tables/
python -m src.plots                          # results/metrics.json -> results/figures/*.png (slides) and *.pdf (LaTeX)
python -m src.write_results                  # -> results/findings.md and report/results_auto.tex
python -m src.data_stats                     # data slide table (only needed if the data changes)
```
`write_results.py` calls an effect real only when its test gives p < 0.05, and says "not significant" otherwise. It cannot tell which result is the most interesting, so read `findings.md`, reorder or trim it to 3–5 points, and check that every sentence still matches the figures.
Point `--raw` at the Drive folder if the raw files haven't been copied into the repo yet. Missing experiments are skipped, so you can run this after each experiment finishes.

## Outputs

- `results/key_numbers.md`: every headline number in one list. Use it to write findings and to check the report and slides (P4 task 5).
- `results/tables/t1_pairwise`, `t2_pointwise`, `t3_fixes`, `t4_verbosity`, `t5_difficulty`, each as `.csv`, `.md` (slides) and `.tex` (paste into Overleaf with `\input`).
- `results/figures/f1_position_bias`, `f2_pointwise`, `f3_fixes`, `f4_verbosity`, `f5_difficulty`. Each model has the same colour in every figure (3B blue, 7B orange).

## Definitions (use the same wording in Methodology)

- **Parse failure** counts as wrong in every accuracy and is also reported on its own.
- **Consistency**: % of pairs where the judge picks the same underlying answer in both orders (right-right or wrong-wrong). If the same letter comes back twice (AA or BB), the verdict followed the position.
- **% A verdicts**: share of parsed pairwise verdicts that are `A`. An unbiased judge gives 50% on this design, since the correct answer is A in exactly half the calls.
- **Exp 4a, order averaging**: keep the verdict only when both orders agree. Report accuracy on decided pairs together with coverage, the % of pairs decided. Accuracy alone overstates the gain.
- **FAR** = incorrect answers judged `correct` / all incorrect answers. **FRR** = correct answers judged `incorrect` / all correct answers.
- **Uncertainty**: 95% Wilson CIs. Paired comparisons (fix vs baseline, padded vs plain) use an exact McNemar test on the same calls. Use these when the viva asks "are 150 pairs enough?".

## Data notes (for Methodology and Limitations)

- **Fix to `sample_data.py`:** the same problem often has a different difficulty label in different splits. Of the 508 problems that appear in more than one split, 416 (82%) disagree. The sampler stratified on one row's label but saved the label of a different, randomly picked row, so the saved split was 56 / 40 / 54 instead of 50 / 50 / 50. It now stratifies on the label of the row it actually saves. **If anyone generated `pairs_150.jsonl` with the old version, discard that copy and use the one committed here.**
- Because the labels are this noisy, treat the by-difficulty results (Table 5, Fig 5) as indicative only, and say so in Limitations.
- Answers are code only: the last fenced block of each generator response, without its reasoning.
- Sample: 78 LeetCode, 71 AtCoder and 1 Codeforces problem. Correct and incorrect answers have almost the same length (mean 64.8 vs 65.3 lines, median 59 vs 59.5). The correct one is longer in 65 of 150 pairs. (An earlier version of this note gave 64.7 vs 58.3; those figures came from the sample before the fix and were wrong.)
- Exp 2 padding adds 16.9 comment lines on average and makes the wrong answer longer than the correct one in 127 of 150 pairs. Exp 2 therefore tests comments and extra length together, not comments alone.
- If P1 ran on the transformers fallback (`--backend hf`), input is cut at 6,000 tokens. By a character-count estimate, 2–5 pairwise prompts are longer than that and would lose their final instruction. Check for parse failures on those pairs. vLLM runs (8,192-token limit) are not affected.

## For P1

Raw files must stay `results/raw/<exp>_<model>.jsonl` (for example `exp1_3b.jsonl`). `metrics.py` skips files with `_smoke` in the name and keeps only the last row for each `job_id`.

## Viva answers for P2's part

- **Why 150 pairs, and are the results reliable?** At a 50% rate, the 95% CI is about ±5.7 points over 300 calls and ±8.0 over 150 pairs. Every rate in the report has its CI, so a reader can see which differences are larger than the noise. Comparisons on the same calls use an exact McNemar test, which counts only the calls where the two conditions disagree, so it is more sensitive than comparing two CIs.
- **How did you measure position bias?** Every pair is judged in both orders. Position bias shows up in three ways: an accuracy gap between the two orders (McNemar test), a share of A verdicts away from 50% (binomial test), and pairs where the same letter comes back twice (AA or BB).
- **Why does FAR matter more than accuracy for code auditing?** A false acceptance ships a bug, while a false rejection only costs a second review. A judge that says "correct" to everything scores 50% accuracy on our balanced data but has a FAR of 100%.
- **Why are parse failures counted as wrong?** Dropping them would reward a judge for refusing to answer on hard items. Their rate is reported separately, so the reader can see their effect.
- **Why does order averaging report coverage?** It only gives a verdict when both orders agree, and the pairs it skips are the hard ones. Accuracy on the decided pairs alone would overstate the gain.
- **Data caveats to admit up front:** difficulty labels in CodeJudgeBench disagree across splits for 82% of the problems that appear in more than one split, so the difficulty breakdown is indicative only. Exp 2 padding also makes the wrong answer longer, so Exp 2 measures comments and length together.
