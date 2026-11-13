# P1 handoff notes

## Files
- `src/judge.py`: `load_backend()` + `judge()` / `judge_batch()` / `judge_both_orders()` (the last one is for P3's demo)
- `src/run_experiment.py`: batch runner (exp1, exp3, exp4b, exp4c, exp2), Drive checkpoints, resume
- `src/sample_data.py`, `src/perturb.py`: data and the Exp 2 perturbation
- `src/setup_summary.py`: meta files → table for the Experimental Setup section
- `notebooks/p1_judge_runs.ipynb`: Colab/Kaggle notebook, run top to bottom
- `report/experimental_setup.tex`: Section IV draft, red [blanks] to fill

## For P2: data and prompts already exist (P1 ran first)
P1 made these so the runs weren't blocked. **Don't regenerate or edit them after the runs start**, or the results won't match the data.
- `src/sample_data.py` → `data/pairs_150.jsonl` + `data/pairs_150_stats.json`
  (mattymchen/codejudgebench, `codegen`, all splits pooled, one pair per question, 50/50/50 by difficulty, seed 42,
  answers = last fenced code block of `pos_response` / `neg_response`)
- `src/perturb.py` → `data/pairs_150_verbose.jsonl` (adds `incorrect_verbose`; only comments are added, checked by AST equality)
- `prompts/pairwise_v1.txt`, `prompts/pairwise_debias.txt`, `prompts/pointwise_v1.txt`: the prompts actually used
- P2's tasks 1–5 become: review these, use the stats for the data slide, and describe them in Methodology. Then go on to `metrics.py`.

## Raw result rows (for metrics.py)
`results/raw/<exp>_<model>.jsonl`, one row per call:
- pairwise: `pair_id, order ("cf" = correct shown as A, "cs" = correct shown as B), gold, verdict, parse_ok, correct, raw`
- pointwise: `pair_id, answer ("correct"/"incorrect"), gold, verdict, parse_ok, correct, raw`
- exp4c also has `samples` (the 3 parsed votes); `verdict` = majority (None on tie)
- Exp 4a (order-averaging) = from exp1: a pair counts as decided only if `cf` and `cs` pick the same answer

## For P3 (demo)
```python
from src.judge import load_backend, judge_both_orders
b = load_backend("ollama", "3b")          # ollama pull qwen2.5-coder:3b
r = judge_both_orders(b, problem, code_1, code_2)
r["position_bias"], r["final"]            # final = 1 / 2 / None (orders disagree)
```
