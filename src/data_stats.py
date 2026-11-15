"""
data_stats.py -- the numbers for the data slide and the Methodology dataset paragraph.

    python -m src.data_stats

Reads data/pairs_150.jsonl, data/pairs_150_verbose.jsonl and data/pairs_150_stats.json.
Writes results/tables/t0_data.{csv,md,tex} and results/data_stats.json.
"""
import json
import statistics as st
from collections import Counter
from pathlib import Path

from src.metrics import write_table

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = {
    "claude_3.7_sonnet": "Claude 3.7 Sonnet", "claude_4_sonnet": "Claude 4 Sonnet",
    "claude_4_opus": "Claude 4 Opus", "gemini_2.5_pro": "Gemini 2.5 Pro",
    "gemini_2.5_flash": "Gemini 2.5 Flash", "gemini_2.5_flash_lite": "Gemini 2.5 Flash-Lite",
    "qwen3_235b": "Qwen3-235B",
}


def lines(s: str) -> int:
    return s.count("\n") + 1


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "pairs_150.jsonl")]
    pool = json.loads((ROOT / "data" / "pairs_150_stats.json").read_text())["pool"]
    vpath = ROOT / "data" / "pairs_150_verbose.jsonl"
    verbose = [json.loads(l) for l in open(vpath)] if vpath.exists() else []

    lc = [lines(r["correct"]) for r in rows]
    li = [lines(r["incorrect"]) for r in rows]
    S = {
        "n_pairs": len(rows), "n_answers": 2 * len(rows),
        "pool_pairs": pool["raw_pairs"], "pool_problems": pool["distinct_questions"],
        "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "platform": dict(Counter(r["platform"] for r in rows)),
        "generator": dict(Counter(r["source_split"] for r in rows)),
        "lines_correct_mean": round(st.mean(lc), 1), "lines_correct_median": st.median(lc),
        "lines_incorrect_mean": round(st.mean(li), 1), "lines_incorrect_median": st.median(li),
        "correct_longer": sum(a > b for a, b in zip(lc, li)),
        "incorrect_longer": sum(a < b for a, b in zip(lc, li)),
        "problem_chars_median": st.median(len(r["problem"]) for r in rows),
    }
    if verbose:
        added = [lines(v["incorrect_verbose"]) - lines(v["incorrect"]) for v in verbose]
        S["exp2_lines_added_mean"] = round(st.mean(added), 1)
        S["exp2_padded_longer_than_correct"] = sum(
            lines(v["incorrect_verbose"]) > lines(v["correct"]) for v in verbose)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "data_stats.json").write_text(json.dumps(S, indent=2))

    d, p = S["difficulty"], S["platform"]
    gens = ", ".join(f"{GENERATOR.get(k, k)} {v}" for k, v in sorted(S["generator"].items(), key=lambda x: -x[1]))
    table = [
        ["Source", f"CodeJudgeBench, code generation ({S['pool_pairs']:,} pairs, {S['pool_problems']} problems)"],
        ["Sample", f"{S['n_pairs']} pairs = {S['n_answers']} answers, one pair per problem, seed 42"],
        ["Ground truth", "Hidden unit tests: one answer passes all, the other fails at least one"],
        ["Difficulty", f"{d.get('easy', 0)} easy / {d.get('medium', 0)} medium / {d.get('hard', 0)} hard"],
        ["Platform", f"{p.get('leetcode', 0)} LeetCode / {p.get('atcoder', 0)} AtCoder / {p.get('codeforces', 0)} Codeforces"],
        ["Answer length (lines)", f"correct {S['lines_correct_mean']} mean, {S['lines_correct_median']:g} median; "
                                  f"incorrect {S['lines_incorrect_mean']} mean, {S['lines_incorrect_median']:g} median"],
        ["Longer answer", f"correct in {S['correct_longer']} pairs, incorrect in {S['incorrect_longer']}"],
        ["Answers written by", gens],
    ]
    if verbose:
        table.append(["Exp 2 padding", f"+{S['exp2_lines_added_mean']} comment lines on average; padded wrong answer "
                                       f"is longer than the correct one in {S['exp2_padded_longer_than_correct']} pairs"])
    write_table(out / "tables", "t0_data", "Evaluation data", ["", "Value"], table,
                colspec=r"p{0.27\columnwidth}p{0.66\columnwidth}")
    print(json.dumps(S, indent=2))
    print("wrote results/data_stats.json and results/tables/t0_data.*")


if __name__ == "__main__":
    main()
