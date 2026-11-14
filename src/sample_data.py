"""
sample_data.py -- build data/pairs_150.jsonl from CodeJudgeBench (codegen task).

    pip install datasets
    python -m src.sample_data                    # 150 pairs, seed 42, code-only answers
    python -m src.sample_data --full-response    # keep the generator's full response (reasoning + code)

Dataset: mattymchen/codejudgebench, config "codegen". Splits are named after the model
that generated the candidate answers (claude_3.7_sonnet, gemini_2.5_pro, ...). Fields used:
question_id, question_content, starter_code, difficulty, pos_response (passes all unit
tests), neg_response (fails at least one).

Sampling: pool all splits -> drop rows whose answers contain no code -> one pair per
distinct question (random pick) -> stratify evenly by difficulty -> seed 42.
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(resp: str):
    """Last fenced block, preferring python-tagged blocks. None if there is no fenced code."""
    blocks = FENCE.findall(resp or "")
    if not blocks:
        return None
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
    code = (py or [b for _, b in blocks])[-1].strip()
    return code or None


def load_rows():
    from datasets import load_dataset
    dd = load_dataset("mattymchen/codejudgebench", "codegen")
    rows = []
    for split, ds in dd.items():
        for r in ds:
            rows.append(dict(r, source_split=split))
    return rows


def build_problem(r):
    p = r["question_content"].strip()
    if (r.get("starter_code") or "").strip():
        p += "\n\nStarter code:\n```python\n" + r["starter_code"].strip() + "\n```"
    return p


def stratified(by_level, n, rng):
    """Equal share per level; if a level runs short, the others fill the gap."""
    levels = sorted(by_level)
    for v in by_level.values():
        rng.shuffle(v)
    take = {l: 0 for l in levels}
    remaining = n
    while remaining:
        open_lv = [l for l in levels if take[l] < len(by_level[l])]
        if not open_lv:
            break
        share = max(remaining // len(open_lv), 1)
        for l in open_lv:
            add = min(share, len(by_level[l]) - take[l], remaining)
            take[l] += add
            remaining -= add
            if not remaining:
                break
    return [x for l in levels for x in by_level[l][: take[l]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-response", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "pairs_150.jsonl"))
    ap.add_argument("--rows-jsonl", default=None, help="offline: read raw rows from a jsonl instead of HF")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = ([json.loads(l) for l in open(args.rows_jsonl)] if args.rows_jsonl else load_rows())
    rows.sort(key=lambda r: (str(r["question_id"]), r["source_split"]))  # deterministic before shuffling
    print(f"loaded {len(rows)} codegen pairs from {len(set(r['source_split'] for r in rows))} splits")

    per_q = defaultdict(list)
    no_code = 0
    for r in rows:
        pc, nc = extract_code(r["pos_response"]), extract_code(r["neg_response"])
        if not (pc and nc):
            no_code += 1
            continue
        if pc.strip() == nc.strip():
            continue
        per_q[str(r["question_id"])].append((r, pc, nc))
    print(f"dropped {no_code} pairs without a fenced code block; {len(per_q)} distinct questions left")

    by_level = defaultdict(list)
    # Stratify on the difficulty label of the row we actually pick. The same question can carry
    # different labels in different splits (e.g. LeetCode 2839 is easy/hard/medium), so using
    # another row's label made the saved difficulty disagree with the stratification.
    for qid in sorted(per_q):
        pick = rng.choice(per_q[qid])
        by_level[(pick[0].get("difficulty") or "unknown").lower()].append(pick)
    picked = stratified(by_level, args.n, rng)
    rng.shuffle(picked)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, (r, pc, nc) in enumerate(picked):
            f.write(json.dumps(dict(
                pair_id=f"p{i:03d}", question_id=str(r["question_id"]),
                question_title=r.get("question_title"), platform=r.get("platform"),
                difficulty=(r.get("difficulty") or "unknown").lower(), source_split=r["source_split"],
                problem=build_problem(r),
                correct=r["pos_response"] if args.full_response else pc,
                incorrect=r["neg_response"] if args.full_response else nc,
            )) + "\n")

    lines = lambda s: s.count("\n") + 1
    pcs, ncs = [p for _, p, _ in picked], [n for _, _, n in picked]
    stats = dict(
        n_pairs=len(picked), n_answers=2 * len(picked), seed=args.seed,
        answer_format="full response" if args.full_response else "code only (last fenced block)",
        difficulty=dict(Counter((r.get("difficulty") or "unknown").lower() for r, _, _ in picked)),
        platform=dict(Counter(r.get("platform") for r, _, _ in picked)),
        source_split=dict(Counter(r["source_split"] for r, _, _ in picked)),
        avg_lines_correct=round(statistics.mean(map(lines, pcs)), 1),
        avg_lines_incorrect=round(statistics.mean(map(lines, ncs)), 1),
        avg_chars_problem=round(statistics.mean(len(build_problem(r)) for r, _, _ in picked)),
        pool=dict(raw_pairs=len(rows), dropped_no_code=no_code, distinct_questions=len(per_q)),
    )
    Path(str(out).replace(".jsonl", "_stats.json")).write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
