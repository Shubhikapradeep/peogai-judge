"""
metrics.py -- turn results/raw/*.jsonl into every number, table and JSON the report needs.

    python -m src.metrics                      # reads results/raw, writes results/tables + results/metrics.json
    python -m src.metrics --raw /content/drive/MyDrive/peogai/results/raw

Inputs: <exp>_<model>.jsonl from src/run_experiment.py (files ending in _smoke are skipped).
Rows are deduplicated by job_id, keeping the last one (a resumed run can rewrite a job).

Conventions (say these in the report):
  * A parse failure (verdict None) counts as WRONG in every accuracy. Parse-failure rates are reported separately.
  * Pairwise "pick": which underlying answer the judge chose. In order cf the correct answer is A,
    in cs it is B. A pair is CONSISTENT when both orders pick the same answer (right-right or wrong-wrong).
    Both orders giving the same letter (AA / BB) is a position flip.
  * Exp 4a, order averaging: keep the verdict only when both orders agree, otherwise "undecided".
    Reported as accuracy on decided pairs plus coverage (% of pairs decided).
  * Pointwise: FAR = incorrect answers judged "correct" / all incorrect answers;
    FRR = correct answers judged "incorrect" / all correct answers.
  * 95% CIs are Wilson intervals. Paired comparisons (same jobs, two conditions) use an exact McNemar test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["3b", "7b"]
MODEL_NAME = {"3b": "Qwen2.5-Coder-3B", "7b": "Qwen2.5-Coder-7B (4-bit)"}
PAIRWISE_EXPS = ["exp1", "exp2", "exp4b", "exp4c"]
LEVELS = ["easy", "medium", "hard"]


# ---------------------------------------------------------------- stats helpers
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def prop(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": (k / n) if n else None, "lo": lo, "hi": hi}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact two-sided McNemar on paired booleans (e.g. correct under condition a vs b)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail)}


# ---------------------------------------------------------------- loading
def load_raw(raw_dir: Path) -> dict:
    """{(exp, model): [rows]} deduplicated by job_id (last row wins)."""
    out = {}
    for f in sorted(raw_dir.glob("*.jsonl")):
        if "_smoke" in f.stem:
            continue
        parts = f.stem.split("_")
        if len(parts) != 2:
            continue
        exp, model = parts
        rows = {}
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[r["job_id"]] = r
        out[(exp, model)] = list(rows.values())
    return out


def by_pair(rows: list[dict]) -> dict:
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair_id"], {"difficulty": r.get("difficulty")})[r["order"]] = r
    return pairs


def pick(order: str, verdict):
    """Which answer the judge chose: 'correct' / 'incorrect' / None."""
    if verdict not in ("A", "B"):
        return None
    correct_letter = "A" if order == "cf" else "B"
    return "correct" if verdict == correct_letter else "incorrect"


# ---------------------------------------------------------------- metrics
def pairwise_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["verdict"] in ("A", "B")]
    m = {
        "n_calls": n,
        "accuracy": prop(sum(r["verdict"] == r["gold"] for r in rows), n),
        "parse_fail": prop(n - len(ok), n),
        "pct_A": prop(sum(r["verdict"] == "A" for r in ok), len(ok)),
    }
    for order in ("cf", "cs"):
        sub = [r for r in rows if r["order"] == order]
        m[f"acc_{order}"] = prop(sum(r["verdict"] == r["gold"] for r in sub), len(sub))

    pairs = [p for p in by_pair(rows).values() if "cf" in p and "cs" in p]
    np_ = len(pairs)
    cons_right = cons_wrong = flip_a = flip_b = undecided_fail = 0
    for p in pairs:
        a, b = pick("cf", p["cf"]["verdict"]), pick("cs", p["cs"]["verdict"])
        va, vb = p["cf"]["verdict"], p["cs"]["verdict"]
        if a is None or b is None:
            undecided_fail += 1
        elif a == b:
            cons_right += a == "correct"
            cons_wrong += a == "incorrect"
        else:
            flip_a += va == "A" and vb == "A"
            flip_b += va == "B" and vb == "B"
    decided = cons_right + cons_wrong
    m.update({
        "n_pairs": np_,
        "consistency": prop(decided, np_),
        "consistent_right": cons_right, "consistent_wrong": cons_wrong,
        "flip_AA": flip_a, "flip_BB": flip_b, "pairs_with_parse_fail": undecided_fail,
        # Exp 4a: order averaging
        "orderavg_acc_decided": prop(cons_right, decided),
        "orderavg_coverage": prop(decided, np_),
        "orderavg_acc_all": prop(cons_right, np_),  # undecided counted as wrong
    })
    m["position_gap_pp"] = (100 * (m["acc_cf"]["p"] - m["acc_cs"]["p"])
                            if m["acc_cf"]["p"] is not None and m["acc_cs"]["p"] is not None else None)
    return m


def pointwise_metrics(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["gold"] == "correct"]
    neg = [r for r in rows if r["gold"] == "incorrect"]
    n = len(rows)
    return {
        "n_calls": n,
        "accuracy": prop(sum(r["verdict"] == r["gold"] for r in rows), n),
        "far": prop(sum(r["verdict"] == "correct" for r in neg), len(neg)),
        "frr": prop(sum(r["verdict"] == "incorrect" for r in pos), len(pos)),
        "pct_said_correct": prop(sum(r["verdict"] == "correct" for r in rows), n),
        "parse_fail": prop(sum(r["verdict"] is None for r in rows), n),
        # both answers of a pair judged right -> the judge actually separated the pair
        "pair_both_right": prop(*_pair_both_right(rows)),
    }


def _pair_both_right(rows):
    g = {}
    for r in rows:
        g.setdefault(r["pair_id"], []).append(r["verdict"] == r["gold"])
    full = [v for v in g.values() if len(v) == 2]
    return sum(all(v) for v in full), len(full)


def by_difficulty(rows: list[dict], fn) -> dict:
    out = {}
    for lv in LEVELS:
        sub = [r for r in rows if (r.get("difficulty") or "").lower() == lv]
        if sub:
            out[lv] = fn(sub)
    return out


def paired(rows_a: list[dict], rows_b: list[dict], key) -> dict:
    """McNemar between two conditions on the job_ids both have."""
    a = {r["job_id"]: key(r) for r in rows_a}
    b = {r["job_id"]: key(r) for r in rows_b}
    common = sorted(set(a) & set(b))
    res = mcnemar([a[j] for j in common], [b[j] for j in common])
    res["n"] = len(common)
    return res


def compute(raw: dict) -> dict:
    M = {"pairwise": {}, "pointwise": {}, "difficulty": {}, "tests": {}, "verbosity": {}}
    for (exp, model), rows in raw.items():
        if exp in PAIRWISE_EXPS:
            M["pairwise"][f"{exp}_{model}"] = pairwise_metrics(rows)
        elif exp == "exp3":
            M["pointwise"][f"{exp}_{model}"] = pointwise_metrics(rows)

    is_right = lambda r: r["verdict"] == r["gold"]
    for model in MODELS:
        base = raw.get(("exp1", model))
        if not base:
            continue
        M["difficulty"][f"exp1_{model}"] = by_difficulty(base, pairwise_metrics)
        if ("exp3", model) in raw:
            M["difficulty"][f"exp3_{model}"] = by_difficulty(raw[("exp3", model)], pointwise_metrics)
        for fix in ("exp4b", "exp4c"):
            if (fix, model) in raw:
                M["tests"][f"exp1_vs_{fix}_{model}"] = paired(base, raw[(fix, model)], is_right)
        # Exp 2: how often the (padded) wrong answer wins
        if ("exp2", model) in raw:
            wins = lambda r: pick(r["order"], r["verdict"]) == "incorrect"
            pad = raw[("exp2", model)]
            v = {"baseline": prop(sum(map(wins, base)), len(base)),
                 "padded": prop(sum(map(wins, pad)), len(pad)),
                 "test": paired(base, pad, wins)}
            for order in ("cf", "cs"):
                b_o = [r for r in base if r["order"] == order]
                p_o = [r for r in pad if r["order"] == order]
                v[f"baseline_{order}"] = prop(sum(map(wins, b_o)), len(b_o))
                v[f"padded_{order}"] = prop(sum(map(wins, p_o)), len(p_o))
            v["delta_pp"] = 100 * (v["padded"]["p"] - v["baseline"]["p"])
            M["verbosity"][model] = v
    return M


# ---------------------------------------------------------------- tables
def pct(x, ci=False):
    if x is None or x.get("p") is None:
        return "--"
    s = f"{100 * x['p']:.1f}"
    if ci:
        s += f" [{100 * x['lo']:.0f}, {100 * x['hi']:.0f}]"
    return s


def pval(p):
    return "--" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def write_table(out_dir: Path, name: str, caption: str, header: list[str], rows: list[list[str]], note=""):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    md = [f"**{caption}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    if note:
        md += ["", f"_{note}_"]
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: (s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
                      .replace("<", r"$<$").replace("Δ", r"$\Delta$"))
    tex = [r"\begin{table}[t]", r"\centering", r"\caption{" + esc(caption) + "}",
           r"\label{tab:" + name + "}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\hline",
           " & ".join(esc(h) for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    tex += [r"\hline", r"\end{tabular}"]
    if note:
        tex += [r"\\[2pt]", r"{\scriptsize " + esc(note) + "}"]
    tex += [r"\end{table}"]
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n")


def make_tables(M: dict, out_dir: Path):
    P, Q = M["pairwise"], M["pointwise"]
    ci_note = "Values in %. Brackets: 95% Wilson CI. Parse failures count as wrong."

    rows = [[MODEL_NAME[m], pct(P[k]["acc_cf"]), pct(P[k]["acc_cs"]), pct(P[k]["accuracy"], True),
             pct(P[k]["consistency"], True), pct(P[k]["pct_A"]), pct(P[k]["parse_fail"])]
            for m in MODELS if (k := f"exp1_{m}") in P]
    if rows:
        write_table(out_dir, "t1_pairwise", "Exp 1: pairwise judging in both answer orders",
                    ["Model", "Acc (correct=A)", "Acc (correct=B)", "Acc overall",
                     "Consistency", "% A verdicts", "Parse fail"], rows, ci_note)

    rows = [[MODEL_NAME[m], pct(Q[k]["accuracy"], True), pct(Q[k]["far"], True), pct(Q[k]["frr"], True),
             pct(Q[k]["pct_said_correct"]), pct(Q[k]["pair_both_right"]), pct(Q[k]["parse_fail"])]
            for m in MODELS if (k := f"exp3_{m}") in Q]
    if rows:
        write_table(out_dir, "t2_pointwise", "Exp 3: pointwise judging of all 300 answers",
                    ["Model", "Accuracy", "FAR", "FRR", "% said correct", "Both of pair right", "Parse fail"],
                    rows, ci_note + " FAR: wrong code accepted. FRR: correct code rejected.")

    rows = []
    for m in MODELS:
        b = P.get(f"exp1_{m}")
        if not b:
            continue
        rows.append([MODEL_NAME[m], "Baseline (Exp 1)", pct(b["accuracy"]), pct(b["consistency"]),
                     pct(b["pct_A"]), "100.0", "--"])
        rows.append(["", "Order averaging (4a)", pct(b["orderavg_acc_decided"]), "100.0", "--",
                     pct(b["orderavg_coverage"]), "--"])
        for fix, label in (("exp4b", "Debias prompt (4b)"), ("exp4c", "Majority of 3 (4c)")):
            f = P.get(f"{fix}_{m}")
            if f:
                t = M["tests"].get(f"exp1_vs_{fix}_{m}", {})
                rows.append(["", label, pct(f["accuracy"]), pct(f["consistency"]), pct(f["pct_A"]),
                             "100.0", pval(t.get("p"))])
    if rows:
        write_table(out_dir, "t3_fixes", "Exp 4: position-bias fixes compared with the baseline",
                    ["Model", "Method", "Accuracy", "Consistency", "% A verdicts", "Coverage", "p vs baseline"],
                    rows, "Values in %. 4a accuracy is on decided pairs only; coverage = % of pairs decided. "
                          "p: exact McNemar test on per-call correctness vs Exp 1.")

    rows = []
    for m, v in M["verbosity"].items():
        rows.append([MODEL_NAME[m], pct(v["baseline"]), pct(v["padded"]), f"{v['delta_pp']:+.1f}",
                     f"{pct(v['baseline_cf'])} -> {pct(v['padded_cf'])}",
                     f"{pct(v['baseline_cs'])} -> {pct(v['padded_cs'])}", pval(v["test"]["p"])])
    if rows:
        write_table(out_dir, "t4_verbosity", "Exp 2: how often the wrong answer wins, before and after padding it with comments",
                    ["Model", "Wrong wins (plain)", "Wrong wins (padded)", "Δ pp",
                     "Correct=A", "Correct=B", "p"], rows,
                    "Values in %. p: exact McNemar test on the same 300 calls.")

    rows = []
    for m in MODELS:
        d1, d3 = M["difficulty"].get(f"exp1_{m}", {}), M["difficulty"].get(f"exp3_{m}", {})
        for lv in LEVELS:
            if lv in d1 or lv in d3:
                rows.append([MODEL_NAME[m] if lv == "easy" else "", lv,
                             pct(d1.get(lv, {}).get("accuracy")), pct(d1.get(lv, {}).get("consistency")),
                             pct(d3.get(lv, {}).get("accuracy")), pct(d3.get(lv, {}).get("far"))])
    if rows:
        write_table(out_dir, "t5_difficulty", "Results by problem difficulty (50 pairs per level)",
                    ["Model", "Difficulty", "Exp 1 acc", "Exp 1 consistency", "Exp 3 acc", "Exp 3 FAR"], rows,
                    "Values in %. Difficulty labels come from CodeJudgeBench and are noisy: the same problem "
                    "often has different labels in different splits.")


def key_numbers(M: dict) -> str:
    """Plain-text cheat sheet for writing findings and checking the report/slides."""
    L = ["# Key numbers (auto-generated by src/metrics.py; do not edit by hand)", ""]
    for k, v in M["pairwise"].items():
        L.append(f"- **{k}**: acc {pct(v['accuracy'], True)}; correct=A {pct(v['acc_cf'])} vs correct=B "
                 f"{pct(v['acc_cs'])} (gap {v['position_gap_pp']:+.1f} pp); consistency {pct(v['consistency'], True)}; "
                 f"%A {pct(v['pct_A'])}; flips AA {v['flip_AA']} / BB {v['flip_BB']} of {v['n_pairs']} pairs; "
                 f"order-avg acc {pct(v['orderavg_acc_decided'])} at coverage {pct(v['orderavg_coverage'])}; "
                 f"parse fail {pct(v['parse_fail'])}")
    for k, v in M["pointwise"].items():
        L.append(f"- **{k}**: acc {pct(v['accuracy'], True)}; FAR {pct(v['far'], True)}; FRR {pct(v['frr'], True)}; "
                 f"said correct {pct(v['pct_said_correct'])}; parse fail {pct(v['parse_fail'])}")
    for k, t in M["tests"].items():
        L.append(f"- **{k}**: only baseline right {t['only_a']}, only fix right {t['only_b']}, McNemar p={pval(t['p'])}")
    for m, v in M["verbosity"].items():
        L.append(f"- **exp2_{m}**: wrong answer wins {pct(v['baseline'])} -> {pct(v['padded'])} "
                 f"({v['delta_pp']:+.1f} pp), McNemar p={pval(v['test']['p'])}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    raw = load_raw(Path(args.raw))
    if not raw:
        raise SystemExit(f"no <exp>_<model>.jsonl files in {args.raw}")
    print("loaded:", ", ".join(f"{e}_{m} ({len(r)} rows)" for (e, m), r in sorted(raw.items())))
    M = compute(raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(M, indent=1))
    make_tables(M, out / "tables")
    (out / "key_numbers.md").write_text(key_numbers(M))
    print(f"wrote {out / 'metrics.json'}, {out / 'key_numbers.md'} and tables in {out / 'tables'}")
    print(key_numbers(M))


if __name__ == "__main__":
    main()
