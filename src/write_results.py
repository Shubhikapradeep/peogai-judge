"""
write_results.py -- write the findings and the Results section from results/metrics.json.

    python -m src.metrics && python -m src.plots && python -m src.write_results

Writes
    results/findings.md      the 3-5 main findings (for the team, slides 12-13 and the viva)
    report/results_auto.tex  Section V with every number filled in (replaces the \\tbd template)

The wording follows the data. An effect is described as real only when its test gives p < 0.05;
otherwise the text says it is not significant. Read both files before using them: the script
cannot judge which result is most interesting, and the Discussion (P4) should interpret, not repeat, them.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05
NAME = {"3b": "3B", "7b": "7B"}


# ---------------------------------------------------------------- formatting
def pc(x, d=1):
    return "--" if x is None or x.get("p") is None else f"{100 * x['p']:.{d}f}%"


def pci(x):
    return f"{pc(x)} [95% CI {100 * x['lo']:.1f}-{100 * x['hi']:.1f}]" if x and x.get("p") is not None else "--"


def pv(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sig(p):
    return p < ALPHA


def tex(s: str) -> str:
    return (s.replace("%", r"\%").replace("p < ", "$p<$ ").replace("p = ", "$p=$ ")
            .replace("->", r"$\rightarrow$").replace("~", r"$\sim$"))


def join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# ---------------------------------------------------------------- analysis
class R:
    def __init__(self, M, D):
        self.M, self.D = M, D
        P = M["pairwise"]
        self.models = [m for m in ("3b", "7b") if f"exp1_{m}" in P]
        self.p_models = [m for m in ("3b", "7b") if f"exp3_{m}" in M["pointwise"]]

    def pw(self, exp, m):
        return self.M["pairwise"].get(f"{exp}_{m}")

    def pt(self, m):
        return self.M["pointwise"].get(f"exp3_{m}")

    def test(self, k):
        return self.M["tests"].get(k)


def subject(ms, r):
    if len(ms) == len(r.models) and len(ms) > 1:
        return "Both judges"
    return "The " + join(NAME[m] for m in ms) + " judge"


def chance(p_acc, acc):
    if not sig(p_acc):
        return f"not significantly better than chance ({pv(p_acc)})"
    return f"significantly {'above' if acc > 0.5 else 'below'} chance ({pv(p_acc)})"


def rare_letter(b):
    return "B" if b["pct_A"]["p"] is not None and b["pct_A"]["p"] > 0.5 else "A"


def finding_position(r: R):
    if not r.models:
        return None
    B = {m: r.pw("exp1", m) for m in r.models}
    biased = [m for m in r.models if sig(B[m]["order_test"]["p"])]
    if biased:
        s = (f"{subject(biased, r)} show{'s' if len(biased) == 1 else ''} strong position bias, answering A in "
             + join(f"{pc(B[m]['pct_A'])} ({NAME[m]})" for m in r.models)
             + " of calls against 50% for an unbiased judge. ")
    else:
        s = "Neither judge shows a significant accuracy gap between the two answer orders. "
    s += "; ".join(f"{NAME[m]} accuracy is {pc(B[m]['acc_cf'])} with the correct answer as A and {pc(B[m]['acc_cs'])} "
                   f"as B ({pv(B[m]['order_test']['p'])}), and only {pc(B[m]['consistency'])} of pairs get the same "
                   f"pick in both orders" for m in r.models) + ". "
    if all(not sig(B[m]["acc_vs_chance_p"]) for m in r.models):
        s += ("Overall pairwise accuracy (" + join(f"{pc(B[m]['accuracy'])} for {NAME[m]}, {pv(B[m]['acc_vs_chance_p'])}"
                                                    for m in r.models) + ") is not significantly better than chance.")
    else:
        s += "Overall pairwise accuracy is " + join(
            f"{pc(B[m]['accuracy'])} for {NAME[m]}, {chance(B[m]['acc_vs_chance_p'], B[m]['accuracy']['p'])}"
            for m in r.models) + "."
    signal = []
    for m in r.models:
        L = rare_letter(B[m])
        x, p = B[m][f"precision_{L}"], B[m][f"precision_{L}_p"]
        if x["n"] >= 5 and sig(p) and x["p"] > 0.5:
            signal.append(f"{NAME[m]} is right in {x['k']} of its {x['n']} {L} verdicts ({pc(x)}, {pv(p)})")
    if signal:
        s += (" The rare verdicts against the default still carry signal: " + join(signal) +
              ". The model does read the code, but its default letter drowns that signal out.")
    return s


def finding_pointwise(r: R):
    if not r.p_models:
        return None
    Q = {m: r.pt(m) for m in r.p_models}
    who = "the judges" if len(r.p_models) > 1 else "the judge"
    s = (f"Judged one answer at a time, {who} call " + join(f"{pc(Q[m]['pct_said_correct'])} ({NAME[m]})" for m in r.p_models)
         + " of answers correct, although exactly half are wrong"
         + ((" (both p < 0.001)." if len(r.p_models) > 1 and all(Q[m]["said_correct_vs_half_p"] < 0.001 for m in r.p_models)
             else f" ({join(pv(Q[m]['said_correct_vs_half_p']) for m in r.p_models)}).")
            if all(sig(Q[m]['said_correct_vs_half_p']) for m in r.p_models) else ".")
         + " They accept " + join(f"{pc(Q[m]['far'])} ({NAME[m]})" for m in r.p_models) + " of incorrect solutions and reject "
         + join(f"{pc(Q[m]['frr'])} ({NAME[m]})" for m in r.p_models) + " of correct ones. Accuracy is "
         + join(f"{pc(Q[m]['accuracy'])} for {NAME[m]}, {chance(Q[m]['acc_vs_chance_p'], Q[m]['accuracy']['p'])}"
                for m in r.p_models) + ".")
    wm = max(r.p_models, key=lambda m: Q[m]["far"]["p"])
    worst = Q[wm]["far"]["p"]
    if worst >= 0.3:
        share = "almost every" if worst >= 0.95 else f"about {round(10 * worst)} in 10"
        s += (f" With the {NAME[wm]} judge, {share} buggy solution{'' if worst >= 0.95 else 's'} would pass review, "
              "so accuracy alone overstates how safe these judges are for code auditing.")
    return s


def finding_scale(r: R):
    if len(r.models) < 2:
        return None
    a, b = r.pw("exp1", "3b"), r.pw("exp1", "7b")
    tA, tC, tAcc = r.test("pctA_3b_vs_7b"), r.test("consistency_3b_vs_7b"), r.test("exp1_3b_vs_7b")
    less = (tA and sig(tA["p"]) and abs(b["pct_A"]["p"] - 0.5) < abs(a["pct_A"]["p"] - 0.5)) or \
           (tC and sig(tC["p"]) and b["consistency"]["p"] > a["consistency"]["p"])
    s = ("The 7B judge is " + ("less position-biased than the 3B judge" if less else "not significantly less position-biased than the 3B judge")
         + f" (A verdicts {pc(b['pct_A'])} vs {pc(a['pct_A'])}, {pv(tA['p'])}; same pick in both orders "
           f"{pc(b['consistency'])} vs {pc(a['consistency'])}, {pv(tC['p'])}), "
         + ("but no more accurate" if not sig(tAcc["p"]) else ("and more accurate" if b["accuracy"]["p"] > a["accuracy"]["p"] else "but less accurate"))
         + f" ({pc(b['accuracy'])} vs {pc(a['accuracy'])}, {pv(tAcc['p'])}).")
    t3 = r.test("exp3_3b_vs_7b")
    if len(r.p_models) == 2 and t3:
        s += (f" In pointwise judging its FAR is {pc(r.pt('7b')['far'])} vs {pc(r.pt('3b')['far'])}, and the accuracy difference is "
              + ("not significant" if not sig(t3["p"]) else "significant") + f" ({pv(t3['p'])}).")
    s += " The larger model does not make either format a reliable judge."
    return s


def finding_fixes(r: R):
    if not r.models:
        return None
    B = {m: r.pw("exp1", m) for m in r.models}
    dec = [m for m in r.models if B[m]["orderavg_acc_decided"]["p"] is not None]
    s = ("Order averaging gives a verdict only when both orders agree, which happens for "
         + join(f"{pc(B[m]['orderavg_coverage'])} of pairs ({NAME[m]})" for m in r.models) + ".")
    if dec:
        s += (" On those pairs accuracy is " + join(
            f"{pc(B[m]['orderavg_acc_decided'])} ({B[m]['orderavg_acc_decided']['k']} of {B[m]['orderavg_acc_decided']['n']}, "
            f"95% CI {100 * B[m]['orderavg_acc_decided']['lo']:.0f}-{100 * B[m]['orderavg_acc_decided']['hi']:.0f}%, {NAME[m]})"
            for m in dec) + ", but the method abstains on almost every pair." if all(B[m]["orderavg_coverage"]["p"] < 0.3 for m in r.models)
            else " On those pairs accuracy is " + join(f"{pc(B[m]['orderavg_acc_decided'])} ({NAME[m]})" for m in dec) + ".")
    # 4b: debias prompt
    worse, better, flat = [], [], []
    for m in r.models:
        f = r.pw("exp4b", m)
        if not f:
            continue
        tA, tC = r.test(f"pctA_exp1_vs_exp4b_{m}"), r.test(f"consistency_exp1_vs_exp4b_{m}")
        more_biased = (sig(tA["p"]) and abs(f["pct_A"]["p"] - 0.5) > abs(B[m]["pct_A"]["p"] - 0.5)) or \
                      (sig(tC["p"]) and f["consistency"]["p"] < B[m]["consistency"]["p"])
        less_biased = (sig(tA["p"]) and abs(f["pct_A"]["p"] - 0.5) < abs(B[m]["pct_A"]["p"] - 0.5)) or \
                      (sig(tC["p"]) and f["consistency"]["p"] > B[m]["consistency"]["p"])
        d = (f"A verdicts {pc(B[m]['pct_A'])} -> {pc(f['pct_A'])}, {pv(tA['p'])}; same pick in both orders "
             f"{pc(B[m]['consistency'])} -> {pc(f['consistency'])}, {pv(tC['p'])}")
        (worse if more_biased else better if less_biased else flat).append((m, d))
    if worse or better or flat:
        fmt = lambda items: " ".join(f"For {NAME[m]}: {d}." for m, d in items)
        if worse and not better:
            s += (f" The debias prompt makes the bias worse for {join(NAME[m] for m, _ in worse)}, not better. "
                  + fmt(worse))
            if flat:
                s += (f" For {join(NAME[m] for m, _ in flat)} the change goes the same way but is not significant: "
                      + "; ".join(d for _, d in flat) + ".")
        elif better and not worse:
            s += f" The debias prompt reduces the bias for {join(NAME[m] for m, _ in better)}. " + fmt(better)
        else:
            s += " The debias prompt has mixed or no significant effects. " + fmt(worse + better + flat)
        acc = [(m, r.pw("exp4b", m), r.test(f"exp1_vs_exp4b_{m}")) for m in r.models if r.pw("exp4b", m)]
        if all(not sig(t["p"]) for _, _, t in acc):
            s += " Accuracy does not change significantly (" + join(
                f"{NAME[m]} {pc(B[m]['accuracy'])} -> {pc(f['accuracy'])}, {pv(t['p'])}" for m, f, t in acc) + ")."
    # 4c: majority vote
    for m in r.models:
        f = r.pw("exp4c", m)
        if not f:
            continue
        tA, tAcc = r.test(f"pctA_exp1_vs_exp4c_{m}"), r.test(f"exp1_vs_exp4c_{m}")
        s += (f" Majority voting over 3 samples at temperature 0.7 ({NAME[m]}) "
              + (f"lowers A verdicts from {pc(B[m]['pct_A'])} to {pc(f['pct_A'])} ({pv(tA['p'])}) " if sig(tA["p"]) else "")
              + (f"but leaves accuracy at {pc(f['accuracy'])} ({pv(tAcc['p'])} vs greedy) and consistency at {pc(f['consistency'])}: "
                 "sampling spreads the verdicts out without adding information." if not sig(tAcc["p"]) else
                 f"and changes accuracy to {pc(f['accuracy'])} ({pv(tAcc['p'])})."))
    return s


def finding_verbosity(r: R):
    V = r.M["verbosity"]
    if not V:
        return None
    out = []
    for m, v in V.items():
        p, pA, pB = v["test"]["p"], v["test_cs"]["p"], v["test_cf"]["p"]   # cs: wrong answer shown as A
        head = (f"Padding the wrong answer with comments changes how often the {NAME[m]} judge picks it from "
                f"{pc(v['baseline'])} to {pc(v['padded'])} ({pv(p)}{'' if sig(p) else ', not significant'}).")
        dA = v["padded_cs"]["p"] - v["baseline_cs"]["p"]
        dB = v["padded_cf"]["p"] - v["baseline_cf"]["p"]
        split = (f" When the padded answer is shown as A it wins {pc(v['baseline_cs'])} -> {pc(v['padded_cs'])} ({pv(pA)}); "
                 f"when it is shown as B, {pc(v['baseline_cf'])} -> {pc(v['padded_cf'])} ({pv(pB)}).")
        e2, e1 = r.pw("exp2", m), r.pw("exp1", m)
        if sig(pA) and dA > 0 and not (sig(pB) and dB > 0):
            tail = (" The rise comes only from position A, so padding strengthens the judge's default to A"
                    + (f" ({pc(e1['pct_A'])} -> {pc(e2['pct_A'])} A verdicts)" if e1 and e2 else "")
                    + " rather than making it prefer the padded answer.")
        elif sig(pA) and sig(pB) and dA > 0 and dB > 0:
            tail = " The padded answer gains in both positions, so the judge does prefer the commented code."
        else:
            tail = " Neither position shows a clear preference for the padded answer."
        out.append(head + split + tail)
    n = r.D.get("exp2_padded_longer_than_correct") if r.D else None
    if n:
        out.append(f"Because the padding also makes the wrong answer the longer one in {n} of 150 pairs, this tests "
                   "comments and length together.")
    return " ".join(out)


def findings(r: R):
    return [f for f in (finding_position(r), finding_pointwise(r), finding_scale(r), finding_fixes(r),
                        finding_verbosity(r)) if f]


# ---------------------------------------------------------------- Results section
def fig(name, caption, label, wide=False):
    env = "figure*" if wide else "figure"
    width = r"0.85\textwidth" if wide else r"\columnwidth"
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{{env}}}\n")


def section(r: R, F: list[str]) -> str:
    M = r.M
    L = [r"% ---- V. Results (P2) -- generated by src/write_results.py from results/metrics.json ----",
         r"% Regenerate instead of editing numbers by hand. Upload results/tables/*.tex to tables/ and",
         r"% results/figures/*.pdf to figures/ on Overleaf.", "",
         r"\section{Results}", r"\label{sec:results}", "",
         tex("Each experiment covers 150 pairs (300 calls per judge). Brackets give 95% Wilson confidence "
             "intervals, and parse failures are scored as wrong. Paired comparisons on the same calls use an exact "
             "McNemar test, and the share of A verdicts is tested against 50% with an exact binomial test."), ""]

    if r.models:
        L += [r"\subsection{Pairwise judging and position bias (Exp.~1)}", r"\input{tables/t1_pairwise.tex}",
              fig("f1_position_bias", "Pairwise accuracy when the correct answer is shown as A or as B. Dashed line: "
                  "chance. Error bars: 95\\% Wilson CIs.", "position")]
        txt = []
        for m in r.models:
            b = r.pw("exp1", m)
            txt.append(
                f"The {NAME[m]} judge is {pc(b['accuracy'])} accurate overall [{100 * b['accuracy']['lo']:.1f}-"
                f"{100 * b['accuracy']['hi']:.1f}]: {pc(b['acc_cf'])} when the correct answer is A and "
                f"{pc(b['acc_cs'])} when it is B ({b['position_gap_pp']:+.1f} points, McNemar {pv(b['order_test']['p'])}). "
                f"It answers A in {pc(b['pct_A'])} of parsed verdicts, against 50% for an unbiased judge "
                f"(binomial {pv(b['pct_A_test_p'])}). It picks the same answer in both orders for {pc(b['consistency'])} "
                f"of pairs. Of the inconsistent pairs, {b['flip_AA']} got A twice and {b['flip_BB']} got B twice, "
                f"so in those the verdict followed the position, not the code."
                + (f" Parse failures: {pc(b['parse_fail'])} of calls." if b["parse_fail"]["p"] else "")
                + f" Its overall accuracy is {chance(b['acc_vs_chance_p'], b['accuracy']['p'])}."
                + (lambda L: f" When it does answer {L}, it is right in {b[f'precision_{L}']['k']} of "
                             f"{b[f'precision_{L}']['n']} cases ({pc(b[f'precision_{L}'])}, binomial "
                             f"{pv(b[f'precision_{L}_p'])})." if b[f"precision_{L}"]["n"] else "")(rare_letter(b)))
        L += [tex(" ".join(txt)), ""]
        if len(r.models) == 2:
            L += [tex(finding_scale(r)), ""]

    if r.p_models:
        L += [r"\subsection{Pointwise judging (Exp.~3)}", r"\input{tables/t2_pointwise.tex}",
              fig("f2_pointwise", "Pointwise judging of all 300 answers. FAR: incorrect code accepted. "
                  "FRR: correct code rejected.", "pointwise")]
        both = join(f"{pc(r.pt(m)['pair_both_right'])} ({NAME[m]})" for m in r.p_models)
        L += [tex(finding_pointwise(r) + f" Both answers of a pair are judged right in only {both} of pairs."), ""]

    if r.models:
        L += [r"\subsection{Effect of the fixes (Exp.~4)}", r"\input{tables/t3_fixes.tex}",
              fig("f3_fixes", "Accuracy (left) and order consistency (right) for the baseline and the fixes. Order "
                  "averaging is consistent by construction and is scored only on the pairs it decides.", "fixes", wide=True),
              tex(finding_fixes(r)), ""]

    if M["verbosity"]:
        L += [r"\subsection{Verbosity (Exp.~2)}", r"\input{tables/t4_verbosity.tex}",
              fig("f4_verbosity", "How often the incorrect answer is chosen before and after it is padded with comments "
                  "claiming correctness. The code logic is unchanged (checked by AST equality).", "verbosity"),
              tex(finding_verbosity(r)), ""]

    D = M.get("difficulty", {})
    if any(f"exp1_{m}" in D for m in r.models):
        L += [r"\subsection{By difficulty}", r"\input{tables/t5_difficulty.tex}"]
        txt = []
        for m in r.models:
            d = D.get(f"exp1_{m}", {})
            txt.append(f"{NAME[m]} pairwise accuracy is " + join(f"{pc(d[l]['accuracy'])} ({l})" for l in d) + ".")
        txt.append("Each level has only 50 pairs and the benchmark's difficulty labels disagree across generator "
                   "splits for 82% of the problems that appear in more than one split, so this breakdown is indicative only.")
        L += [tex(" ".join(txt)), ""]

    L += [r"\subsection{Summary of findings}", r"\begin{enumerate}"]
    L += [r"\item " + tex(f) for f in F]
    L += [r"\end{enumerate}", ""]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"), help="folder with metrics.json and data_stats.json")
    ap.add_argument("--report", default=str(ROOT / "report" / "results_auto.tex"))
    ap.add_argument("--allow-mixed-data", action="store_true",
                    help="write findings even though metrics.py's data check failed (never for the report)")
    args = ap.parse_args()
    res = Path(args.results)
    M = json.loads((res / "metrics.json").read_text())
    chk = M.get("data_check", {})
    if chk.get("ok") is False and not args.allow_mixed_data:
        bad = {k: v["mismatched"] for k, v in chk["files"].items() if v["mismatched"]}
        raise SystemExit(f"Refusing to write findings: these runs were judged on a different sample than "
                         f"{chk['data']}: {bad}. Re-run them (see P2_NOTES.md), then run metrics again.")
    dpath = res / "data_stats.json"
    if not dpath.exists():
        dpath = ROOT / "results" / "data_stats.json"
    D = json.loads(dpath.read_text()) if dpath.exists() else {}
    r = R(M, D)
    F = findings(r)
    md = ["# Main findings (generated by src/write_results.py from results/metrics.json)", "",
          "Effects are called significant only when p < 0.05. Re-run the script after any new results come in.", ""]
    md += [f"{i}. {f}" for i, f in enumerate(F, 1)]
    (res / "findings.md").write_text("\n".join(md) + "\n")
    Path(args.report).write_text(section(r, F))
    print("\n".join(md))
    print(f"\nwrote {res / 'findings.md'} and {args.report}")


if __name__ == "__main__":
    main()
