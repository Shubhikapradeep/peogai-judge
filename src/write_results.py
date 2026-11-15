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

from src.metrics import binom_two_sided

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


def finding_position(r: R):
    if not r.models:
        return None
    parts, biased = [], []
    for m in r.models:
        b = r.pw("exp1", m)
        p = b["order_test"]["p"]
        if sig(p):
            biased.append(m)
        parts.append(f"{NAME[m]}: {pc(b['acc_cf'])} with the correct answer as A vs {pc(b['acc_cs'])} as B "
                     f"({b['position_gap_pp']:+.1f} points, {pv(p)}); same pick in both orders for "
                     f"{pc(b['consistency'])} of pairs")
    if biased:
        lead = (f"{'Both judges show' if len(biased) == len(r.models) and len(r.models) > 1 else join(NAME[m] for m in biased) + ' shows'}"
                " clear position bias: accuracy depends on where the correct answer is placed.")
    else:
        lead = "Neither judge shows a significant accuracy gap between the two answer orders."
    return lead + " " + "; ".join(parts) + "."


def finding_pointwise(r: R):
    if not r.p_models:
        return None
    far = join(f"{pc(r.pt(m)['far'])} ({NAME[m]})" for m in r.p_models)
    frr = join(f"{pc(r.pt(m)['frr'])} ({NAME[m]})" for m in r.p_models)
    wm = max(r.p_models, key=lambda m: r.pt(m)["far"]["p"])
    worst = r.pt(wm)["far"]["p"]
    who = "the judges" if len(r.p_models) > 1 else "the judge"
    s = (f"Judged one answer at a time, {who} accept{'' if len(r.p_models) > 1 else 's'} {far} of incorrect solutions, "
         f"and reject {frr} of correct ones.")
    if worst >= 0.3:
        share = "almost every" if worst >= 0.95 else f"about {round(10 * worst)} in 10"
        s += (f" With the {NAME[wm]} judge, {share} buggy solution{'' if worst >= 0.95 else 's'} would pass review, "
              "so accuracy alone overstates how safe these judges are for code auditing.")
    return s


def finding_fixes(r: R):
    if not r.models:
        return None
    bits, none = [], []
    for m in r.models:
        b = r.pw("exp1", m)
        if b["orderavg_acc_decided"]["p"] is None:
            none.append(NAME[m])
            continue
        verb = "raises" if b["orderavg_acc_decided"]["p"] > b["accuracy"]["p"] else "changes"
        bits.append((verb, f"from {pc(b['accuracy'])} to {pc(b['orderavg_acc_decided'])} for {NAME[m]} "
                           f"(deciding only {pc(b['orderavg_coverage'])} of pairs)"))
    s = "Order averaging is the only fix that removes order inconsistency by construction."
    if bits:
        verb = "raises" if all(v == "raises" for v, _ in bits) else "changes"
        s += f" It {verb} accuracy " + join(t for _, t in bits) + ", but leaves every other pair undecided."
    if none:
        s += (f" For {join(none)} it decides no pairs at all, because the judge never picks the same answer "
              "in both orders.")
    for fix, label in (("exp4b", "the debias prompt"), ("exp4c", "majority voting over 3 samples")):
        eff = []
        for m in r.models:
            f, t = r.pw(fix, m), r.test(f"exp1_vs_{fix}_{m}")
            if f and t:
                d = 100 * (f["accuracy"]["p"] - r.pw("exp1", m)["accuracy"]["p"])
                eff.append((m, d, t["p"], f))
        if not eff:
            continue
        if any(sig(p) for _, _, p, _ in eff):
            desc = join(f"{d:+.1f} points for {NAME[m]} ({pv(p)})" for m, d, p, _ in eff)
            s += f" {label[0].upper() + label[1:]} changes accuracy by {desc}."
        else:
            desc = join(f"{d:+.1f} points for {NAME[m]}, {pv(p)}" for m, d, p, _ in eff)
            s += f" {label[0].upper() + label[1:]} does not significantly change accuracy ({desc})."
        if fix == "exp4b":
            s += " It moves the share of A verdicts " + join(
                f"from {pc(r.pw('exp1', m)['pct_A'])} to {pc(f['pct_A'])} ({NAME[m]})" for m, _, _, f in eff) + "."
    return s


def finding_scale(r: R):
    if len(r.models) < 2:
        return None
    a, b = r.pw("exp1", "3b"), r.pw("exp1", "7b")
    t = r.test("exp1_3b_vs_7b")
    d = 100 * (b["accuracy"]["p"] - a["accuracy"]["p"])
    s = (f"Moving from 3B to 7B {'raises' if d > 0 else 'lowers'} pairwise accuracy from {pc(a['accuracy'])} to "
         f"{pc(b['accuracy'])} ({pv(t['p'])}{'' if sig(t['p']) else ', not significant'}), and order consistency "
         f"from {pc(a['consistency'])} to {pc(b['consistency'])}.")
    if b["position_gap_pp"] is not None and a["position_gap_pp"] is not None:
        s += (f" The gap between orders is {abs(b['position_gap_pp']):.1f} points for 7B vs "
              f"{abs(a['position_gap_pp']):.1f} for 3B, so "
              + ("the larger model is less position-biased." if abs(b["position_gap_pp"]) < abs(a["position_gap_pp"]) - 5
                 else "scale alone does not remove the position bias."))
    if len(r.p_models) == 2:
        s += f" In pointwise judging, FAR is {pc(r.pt('3b')['far'])} (3B) vs {pc(r.pt('7b')['far'])} (7B)."
    return s


def finding_verbosity(r: R):
    V = r.M["verbosity"]
    if not V:
        return None
    out = []
    for m, v in V.items():
        p = v["test"]["p"]
        verb = ("significantly more often" if v["delta_pp"] > 0 else "significantly less often") if sig(p) else \
            "about as often (no significant change)"
        out.append(f"After the wrong answer is padded with comments claiming it is correct, the {NAME[m]} judge picks it "
                   f"{verb}: {pc(v['baseline'])} -> {pc(v['padded'])} ({v['delta_pp']:+.1f} points, {pv(p)}).")
    n = r.D.get("exp2_padded_longer_than_correct") if r.D else None
    if n:
        out.append(f"The padding also makes the wrong answer the longer one in {n} of 150 pairs, so this measures "
                   "comments and length together.")
    return " ".join(out)


def findings(r: R):
    return [f for f in (finding_position(r), finding_pointwise(r), finding_fixes(r), finding_scale(r),
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
                + (f" Parse failures: {pc(b['parse_fail'])} of calls." if b["parse_fail"]["p"] else ""))
        L += [tex(" ".join(txt)), ""]
        if len(r.models) == 2:
            L += [tex(finding_scale(r)), ""]

    if r.p_models:
        L += [r"\subsection{Pointwise judging (Exp.~3)}", r"\input{tables/t2_pointwise.tex}",
              fig("f2_pointwise", "Pointwise judging of all 300 answers. FAR: incorrect code accepted. "
                  "FRR: correct code rejected.", "pointwise")]
        txt = []
        for m in r.p_models:
            q = r.pt(m)
            k, n = q["pct_said_correct"]["k"], q["pct_said_correct"]["n"]
            p_len = binom_two_sided(k, n)
            lenient = ("more often than the true rate of 50%, a leniency bias" if q["pct_said_correct"]["p"] > 0.5
                       else "less often than the true rate of 50%, a strictness bias")
            txt.append(
                f"The {NAME[m]} judge reaches {pci(q['accuracy'])} accuracy, with FAR {pc(q['far'])} and FRR "
                f"{pc(q['frr'])}. It labels {pc(q['pct_said_correct'])} of answers correct"
                + (f", {lenient} (binomial {pv(p_len)})." if sig(p_len) else
                   f", not significantly different from 50% (binomial {pv(p_len)}).")
                + f" Both answers of a pair are judged right in only {pc(q['pair_both_right'])} of pairs.")
        L += [tex(" ".join(txt) + " " + (finding_pointwise(r).split(". ", 1)[1] if ". " in finding_pointwise(r) else "")), ""]

    if r.models:
        L += [r"\subsection{Effect of the fixes (Exp.~4)}", r"\input{tables/t3_fixes.tex}",
              fig("f3_fixes", "Accuracy (left) and order consistency (right) for the baseline and the fixes. Order "
                  "averaging is consistent by construction and is scored only on the pairs it decides.", "fixes", wide=True),
              tex(finding_fixes(r))]
        c = [m for m in r.models if r.pw("exp4c", m)]
        if c:
            t = r.test(f"exp1_vs_exp4c_{c[0]}")
            L.append(tex(" Because three samples at temperature 0.7 " +
                         ("do not change accuracy significantly, the bias appears systematic rather than random noise."
                          if not sig(t["p"]) else "do change accuracy significantly, part of the error is sampling noise "
                          "that voting averages out.")))
        L += [""]

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
    args = ap.parse_args()
    res = Path(args.results)
    M = json.loads((res / "metrics.json").read_text())
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
