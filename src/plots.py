"""
plots.py -- figures for the report and slides, drawn from results/metrics.json.

    python -m src.metrics && python -m src.plots

Writes results/figures/<name>.png (slides, 200 dpi) and .pdf (LaTeX, vector).
Colour follows the entity in every figure: 3B is always blue and 7B is always orange.
These are the first two slots of the colour-blind-safe reference palette.
Error bars are 95% Wilson CIs. The dashed line is chance (50%) where that applies.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLOR = {"3b": "#2a78d6", "7b": "#eb6834"}
NEUTRAL = "#a3a29c"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
LABEL = {"3b": "Qwen2.5-Coder-3B", "7b": "Qwen2.5-Coder-7B (4-bit)"}

plt.rcParams.update({
    "font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _p(x):
    return None if x is None or x.get("p") is None else 100 * x["p"]


def _err(x):
    p = _p(x)
    return [[p - 100 * x["lo"]], [100 * x["hi"] - p]] if p is not None else None


def grouped_bars(ax, groups, series, get, colors, labels, chance=None, ylim=100):
    """groups: x categories; series: one bar per series within each group; get(series, group) -> prop."""
    n = len(series)
    width = 0.8 / n
    for i, s in enumerate(series):
        for j, g in enumerate(groups):
            x = get(s, g)
            p = _p(x)
            if p is None:
                continue
            xpos = j + (i - (n - 1) / 2) * width
            ax.bar(xpos, p, width * 0.92, color=colors[i], label=labels[i] if j == 0 else None,
                   edgecolor=SURFACE, linewidth=1)
            ax.errorbar(xpos, p, yerr=_err(x), fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
            ax.text(xpos, min(100 * x["hi"], ylim) + 1.5, f"{p:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    if chance is not None:
        ax.axhline(chance, color=INK2, linestyle=(0, (3, 3)), linewidth=0.8)
        # label the line just outside the plot area so it never collides with a bar
        ax.text(1.01, chance, "chance", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(0, ylim + 12)
    ax.set_yticks(range(0, ylim + 1, 25))


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def models_in(d, prefix):
    return [m for m in ("3b", "7b") if f"{prefix}_{m}" in d]


def fig_position_bias(M, out):
    P = M["pairwise"]
    ms = models_in(P, "exp1")
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    grouped_bars(ax, ["cf", "cs"], ms, lambda m, o: P[f"exp1_{m}"][f"acc_{o}"],
                 [COLOR[m] for m in ms], [LABEL[m] for m in ms], chance=50)
    ax.set_xticklabels(["Correct answer\nshown as A", "Correct answer\nshown as B"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp 1: accuracy by answer order")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    save(fig, out, "f1_position_bias")


def fig_pointwise(M, out):
    Q = M["pointwise"]
    ms = models_in(Q, "exp3")
    if not ms:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    grouped_bars(ax, ["accuracy", "far", "frr"], ms, lambda m, k: Q[f"exp3_{m}"][k],
                 [COLOR[m] for m in ms], [LABEL[m] for m in ms])
    ax.set_xticklabels(["Accuracy", "False accept.\n(FAR)", "False reject.\n(FRR)"])
    ax.set_ylabel("%")
    ax.set_title("Exp 3: pointwise judging")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5)
    save(fig, out, "f2_pointwise")


def fig_fixes(M, out):
    P = M["pairwise"]
    ms = models_in(P, "exp1")
    if not ms:
        return
    methods = [("base", "Baseline"), ("4a", "Order\naveraging"), ("exp4b", "Debias\nprompt"), ("exp4c", "Majority\nof 3")]
    methods = [(k, l) for k, l in methods if k in ("base", "4a") or any(f"{k}_{m}" in P for m in ms)]

    def get(metric):
        def f(m, k):
            b = P[f"exp1_{m}"]
            if k == "base":
                return b[metric]
            if k == "4a":
                return b["orderavg_acc_decided"] if metric == "accuracy" else None
            return P.get(f"{k}_{m}", {}).get(metric)
        return f

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, metric, title in ((axes[0], "accuracy", "Accuracy (%)"),
                              (axes[1], "consistency", "Same pick in both orders (%)")):
        grouped_bars(ax, [k for k, _ in methods], ms, get(metric), [COLOR[m] for m in ms],
                     [LABEL[m] for m in ms], chance=50 if metric == "accuracy" else None)
        ax.set_xticklabels([l for _, l in methods])
        ax.set_title(title)
    # 4a is consistent by construction; say so instead of drawing a 100% bar
    ax = axes[1]
    ax.text([k for k, _ in methods].index("4a"), 4, "100 by\ndefinition", ha="center", va="bottom",
            fontsize=7, color=INK2)
    cov = ", ".join(f"{m.upper()} {_p(P[f'exp1_{m}']['orderavg_coverage']):.0f}%" for m in ms)
    fig.suptitle("Exp 4: position-bias fixes vs baseline", x=0.01, y=1.0, ha="left", fontsize=10,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.04), ncol=2, fontsize=7.5)
    fig.text(0.01, 0.0, f"Order averaging: accuracy on decided pairs only (decided: {cov})",
             ha="left", va="bottom", fontsize=7, color=INK2)
    save(fig, out, "f3_fixes")


def fig_verbosity(M, out):
    V = M["verbosity"]
    if not V:
        return
    m = "3b" if "3b" in V else next(iter(V))
    v = V[m]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    grouped_bars(ax, ["cf", "cs", ""], ["baseline", "padded"],
                 lambda s, o: v[f"{s}_{o}"] if o else v[s], [NEUTRAL, COLOR[m]],
                 ["Plain wrong answer", "Wrong answer padded with comments"], chance=None)
    ax.set_xticklabels(["Correct=A", "Correct=B", "Overall"])
    ax.set_ylabel("Wrong answer chosen (%)")
    ax.set_title(f"Exp 2: verbosity ({m.upper()})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=7.5)
    save(fig, out, "f4_verbosity")


def fig_difficulty(M, out):
    D = M["difficulty"]
    ms = models_in(D, "exp1")
    if not ms:
        return
    lv = ["easy", "medium", "hard"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    grouped_bars(ax, lv, ms, lambda m, l: D[f"exp1_{m}"].get(l, {}).get("accuracy"),
                 [COLOR[m] for m in ms], [LABEL[m] for m in ms], chance=50)
    ax.set_xticklabels([l.capitalize() for l in lv])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp 1 accuracy by difficulty")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5)
    save(fig, out, "f5_difficulty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "results" / "metrics.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()
    M = json.loads(Path(args.metrics).read_text())
    out = Path(args.out)
    for f in (fig_position_bias, fig_pointwise, fig_fixes, fig_verbosity, fig_difficulty):
        f(M, out)


if __name__ == "__main__":
    main()
