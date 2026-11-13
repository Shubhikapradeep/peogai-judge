"""
setup_summary.py -- turn results/raw/*_meta.json into the numbers for the
Experimental Setup section (calls, wall-clock time, s/call, GPU, quantisation).

    python -m src.setup_summary --raw results/raw
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    args = ap.parse_args()
    rows, total_calls, total_s = [], 0, 0.0
    for m in sorted(Path(args.raw).glob("*_meta.json")):
        if "_smoke" in m.name:
            continue
        d = json.loads(m.read_text())
        s = d["sessions"]
        calls = sum(x["calls"] for x in s)
        n = d.get("summary", {}).get("n", calls)
        last = s[-1]
        rows.append((d["exp"], d["model_key"], last["model_id"], last["quant"], last["backend"],
                     sorted({x["gpu"] for x in s}), n, d["compute_seconds"], len(s)))
        total_calls += n
        total_s += d["compute_seconds"]

    print("| Exp | Model | Checkpoint | Quant | Backend | GPU | Calls | Time (min) | s/call | Sessions |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for e, k, mid, q, b, g, n, sec, ns in rows:
        print(f"| {e} | {k.upper()} | {mid} | {q} | {b} | {', '.join(g)} | {n} | "
              f"{sec / 60:.1f} | {sec / max(n, 1):.2f} | {ns} |")
    print(f"\nTotal: {total_calls} judge calls, {total_s / 3600:.2f} GPU-hours")


if __name__ == "__main__":
    main()
