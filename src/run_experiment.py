"""
run_experiment.py -- batch runner with Drive checkpointing and resume.

    python -m src.run_experiment --exp exp1 --model 3b --backend vllm \
        --data data/pairs_150.jsonl --out /content/drive/MyDrive/peogai/results/raw

Experiments
    exp1   pairwise_v1, both orders, greedy                 -> 300 calls / model
    exp3   pointwise_v1, every answer on its own, greedy    -> 300 calls / model
    exp4b  pairwise_debias, both orders, greedy             -> 300 calls / model
    exp4c  pairwise_v1, both orders, 3 samples @ T=0.7      -> 900 generations (3B)
    exp2   pairwise_v1, both orders, incorrect answer padded
           with comments (needs field `incorrect_verbose`)  -> 300 calls (3B)
    (exp4a = order-averaging is computed by P2 from exp1 files; no new calls.)

Output: <out>/<exp>_<model>.jsonl (one row per call) + <exp>_<model>_meta.json.
Re-running the same command skips job_ids already in the file, so a dropped
Colab session just needs the same cell run again.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.judge import judge_batch, load_backend  # noqa: E402

EXPS = {
    "exp1":  dict(prompt="pairwise_v1",     kind="pairwise",  temperature=0.0, n=1),
    "exp3":  dict(prompt="pointwise_v1",    kind="pointwise", temperature=0.0, n=1),
    "exp4b": dict(prompt="pairwise_debias", kind="pairwise",  temperature=0.0, n=1),
    "exp4c": dict(prompt="pairwise_v1",     kind="pairwise",  temperature=0.7, n=3),
    "exp2":  dict(prompt="pairwise_v1",     kind="pairwise",  temperature=0.0, n=1, verbose=True),
}

# Field-name aliases so we don't break if P2's file (or the raw dataset) names things differently.
ALIASES = {
    "pair_id":   ["pair_id", "id", "question_id", "problem_id"],
    "problem":   ["problem", "question_content", "question", "prompt", "instruction"],
    "correct":   ["correct", "correct_code", "pos_response", "chosen", "positive"],
    "incorrect": ["incorrect", "incorrect_code", "neg_response", "rejected", "negative"],
    "difficulty": ["difficulty", "level"],
}


def _get(row, key, required=True):
    for k in ALIASES[key]:
        if k in row and row[k] not in (None, ""):
            return row[k]
    if required:
        raise KeyError(f"None of {ALIASES[key]} found in row keys {list(row)}")
    return None


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            pairs.append(dict(
                pair_id=str(_get(r, "pair_id", required=False) or f"p{i:03d}"),
                problem=_get(r, "problem"), correct=_get(r, "correct"),
                incorrect=_get(r, "incorrect"),
                incorrect_verbose=r.get("incorrect_verbose"),
                difficulty=_get(r, "difficulty", required=False)))
    return pairs


def build_jobs(pairs, cfg):
    """Each job = one judge call. Pairwise: order 'cf' (correct shown as A) and 'cs'
    (correct shown as B). Pointwise: one job per answer."""
    jobs = []
    for p in pairs:
        bad = p["incorrect"]
        if cfg.get("verbose"):
            if not p["incorrect_verbose"]:
                raise ValueError(f"{p['pair_id']}: exp2 needs `incorrect_verbose` (run src/perturb.py)")
            bad = p["incorrect_verbose"]
        base = dict(pair_id=p["pair_id"], difficulty=p["difficulty"])
        if cfg["kind"] == "pairwise":
            jobs.append(dict(base, job_id=f"{p['pair_id']}_cf", order="cf", gold="A",
                             item=dict(problem=p["problem"], code_a=p["correct"], code_b=bad)))
            jobs.append(dict(base, job_id=f"{p['pair_id']}_cs", order="cs", gold="B",
                             item=dict(problem=p["problem"], code_a=bad, code_b=p["correct"])))
        else:
            jobs.append(dict(base, job_id=f"{p['pair_id']}_pos", answer="correct", gold="correct",
                             item=dict(problem=p["problem"], code=p["correct"])))
            jobs.append(dict(base, job_id=f"{p['pair_id']}_neg", answer="incorrect", gold="incorrect",
                             item=dict(problem=p["problem"], code=bad)))
    return jobs


def repair_tail(path):
    """Drop a half-written last line (session killed mid-write) so appends stay valid JSONL."""
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        path.write_bytes(data[:cut])
        print(f"repaired truncated last line in {path.name}")


def done_ids(path):
    repair_tail(path)
    ids = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["job_id"])
                except (json.JSONDecodeError, KeyError):
                    pass  # half-written last line from a killed session
    return ids


def flush(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def quick_summary(path, kind):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    ok = [r for r in rows if r["parse_ok"]]
    acc = sum(r["verdict"] == r["gold"] for r in rows) / n if n else 0
    s = dict(n=n, accuracy=round(acc, 4), parse_fail=n - len(ok))
    if kind == "pairwise":
        s["pct_A"] = round(sum(r["verdict"] == "A" for r in ok) / max(len(ok), 1), 4)
        by = {}
        for r in rows:
            by.setdefault(r["pair_id"], {})[r["order"]] = r["verdict"] == r["gold"]
        both = [v for v in by.values() if len(v) == 2]
        s["consistent_pairs"] = sum((v["cf"] == v["cs"]) for v in both)  # right-right or wrong-wrong
        s["pairs_both_orders"] = len(both)
    else:
        neg = [r for r in rows if r["gold"] == "incorrect"]
        s["false_accept_rate"] = round(sum(r["verdict"] == "correct" for r in neg) / max(len(neg), 1), 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--model", required=True, choices=["3b", "7b"])
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf", "ollama", "mock"])
    ap.add_argument("--data", default="data/pairs_150.jsonl")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for output file, e.g. _smoke")
    args = ap.parse_args()

    cfg = EXPS[args.exp]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.exp}_{args.model}{args.tag}.jsonl"
    meta_path = out_dir / f"{args.exp}_{args.model}{args.tag}_meta.json"

    pairs = load_pairs(args.data)[: args.limit]
    jobs = build_jobs(pairs, cfg)
    seen = done_ids(out_path)
    todo = [j for j in jobs if j["job_id"] not in seen]
    print(f"[{args.exp}/{args.model}] {len(jobs)} jobs, {len(seen)} already done, {len(todo)} to run")
    if not todo:
        print(json.dumps(quick_summary(out_path, cfg["kind"]), indent=2))
        return

    t_load = time.time()
    backend = load_backend(args.backend, args.model)
    load_s = time.time() - t_load
    print(f"loaded {backend.model_id} [{backend.quant}] in {load_s:.0f}s on {gpu_name()}")

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {
        "exp": args.exp, "model_key": args.model, "sessions": [], "compute_seconds": 0.0}
    session = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), backend=backend.name,
                   model_id=backend.model_id, quant=backend.quant, gpu=gpu_name(),
                   python=platform.python_version(), load_seconds=round(load_s, 1), calls=0)
    meta["sessions"].append(session)

    buf, t_run = [], time.time()
    try:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            res = judge_batch(backend, cfg["prompt"], [j["item"] for j in chunk],
                              temperature=cfg["temperature"], n=cfg["n"],
                              max_new_tokens=args.max_new_tokens)
            for j, r in zip(chunk, res):
                row = {k: v for k, v in j.items() if k != "item"}
                row.update(r, exp=args.exp, model=backend.model_id, prompt=cfg["prompt"],
                           temperature=cfg["temperature"], n_samples=cfg["n"],
                           correct=r["verdict"] == j["gold"])
                buf.append(row)
            session["calls"] += len(chunk)
            if len(buf) >= args.save_every:
                flush(out_path, buf); buf = []
            done = len(seen) + session["calls"]
            el = time.time() - t_run
            eta = el / session["calls"] * (len(todo) - session["calls"])
            print(f"  {done}/{len(jobs)}  {el / session['calls']:.2f}s/call  ETA {eta / 60:.1f} min",
                  flush=True)
    finally:  # also runs on KeyboardInterrupt / Colab stop
        if buf:
            flush(out_path, buf)
        run_s = time.time() - t_run
        session["run_seconds"] = round(run_s, 1)
        session["sec_per_call"] = round(run_s / max(session["calls"], 1), 3)
        meta["compute_seconds"] = round(meta["compute_seconds"] + run_s, 1)
        meta_path.write_text(json.dumps(meta, indent=2))

    s = quick_summary(out_path, cfg["kind"])
    meta["summary"] = s
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
