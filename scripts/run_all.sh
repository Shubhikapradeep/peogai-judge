#!/usr/bin/env bash
# P1: run every experiment end to end, then metrics + plots.
#   bash scripts/run_all.sh <OUT_DIR>          (OUT_DIR on Drive, e.g. /content/drive/MyDrive/peogai)
# Safe to re-run after a disconnect: finished calls are skipped.
set -u
OUT=${1:-results}
RAW=$OUT/results/raw
mkdir -p "$RAW"
LOG=$OUT/run_all.log
exec > >(tee -a "$LOG") 2>&1
echo "===== run_all started $(date) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU"

# ---- backend: vLLM if it installs and passes the smoke test, else transformers+bitsandbytes
BACKEND=$(cat "$OUT/backend.txt" 2>/dev/null || echo "")
smoke() {  # $1 backend
  python -m src.run_experiment --exp exp1 --model 3b --backend "$1" --limit 5 --tag _smoke --out "$RAW"
}
if [ -z "$BACKEND" ]; then
  pip install -q vllm && smoke vllm && BACKEND=vllm
  if [ -z "$BACKEND" ]; then
    echo ">>> vLLM failed, switching to transformers + bitsandbytes"
    pip install -q -U transformers accelerate bitsandbytes
    smoke hf && BACKEND=hf
  fi
  [ -z "$BACKEND" ] && { echo "!!! both backends failed, see log above"; exit 1; }
  echo "$BACKEND" > "$OUT/backend.txt"
elif [ "$BACKEND" = vllm ]; then pip install -q vllm
else pip install -q -U transformers accelerate bitsandbytes; fi
BS=8; [ "$BACKEND" = hf ] && BS=2
echo ">>> backend: $BACKEND, batch size $BS"

python -m src.run_experiment --exp exp1 --model 7b --backend "$BACKEND" --limit 5 --tag _smoke --out "$RAW" --batch-size $BS

# ---- smoke gate: stop if the judge's output isn't parsing
python - "$RAW" <<'EOF' || exit 1
import json, sys, glob
for f in sorted(glob.glob(sys.argv[1] + "/*_smoke.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    bad = sum(not r["parse_ok"] for r in rows)
    print(f, f"parse failures {bad}/{len(rows)}", [r["raw"][:30] for r in rows[:4]])
    if bad > len(rows) * 0.3:
        sys.exit("!!! too many parse failures in smoke test; send this log to the team")
EOF

# ---- full runs, most important first (a session cut-off loses the least)
run() { python -m src.run_experiment --backend "$BACKEND" --batch-size $BS --out "$RAW" "$@" || echo "!!! failed: $*"; }
run --exp exp1  --model 3b
run --exp exp3  --model 3b
run --exp exp1  --model 7b
run --exp exp3  --model 7b
run --exp exp4b --model 3b
run --exp exp4b --model 7b
run --exp exp4c --model 3b
run --exp exp2  --model 3b --data data/pairs_150_verbose.jsonl

# ---- analysis (P2's scripts) + setup numbers for Section IV
pip install -q matplotlib
python -m src.metrics --raw "$RAW" --out "$OUT/results"
python -m src.plots --metrics "$OUT/results/metrics.json" --out "$OUT/results/figures"
python -m src.write_results --results "$OUT/results" --report "$OUT/results/results_auto.tex"
python -m src.setup_summary --raw "$RAW" | tee "$OUT/results/setup_summary.md"
echo "===== run_all finished $(date) ====="
echo "Everything is in $OUT/results (Drive). Key numbers: $OUT/results/key_numbers.md; findings: $OUT/results/findings.md"
