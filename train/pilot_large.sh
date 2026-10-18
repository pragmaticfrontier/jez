#!/usr/bin/env bash
# Recipe pilot for Jez-large: a short LoRA run that saves adapter snapshots, each scored on Tier 1 samples against the
# Gemma zero-shot run on the same items, and a gate-2 table at the end. Cheaper than a release run: the snapshots are
# evaluated straight from the adapter directory (merged in memory), nothing else is written.
#
#   train/pilot_large.sh <mix-dir> <out-dir> <label> [train.finetune flags...]
#   GUARD="python3 tools/no_guard.py" PY=python3 train/pilot_large.sh /data/mix-v2 /releases/mixv2/pilots/gentle gentle \
#       --lr 5e-5 --steps 1000 --snapshot-every 250 [--base-kl-weight 0.5]
#
# PILOT_N sets the items per task (default 300). Logs under runs/pilot-<label>/; results runs/<label>-step-N-n<N>.json.
set -euo pipefail
cd "$(dirname "$0")/.."
MIX=$1; OUT=$2; LABEL=$3; shift 3
PY=${PY:-.venv/bin/python}
GUARD=${GUARD:-"tools/gpu_guard.py --cuda-fraction 0.5 --telemetry-period 0.5 --max-gpu-temp 82 --max-soc-temp 88 --min-tlimit 8"}
N=${PILOT_N:-300}
BASE=google/gemma-4-E4B-it
BASE_RUN=runs/gemma-4-E4B-it-zeroshot-n$N.json
LOGS=runs/pilot-$LABEL
mkdir -p "$LOGS" "$OUT"

step() { echo "$(date '+%F %T') == $*"; }

if [ ! -f "$BASE_RUN" ]; then
  step "Gemma 4 E4B zero-shot on $N-item samples"
  $GUARD --timeout 7200 -- $PY -m eval.run_eval --checkpoint "$BASE" --label "gemma-4-E4B-it-zeroshot-n$N" \
      --model-id jez-large-1.0.0 --tasks tier1 --n "$N" --latency-items 5 \
      2>&1 | tee "$LOGS/baseline.log" | { grep --line-buffered -E "INFO eval|exited" || true; }
fi

if [ ! -f "$OUT/training.json" ] && [ ! -f "$OUT/KEEP_BASE_MODEL" ]; then
  step "pilot training: $*"
  $GUARD --timeout 43200 -- $PY -m train.finetune --model jez-large-1.0.0 --data "$MIX" --out "$OUT" --micro-tokens 16384 \
      --eval-every 250 --log-every 20 "$@" 2>&1 | tee -a "$LOGS/train.log" | { grep --line-buffered -E "INFO train|exited" || true; }
fi

for snap in "$OUT"/snapshots/step-*; do
  [ -d "$snap/adapter" ] || continue
  s=$(basename "$snap")
  if [ ! -f "runs/$LABEL-$s-n$N.json" ]; then
    step "Tier 1 on $N-item samples: $s"
    $GUARD --timeout 7200 -- $PY -m eval.run_eval --checkpoint "$snap/adapter" --label "$LABEL-$s-n$N" \
        --model-id jez-large-1.0.0 --tasks tier1 --n "$N" --latency-items 5 \
        2>&1 | tee "$LOGS/eval-$s.log" | { grep --line-buffered -E "INFO eval|exited" || true; }
  fi
done

step "gate 2 table"
$PY -m eval.gate2 "$BASE_RUN" $(ls "runs/$LABEL"-step-*-n"$N".json | sort -t- -k3,3n) | tee "$LOGS/gate2.md"
step "done"
