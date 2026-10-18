#!/usr/bin/env bash
# The whole Jez-small release pipeline, every GPU step under tools/gpu_guard.py, logs under runs/pipeline-small/.
#
#   train/pipeline_small.sh [mix-dir] [release-dir]
#   GUARD="python3 tools/no_guard.py" PY=python3 SKIP_LATENCY=1 train/pipeline_small.sh /data/mix-v1 /releases/jez-small-1.0.0   # cloud
#
# Steps: learning-rate sweep on 5% of the data -> full run at the chosen rate -> temperatures on the calibration
# split -> Tier 1 evaluation (1,000-item samples, release temperatures) -> order stability -> latency -> model card.
# Each step is skipped when its output already exists, so the script can be rerun after an interruption; an
# interrupted full run resumes from its last saved state (every 200 steps).
# A failing guard (memory watchdog, timeout, thermal kill, or the job itself) stops the script (pipefail).
set -euo pipefail
cd "$(dirname "$0")/.."
MIX=${1:-train/data/mix-v1}
REL=${2:-releases/jez-small-1.0.0}
PY=${PY:-.venv/bin/python}
GUARD=${GUARD:-"tools/gpu_guard.py --cuda-fraction 0.5 --telemetry-period 0.5 --max-gpu-temp 82 --max-soc-temp 88 --min-tlimit 8"}
LOGS=runs/pipeline-small
BASE_RUN=runs/qwen3.5-0.8b-zeroshot-n1000.json
mkdir -p "$LOGS" "$REL"

step() { echo "$(date '+%F %T') == $*"; }

if [ ! -f "$LOGS/sweep.json" ]; then
  step "learning-rate sweep (1e-5, 2e-5, 4e-5 on 5% of the data)"
  $GUARD --timeout 14400 -- $PY -m train.finetune --model jez-small-1.0.0 --data "$MIX" \
      --sweep 1e-5,2e-5,4e-5 --sweep-fraction 0.05 --eval-every 50 --log-every 10 \
      2>&1 | tee "$LOGS/sweep.log" | { grep --line-buffered -E "INFO train|chosen_lr|exited" || true; }
  $PY - "$LOGS/sweep.log" "$LOGS/sweep.json" <<'EOF'
import json, re, sys
text = open(sys.argv[1]).read()
start = text.index('{\n "sweep"')
end = text.index('\n}', start) + 2
json.dump(json.loads(text[start:end]), open(sys.argv[2], "w"), indent=1)
EOF
fi
LR=$($PY -c "import json; print(json.load(open('$LOGS/sweep.json'))['chosen_lr'])")
step "chosen learning rate: $LR"

# state.pt is the resumable training state of an interrupted full run; its presence means the run is not finished even
# when an earlier best checkpoint already wrote training.json.
if [ -f "$REL/state.pt" ] || { [ ! -f "$REL/training.json" ] && [ ! -f "$REL/KEEP_BASE_MODEL" ]; }; then
  step "full run$( [ -f "$REL/state.pt" ] && echo " (resuming from $REL/state.pt)" )"
  $GUARD --timeout 86400 -- $PY -m train.finetune --model jez-small-1.0.0 --data "$MIX" --lr "$LR" --out "$REL" \
      --eval-every 200 --log-every 20 2>&1 | tee -a "$LOGS/train.log" | { grep --line-buffered -E "INFO train|exited" || true; }
fi
if [ -f "$REL/KEEP_BASE_MODEL" ]; then
  step "no checkpoint beat the base model; the release keeps the base weights"
  WEIGHTS=Qwen/Qwen3.5-0.8B
else
  WEIGHTS=$REL/weights
fi

if [ ! -f "$REL/calibration.json" ]; then
  step "temperatures on the calibration split"
  $GUARD --timeout 7200 -- $PY -m train.fit_calibration --checkpoint "$WEIGHTS" --data "$MIX/calibration.jsonl" \
      --out "$REL/calibration.json" 2>&1 | tee "$LOGS/calibration.log" | tail -3
fi

if [ ! -f runs/jez-small-1.0.0-n1000.json ]; then
  step "Tier 1 evaluation, 1,000-item samples, release temperatures"
  $GUARD --timeout 14400 -- $PY -m eval.run_eval --checkpoint "$WEIGHTS" --label jez-small-1.0.0-n1000 \
      --model-id jez-small-1.0.0 --tasks tier1 --n 1000 --latency-items 20 --calibration "$REL/calibration.json" \
      2>&1 | tee "$LOGS/eval.log" | { grep --line-buffered -E "INFO eval|exited" || true; }
fi

if [ ! -f runs/order-stability-agnews-jez-small-1.0.0.json ]; then
  step "order stability (AG News proxy, 200 items)"
  $GUARD --timeout 3600 -- $PY -m eval.order_stability --checkpoint "$WEIGHTS" --task agnews --n 200 \
      --calibration "$REL/calibration.json" --out runs/order-stability-agnews-jez-small-1.0.0.json 2>&1 | tail -4
fi

LATENCY=runs/latency-jez-small-1.0.0.json
if [ -z "${SKIP_LATENCY:-}" ] && [ ! -f "$LATENCY" ]; then
  step "latency on the reference request"
  $GUARD --timeout 1800 -- $PY tools/latency.py "$WEIGHTS" --runs 60 2>&1 | sed -n '/^{/,/^}/p' > "$LATENCY"
fi

step "model card"
LAT_ARG=()
[ -f "$LATENCY" ] && LAT_ARG=(--latency "$LATENCY") || echo "(no latency file; gate 5 is measured on the target machine)"
$PY -m train.model_card --release "$REL" --model-id jez-small-1.0.0 --run runs/jez-small-1.0.0-n1000.json \
    --base-run "$BASE_RUN" "${LAT_ARG[@]}" --stability runs/order-stability-agnews-jez-small-1.0.0.json
$PY -m eval.compare "$BASE_RUN" runs/jez-small-1.0.0-n1000.json | tee "$LOGS/compare.md"
step "done"
