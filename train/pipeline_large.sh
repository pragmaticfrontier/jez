#!/usr/bin/env bash
# The Jez-large release pipeline: Gemma 4 E4B, LoRA. Same shape as pipeline_small.sh plus the Gemma zero-shot baseline
# (gate 2 compares against it) and the adapter merge that gives the engine plain weights to serve. Logs under
# runs/pipeline-large/.
#
#   train/pipeline_large.sh [mix-dir] [release-dir]
#   GUARD="python3 tools/no_guard.py" PY=python3 SKIP_LATENCY=1 train/pipeline_large.sh /data/mix-v1 /releases/jez-large-1.0.0
#
# Each step is skipped when its output already exists, so the script can be rerun after an interruption; an interrupted
# full run resumes from its last saved state. A failing guard or job stops the script (pipefail). Gemma is too heavy for
# this machine's cooling, so on the Spark it needs the clock cap in tools/gb10-clock-cap.service; train/modal_app.py
# runs the same script on a rented GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
MIX=${1:-train/data/mix-v1}
REL=${2:-releases/jez-large-1.0.0}
PY=${PY:-.venv/bin/python}
GUARD=${GUARD:-"tools/gpu_guard.py --cuda-fraction 0.5 --telemetry-period 0.5 --max-gpu-temp 82 --max-soc-temp 88 --min-tlimit 8"}
LOGS=runs/pipeline-large
BASE=google/gemma-4-E4B-it
BASE_RUN=runs/gemma-4-E4B-it-zeroshot-n1000.json
MICRO="--micro-tokens 16384"   # Gemma's activations run several times Qwen3.5-0.8B's per token
mkdir -p "$LOGS" "$REL"

step() { echo "$(date '+%F %T') == $*"; }

if [ ! -f "$BASE_RUN" ]; then
  step "Gemma 4 E4B zero-shot baseline, 1,000-item samples"
  $GUARD --timeout 14400 -- $PY -m eval.run_eval --checkpoint "$BASE" --label gemma-4-E4B-it-zeroshot-n1000 \
      --model-id jez-large-1.0.0 --tasks tier1 --n 1000 --latency-items 20 \
      --save-calibration runs/gemma-4-E4B-it-zeroshot-calibration.json \
      2>&1 | tee "$LOGS/baseline.log" | { grep --line-buffered -E "INFO eval|exited" || true; }
fi

if [ ! -f "$LOGS/sweep.json" ]; then
  step "learning-rate sweep (5e-5, 1e-4, 2e-4 on 5% of the data)"
  $GUARD --timeout 28800 -- $PY -m train.finetune --model jez-large-1.0.0 --data "$MIX" $MICRO \
      --sweep 5e-5,1e-4,2e-4 --sweep-fraction 0.05 --eval-every 50 --log-every 10 \
      2>&1 | tee "$LOGS/sweep.log" | { grep --line-buffered -E "INFO train|chosen_lr|exited" || true; }
  $PY - "$LOGS/sweep.log" "$LOGS/sweep.json" <<'PYEOF'
import json, sys
text = open(sys.argv[1]).read()
start = text.index('{\n "sweep"')
end = text.index('\n}', start) + 2
json.dump(json.loads(text[start:end]), open(sys.argv[2], "w"), indent=1)
PYEOF
fi
LR=$($PY -c "import json; print(json.load(open('$LOGS/sweep.json'))['chosen_lr'])")
step "chosen learning rate: $LR"

if [ -f "$REL/state.pt" ] || { [ ! -f "$REL/training.json" ] && [ ! -f "$REL/KEEP_BASE_MODEL" ]; }; then
  step "full run$( [ -f "$REL/state.pt" ] && echo " (resuming from $REL/state.pt)" )"
  $GUARD --timeout 86400 -- $PY -m train.finetune --model jez-large-1.0.0 --data "$MIX" --lr "$LR" --out "$REL" $MICRO \
      --eval-every 200 --log-every 20 2>&1 | tee -a "$LOGS/train.log" | { grep --line-buffered -E "INFO train|exited" || true; }
fi
if [ -f "$REL/KEEP_BASE_MODEL" ]; then
  step "no checkpoint beat the base model; the release keeps the base weights"
  WEIGHTS=$BASE
else
  if [ ! -f "$REL/weights/config.json" ]; then
    step "merging the adapter into the base weights"
    $GUARD --timeout 3600 -- $PY -m train.merge_adapter --model jez-large-1.0.0 --release "$REL" 2>&1 | tail -3
  fi
  WEIGHTS=$REL/weights
fi

if [ ! -f "$REL/calibration.json" ]; then
  step "temperatures on the calibration split"
  $GUARD --timeout 7200 -- $PY -m train.fit_calibration --checkpoint "$WEIGHTS" --data "$MIX/calibration.jsonl" \
      --model-id jez-large-1.0.0 --out "$REL/calibration.json" 2>&1 | tee "$LOGS/calibration.log" | tail -3
fi

if [ ! -f runs/jez-large-1.0.0-n1000.json ]; then
  step "Tier 1 evaluation, 1,000-item samples, release temperatures"
  $GUARD --timeout 14400 -- $PY -m eval.run_eval --checkpoint "$WEIGHTS" --label jez-large-1.0.0-n1000 \
      --model-id jez-large-1.0.0 --tasks tier1 --n 1000 --latency-items 20 --calibration "$REL/calibration.json" \
      2>&1 | tee "$LOGS/eval.log" | { grep --line-buffered -E "INFO eval|exited" || true; }
fi

if [ ! -f runs/order-stability-agnews-jez-large-1.0.0.json ]; then
  step "order stability (AG News proxy, 200 items)"
  $GUARD --timeout 3600 -- $PY -m eval.order_stability --checkpoint "$WEIGHTS" --task agnews --n 200 \
      --calibration "$REL/calibration.json" --out runs/order-stability-agnews-jez-large-1.0.0.json 2>&1 | tail -4
fi

LATENCY=runs/latency-jez-large-1.0.0.json
if [ -z "${SKIP_LATENCY:-}" ] && [ ! -f "$LATENCY" ]; then
  step "latency on the reference request"
  $GUARD --timeout 1800 -- $PY tools/latency.py "$WEIGHTS" --runs 60 2>&1 | sed -n '/^{/,/^}/p' > "$LATENCY"
fi

step "model card"
LAT_ARG=()
[ -f "$LATENCY" ] && LAT_ARG=(--latency "$LATENCY") || echo "(no latency file; gate 5 is measured on the target machine)"
$PY -m train.model_card --release "$REL" --model-id jez-large-1.0.0 --run runs/jez-large-1.0.0-n1000.json \
    --base-run "$BASE_RUN" "${LAT_ARG[@]}" --stability runs/order-stability-agnews-jez-large-1.0.0.json
$PY -m eval.compare "$BASE_RUN" runs/jez-large-1.0.0-n1000.json | tee "$LOGS/compare.md"
step "done"
