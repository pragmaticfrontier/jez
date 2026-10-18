#!/usr/bin/env bash
# Build a Jez-large release from a trained adapter: merge, calibration temperatures, Tier 1 at 1,000 items against the
# Gemma baseline, order stability, a latency reference and the model card. Used when the adapter comes from a recipe
# pilot (train/pilot_large.sh) rather than from pipeline_large.sh. Each step is skipped when its output exists.
#
#   train/release_from_adapter.sh <adapter-dir> <training.json> <release-dir> [label] [training-json-patch]
# The optional fifth argument is a JSON object merged into the copied training.json (run facts an older record lacks,
# e.g. '{"base_kl_weight": 0.5, "steps": 1000}'). LR_NOTE and SELECTION_NOTE feed the card's schedule and selection rows.
#   GUARD="python3 tools/no_guard.py" PY=python3 train/release_from_adapter.sh \
#       /releases/mixv2/pilots/anchored/snapshots/step-1000/adapter /releases/mixv2/pilots/anchored/training.json \
#       /releases/mixv2/jez-large-1.0.0-anchored jez-large-1.0.0
#
# The base run runs/gemma-4-E4B-it-zeroshot-n1000.json must exist (gate 2 compares against it). The latency figure is a
# reference for the machine the script runs on; gate 5 is measured on the target machine and is not passed to the card.
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER=$1
TRAINING=$2
REL=$3
LABEL=${4:-jez-large-1.0.0}
PATCH=${5:-}
LR_NOTE=${LR_NOTE:-"chosen by a 3-rate sweep on 5% of the data"}
SELECTION_NOTE=${SELECTION_NOTE:-}
MIX=${MIX:-/data/mix-v2}
PY=${PY:-.venv/bin/python}
GUARD=${GUARD:-"tools/gpu_guard.py --cuda-fraction 0.5 --telemetry-period 0.5 --max-gpu-temp 82 --max-soc-temp 88 --min-tlimit 8"}
BASE_RUN=runs/gemma-4-E4B-it-zeroshot-n1000.json
LOGS=runs/release-$(basename "$REL")
RUN=runs/$LABEL-n1000.json
STAB=runs/order-stability-agnews-$LABEL.json
LAT=runs/latency-$LABEL-$(hostname -s 2>/dev/null || echo host).json
mkdir -p "$LOGS" "$REL"
[ -f "$BASE_RUN" ] || { echo "missing $BASE_RUN"; exit 2; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter at $ADAPTER"; exit 2; }

step() { echo "$(date '+%F %T') == $*"; }

if [ ! -f "$REL/adapter/adapter_config.json" ]; then
  step "copying the adapter and the training record into $REL"
  cp -r "$ADAPTER" "$REL/adapter"
  cp "$TRAINING" "$REL/training.json"
  if [ -n "$PATCH" ]; then
    $PY - "$REL/training.json" "$PATCH" <<'PYEOF'
import json, sys
path, patch = sys.argv[1], json.loads(sys.argv[2])
rec = json.load(open(path)); rec.update(patch); json.dump(rec, open(path, "w"), indent=1)
print("training.json patched with", sorted(patch))
PYEOF
  fi
fi

if [ ! -f "$REL/weights/config.json" ]; then
  step "merging the adapter into the base weights"
  $GUARD --timeout 3600 -- $PY -m train.merge_adapter --model jez-large-1.0.0 --release "$REL" 2>&1 | tail -3
fi
WEIGHTS=$REL/weights

if [ ! -f "$REL/calibration.json" ]; then
  step "temperatures on the calibration split"
  $GUARD --timeout 7200 -- $PY -m train.fit_calibration --checkpoint "$WEIGHTS" --data "$MIX/calibration.jsonl" \
      --model-id jez-large-1.0.0 --out "$REL/calibration.json" 2>&1 | tee "$LOGS/calibration.log" | tail -3
fi

if [ ! -f "$RUN" ]; then
  step "Tier 1 evaluation, 1,000-item samples, release temperatures"
  $GUARD --timeout 14400 -- $PY -m eval.run_eval --checkpoint "$WEIGHTS" --label "$LABEL-n1000" \
      --model-id jez-large-1.0.0 --tasks tier1 --n 1000 --latency-items 20 --calibration "$REL/calibration.json" \
      2>&1 | tee "$LOGS/eval.log" | { grep --line-buffered -E "INFO eval|exited|wrote" || true; }
fi

if [ ! -f "$STAB" ]; then
  step "order stability (AG News proxy, 200 items)"
  $GUARD --timeout 3600 -- $PY -m eval.order_stability --checkpoint "$WEIGHTS" --task agnews --n 200 \
      --calibration "$REL/calibration.json" --out "$STAB" 2>&1 | tail -4
fi

if [ ! -f "$LAT" ]; then
  step "latency reference on this machine"
  $GUARD --timeout 1800 -- $PY tools/latency.py "$WEIGHTS" --runs 60 2>&1 | sed -n '/^{/,/^}/p' > "$LAT" || true
fi

step "model card (gate 5 left to the target machine)"
$PY -m train.model_card --release "$REL" --model-id jez-large-1.0.0 --run "$RUN" --base-run "$BASE_RUN" --stability "$STAB" \
    --lr-note "$LR_NOTE" --selection-note "$SELECTION_NOTE"
$PY -m eval.compare "$BASE_RUN" "$RUN" | tee "$LOGS/compare.md"
$PY -m eval.gate2 "$BASE_RUN" "$RUN" --bootstrap > "$LOGS/gate2.md" 2>&1 || true
head -n 20 "$LOGS/gate2.md"
step "done"
