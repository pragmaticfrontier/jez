#!/usr/bin/env bash
# Jez-large 1.0.0 release from the anchored recipe pilot (2026-10-18): the step-1000 LoRA snapshot trained with
# lr 5e-5, 1,000 steps and a 0.5 KL anchor to the base (reports/mixv2/pilots/README.md). Runs on Modal:
#   modal run --detach train/modal_app.py --cmd train/release_anchored.sh --no-wait
set -euo pipefail
cd "$(dirname "$0")/.."
export LR_NOTE="chosen on Tier 1 development samples at n = 300 (recipe pilots, reports/mixv2/pilots)"
export SELECTION_NOTE="the step-1000 snapshot of a 1,000-step run was picked on Tier 1 samples, not on in-mix validation"
exec train/release_from_adapter.sh \
  /releases/mixv2/pilots/anchored/snapshots/step-1000/adapter \
  /releases/mixv2/pilots/anchored/training.json \
  /releases/mixv2/jez-large-1.0.0-anchored \
  jez-large-1.0.0-anchored \
  '{"base_kl_weight": 0.5, "kl_weight": 0.5, "steps": 1000, "epoch_steps": 4359, "batch": 64, "n_train": 278999, "data": "/data/mix-v2"}'
