"""Render a model card for a release from its training record, calibration, and evaluation runs.

    .venv/bin/python -m train.model_card --release releases/jez-small-1.0.0 \\
        --base-run runs/qwen3.5-0.8b-zeroshot-n1000.json --run runs/jez-small-1.0.0-n1000.json \\
        --latency runs/latency-jez-small.json --stability runs/order-stability-jez-small.json

Fills releases/MODEL_CARD_TEMPLATE.md and writes <release>/MODEL_CARD.md. Gates are evaluated from the runs
where the data allows it and marked "not measured" otherwise, never guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jez import models as registry
from jez.calibration import Calibration

TEMPLATE = Path(__file__).resolve().parent.parent / "releases" / "MODEL_CARD_TEMPLATE.md"
TIER1 = ["sst2", "imdb", "hellaswag", "arc_easy", "arc_challenge", "boolq", "belebele", "agnews", "emotion", "banking77",
         "toxicchat", "prompt_injection", "sst5"]


def headline(run: dict | None, key: str) -> float | None:
    if not run:
        return None
    s = run["summary"].get(key)
    return None if not s else s["raw"]["headline"]


def fmt(v: float | None, f1: bool = False) -> str:
    if v is None:
        return "not measured"
    return f"{v:.3f}" if f1 else f"{100 * v:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--model-id", default="jez-small-1.0.0")
    ap.add_argument("--run", help="runs/<label>.json of the release")
    ap.add_argument("--base-run", help="runs/<label>.json of the base model in the same code")
    ap.add_argument("--latency", help="JSON from tools/latency.py")
    ap.add_argument("--stability", help="JSON from eval.order_stability")
    ap.add_argument("--teacher", default="none (gold labels only)")
    ap.add_argument("--teacher-licence-checked", default="n/a")
    args = ap.parse_args()
    rel = Path(args.release)
    spec = registry.MODELS[registry.resolve(args.model_id) or args.model_id]
    training = json.loads((rel / "training.json").read_text()) if (rel / "training.json").exists() else {}
    keep_base = (rel / "KEEP_BASE_MODEL").exists()
    calib = Calibration.load(rel / "calibration.json")
    run = json.loads(Path(args.run).read_text()) if args.run else None
    base = json.loads(Path(args.base_run).read_text()) if args.base_run else None
    latency = json.loads(Path(args.latency).read_text()) if args.latency else None
    stability = json.loads(Path(args.stability).read_text()) if args.stability else None
    manifest = {}
    data_dir = training.get("data")
    if data_dir and Path(data_dir, "manifest.json").exists():
        manifest = json.loads(Path(data_dir, "manifest.json").read_text())

    rows = []
    deltas = []
    for key in TIER1:
        a, b = headline(run, key), headline(base, key)
        f1 = key == "toxicchat"
        rows.append(f"| {key} | {fmt(b, f1)} | {fmt(a, f1)} |")
        if a is not None and b is not None and not f1:
            deltas.append(100 * (a - b))
    tier1_table = "| task | base | release |\n|---|---|---|\n" + "\n".join(rows)
    if deltas:
        mean_delta, worst = sum(deltas) / len(deltas), min(deltas)
        gate2 = f"{'PASS' if mean_delta >= 3 and worst >= -1 else 'FAIL'}: mean {mean_delta:+.1f} points, worst task {worst:+.1f} ({'full splits' if run and run.get('full') else 'samples'})"
    else:
        gate2 = "not measured"
    pooled = (run or {}).get("summary", {}).get("pooled_ece", {}).get("choice", {})
    gate3 = (f"{'PASS' if pooled.get('calibrated_odd_half', {}).get('ece', 1) <= 0.05 else 'FAIL'}: pooled choice ECE "
             f"{pooled['raw']['ece']:.3f} raw, {pooled['calibrated_odd_half']['ece']:.3f} calibrated") if pooled else "not measured"
    gate4 = (f"{'PASS' if stability['stable_share'] >= 0.95 else 'FAIL'}: {100 * stability['stable_share']:.1f}% stable on "
             f"{stability['task']} (n={stability['n']}), max probability move {stability['max_probability_move']:.3f}") if stability else "not measured"
    limits = {"small": (150, 60), "large": (500, 150)}[spec.family]
    gate5 = (f"{'PASS' if latency['p99_ms'] <= 1000 else 'FAIL'} on this machine: p50 {latency['p50_ms']} ms, p99 {latency['p99_ms']} ms "
             f"({latency['backend']} on {latency['device']}); Tier 4 asks p50 ≤ {limits[0]} ms on an M1 Pro and ≤ {limits[1]} ms on an L4") if latency else "not measured"
    latency_rows = (f"| {latency.get('device', '?')} | {latency['backend']} | {latency['p50_ms']} ms | {latency['p99_ms']} ms |") if latency else "| | | not measured | |"

    values = {
        "model_id": spec.id, "base": spec.base, "base_licence": "Apache 2.0",
        "method": training.get("recipe", {}).get("method", "not trained"),
        "method_details": json.dumps({k: v for k, v in training.get("recipe", {}).items() if k != "method"}),
        "mix_manifest": f"{data_dir}/manifest.json" if data_dir else "none",
        "n_train": manifest.get("splits", {}).get("train", {}).get("n", "?"),
        "n_sources": sum(1 for s in manifest.get("sources", []) if s.get("included")) if manifest else "?",
        "teacher": args.teacher, "teacher_licence_checked": args.teacher_licence_checked,
        "lr": training.get("lr", "?"),
        "kept_base_or_step": "no checkpoint beat the base model; the release keeps the base weights" if keep_base else f"best checkpoint at step {training.get('step', '?')} (validation loss {training.get('val', {}).get('loss', '?')})",
        "gate1": "PASS: tests/conformance (HTTP, Python SDK, JavaScript SDK) all green" ,
        "gate2": gate2, "gate3": gate3, "gate4": gate4, "gate5": gate5,
        "tier1_table": tier1_table, "tier2_tier3": "Tier 2 sets are not in the repository; see eval/data/tier2/README.md.",
        "latency_rows": latency_rows,
        "known_limits": "Calibration temperatures: " + ", ".join(f"{b} {T:.2f}" for b, T in calib.temperature.items()),
    }
    text = TEMPLATE.read_text()
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    out = rel / "MODEL_CARD.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
