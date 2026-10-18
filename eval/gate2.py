"""Gate 2 at a glance: per-task deltas of one or more runs against a base run, with the verdict for each run.

    .venv/bin/python -m eval.gate2 runs/gemma-4-E4B-it-zeroshot-n300.json runs/gentle-step-*-n300.json
    .venv/bin/python -m eval.gate2 <base.json> <run.json>... --bootstrap    # adds paired 95% intervals per task

Gate 2 asks for a Tier 1 mean at least 3 points above the base with no task more than 1 point below it; ToxicChat is
reported as F1 and left out of the mean, Belebele counts once as the mean of its 12 languages. The optional bootstrap
resamples items, paired between base and run, so the interval says whether a drop is more than sampling noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.model_card import TIER1, headline


def deltas(base: dict, run: dict) -> dict[str, float | None]:
    """Per Tier 1 task: run minus base in points (F1 tasks in F1 units), None when either side is missing."""
    out = {}
    for key in TIER1:
        a, b = headline(run, key), headline(base, key)
        out[key] = None if a is None or b is None else (a - b if key == "toxicchat" else 100 * (a - b))
    return out


def verdict(d: dict[str, float | None]) -> dict:
    scored = [v for k, v in d.items() if v is not None and k != "toxicchat"]
    if not scored:
        return {"mean": None, "worst": None, "passed": False, "n_tasks": 0}
    mean, worst = sum(scored) / len(scored), min(scored)
    return {"mean": mean, "worst": worst, "passed": mean >= 3 and worst >= -1, "n_tasks": len(scored)}


def paired_intervals(base: dict, run: dict, resamples: int = 2000, seed: int = 0) -> dict[str, tuple[float, float]]:
    """95% bootstrap interval of the accuracy delta per task, from the per-item records (same items, same order)."""
    rng = np.random.default_rng(seed)
    bt = {t["task"]: t for t in base.get("tasks", [])}
    out = {}
    for t in run.get("tasks", []):
        b = bt.get(t["task"])
        if not b or len(b["items"]) != len(t["items"]):
            continue
        a = np.array([int(np.argmax(i["logprobs"]) == i["gold"]) for i in b["items"]])
        s = np.array([int(np.argmax(i["logprobs"]) == i["gold"]) for i in t["items"]])
        d = s - a
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(resamples)]) * 100
        out[t["task"]] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


def fmt(v: float | None, key: str) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if key == "toxicchat" else f"{100 * v:.1f}"


def table(base: dict, runs: list[dict]) -> str:
    names = [r.get("label", f"run{i}") for i, r in enumerate(runs)]
    ds = [deltas(base, r) for r in runs]
    lines = ["| task | " + base.get("label", "base") + " | " + " | ".join(names) + " |",
             "|---|---|" + "---|" * len(runs)]
    for key in TIER1:
        cells = []
        for r, d in zip(runs, ds):
            v = headline(r, key)
            dv = d[key]
            if v is None:
                cells.append("-")
            elif key == "toxicchat":
                cells.append(f"{v:.3f} ({dv:+.3f})")
            else:
                cells.append(f"{100 * v:.1f} ({dv:+.1f})")
        lines.append(f"| {key} | {fmt(headline(base, key), key)} | " + " | ".join(cells) + " |")
    verdicts = [verdict(d) for d in ds]
    lines.append("| **mean delta** | | " + " | ".join("-" if v["mean"] is None else f"**{v['mean']:+.1f}**" for v in verdicts) + " |")
    lines.append("| **worst task** | | " + " | ".join("-" if v["worst"] is None else f"**{v['worst']:+.1f}**" for v in verdicts) + " |")
    lines.append("| **gate 2** | | " + " | ".join(("PASS" if v["passed"] else "FAIL") if v["n_tasks"] else "-" for v in verdicts) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--bootstrap", action="store_true", help="paired 95%% intervals per task from the item records")
    args = ap.parse_args()
    base = json.loads(Path(args.base).read_text())
    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    print(f"## Gate 2 against `{base.get('label', args.base)}`" + (" (full splits)" if base.get("full") else f" ({base.get('n')} items per task)"))
    print()
    print(table(base, runs))
    if args.bootstrap:
        for r in runs:
            print(f"\n### {r.get('label')}: paired 95% intervals of the accuracy delta\n")
            print("| task | delta | 95% interval | |\n|---|---|---|---|")
            bt = {t["task"]: t for t in base["tasks"]}
            for t in r["tasks"]:
                lo, hi = paired_intervals(base, {"tasks": [t]})[t["task"]]
                a = np.mean([int(np.argmax(i["logprobs"]) == i["gold"]) for i in bt[t["task"]]["items"]])
                s = np.mean([int(np.argmax(i["logprobs"]) == i["gold"]) for i in t["items"]])
                sig = "significant" if lo > 0 or hi < 0 else ""
                print(f"| {t['task']} | {100 * (s - a):+.1f} | [{lo:+.1f}, {hi:+.1f}] | {sig} |")


if __name__ == "__main__":
    main()
