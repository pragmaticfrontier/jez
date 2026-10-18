"""Accuracy, latency and calibration of the decision engine on public datasets.

    .venv/bin/python -m bench.eval --model Qwen/Qwen3-1.7B --n 300
    .venv/bin/python -m bench.eval --model Qwen/Qwen3-1.7B --n 300 --fit-temperature calibration.json

With --fit-temperature, examples are split in half (even/odd index): the
temperature per question type is fitted on the even half and all calibrated
metrics are reported on the odd half only.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sysone.calibration import (
    Calibration,
    expected_calibration_error,
    fit_temperature,
    per_example_stats,
    selective_table,
)
from sysone.decider import Decider
from sysone.engine import Scorer
from sysone.schema import DecideRequest

from .data import TASKS, load_rows

log = logging.getLogger("bench")


def run_task(decider: Decider, task, rows: list[dict], debias: bool) -> dict:
    keys_order: list[str] | None = None
    logits, gold_idx, pred_idx, lat, expected_vals, toks = [], [], [], [], [], []
    for row in rows:
        req = DecideRequest(state=task.state(row), questions={"q": task.question(row)},
                            options={"return_logits": True, "debias": debias, "temperature": 1.0})
        t = time.perf_counter()
        res = decider.decide(req)
        lat.append((time.perf_counter() - t) * 1000)
        toks.append(res.usage.input_tokens)
        ans = res.answers["q"]
        if keys_order is None:
            keys_order = list(ans.logits.keys())
        logits.append([ans.logits[k] for k in keys_order])
        gold_idx.append(keys_order.index(task.gold(row)))
        if ans.type == "choice":
            pred = ans.choice
        elif ans.type == "bool":
            pred = "true" if ans.value else "false"
        else:
            pred = str(ans.score)
            expected_vals.append(ans.expected)
        pred_idx.append(keys_order.index(pred))
    return {
        "task": task.name, "qtype": task.qtype, "keys": keys_order, "n": len(rows),
        "logits": np.array(logits, dtype=np.float64), "gold": np.array(gold_idx), "pred": np.array(pred_idx),
        "latency_ms": np.array(lat), "expected": np.array(expected_vals) if expected_vals else None,
        "tokens": np.array(toks),
    }


def metrics(logits, gold, temperature: float) -> dict:
    """Accuracy / calibration summary; `logits` may be a 2-D array or a list of ragged rows."""
    conf, correct, nlls = per_example_stats(logits, gold, temperature)
    return {
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct),
        "nll": float(nlls.mean()),
        "mean_confidence": float(conf.mean()),
        "selective": selective_table(conf, correct),
    }


def fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--n", type=int, default=300, help="examples per task")
    ap.add_argument("--debias", action="store_true", help="rotate options for choice questions")
    ap.add_argument("--fit-temperature", metavar="PATH", help="fit T per question type and write a calibration file")
    ap.add_argument("--out", metavar="PATH", help="write raw results JSON here")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scorer = Scorer(args.model)
    scorer.warmup()
    decider = Decider(scorer)

    results = []
    for name in args.tasks:
        task = TASKS[name]
        rows = load_rows(task, args.n)
        log.info("task %s: %d rows", name, len(rows))
        results.append(run_task(decider, task, rows, args.debias))

    # ---- raw (T = 1) report on all examples
    print(f"\n## {args.model}  (n={args.n}/task, debias={args.debias})\n")
    print("| task | type | n | accuracy | ECE | mean conf | latency p50 | p95 | tokens/ex |")
    print("|---|---|---|---|---|---|---|---|---|")
    summary = {}
    for r in results:
        m = metrics(r["logits"], r["gold"], 1.0)
        lat = r["latency_ms"]
        extra = ""
        if r["expected"] is not None:
            levels = np.array([int(k) for k in r["keys"]])
            gold_lv = levels[r["gold"]]
            mae_arg = float(np.abs(levels[r["pred"]] - gold_lv).mean())
            mae_exp = float(np.abs(r["expected"] - gold_lv).mean())
            within1 = float((np.abs(levels[r["pred"]] - gold_lv) <= 1).mean())
            extra = f"  (score task: MAE argmax {mae_arg:.2f}, MAE expected {mae_exp:.2f}, within±1 {fmt_pct(within1)})"
            m.update({"mae_argmax": mae_arg, "mae_expected": mae_exp, "within_1": within1})
        print(f"| {r['task']} | {r['qtype']} | {r['n']} | {fmt_pct(m['accuracy'])} | {m['ece']:.3f} | {m['mean_confidence']:.3f} "
              f"| {statistics.median(lat):.0f} ms | {np.percentile(lat, 95):.0f} ms | {r['tokens'].mean():.0f} |{extra}")
        summary[r["task"]] = {"raw": m, "latency_p50_ms": float(statistics.median(lat)), "latency_p95_ms": float(np.percentile(lat, 95)),
                              "tokens_mean": float(r["tokens"].mean())}
        if r["qtype"] == "score":
            levels = [int(k) for k in r["keys"]]
            cm = np.zeros((len(levels), len(levels)), dtype=int)
            for g, pr in zip(r["gold"], r["pred"]):
                cm[g, pr] += 1
            print(f"\nConfusion for {r['task']} (rows = gold level, cols = predicted level {levels}):")
            for lv, row in zip(levels, cm):
                print(f"  {lv}: " + " ".join(f"{v:4d}" for v in row))
            summary[r["task"]]["confusion"] = cm.tolist()

    # ---- temperature scaling: fit on even indices, report on odd indices
    calib = Calibration(model=args.model)
    if args.fit_temperature:
        print("\n### Temperature scaling (fit on even half, evaluated on odd half)\n")
        print("| type | tasks | T | ECE before | ECE after | NLL before | NLL after | acc (odd half) |")
        print("|---|---|---|---|---|---|---|---|")
        fitted = []
        for qtype in ("choice", "bool", "score"):
            group = [r for r in results if r["qtype"] == qtype]
            if not group:
                continue
            fit_logits = [row for r in group for row in r["logits"][0::2]]
            fit_gold = np.concatenate([r["gold"][0::2] for r in group])
            T = fit_temperature(fit_logits, fit_gold)
            calib.temperature[qtype] = round(T, 4)
            ev_logits = [row for r in group for row in r["logits"][1::2]]
            ev_gold = np.concatenate([r["gold"][1::2] for r in group])
            before, after = metrics(ev_logits, ev_gold, 1.0), metrics(ev_logits, ev_gold, T)
            print(f"| {qtype} | {'+'.join(r['task'] for r in group)} | {T:.2f} | {before['ece']:.3f} | {after['ece']:.3f} "
                  f"| {before['nll']:.3f} | {after['nll']:.3f} | {fmt_pct(after['accuracy'])} |")
            summary[f"calibrated_{qtype}"] = {"T": T, "before": before, "after": after}
            fitted.append((qtype, after))
        for qtype, after in fitted:
            print(f"\nSelective prediction after scaling ({qtype}, odd half): act only when confidence >= threshold\n")
            print("| threshold | coverage | accuracy |\n|---|---|---|")
            for row in after["selective"]:
                print(f"| {row['threshold']:.2f} | {fmt_pct(row['coverage'])} | {fmt_pct(row['accuracy'])} |")
        print()
        calib.save(args.fit_temperature)
        print(f"wrote {args.fit_temperature}: {calib.temperature}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "n": args.n, "debias": args.debias, "summary": summary,
                                              "temperature": calib.temperature}, indent=2, default=float))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
