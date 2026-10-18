"""Score a checkpoint on the benchmark tasks and report accuracy, ECE, Brier and latency.

    tools/gpu_guard.py --timeout 7200 -- .venv/bin/python -m eval.run_eval \\
        --checkpoint google/gemma-4-E4B-it --label gemma-4-E4B-it-zeroshot --tasks tier1 --n 1000

Items are sampled evenly across the split (``--n``) or taken in full (``--full``). Every
item is one Jev request with one question, planned and scored exactly as the server
does it (both option orders, averaged). Temperatures are fitted per bucket on the
even-indexed items and the calibrated metrics are reported on the odd-indexed items,
so no item calibrates itself. Latency is measured separately, one request at a time
through the decider, on the first ``--latency-items`` items of each task.

Outputs ``runs/<label>.json`` (everything, including per-item log-probabilities) and
``runs/<label>.md`` (the tables).
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

from jez.calibration import Calibration, Record, bucket, fit
from jez.decider import Decider, _softmax
from jez.engine import load_backend
from jez.schema import validate_request
from sysone.calibration import expected_calibration_error

from .tasks import Task, expand

log = logging.getLogger("eval")


def confidence_and_correct(qtype: str, probs: np.ndarray, gold: int) -> tuple[float, float, float]:
    """(confidence, correct, brier) for one item under a probability vector in target order."""
    pred = int(np.argmax(probs))
    onehot = np.zeros_like(probs)
    onehot[gold] = 1.0
    brier = float(((probs - onehot) ** 2).sum())
    return float(probs.max()), float(pred == gold), brier


def f1_positive(preds: list[int], golds: list[int], positive: int) -> float:
    tp = sum(p == positive and g == positive for p, g in zip(preds, golds))
    fp = sum(p == positive and g != positive for p, g in zip(preds, golds))
    fn = sum(p != positive and g == positive for p, g in zip(preds, golds))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def task_metrics(task: Task, items: list[dict], calib: Calibration | None, subset: slice = slice(None)) -> dict:
    rows = items[subset]
    if not rows:
        return {"n": 0}
    conf, corr, briers, preds, golds = [], [], [], [], []
    for it in rows:
        lp = np.asarray(it["logprobs"])
        T = calib.temperature_for(task.qtype, len(lp)) if calib else 1.0
        p = _softmax(lp, T)
        c, ok, b = confidence_and_correct(task.qtype, p, it["gold"])
        conf.append(c); corr.append(ok); briers.append(b)
        preds.append(int(np.argmax(p))); golds.append(it["gold"])
    out = {"n": len(rows), "accuracy": float(np.mean(corr)), "ece": expected_calibration_error(np.array(conf), np.array(corr)),
           "brier": float(np.mean(briers)), "mean_confidence": float(np.mean(conf))}
    if task.metric == "f1_positive":
        pos = rows[0]["targets"].index("true")
        out["f1_positive"] = f1_positive(preds, golds, pos)
        out["headline"] = out["f1_positive"]
    else:
        out["headline"] = out["accuracy"]
    if task.qtype == "score":
        exp = [float(sum(i * p for i, p in enumerate(_softmax(np.asarray(it["logprobs"]), 1.0)))) for it in rows]
        out["mae_expected"] = float(np.mean([abs(e - it["gold"]) for e, it in zip(exp, rows)]))
    return out


def run_task(decider: Decider, task: Task, n: int | None, batch: int, latency_items: int) -> dict:
    rows = task.rows(n)
    examples = [task.build(r) for r in rows]
    reqs = [validate_request({"state": s, "model": decider.model_id, "questions": {"q": q}}) for s, q, _ in examples]
    items: list[dict] = []
    t0 = time.perf_counter()
    for start in range(0, len(reqs), batch):
        chunk = reqs[start:start + batch]
        scored = decider.score_requests(chunk)
        for (s, q, gold), sc in zip(examples[start:start + batch], scored):
            targets, lp = sc["q"]
            items.append({"targets": list(targets), "logprobs": [round(float(x), 6) for x in lp], "gold": targets.index(gold)})
    throughput_ms = (time.perf_counter() - t0) * 1000 / max(1, len(reqs))
    lat = []
    for req in reqs[:latency_items]:
        t1 = time.perf_counter()
        decider.decide(req)
        lat.append((time.perf_counter() - t1) * 1000)
    lat.sort()
    tokens = [decider.check_limits(r) for r in reqs[:latency_items]]
    return {"task": task.name, "qtype": task.qtype, "metric": task.metric, "tier": task.tier, "group": task.group,
            "language": task.language, "n": len(items), "items": items,
            "latency_p50_ms": statistics.median(lat) if lat else None,
            "latency_p99_ms": lat[min(len(lat) - 1, int(round(0.99 * (len(lat) - 1))))] if lat else None,
            "ms_per_item_batched": throughput_ms, "tokens_mean": float(np.mean(tokens)) if tokens else None}


def pooled_ece(results: list[dict], qtype: str, calib: Calibration | None, subset: slice = slice(None)) -> dict:
    """Tier 3: ECE pooled over every answer of one question type across tasks (raw, or calibrated on a subset)."""
    conf, corr = [], []
    for r in results:
        if r["qtype"] != qtype:
            continue
        for it in r["items"][subset]:
            lp = np.asarray(it["logprobs"])
            p = _softmax(lp, calib.temperature_for(qtype, len(lp)) if calib else 1.0)
            conf.append(float(p.max()))
            corr.append(float(int(np.argmax(p)) == it["gold"]))
    if not conf:
        return {"n": 0, "ece": float("nan")}
    return {"n": len(conf), "ece": expected_calibration_error(np.array(conf), np.array(corr))}


def fmt(x, pct=True) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True, help="name of the run; files go to runs/<label>.*")
    ap.add_argument("--model-id", default="jez-small-1.0.0")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--tasks", nargs="+", default=["tier1"])
    ap.add_argument("--n", type=int, default=1000, help="items per task (development runs)")
    ap.add_argument("--full", action="store_true", help="full splits (release runs)")
    ap.add_argument("--batch", type=int, default=64, help="requests per scoring call")
    ap.add_argument("--max-batch", type=int, default=16, help="probes per forward pass")
    ap.add_argument("--latency-items", type=int, default=50)
    ap.add_argument("--calibration", help="an existing calibration.json to report with (instead of fitting)")
    ap.add_argument("--save-calibration", help="write the fitted temperatures here")
    ap.add_argument("--no-debias", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    backend = load_backend(args.checkpoint, args.backend, max_batch=args.max_batch)
    if hasattr(backend, "warmup"):
        backend.warmup()
    decider = Decider(backend, args.model_id, calibration=Calibration(), debias=not args.no_debias)
    tasks = expand(args.tasks)
    results = []
    for task in tasks:
        try:
            r = run_task(decider, task, None if args.full else args.n, args.batch, args.latency_items)
        except FileNotFoundError as e:
            log.warning("skipping %s: %s", task.name, e)
            continue
        results.append(r)
        raw = task_metrics(task, r["items"], None)
        log.info("%s: n=%d %s=%s ece=%.3f p50=%.0fms", task.name, raw["n"], task.metric, fmt(raw["headline"]), raw["ece"],
                 r["latency_p50_ms"] or 0)

    # temperatures: fit on even items, report on odd items
    if args.calibration:
        calib = Calibration.load(args.calibration)
    else:
        records = [Record(r["qtype"], np.asarray(it["logprobs"]), it["gold"]) for r in results for it in r["items"][0::2]]
        calib = fit(records, model=args.checkpoint)
    if args.save_calibration:
        calib.save(args.save_calibration)

    by_name = {t.name: t for t in tasks}
    lines = [f"# {args.label}", "", f"checkpoint `{args.checkpoint}` · backend {backend.name} on {backend.device} · "
             f"{'full splits' if args.full else f'{args.n} items per task'} · debias {not args.no_debias}", "",
             "| task | type | n | " + "accuracy/metric" + " | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for r in results:
        t = by_name[r["task"]]
        raw = task_metrics(t, r["items"], None)
        cal = task_metrics(t, r["items"], calib, slice(1, None, 2))
        summary[r["task"]] = {"raw": raw, "calibrated_odd_half": cal, "latency_p50_ms": r["latency_p50_ms"],
                              "latency_p99_ms": r["latency_p99_ms"], "ms_per_item_batched": r["ms_per_item_batched"],
                              "tokens_mean": r["tokens_mean"], "n": r["n"], "qtype": r["qtype"], "metric": r["metric"],
                              "group": r["group"], "tier": r["tier"]}
        lines.append(f"| {r['task']} | {r['qtype']} | {r['n']} | {fmt(raw['headline'])} | {raw['ece']:.3f} | {cal.get('ece', float('nan')):.3f} "
                     f"| {raw['brier']:.3f} | {r['latency_p50_ms']:.0f} | {r['latency_p99_ms']:.0f} | {r['tokens_mean']:.0f} |")
    groups = {}
    for name, s in summary.items():
        if s["group"]:
            groups.setdefault(s["group"], []).append(s["raw"]["headline"])
    for g, vals in groups.items():
        summary[g] = {"raw": {"headline": float(np.mean(vals)), "accuracy": float(np.mean(vals))}, "n_tasks": len(vals), "group_mean": True}
        lines.append(f"| **{g} (mean of {len(vals)})** | | | {fmt(float(np.mean(vals)))} | | | | | | |")
    tier1 = [s["raw"]["headline"] for n, s in summary.items() if s.get("tier") == 1 and not s.get("group")]
    tier1 += [s["raw"]["headline"] for n, s in summary.items() if s.get("group_mean")]
    if tier1:
        lines.append(f"| **Tier 1 mean** | | | {fmt(float(np.mean(tier1)))} | | | | | | |")
    pooled = {q: {"raw": pooled_ece(results, q, None), "calibrated_odd_half": pooled_ece(results, q, calib, slice(1, None, 2))}
              for q in ("choice", "noul", "score")}
    summary["pooled_ece"] = pooled
    lines += ["", "## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)", "", "| type | answers | ECE raw | ECE calibrated (odd half) |", "|---|---|---|---|"]
    lines += [f"| {q} | {v['raw']['n']} | {v['raw']['ece']:.3f} | {v['calibrated_odd_half']['ece']:.3f} |" for q, v in pooled.items() if v["raw"]["n"]]
    lines += ["", "## Temperatures (fitted on even-indexed items)", "", "| bucket | T |", "|---|---|"]
    lines += [f"| {b} | {T:.3f} |" for b, T in calib.temperature.items()]
    out = Path("runs") / args.label
    out.parent.mkdir(exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps({"label": args.label, "checkpoint": args.checkpoint, "model_id": args.model_id,
                                               "backend": backend.name, "device": backend.device, "n": None if args.full else args.n,
                                               "full": args.full, "summary": summary, "temperature": calib.temperature,
                                               "tasks": results}, indent=1))
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}.json and {out}.md")
    if hasattr(backend, "release"):
        backend.release()


if __name__ == "__main__":
    sys.exit(main())
