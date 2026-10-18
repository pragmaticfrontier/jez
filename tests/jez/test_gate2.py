"""eval.gate2 reads Tier 1 deltas off run summaries and applies the gate."""
from __future__ import annotations

from eval.gate2 import deltas, paired_intervals, table, verdict
from train.model_card import TIER1


def run(label, values, f1=0.5, items=None):
    summary = {k: {"raw": {"headline": v}} for k, v in values.items()}
    summary["toxicchat"] = {"raw": {"headline": f1}}
    return {"label": label, "summary": summary, "tasks": items or []}


BASE = {k: 0.60 for k in TIER1 if k != "toxicchat"}


def test_verdict_needs_mean_plus_three_and_no_task_below_minus_one():
    good = dict(BASE, **{k: 0.64 for k in BASE})
    assert verdict(deltas(run("b", BASE), run("g", good)))["passed"]
    one_drop = dict(good, boolq=0.585)  # -1.5 on one task sinks it despite a big mean
    v = verdict(deltas(run("b", BASE), run("d", one_drop)))
    assert not v["passed"] and round(v["worst"], 1) == -1.5
    small_mean = dict(BASE, **{k: 0.62 for k in BASE})  # +2.0 mean, nothing below -1
    assert not verdict(deltas(run("b", BASE), run("s", small_mean)))["passed"]


def test_f1_task_is_reported_but_not_averaged():
    d = deltas(run("b", BASE, f1=0.40), run("r", BASE, f1=0.55))
    assert round(d["toxicchat"], 2) == 0.15
    v = verdict(d)
    assert v["mean"] == 0 and v["n_tasks"] == 12


def test_table_lists_every_run_with_its_verdict():
    text = table(run("base", BASE), [run("a", dict(BASE, sst5=0.70)), run("b", BASE)])
    assert "| sst5 | 60.0 | 70.0 (+10.0) | 60.0 (+0.0) |" in text
    assert text.splitlines()[-1].startswith("| **gate 2** | | FAIL | FAIL |")


def test_paired_intervals_cover_a_true_difference():
    items_base = [{"logprobs": [0.0, -1.0], "gold": 0} for _ in range(100)]  # base always right
    items_run = [{"logprobs": [0.0, -1.0] if i % 2 else [-1.0, 0.0], "gold": 0} for i in range(100)]  # run right half the time
    lo, hi = paired_intervals({"tasks": [{"task": "x", "items": items_base}]}, {"tasks": [{"task": "x", "items": items_run}]})["x"]
    assert lo < -50 < hi and hi < 0
