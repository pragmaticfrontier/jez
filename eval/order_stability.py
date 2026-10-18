"""Order stability: does the chosen option survive every permutation of the option order?

    tools/gpu_guard.py -- .venv/bin/python -m eval.order_stability --checkpoint ... --task withdrawal_routing

For each item the question is answered under every permutation of its options (all 6 for three
options; 6 random permutations when there are more). Reports the share of items whose top choice
is the same under all orders (the Tier 3 gate is 95%) and the largest probability movement.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from jez.calibration import Calibration
from jez.decider import Decider
from jez.engine import load_backend
from jez.schema import validate_request

from .tasks import TASKS


def permutations_of(names: list[str], limit: int = 6, seed: int = 0) -> list[list[str]]:
    if len(names) <= 3:
        return [list(p) for p in itertools.permutations(names)]
    rng = random.Random(seed)
    seen = {tuple(names)}
    out = [list(names)]
    while len(out) < limit:
        p = names[:]
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--task", default="withdrawal_routing")
    ap.add_argument("--n", type=int)
    ap.add_argument("--calibration")
    ap.add_argument("--out")
    args = ap.parse_args()
    task = TASKS[args.task]
    rows = task.rows(args.n)
    backend = load_backend(args.checkpoint, args.backend)
    dec = Decider(backend, "order-stability", calibration=Calibration.load(args.calibration))
    stable = 0
    max_move = 0.0
    per_item = []
    for r in rows:
        state, q, gold = task.build(r)
        if q["type"] != "choice":
            raise SystemExit("order stability is defined for choice questions")
        names = list(q["criteria"])
        choices, moves = [], {}
        for order in permutations_of(names):
            qq = dict(q, criteria={k: q["criteria"][k] for k in order})
            req = validate_request({"state": state, "model": "jez-small-latest", "questions": {"q": qq}})
            ans = dec.decide(req).answers["q"]
            choices.append(ans["choice"])
            for k, p in ans["probabilities"].items():
                moves.setdefault(k, []).append(p)
        item_move = max(max(v) - min(v) for v in moves.values())
        max_move = max(max_move, item_move)
        same = len(set(choices)) == 1
        stable += same
        per_item.append({"stable": same, "choices": choices, "max_move": item_move, "gold": gold})
    report = {"task": args.task, "checkpoint": args.checkpoint, "n": len(rows), "stable_share": stable / max(1, len(rows)),
              "max_probability_move": max_move, "gate_95": stable / max(1, len(rows)) >= 0.95, "items": per_item}
    print(json.dumps({k: v for k, v in report.items() if k != "items"}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    if hasattr(backend, "release"):
        backend.release()


if __name__ == "__main__":
    main()
