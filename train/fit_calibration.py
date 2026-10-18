"""Fit the release temperatures on the calibration split and write them into the release directory.

    tools/gpu_guard.py -- .venv/bin/python -m train.fit_calibration --checkpoint releases/jez-small-1.0.0/weights \\
        --data train/data/mix-v1/calibration.jsonl --out releases/jez-small-1.0.0/calibration.json

One temperature per question type and option-count bucket (see jez.calibration). The split is the
2% ``calibration.jsonl`` that ``train.build_mix`` sets aside: it never overlaps the training, validation
or any test split. Buckets with fewer than ``--min-examples`` items keep T = 1.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from jez.calibration import Record, ece, fit
from jez.decider import Decider
from jez.engine import load_backend
from jez.schema import validate_request

log = logging.getLogger("train.calibration")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="calibration.jsonl")
    ap.add_argument("--out", required=True, help="calibration.json inside the release directory")
    ap.add_argument("--model-id", default="jez-small-1.0.0")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-examples", type=int, default=50)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = [json.loads(l) for l in Path(args.data).read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    backend = load_backend(args.checkpoint, args.backend)
    dec = Decider(backend, args.model_id)
    records: list[Record] = []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        reqs = [validate_request({"state": e["state"], "model": args.model_id, "questions": {"q": e["question"]}}) for e in chunk]
        for e, scored in zip(chunk, dec.score_requests(reqs)):
            targets, lp = scored["q"]
            if str(e["gold"]) not in targets:
                continue
            records.append(Record(e["question"]["type"], np.asarray(lp), targets.index(str(e["gold"]))))
    calib = fit(records, model=args.model_id, min_examples=args.min_examples)
    before, after = ece(records, None), ece(records, calib)
    calib.save(args.out)
    counts = {}
    for r in records:
        from jez.calibration import bucket

        counts[bucket(r.qtype, len(r.logprobs))] = counts.get(bucket(r.qtype, len(r.logprobs)), 0) + 1
    print(json.dumps({"n": len(records), "per_bucket": counts, "temperature": calib.temperature,
                      "ece_before": round(before, 4), "ece_after": round(after, 4), "wrote": args.out}, indent=1))
    if hasattr(backend, "release"):
        backend.release()


if __name__ == "__main__":
    main()
