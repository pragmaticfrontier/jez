"""Soft labels from an open-weight teacher, read through the same label read-out as serving.

    tools/gpu_guard.py --cuda-fraction 0.7 --timeout 86400 -- .venv/bin/python -m train.teacher_label \\
        --teacher Qwen/Qwen3.8-27B --data train/data/mix-v1/train.jsonl --licence-checked

Every example gets a ``teacher`` field: the teacher's probabilities over the question's answers
(option names, "true"/"false", or level indices), averaged over both option orders exactly as
Jez answers. The ``--licence-checked`` flag is the record that someone read the teacher's licence
and confirmed it permits training on its outputs; without it the script refuses to run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from jez.decider import Decider, _softmax
from jez.engine import load_backend
from jez.schema import validate_request

log = logging.getLogger("train.teacher")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", help="default: <data> with .teacher.jsonl")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--licence-checked", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.licence_checked:
        raise SystemExit("refusing to label: pass --licence-checked once the teacher's licence has been read "
                         "and permits training on its outputs")
    src = Path(args.data)
    out = Path(args.out) if args.out else src.with_suffix(".teacher.jsonl")
    rows = [json.loads(l) for l in src.read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    backend = load_backend(args.teacher, "auto", max_batch=args.max_batch)
    dec = Decider(backend, "teacher")
    done = 0
    with out.open("w") as f:
        for start in range(0, len(rows), args.batch):
            chunk = rows[start:start + args.batch]
            reqs = [validate_request({"state": e["state"], "model": "jez-small-latest", "questions": {"q": e["question"]}}) for e in chunk]
            for e, scored in zip(chunk, dec.score_requests(reqs)):
                targets, lp = scored["q"]
                p = _softmax(np.asarray(lp), 1.0)
                e["teacher"] = {t: round(float(x), 6) for t, x in zip(targets, p)}
                e["teacher_model"] = args.teacher
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            done += len(chunk)
            if done % (args.batch * 20) == 0:
                log.info("%d / %d", done, len(rows))
    log.info("wrote %s", out)
    if hasattr(backend, "release"):
        backend.release()


if __name__ == "__main__":
    main()
