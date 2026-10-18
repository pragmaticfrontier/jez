#!/usr/bin/env python3
"""Latency of the reference request (examples/claim.json) through the decider, one request at a time.

    tools/gpu_guard.py -- .venv/bin/python tools/latency.py Qwen/Qwen3.5-0.8B --runs 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jez.decider import Decider  # noqa: E402
from jez.engine import load_backend  # noqa: E402
from jez.schema import validate_request  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--request", default=str(Path(__file__).resolve().parent.parent / "examples" / "claim.json"))
    ap.add_argument("--no-share", action="store_true", help="disable state-cache sharing")
    args = ap.parse_args()
    body = json.loads(Path(args.request).read_text())
    req = validate_request(body)
    be = load_backend(args.checkpoint, args.backend)
    if args.no_share:
        be.share_state = False
    dec = Decider(be, "latency")
    for _ in range(3):
        dec.decide(req)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        d = dec.decide(req)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50 = statistics.median(times)
    p99 = times[min(len(times) - 1, int(round(0.99 * (len(times) - 1))))]
    print(json.dumps({
        "checkpoint": args.checkpoint, "backend": be.name, "device": be.device, "share_state": getattr(be, "share_state", None),
        "runs": args.runs, "p50_ms": round(p50, 1), "p99_ms": round(p99, 1), "min_ms": round(times[0], 1),
        "input_tokens": d.input_tokens, "probes": d.probes,
        "answers": {k: v.get("choice", v.get("noul", v.get("score"))) for k, v in d.answers.items()},
    }, indent=1))
    if hasattr(be, "release"):
        be.release()


if __name__ == "__main__":
    main()
