#!/usr/bin/env python3
"""Check the torch backend on real checkpoints, one model at a time, under the memory guard.

For every probe of a reference request it compares three ways of reading the label log-probabilities:

    single     each prompt alone, no padding, no cache          (the reference)
    unshared   all prompts in one left-padded batch
    shared     the common prefix once, suffixes batched against the replicated cache

and prints the worst absolute difference, plus latency and determinism. Run it through tools/gpu_guard.py:

    tools/gpu_guard.py --timeout 1200 -- .venv/bin/python tools/engine_check.py Qwen/Qwen3.5-0.8B:bf16,fp32 google/gemma-4-E4B-it:bf16
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jez.decider import Decider  # noqa: E402
from jez.engine.torch_backend import TorchBackend, common_prefix_len  # noqa: E402
from jez.prompt import plan  # noqa: E402
from jez.schema import validate_request  # noqa: E402

REQUEST = {
    "state": {"claim_id": "C-88213", "amount_usd": 4200, "policy_age_days": 11, "prior_claims": 3,
              "description": "Phone screen cracked when it fell off the kitchen counter. Also lost my laptop on the "
                             "same day, unrelated incident. No receipts, bought both for cash."},
    "model": "jez-latest",
    "questions": {
        "likely_fraud": {"type": "noul", "instructions": "Does this claim show strong indicators of fraud?"},
        "claim_type": {"type": "choice", "instructions": "Which claim type is this?",
                       "criteria": {"accidental_damage": "Item broken by accident", "theft": "Item stolen",
                                    "loss": "Item lost or misplaced", "liability": "Damage to a third party"}},
        "severity": {"type": "score", "instructions": "How severe is this claim for the insurer?",
                     "criteria": ["negligible", "minor", "moderate", "major", "catastrophic"]},
        "above_1000": {"type": "noul", "instructions": "Is the amount above 1000 USD?"},
        "many": {"type": "choice", "instructions": "Pick the colour named in `description`, if any.",
                 "criteria": {f"opt{i}": f"colour number {i}" for i in range(30)}},
    },
}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


def check(checkpoint: str, dtype: torch.dtype) -> dict:
    req = validate_request(REQUEST)
    be = TorchBackend(checkpoint, dtype=dtype)
    try:
        probes = plan(req, be.table)
        prompts = [be.render(p.messages) for p in probes]
        dec = Decider(be, "check")
        labels = [dec._labels_for(p) for p in probes]
        ids = [be.encode(p) for p in prompts]
        be.share_state = False
        single = [be.score([p], [l])[0] for p, l in zip(prompts, labels)]
        t0 = time.perf_counter()
        unshared = be.score(prompts, labels)
        t_un = time.perf_counter() - t0
        be.share_state = True
        t0 = time.perf_counter()
        shared = be.score(prompts, labels)
        t_sh = time.perf_counter() - t0
        shared2 = be.score(prompts, labels)
        d_un = [float(np.abs(a - b).max()) for a, b in zip(single, unshared)]
        d_sh = [float(np.abs(a - b).max()) for a, b in zip(single, shared)]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            decision = dec.decide(req)
            ts.append((time.perf_counter() - t0) * 1000)
        result = {
            "checkpoint": checkpoint, "dtype": str(dtype).replace("torch.", ""), "family": be.family,
            "prefix_tokens": common_prefix_len(ids), "prompt_tokens": [len(i) for i in ids],
            "worst_unshared_vs_single": max(d_un), "worst_shared_vs_single": max(d_sh),
            "per_probe_unshared": [round(d, 4) for d in d_un], "per_probe_shared": [round(d, 4) for d in d_sh],
            "ms_unshared": round(t_un * 1000), "ms_shared": round(t_sh * 1000),
            "deterministic": all(np.array_equal(a, b) for a, b in zip(shared, shared2)),
            "decide_ms": [round(t) for t in ts],
            "answers": {k: v.get("choice", v.get("noul", v.get("score"))) for k, v in decision.answers.items()},
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
        }
        dec = None
        return result
    finally:
        be.release()
        del be
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def main() -> None:
    results = []
    for spec in sys.argv[1:]:
        checkpoint, _, dtypes = spec.partition(":")
        for name in (dtypes or "bf16").split(","):
            r = check(checkpoint, DTYPES[name])
            results.append(r)
            print(f"\n== {r['checkpoint']} {r['dtype']} ({r['family']}): prefix {r['prefix_tokens']} tokens; "
                  f"peak {r['peak_gb']} GB", flush=True)
            print(f"   worst |dlogp|  unshared-vs-single {r['worst_unshared_vs_single']:.4f}   "
                  f"shared-vs-single {r['worst_shared_vs_single']:.4f}")
            print(f"   per probe unshared {r['per_probe_unshared']}")
            print(f"   per probe shared   {r['per_probe_shared']}")
            print(f"   batch ms unshared {r['ms_unshared']} shared {r['ms_shared']} | decide() ms {r['decide_ms']} "
                  f"| deterministic {r['deterministic']}")
            print(f"   answers {json.dumps(r['answers'])}", flush=True)
    out = Path("runs") / "engine_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
