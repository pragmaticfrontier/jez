"""Cross-backend agreement on the conformance requests.

    # on each machine (CUDA box, Apple Silicon):
    python -m eval.backend_agreement dump --checkpoint Qwen/Qwen3.5-0.8B --backend torch --dtype fp32 --out runs/agree-torch.json
    python -m eval.backend_agreement dump --checkpoint Qwen/Qwen3.5-0.8B --backend mlx --out runs/agree-mlx.json
    # then anywhere:
    python -m eval.backend_agreement compare runs/agree-torch.json runs/agree-mlx.json --tolerance 0.02

Dumps the restricted log-probabilities of every probe of every conformance example plus
examples/claim.json. In fp32 the torch shared, unshared and single paths agree to 2e-4 on
both bases; in bf16 the accumulation noise alone is 0.1 to 0.2 (Qwen3.5-0.8B) and up to a
few nats on 30-option questions (Gemma 4 E4B), so a 0.02 tolerance is meaningful only for
fp32 dumps. Compare bf16 dumps with a looser tolerance and read the per-probe table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jez.decider import Decider
from jez.engine import load_backend
from jez.prompt import plan
from jez.schema import validate_request

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "tests" / "conformance" / "examples").glob("*.json")) + [ROOT / "examples" / "claim.json"]


def dump(args) -> None:
    kwargs = {}
    if args.dtype and args.backend == "torch":
        import torch

        kwargs["dtype"] = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    backend = load_backend(args.checkpoint, args.backend, **kwargs)
    dec = Decider(backend, "agree")
    out = {"checkpoint": args.checkpoint, "backend": backend.name, "device": backend.device, "dtype": args.dtype, "probes": []}
    for path in EXAMPLES:
        body = json.loads(path.read_text())
        body["model"] = "jez-small-latest"
        req = validate_request(body)
        probes = plan(req, backend.table)
        prompts = [backend.render(p.messages) for p in probes]
        labels = [dec._labels_for(p) for p in probes]
        for p, lp in zip(probes, backend.score(prompts, labels)):
            out["probes"].append({"example": path.name, "question": p.question_id, "order": p.order,
                                  "targets": list(p.targets), "logprobs": [float(x) for x in lp]})
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}: {len(out['probes'])} probes from {len(EXAMPLES)} requests")
    if hasattr(backend, "release"):
        backend.release()


def compare(args) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in args.dumps)
    assert len(a["probes"]) == len(b["probes"]), "the dumps cover different probes"
    worst = 0.0
    rows = []
    for pa, pb in zip(a["probes"], b["probes"]):
        assert pa["targets"] == pb["targets"], (pa["example"], pa["question"])
        d = float(np.abs(np.array(pa["logprobs"]) - np.array(pb["logprobs"])).max())
        worst = max(worst, d)
        rows.append((pa["example"], pa["question"], pa["order"], d))
    print(f"{a['backend']} ({a.get('dtype')}) vs {b['backend']} ({b.get('dtype')}) on {a['checkpoint']}")
    print("| example | question | order | max |dlogp| |\n|---|---|---|---|")
    for ex, q, o, d in rows:
        print(f"| {ex} | {q} | {o} | {d:.4f} |")
    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(f"\nworst {worst:.4f} vs tolerance {args.tolerance}: {verdict}")
    raise SystemExit(0 if verdict == "PASS" else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--backend", default="auto")
    d.add_argument("--dtype", choices=["bf16", "fp32", "fp16"])
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("dumps", nargs=2)
    c.add_argument("--tolerance", type=float, default=0.02)
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
