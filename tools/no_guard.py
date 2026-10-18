#!/usr/bin/env python3
"""Stand-in for tools/gpu_guard.py on machines that need no guard, such as a rented cloud GPU.

Accepts the guard's flags, ignores all but --cuda-fraction, and replaces itself with the command after ``--`` so the
pipeline scripts run unchanged:

    GUARD="python3 tools/no_guard.py" train/pipeline_small.sh /data/mix-v1 /releases/jez-small-1.0.0
"""
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cuda-fraction", type=float, default=0.9)
    args, rest = ap.parse_known_args()
    if "--" not in rest:
        sys.exit("no_guard: expected -- followed by the command to run")
    cmd = rest[rest.index("--") + 1:]
    os.environ.setdefault("JEZ_CUDA_MEMORY_FRACTION", str(args.cuda_fraction))
    os.environ.setdefault("JEZ_GPU_GUARD", "none")
    print(f"no_guard: running {' '.join(cmd)}", file=sys.stderr, flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
