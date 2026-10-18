"""Re-report an evaluation run with different temperatures, without touching a GPU.

    python -m eval.recalibrate_run runs/<label>.json --fit-even --save-calibration releases/<id>/calibration.json \\
        --model-id <id> --out runs/<label>-tier1cal

``--fit-even`` fits one temperature per bucket on the even-indexed items of every task in the run (the protocol
``eval.run_eval`` follows when no calibration file is given) and reports every calibrated metric on the odd-indexed
items; ``--calibration <file>`` reports with an existing file instead. The raw metrics and the items are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jez.calibration import Calibration, Record, fit

from .run_eval import build_report
from .tasks import TASKS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="runs/<label>.json written by eval.run_eval")
    ap.add_argument("--fit-even", action="store_true", help="fit temperatures on the even-indexed items")
    ap.add_argument("--calibration", help="report with this calibration.json instead")
    ap.add_argument("--save-calibration", help="write the temperatures used here")
    ap.add_argument("--model-id", default=None, help="model field of the saved calibration")
    ap.add_argument("--out", required=True, help="runs/<new label> (.json and .md are written)")
    args = ap.parse_args()
    if bool(args.fit_even) == bool(args.calibration):
        raise SystemExit("choose exactly one of --fit-even and --calibration")
    run = json.loads(Path(args.run).read_text())
    results = run["tasks"]
    if args.fit_even:
        records = [Record(r["qtype"], np.asarray(it["logprobs"]), it["gold"]) for r in results for it in r["items"][0::2]]
        calib = fit(records, model=args.model_id or run.get("model_id"))
        how = "temperatures fitted on the even-indexed items of this run"
    else:
        calib = Calibration.load(args.calibration)
        how = f"temperatures from `{args.calibration}`"
    if args.save_calibration:
        calib.save(args.save_calibration)
    label = Path(args.out).name
    n = run.get("n")
    header = (f"checkpoint `{run.get('checkpoint')}` · backend {run.get('backend')} on {run.get('device')} · "
              f"{'full splits' if run.get('full') else f'{n} items per task'} · re-reported from `{Path(args.run).name}` with {how}")
    summary, lines = build_report(results, calib, TASKS, label, header)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.json").write_text(json.dumps({**run, "label": label, "summary": summary, "temperature": calib.temperature,
                                               "recalibrated_from": Path(args.run).name, "calibration_fit": how, "tasks": results}, indent=1))
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    pooled = summary["pooled_ece"]
    print(json.dumps({"temperature": calib.temperature,
                      "pooled_ece_odd_half": {q: round(v["calibrated_odd_half"]["ece"], 4) for q, v in pooled.items() if v["raw"]["n"]},
                      "wrote": [f"{out}.json", f"{out}.md"] + ([args.save_calibration] if args.save_calibration else [])}, indent=1))


if __name__ == "__main__":
    main()
