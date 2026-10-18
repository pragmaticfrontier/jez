"""Set Jez runs beside the published figures in research/decision-model-benchmarks.csv.

    .venv/bin/python -m eval.compare runs/gemma-4-E4B-it-zeroshot.json runs/jez-large-1.0.0.json

Prints the Tier 1 table (Jev, Microsoft-Decision-1, Decisions API, Laya, Gemma 4 E4B zero-shot,
the Jez targets) with one column per run, then the Tier 2 and Tier 3 rows where a run reports them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "research" / "decision-model-benchmarks.csv"

TIER1_ROWS = [  # (csv dataset key, run summary key, display)
    ("sst2", "sst2", "SST-2"), ("imdb", "imdb", "IMDB"), ("hellaswag", "hellaswag", "HellaSwag"),
    ("arc_easy", "arc_easy", "ARC-Easy"), ("arc_challenge", "arc_challenge", "ARC-Challenge"), ("boolq", "boolq", "BoolQ"),
    ("belebele", "belebele", "Belebele, 12 languages"), ("agnews", "agnews", "AG News"), ("emotion", "emotion", "DAIR Emotion"),
    ("banking77", "banking77", "BANKING77"), ("toxicchat", "toxicchat", "ToxicChat (F1 toxic)"),
    ("prompt_injection", "prompt_injection", "Prompt injections (deepset)"), ("sst5", "sst5", "SST-5"),
]
COLUMNS = ["jev", "microsoft-decision-1", "decisions-api", "laya", "gemma-4-E4B-it-zeroshot", "target:jez-large", "target:jez-small"]
HEADERS = ["Jev", "Microsoft-Decision-1", "Decisions API", "Laya", "Gemma 4 E4B zero-shot", "Jez-large target", "Jez-small target"]


def load_csv() -> dict[tuple[str, str], dict]:
    table = {}
    with CSV.open() as f:
        for row in csv.DictReader(f):
            table[(row["dataset"], row["model"])] = row
    return table


_QUIET_NOTES = {"paper prompts", "half the gap to Jev", "pf-jev Qwen3.5-2B figure", "Laya figure", "Laya held-out figure"}


def fmt_published(row: dict | None) -> str:
    """A published cell: percentages for accuracy, three decimals for F1, text for non-numeric targets."""
    if row is None:
        return "not tested"
    v = row["value"]
    note = row.get("note", "")
    suffix = f" ({note})" if note and note not in _QUIET_NOTES and len(note) <= 40 else ""
    try:
        x = float(v)
    except ValueError:
        return v
    if row.get("metric", "").startswith("f1") or row.get("metric") == "ece":
        return f"{x:.3f}{suffix}"
    if row.get("metric") == "ms":
        return f"{x:.0f} ms{suffix}"
    return f"{x:.1f}%{suffix}"


def fmt_run(summary: dict, key: str, metric: str) -> str:
    s = summary.get(key)
    if not s:
        return "-"
    v = s["raw"]["headline"]
    n = s.get("n") or s.get("n_tasks")
    return (f"{v:.3f}" if metric.startswith("f1") else f"{100 * v:.1f}%") + (f" (n={n})" if n and not s.get("group_mean") else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="runs/<label>.json files")
    args = ap.parse_args()
    pub = load_csv()
    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    head = "| Dataset | Metric | " + " | ".join(HEADERS) + "".join(f" | {r['label']}" for r in runs) + " |"
    print("## Tier 1\n")
    print(head)
    print("|" + "---|" * (2 + len(HEADERS) + len(runs)))
    for key, run_key, display in TIER1_ROWS:
        jev = pub.get((key, "jev"))
        metric = jev["metric"] if jev else "accuracy"
        cells = [fmt_published(pub.get((key, c))) for c in COLUMNS]
        cells += [fmt_run(r["summary"], run_key, metric) for r in runs]
        print(f"| {display} | {metric} | " + " | ".join(cells) + " |")
    if runs:
        means = []
        for r in runs:
            vals = [r["summary"][k]["raw"]["headline"] for _, k, _ in TIER1_ROWS if k in r["summary"] and k != "toxicchat"]
            means.append(f"{100 * sum(vals) / len(vals):.1f}% over {len(vals)} tasks" if vals else "-")
        print("| **Mean (accuracy tasks present)** | | " + " | ".join([""] * len(HEADERS)) + " | " + " | ".join(means) + " |")
    print("\n## Tier 2\n")
    print("| Test set | n | Jev | Microsoft-Decision-1 | Decisions API | Laya | Jez-large target | Jez-small target" + "".join(f" | {r['label']}" for r in runs) + " |")
    print("|" + "---|" * (8 + len(runs)))
    for key, display in [("typed_decisions", "typed-decisions test split (in-domain)"), ("decision_bench", "Decision Bench bench-v4, text rows"),
                         ("laya_battery", "Laya battery, 5 held-out tasks")]:
        jev = pub.get((key, "jev"))
        cells = [jev["n"] if jev else "-"] + [fmt_published(pub.get((key, c))) for c in ["jev", "microsoft-decision-1", "decisions-api", "laya", "target:jez-large", "target:jez-small"]]
        cells += [fmt_run(r["summary"], key, "accuracy") for r in runs]
        print(f"| {display} | " + " | ".join(cells) + " |")
    print("\n## Tier 3\n")
    for key, display in [("pooled_choice_ece", "Pooled ECE over Choice answers"), ("withdrawal_routing_order_stability", "Same route under all 6 option orders, withdrawal routing")]:
        jev = pub.get((key, "jev"))
        others = [f"{c}: {fmt_published(pub[(key, c)])}" for c in ["microsoft-decision-1", "decisions-api", "gemma-4-E4B-it-zeroshot"] if (key, c) in pub]
        print(f"- {display}: Jev {fmt_published(jev)}; " + "; ".join(others) + (f"; Jez gate {pub[(key, 'target:jez')]['value']}" if (key, "target:jez") in pub else ""))


if __name__ == "__main__":
    main()
