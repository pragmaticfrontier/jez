# Decision-model benchmarks

`decision-model-benchmarks.csv` holds the published figures Jez is compared against, one row per
(dataset, model). Columns: `dataset`, `metric`, `model`, `value`, `n`, `split`, `note`, `source_url`.

## Provenance

The pf-jev repository this work started from did not contain a research directory, so every row
was transcribed from the Jez technical task document (2026-10-16), which attributes them to:

- Deußer, Sparrenberg and Sifa, arXiv 2609.37647: https://arxiv.org/html/2609.37647v1 (Jev figures, 37 datasets)
- per-model results, including Gemma-4-E4B-it zero-shot: https://github.com/AppliedMachineLearning-Lab/jev-benchmarking/tree/main/results/eval
- Microsoft-Decision-1: https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision
- Decisions API, Laya, typed-decisions, Decision Bench, the Laya battery and withdrawal routing: figures given in the
  task document without a URL; their `source_url` points back here until the original sources are recorded.

Rows whose `model` starts with `target:` are the task document's acceptance targets, not measurements:
the Jez-large target closes half of the gap between Gemma 4 E4B zero-shot and Jev; the Jez-small target is the
higher of pf-jev's Qwen3.5-2B figure and Laya's held-out figure where either exists. Neither the Qwen3.5-2B run
nor its figures exist in this repository; they are kept as the document states them.

## Belebele languages

The document refers to "the 12 Belebele languages pf-jev already tests" without listing them, and the repository
never had a Belebele loader. `eval/tasks.py` uses: eng_Latn, deu_Latn, fra_Latn, spa_Latn, ita_Latn, por_Latn,
rus_Cyrl, ukr_Cyrl, arb_Arab, hin_Deva, zho_Hans, jpn_Jpan. Change `BELEBELE_LANGUAGES` if the original set is found;
the Belebele comparison against the paper is only like-for-like once the sets match.

## Tier 2 data

typed-decisions, Decision Bench bench-v4, the Laya battery and the withdrawal-routing cases are not public
Hub datasets. `eval/tasks.py` reads them from `eval/data/tier2/<name>.jsonl`; see the README there for the row format.
