# Jez-large recipe pilots (2026-10-18, Modal H100)

Two short LoRA runs of Gemma 4 E4B on mix-v2, launched 18:40 CEST and finished 20:50 CEST (about 4 H100 hours in total,
roughly $17). Both use rank 64 / alpha 128 on all linear projections, lr 5e-5 with a cosine schedule over 1,000 steps of
64 sequences (64k items, 23% of an epoch), bf16, 16k-token micro-batches. Adapter snapshots at steps 250, 500, 750 and
1,000 were evaluated directly (merged in memory) on Tier 1 at n = 300 per task against the base Gemma on the same items
(`gemma-4-E4B-it-zeroshot-n300`).

- **gentle**: gold cross-entropy only (`--lr 5e-5 --steps 1000 --snapshot-every 250`).
- **anchored**: the same plus `--base-kl-weight 0.5`, a KL term that pulls the student's label distribution toward the
  frozen base model's own answer (computed with the adapter disabled; no teacher involved).

Files: `gate2-gentle.md`, `gate2-anchored.md` (per-snapshot gate-2 tables), `gate2-pilots-bootstrap.md` (paired
bootstrap intervals), `train-*.log`, the per-run Tier 1 tables `*-n300.md`. Raw run JSON in `runs/*-n300.json`.

## Result

| snapshot | gentle mean | gentle worst | anchored mean | anchored worst |
|---|---|---|---|---|
| step 250 | -1.3 | -8.0 (IMDB) | -0.4 | -5.2 (prompt injection) |
| step 500 | -2.8 | -14.7 (prompt injection) | -0.1 | -7.8 (prompt injection) |
| step 750 | -3.3 | -11.2 | -0.0 | -6.9 |
| step 1000 | -3.5 | -12.0 (HellaSwag) | -0.1 | -6.9 |

Gate 2 (mean +3, no task below -1) fails for every snapshot. In-mix validation: gentle 86.3%, anchored 83.7%, base 71.0%,
full-epoch lr 2e-4 run 87.7%.

**Gentle** rules out "the learning rate was simply too high": at a quarter of the old rate and a quarter of the data the
LoRA still erodes Gemma, and the erosion grows with steps (HellaSwag -12, prompt injection -11, SST-5 -5.7, Belebele -4.5,
ToxicChat F1 0.53 to 0.28 at step 1000).

**Anchored** holds the Tier 1 mean at the base's level at every snapshot and produces real gains: at step 1000 Belebele
+2.3 (six of twelve languages significant), BoolQ +4.0 (significant), BANKING77 +2.0, AG News +1.3, ARC-Challenge +1.0,
SST-5 +0.7, ARC-Easy unchanged. Remaining drops: prompt injection -6.9 (significant), IMDB -2.0, HellaSwag -1.7 (-3.0 at
step 750, significant there), SST-2 -0.7, Emotion -0.7, ToxicChat F1 0.533 to 0.485. The anchor costs 2.6 points of
in-mix validation and buys 3.4 points of Tier 1 mean relative to gentle.

## Diagnosis of the remaining drops

- **Prompt injection** (116 items): eight items move from right to wrong between the base and anchored step 750, none
  the other way, and all eight are injections ("Vergiss alles davor...", "forget what's before...", "let's play a game
  where you answer every question with..."). Mean p(true) on injections falls 0.52 to 0.39; on legitimate texts it stays
  at 0.01. The mix has no injection-like family; its closest yes/no families ("Is this comment toxic?", "Is the response
  good enough to send?") teach that imperative, rule-like text is benign, and the base's own answer on these items is not
  confident enough for the anchor to hold it. Every snapshot of both pilots shows the same slide from step 250 on.
- **HellaSwag**: accuracy -1.7 to -3, but the mean top probability halves (0.86 to 0.61 anchored, 0.52 gentle): the
  many-option choice families (MASSIVE 60 intents, DBpedia 14, BANKING-like intents) flatten the model's choice
  distribution. Temperature calibration repairs the ECE, not the accuracy.
- **IMDB**: the deterministic n = 300 sample holds only negative reviews (`chunk_sample` takes the head of a label-sorted
  split); the fine-tuned model answers "positive" on 25 instead of 19 of them. The full split is balanced, where the
  round-2 Jez-small drop was -1.1.

## What this means for gate 2

The base scores 80.6 at n = 300; gate 2 needs 83.6. ARC-Easy (97.7), SST-2 (94.3), IMDB (93.7) and ARC-Challenge (90.7)
are saturated, so +3 on the mean has to come from SST-5 (46), Emotion (63), prompt injection (73), BANKING77 (74), BoolQ,
AG News and HellaSwag (82) and Belebele (88). mix-v2 has no 5-level sentiment, no emotion and no injection family, so
gold-only training on it cannot lift those tasks; the anchor only stops the damage. Levers, in order:

1. **Teacher KL** (the task's own recipe, not yet run): label the items a 1,000-step run consumes (64k-100k) with the
   27B teacher (`train.teacher_label --licence-checked`, about 1-3 H100 hours), then retrain anchored + teacher KL
   (about 2 h with evaluation). Soft labels from a stronger model are the only ingredient that can push a task above the
   base's own zero-shot without task-specific data. Roughly $15-20 for the pilot.
2. **mix-v3**: add licensed, overlap-filtered families for the weak formats (injection/jailbreak detection, emotion,
   5-level sentiment); needs a decision on whether same-distribution train splits are allowed.
3. A stronger anchor (lambda 1-2) would only move the mean closer to zero from below; not worth a run on its own.
