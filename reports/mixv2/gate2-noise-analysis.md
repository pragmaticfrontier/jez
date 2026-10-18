# Gate 2 for Jez-small (mix-v2): how much of the miss is sampling noise?

Paired bootstrap (4,000 resamples, seed 0) over the per-item records of `runs/qwen3.5-0.8b-zeroshot-n1000.json` and
`runs/jez-small-1.0.0-n1000.json`. Both runs score the same items in the same order, so each task's delta is a paired
difference. "won/lost" counts items the tuned model gets right and the base wrong, and the reverse.

| task | n | base | tuned | delta | 95% CI | won/lost |
|---|---|---|---|---|---|---|
| prompt_injection | 116 | 52.6 | 49.1 | −3.4 | [−8.6, +0.9] | 2/6 |
| sst2 | 872 | 83.3 | 81.3 | −1.9 | [−4.8, +0.9] | 73/90 |
| imdb | 1000 | 88.4 | 86.9 | −1.5 | [−3.5, +0.6] | 48/63 |
| boolq | 1000 | 75.2 | 73.7 | −1.5 | [−4.4, +1.4] | 108/123 |
| arc_easy | 1000 | 80.8 | 79.5 | −1.3 | [−4.1, +1.4] | 98/111 |
| emotion | 1000 | 55.7 | 54.4 | −1.3 | [−4.6, +1.8] | 117/130 |
| agnews | 1000 | 76.1 | 75.4 | −0.7 | [−2.6, +1.2] | 45/52 |
| toxicchat (accuracy) | 1000 | 92.2 | 92.1 | −0.1 | [−1.6, +1.4] | 30/31 |
| hellaswag | 1000 | 44.7 | 46.2 | +1.5 | [−2.7, +5.6] | 228/213 |
| arc_challenge | 1000 | 59.4 | 61.1 | +1.7 | [−1.6, +5.0] | 151/134 |
| belebele (12 languages) | 900 each | 49.4 | 60.1 | +5.0 to +14.7 | every CI above +1 | |
| sst5 | 1000 | 24.4 | 34.5 | +10.1 | [+6.8, +13.4] | 202/101 |
| banking77 | 1000 | 14.7 | 60.6 | +45.9 | [+42.7, +49.3] | 477/18 |

ToxicChat F1 (toxic class): base 0.391 (tp 25, fp 17, fn 61) → tuned 0.490 (tp 38, fp 31, fn 48).

## Reading

- None of the six tasks below −1 is distinguishable from "no change" at this sample size: every interval includes 0.
  The gate is defined on full splits, so the n = 1,000 run cannot decide it either way.
- Taken together the six small drops point the same way (about −1.5 on average), which is consistent with mild
  general-knowledge drift from gold-only cross-entropy on a 0.8B model. A teacher/KL term or a weaker update is the
  remedy if the full-split evaluation confirms the drift.
- Full splits only help where the split is large: IMDB (25,000) and BoolQ (3,270) tighten to about ±1 and ±2;
  ARC-Easy (2,376) and Emotion (2,000) to about ±2. SST-2's validation set **is** 872 items and the deepset prompt
  injection test set 116, so those two stay at roughly ±3 and ±9 points however the gate is measured. A strict
  "no task −1" on them is partly luck; the task document itself quotes ±8 points for prompt injections.
- The gains are large and certain: BANKING77 +46, Belebele +11 (all 12 languages individually significant), SST-5 +10,
  ToxicChat F1 +0.10.
