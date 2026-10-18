# Gate 2 for Jez-large (mix-v2): the drop is real

Paired bootstrap (4,000 resamples, seed 0) over the per-item records of `runs/gemma-4-E4B-it-zeroshot-n1000.json`
and `runs/jez-large-1.0.0-n1000.json` (same items, same order). "sig" marks a 95% interval that excludes 0.

| task | n | base | tuned | delta | 95% CI | won/lost | |
|---|---|---|---|---|---|---|---|
| belebele_deu_Latn | 900 | 91.1 | 83.4 | −7.7 | [−10.0, −5.3] | 25/94 | sig |
| belebele_por_Latn | 900 | 89.9 | 83.7 | −6.2 | [−8.3, −4.1] | 22/78 | sig |
| belebele_ita_Latn | 900 | 89.1 | 83.0 | −6.1 | [−8.3, −3.9] | 25/80 | sig |
| belebele_spa_Latn | 900 | 89.0 | 83.0 | −6.0 | [−8.2, −3.9] | 26/80 | sig |
| belebele_arb_Arab | 900 | 88.1 | 82.7 | −5.4 | [−7.8, −3.1] | 34/83 | sig |
| belebele_eng_Latn | 900 | 92.2 | 86.9 | −5.3 | [−7.6, −3.2] | 26/74 | sig |
| prompt_injection | 116 | 73.3 | 68.1 | −5.2 | [−10.3, −0.9] | 1/7 | sig |
| belebele_fra_Latn | 900 | 90.2 | 85.2 | −5.0 | [−7.2, −2.8] | 30/75 | sig |
| banking77 | 1000 | 75.4 | 70.8 | −4.6 | [−7.0, −2.2] | 49/95 | sig |
| sst5 | 1000 | 48.7 | 44.2 | −4.5 | [−8.9, −0.2] | 226/271 | sig |
| belebele_hin_Deva | 900 | 78.0 | 73.6 | −4.4 | [−7.1, −1.8] | 53/93 | sig |
| belebele_ukr_Cyrl | 900 | 87.9 | 83.7 | −4.2 | [−6.4, −2.0] | 36/74 | sig |
| belebele_jpn_Jpan | 900 | 85.8 | 81.8 | −4.0 | [−6.3, −1.8] | 37/73 | sig |
| belebele_zho_Hans | 900 | 88.9 | 85.2 | −3.7 | [−5.9, −1.6] | 35/68 | sig |
| hellaswag | 1000 | 83.9 | 80.3 | −3.6 | [−6.2, −1.0] | 65/101 | sig |
| belebele_rus_Cyrl | 900 | 88.1 | 85.1 | −3.0 | [−5.2, −0.8] | 38/65 | sig |
| arc_challenge | 1000 | 88.8 | 86.4 | −2.4 | [−4.3, −0.7] | 32/56 | sig |
| arc_easy | 1000 | 97.8 | 95.7 | −2.1 | [−3.4, −0.8] | 11/32 | sig |
| emotion | 1000 | 59.2 | 57.7 | −1.5 | [−3.7, +0.7] | 56/71 | |
| agnews | 1000 | 83.7 | 82.3 | −1.4 | [−2.6, −0.2] | 11/25 | sig |
| imdb | 1000 | 95.7 | 94.8 | −0.9 | [−2.0, +0.1] | 11/20 | |
| toxicchat (accuracy) | 1000 | 94.2 | 93.5 | −0.7 | [−1.9, +0.5] | 16/23 | |
| sst2 | 872 | 93.6 | 94.2 | +0.6 | [−0.9, +2.2] | 26/21 | |
| boolq | 1000 | 83.1 | 85.2 | +2.1 | [−0.2, +4.3] | 80/59 | |

ToxicChat F1 (toxic class): base 0.597 → tuned 0.526.

## Reading

- Unlike the small model, this is not noise: 18 of 24 lines are significant drops, every Belebele language loses 3 to
  8 points, and the mean over the 12 accuracy tasks falls from 80.9 to 78.6. Only BoolQ (+2.1) and SST-2 (+0.6) move
  up, neither significantly.
- The in-mix validation accuracy climbed to 87.7% (base 38.0% in the mix's answer format), so the LoRA learned the
  format and the in-domain labels very well while damaging the instruct model's general knowledge. The in-mix
  validation set is therefore not a usable proxy for Tier 1 transfer on a base model that already scores 80.9.
- The learning rate 2e-4 was chosen by the sweep on that same in-mix validation accuracy (85.7% against 85.1% at 1e-4
  and 83.8% at 5e-5 after 217 steps), which rewards the most aggressive update. For a strong base model the recipe
  must be gentler: a lower rate (5e-5 or less), far fewer steps or early stopping on a Tier 1 development slice, a
  smaller rank, or a KL term toward the base model's own answer distribution or toward the 27B teacher.
