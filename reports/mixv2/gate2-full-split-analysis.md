# Gate 2 for Jez-small (mix-v2) on the full Tier 1 splits

Both models were evaluated on the complete Tier 1 splits on Modal H100s on 2026-10-18 (about 73,000 items each,
30 minutes per model): `runs/qwen3.5-0.8b-zeroshot-full.json` and `runs/jez-small-1.0.0-full.json`. Paired bootstrap,
2,000 resamples, seed 0; "sig" marks a 95% interval that excludes 0.

| task | n | base | tuned | delta | 95% CI | won/lost | |
|---|---|---|---|---|---|---|---|
| prompt_injection | 116 | 52.6 | 49.1 | −3.4 | [−7.8, +1.7] | 2/6 | |
| boolq | 3270 | 74.8 | 72.2 | −2.7 | [−4.4, −1.0] | 361/448 | sig |
| arc_easy | 2376 | 80.0 | 77.7 | −2.3 | [−4.1, −0.5] | 210/265 | sig |
| sst2 | 872 | 83.3 | 81.3 | −1.9 | [−4.8, +0.9] | 73/90 | |
| imdb | 25000 | 89.2 | 88.1 | −1.1 | [−1.5, −0.7] | 1290/1569 | sig |
| emotion | 2000 | 54.9 | 54.4 | −0.4 | [−2.6, +1.8] | 247/256 | |
| agnews | 7600 | 73.9 | 73.5 | −0.4 | [−1.1, +0.3] | 378/406 | |
| toxicchat (accuracy) | 5083 | 93.5 | 93.9 | +0.5 | [−0.2, +1.1] | 155/132 | |
| arc_challenge | 1172 | 59.8 | 61.5 | +1.7 | [−1.4, +4.9] | 178/158 | |
| hellaswag | 10042 | 43.6 | 46.9 | +3.3 | [+2.0, +4.5] | 2310/1979 | sig |
| belebele (12 languages) | 900 each | 49.4 | 60.1 | +5.0 to +14.7 | every language significant | | sig |
| sst5 | 2210 | 25.2 | 32.5 | +7.4 | [+5.2, +9.7] | 421/258 | sig |
| banking77 | 3080 | 12.4 | 60.1 | +47.6 | [+45.8, +49.5] | 1531/64 | sig |

ToxicChat F1 (toxic class): base 0.379 (tp 101, fp 70, fn 261) → tuned 0.554 (tp 191, fp 137, fn 171).
Mean over the 12 accuracy tasks: 58.2 → 63.1 (+4.9).

## Verdict

- **Gate 2 fails on the full splits, and not by noise.** BoolQ (−2.7), ARC-Easy (−2.3) and IMDB (−1.1) are
  significant drops; SST-2 (−1.9) and prompt injection (−3.4) remain inside their intervals because those splits are
  small (872 and 116 items), but they point the same way. The mean gain (+4.9) clears the +3 requirement easily.
- The pattern is mild general-knowledge drift from gold-only cross-entropy: the model wins where the mix teaches a
  new format or domain (BANKING77, Belebele, SST-5, ToxicChat) and loses a little where the base already knew the
  answer (reading comprehension, yes/no QA, sentiment).
- Remedies, in the order the task document anticipates: teacher soft labels with the 0.5 × KL term (keeps the model
  near a strong distribution instead of hard gold labels), then a gentler update (lower learning rate, early stop on a
  Tier 1 development slice, or mixing in base-model answers as replay).
- Gate 3 on the full splits: pooled choice ECE 0.023 raw, 0.038 calibrated (PASS). The base model's calibrated
  figures are 0.019 (choice) and 0.057 (yes/no).
