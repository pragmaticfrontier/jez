## Tier 1

| Dataset | Metric | Jev | Microsoft-Decision-1 | Decisions API | Laya | Gemma 4 E4B zero-shot | Jez-large target | Jez-small target | qwen3.5-0.8b-zeroshot-n1000 | jez-small-1.0.0-n1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| SST-2 | accuracy | 96.4% | not tested | not tested | not tested | 95.2% | 95.8% | 82.9% | 83.3% (n=872) | 81.3% (n=872) |
| IMDB | accuracy | 96.5% | not tested | not tested | not tested | 95.6% | 96.0% | 94.3% | 88.4% (n=1000) | 86.9% (n=1000) |
| HellaSwag | accuracy | 95.5% | not tested | not tested | not tested | 82.9% | 89.2% | 65.0% | 44.7% (n=1000) | 46.2% (n=1000) |
| ARC-Easy | accuracy | 99.3% | not tested | not tested | not tested | 95.9% | 97.6% | 94.3% | 80.8% (n=1000) | 79.5% (n=1000) |
| ARC-Challenge | accuracy | 97.7% | not tested | not tested | not tested | 87.8% | 92.7% | 82.9% | 59.4% (n=1000) | 61.1% (n=1000) |
| BoolQ | accuracy | 91.3% | not tested | not tested | 83.0% (held-out) | 82.2% | 86.8% | 83.0% | 75.2% (n=1000) | 73.7% (n=1000) |
| Belebele, 12 languages | accuracy | 92.7% (mean over 12 languages (900 each)) | not tested | not tested | not tested | 80.4% (mean over 12 languages; paper prompts) | 86.5% | 74.6% | 49.4% | 60.1% |
| AG News | accuracy | 88.5% | 88.2% | 89.0% | 95.0% (trained on it) | 86.2% | 87.4% | beat base | 76.1% (n=1000) | 75.4% (n=1000) |
| DAIR Emotion | accuracy | 58.5% | 62.3% | not tested | 59.5% | 56.9% | 57.7% | 59.5% | 55.7% (n=1000) | 54.4% (n=1000) |
| BANKING77 | accuracy | 79.7% | 89.0% | 79.0% | 42.5% | 67.0% | 73.4% | 42.5% | 14.7% (n=1000) | 60.6% (n=1000) |
| ToxicChat (F1 toxic) | f1_positive | 0.786 | 53.0% (accuracy not F1) | not tested | 53.0% (accuracy not F1) | 0.703 | 0.745 | beat base | 0.391 (n=1000) | 0.490 (n=1000) |
| Prompt injections (deepset) | accuracy | 74.1% (interval about ±8 points) | not tested | not tested | 69.8% | 66.4% | 70.2% | 69.8% | 52.6% (n=116) | 49.1% (n=116) |
| SST-5 | accuracy | 57.9% | not tested | not tested | 37.2% | 52.8% | 55.4% | 37.2% | 24.4% (n=1000) | 34.5% (n=1000) |
| **Mean (accuracy tasks present)** | |  |  |  |  |  |  |  | 58.7% over 12 tasks | 63.6% over 12 tasks |

## Tier 2

| Test set | n | Jev | Microsoft-Decision-1 | Decisions API | Laya | Jez-large target | Jez-small target | qwen3.5-0.8b-zeroshot-n1000 | jez-small-1.0.0-n1000 |
|---|---|---|---|---|---|---|---|---|---|
| typed-decisions test split (in-domain) | 2000 | 75.4% | 78.3% | not tested | 76.6% (fine-tuned) | 78.3% | 76.6% | - | - |
| Decision Bench bench-v4, text rows | 949 | 93.2% | 90.7% | 89.5% | 52.8% | 89.5% | report | - | - |
| Laya battery, 5 held-out tasks | 2000 | 80.3% | 78.2% | not tested | 57.9% | report | 57.9% | - | - |

## Tier 3

- Pooled ECE over Choice answers: Jev 0.028; gemma-4-E4B-it-zeroshot: 0.208 (raw); Jez gate at most 0.05 for both models
- Same route under all 6 option orders, withdrawal routing: Jev 91.6% (same route under all 6 option orders); microsoft-decision-1: 97.6%; decisions-api: 64.2%; Jez gate at least 95%
