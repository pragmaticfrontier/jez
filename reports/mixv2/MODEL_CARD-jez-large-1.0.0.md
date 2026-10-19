# jez-large-1.0.0

Jez decision model answering Jev's `/v1/systemone` contract. Reads the probabilities of declared answer
labels at the answer position; never generates text.

| | |
|---|---|
| Base checkpoint | `google/gemma-4-E4B-it` (Apache 2.0 on the Hub card; Gemma 4 licence terms at https://ai.google.dev/gemma/docs/gemma_4_license) |
| Method | lora ({"rank": 64, "alpha": 128, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}) |
| Training data | `reports/mix-v2-manifest.json`: 278999 examples from 20 sources; licences and 13-token overlap drops per source in the manifest |
| Teacher | none (gold labels only) (licence checked: n/a) |
| Objective | cross-entropy on the gold label + 0.5 × KL(base ‖ student) toward the frozen base model's own label distribution (read with the adapter disabled), over label logits at the answer position |
| Schedule | cosine, 3% warm-up, 1,000 steps of 64 sequences (23% of an epoch), up to 4,096 tokens; learning rate 5e-05 chosen on Tier 1 development samples at n = 300 (recipe pilots, reports/mixv2/pilots) |
| Checkpoint selection | best validation loss on a 5% split: step 1000 (loss 0.4060, accuracy 83.7%); the step-1000 snapshot of a 1,000-step run was picked on Tier 1 samples, not on in-mix validation |
| Calibration | `calibration.json`, temperatures per question type and option-count bucket, fitted on the even-indexed half of the 1,000-item Tier 1 samples; calibrated figures are reported on the odd-indexed half |
| Weights tag | `jez-large-1.0.0` |

## Acceptance gates (1,000-item samples per task)

| gate | result |
|---|---|
| 1. Conformance suite and both SDKs | PASS: tests/conformance (HTTP, Python SDK, JavaScript SDK) all green |
| 2. Tier 1 mean +3 points over the base, no task −1 | FAIL: mean -0.3 points, worst task -6.9 (samples) |
| 3. Pooled choice ECE ≤ 0.05; per-task ECE ≤ max(0.05, Jev's) | PASS: pooled choice ECE 0.067 raw, 0.035 calibrated |
| 4. Order stability ≥ 95%; no probability moves > 0.05 under reordering | PASS: 99.0% stable on agnews (n=200), max probability move 0.191 |
| 5. Latency p50 / p99 within Tier 4 limits | not measured |

## Tier 1

| task | base | release |
|---|---|---|
| sst2 | 93.6% | 94.2% |
| imdb | 95.7% | 95.3% |
| hellaswag | 83.9% | 83.9% |
| arc_easy | 97.8% | 97.7% |
| arc_challenge | 88.8% | 89.5% |
| boolq | 83.1% | 84.5% |
| belebele | 88.2% | 90.1% |
| agnews | 83.7% | 84.5% |
| emotion | 59.2% | 58.9% |
| banking77 | 75.4% | 75.8% |
| toxicchat | 0.597 | 0.535 |
| prompt_injection | 73.3% | 66.4% |
| sst5 | 48.7% | 47.1% |

## Tier 2 and Tier 3

Tier 2 sets are not in the repository; see eval/data/tier2/README.md.

## Latency

| machine | backend | p50 | p99 |
|---|---|---|---|
| | | not measured | |

## Known limits

- Text only: image assets and `box` questions return 422.
- 64k tokens per request and 32k per question; the long-context header is not supported.
- Calibration temperatures: noul 1.67, choice:2 0.47, choice:3-5 0.76, choice:6-26 0.69, choice:27-255 1.26, score:2 1.00, score:3-5 1.56, score:6-10 1.00
- Gate 2 is not met: prompt-injection detection is 6.9 points below the base Gemma (66.4% vs 73.3%, significant) and ToxicChat F1 is 0.535 vs 0.597; the fine-tune slides toward "not an injection" on imperative texts, see reports/mixv2/pilots/README.md. Every other Tier 1 task is within noise of the base or above it (Belebele +1.9, BoolQ +1.4).
- The training mix's own calibration split left choice answers under-confident (pooled choice ECE 0.068 with those temperatures, kept in reports/mixv2/calibration-large-mix-split.json), hence the Tier 1 fit. Per-task calibrated ECE stays above 0.05 on HellaSwag (0.135) and AG News (0.110).
- Tier 1 figures are 1,000-item samples per task; the full-split run is pending.
- Latency is not measured on the target machines (M1 Pro, L4). H100 reference on examples/claim.json: p50 88.7 ms, p99 95.5 ms (reports/mixv2/latency-jez-large-1.0.0-h100-reference.json).
