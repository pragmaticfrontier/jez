# jez-large-1.0.0

Jez decision model answering Jev's `/v1/systemone` contract. Reads the probabilities of declared answer
labels at the answer position; never generates text.

| | |
|---|---|
| Base checkpoint | `google/gemma-4-E4B-it` (Apache 2.0) |
| Method | lora ({"lr": 0.0001, "rank": 64, "alpha": 128, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}) |
| Training data | `none`: ? examples from ? sources; licences and 13-token overlap drops per source in the manifest |
| Teacher | none (gold labels only) (licence checked: n/a) |
| Objective | cross-entropy on the gold label + 0.5 × KL(teacher ‖ student) over label logits |
| Schedule | cosine, 3% warm-up, 1 epoch, 64 sequences per step, up to 4,096 tokens; learning rate 0.0002 chosen by a 3-rate sweep on 5% of the data |
| Checkpoint selection | validation loss on a 5% split; best checkpoint at step 4359 (validation loss 0.30767231346620244) |
| Calibration | `calibration.json`, temperatures per question type and option-count bucket, fitted on a 2% calibration split |
| Weights tag | `jez-large-1.0.0` |

## Acceptance gates (full splits)

| gate | result |
|---|---|
| 1. Conformance suite and both SDKs | PASS: tests/conformance (HTTP, Python SDK, JavaScript SDK) all green |
| 2. Tier 1 mean +3 points over the base, no task −1 | FAIL: mean -2.4 points, worst task -5.2 (samples) |
| 3. Pooled choice ECE ≤ 0.05; per-task ECE ≤ max(0.05, Jev's) | PASS: pooled choice ECE 0.019 raw, 0.016 calibrated |
| 4. Order stability ≥ 95%; no probability moves > 0.05 under reordering | PASS: 97.5% stable on agnews (n=200), max probability move 0.268 |
| 5. Latency p50 / p99 within Tier 4 limits | not measured |

## Tier 1

| task | base | release |
|---|---|---|
| sst2 | 93.6% | 94.2% |
| imdb | 95.7% | 94.8% |
| hellaswag | 83.9% | 80.3% |
| arc_easy | 97.8% | 95.7% |
| arc_challenge | 88.8% | 86.4% |
| boolq | 83.1% | 85.2% |
| belebele | 88.2% | 83.1% |
| agnews | 83.7% | 82.3% |
| emotion | 59.2% | 57.7% |
| banking77 | 75.4% | 70.8% |
| toxicchat | 0.597 | 0.526 |
| prompt_injection | 73.3% | 68.1% |
| sst5 | 48.7% | 44.2% |

## Tier 2 and Tier 3

Tier 2 sets are not in the repository; see eval/data/tier2/README.md.

## Latency

| machine | backend | p50 | p99 |
|---|---|---|---|
| | | not measured | |

## Known limits

- Text only: image assets and `box` questions return 422.
- 64k tokens per request and 32k per question; the long-context header is not supported.
- Calibration temperatures: noul 1.10, choice:2 0.58, choice:3-5 1.09, choice:6-26 1.09, choice:27-255 1.08, score:2 1.00, score:3-5 1.02, score:6-10 1.18
