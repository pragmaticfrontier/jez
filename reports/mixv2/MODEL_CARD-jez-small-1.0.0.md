# jez-small-1.0.0

Jez decision model answering Jev's `/v1/systemone` contract. Reads the probabilities of declared answer
labels at the answer position; never generates text.

| | |
|---|---|
| Base checkpoint | `Qwen/Qwen3.5-0.8B` (Apache 2.0) |
| Method | full ({"lr": 2e-05}) |
| Training data | `none`: ? examples from ? sources; licences and 13-token overlap drops per source in the manifest |
| Teacher | none (gold labels only) (licence checked: n/a) |
| Objective | cross-entropy on the gold label + 0.5 × KL(teacher ‖ student) over label logits |
| Schedule | cosine, 3% warm-up, 1 epoch, 64 sequences per step, up to 4,096 tokens; learning rate 1e-05 chosen by a 3-rate sweep on 5% of the data |
| Checkpoint selection | validation loss on a 5% split; best checkpoint at step 3800 (validation loss 0.39027396082133053) |
| Calibration | `calibration.json`, temperatures per question type and option-count bucket, fitted on a 2% calibration split |
| Weights tag | `jez-small-1.0.0` |

## Acceptance gates (full splits)

| gate | result |
|---|---|
| 1. Conformance suite and both SDKs | PASS: tests/conformance (HTTP, Python SDK, JavaScript SDK) all green |
| 2. Tier 1 mean +3 points over the base, no task −1 | FAIL: mean +4.8 points, worst task -3.4 (samples) |
| 3. Pooled choice ECE ≤ 0.05; per-task ECE ≤ max(0.05, Jev's) | PASS: pooled choice ECE 0.025 raw, 0.048 calibrated |
| 4. Order stability ≥ 95%; no probability moves > 0.05 under reordering | PASS: 98.5% stable on agnews (n=200), max probability move 0.151 |
| 5. Latency p50 / p99 within Tier 4 limits | PASS on this machine: p50 58.3 ms, p99 61.5 ms (torch on cuda); Tier 4 asks p50 ≤ 150 ms on an M1 Pro and ≤ 60 ms on an L4 |

## Tier 1

| task | base | release |
|---|---|---|
| sst2 | 83.3% | 81.3% |
| imdb | 88.4% | 86.9% |
| hellaswag | 44.7% | 46.2% |
| arc_easy | 80.8% | 79.5% |
| arc_challenge | 59.4% | 61.1% |
| boolq | 75.2% | 73.7% |
| belebele | 49.4% | 60.1% |
| agnews | 76.1% | 75.4% |
| emotion | 55.7% | 54.4% |
| banking77 | 14.7% | 60.6% |
| toxicchat | 0.391 | 0.490 |
| prompt_injection | 52.6% | 49.1% |
| sst5 | 24.4% | 34.5% |

## Tier 2 and Tier 3

Tier 2 sets are not in the repository; see eval/data/tier2/README.md.

## Latency

| machine | backend | p50 | p99 |
|---|---|---|---|
| cuda | torch | 58.3 ms | 61.5 ms |

## Known limits

- Text only: image assets and `box` questions return 422.
- 64k tokens per request and 32k per question; the long-context header is not supported.
- Calibration temperatures: noul 1.09, choice:2 0.74, choice:3-5 1.16, choice:6-26 1.08, choice:27-255 1.00, score:2 1.00, score:3-5 1.16, score:6-10 1.25
