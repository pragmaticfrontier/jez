# {model_id}

Jez decision model answering Jev's `/v1/systemone` contract. Reads the probabilities of declared answer
labels at the answer position; never generates text.

| | |
|---|---|
| Base checkpoint | `{base}` ({base_licence}) |
| Method | {method}{method_details} |
| Training data | `{mix_manifest}`: {n_train} examples from {n_sources} sources; licences and 13-token overlap drops per source in the manifest |
| Teacher | {teacher} (licence checked: {teacher_licence_checked}) |
| Objective | {objective} |
| Schedule | {schedule} |
| Checkpoint selection | {selection} |
| Calibration | `calibration.json`, temperatures per question type and option-count bucket, {calibration_fit} |
| Weights tag | `{model_id}` |

## Acceptance gates ({gate_scope})

| gate | result |
|---|---|
| 1. Conformance suite and both SDKs | {gate1} |
| 2. Tier 1 mean +3 points over the base, no task −1 | {gate2} |
| 3. Pooled choice ECE ≤ 0.05; per-task ECE ≤ max(0.05, Jev's) | {gate3} |
| 4. Order stability ≥ 95%; no probability moves > 0.05 under reordering | {gate4} |
| 5. Latency p50 / p99 within Tier 4 limits | {gate5} |

## Tier 1

{tier1_table}

## Tier 2 and Tier 3

{tier2_tier3}

## Latency

| machine | backend | p50 | p99 |
|---|---|---|---|
{latency_rows}

## Known limits

- Text only: image assets and `box` questions return 422.
- 64k tokens per request and 32k per question; the long-context header is not supported.
- {known_limits}
