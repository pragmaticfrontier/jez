
## Qwen/Qwen3-1.7B  (n=300/task, debias=False)

| task | type | n | accuracy | ECE | mean conf | latency p50 | p95 | tokens/ex |
|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 |  90.3% | 0.079 | 0.977 | 33 ms | 36 ms | - |
| agnews | choice | 300 |  84.3% | 0.157 | 0.999 | 34 ms | 36 ms | - |
| boolq | bool | 300 |  76.3% | 0.232 | 0.980 | 34 ms | 37 ms | - |
| yelp | score | 300 |  34.0% | 0.600 | 0.940 | 37 ms | 65 ms | - |  (score task: MAE argmax 0.79, MAE expected 0.80, within±1  86.7%)

### Temperature scaling (fit on even half, evaluated on odd half)

| type | tasks | T | ECE before | ECE after | NLL before | NLL after | acc (odd half) |
|---|---|---|---|---|---|---|---|
| choice | sst2+agnews | 6.69 | 0.142 | 0.050 | 2.352 | 0.478 |  85.0% |

Selective prediction after scaling (choice, odd half): act only when confidence >= threshold
| threshold | coverage | accuracy |
|---|---|---|
| 0.00 | 100.0% |  85.0% |
| 0.50 |  99.7% |  85.3% |
| 0.60 |  94.3% |  86.9% |
| 0.70 |  90.0% |  87.4% |
| 0.80 |  84.7% |  89.0% |
| 0.90 |  69.0% |  90.8% |
| 0.95 |  30.0% |  96.7% |
| 0.99 |   0.0% |   nan% |

| bool | boolq | 11.42 | 0.206 | 0.042 | 2.464 | 0.502 |  78.7% |

Selective prediction after scaling (bool, odd half): act only when confidence >= threshold
| threshold | coverage | accuracy |
|---|---|---|
| 0.00 | 100.0% |  78.7% |
| 0.50 | 100.0% |  78.7% |
| 0.60 |  89.3% |  81.3% |
| 0.70 |  70.7% |  83.0% |
| 0.80 |  49.3% |  86.5% |
| 0.90 |   2.7% |  75.0% |
| 0.95 |   0.0% |   nan% |
| 0.99 |   0.0% |   nan% |

| score | yelp | 11.94 | 0.584 | 0.117 | 6.669 | 1.450 |  35.3% |

Selective prediction after scaling (score, odd half): act only when confidence >= threshold
| threshold | coverage | accuracy |
|---|---|---|
| 0.00 | 100.0% |  35.3% |
| 0.50 |  20.7% |  35.5% |
| 0.60 |   0.7% | 100.0% |
| 0.70 |   0.0% |   nan% |
| 0.80 |   0.0% |   nan% |
| 0.90 |   0.0% |   nan% |
| 0.95 |   0.0% |   nan% |
| 0.99 |   0.0% |   nan% |

wrote bench/results/calibration-Qwen3-1.7B.json: {'choice': 6.6854, 'bool': 11.4219, 'score': 11.9368}
wrote bench/results/Qwen3-1.7B.json
