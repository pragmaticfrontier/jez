# qwen3.5-0.8b-zeroshot-n1000

checkpoint `Qwen/Qwen3.5-0.8B` · backend torch on cuda · 1000 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 83.3% | 0.019 | 0.030 | 0.241 | 124 | 808 | 79 |
| imdb | choice | 1000 | 88.4% | 0.029 | 0.019 | 0.167 | 130 | 136 | 312 |
| hellaswag | choice | 1000 | 44.7% | 0.012 | 0.048 | 0.673 | 124 | 134 | 147 |
| arc_easy | choice | 1000 | 80.8% | 0.123 | 0.227 | 0.300 | 96 | 784 | 89 |
| arc_challenge | choice | 1000 | 59.4% | 0.041 | 0.140 | 0.524 | 113 | 130 | 114 |
| boolq | noul | 1000 | 75.2% | 0.048 | 0.075 | 0.334 | 127 | 136 | 170 |
| agnews | choice | 1000 | 76.1% | 0.080 | 0.046 | 0.348 | 131 | 136 | 171 |
| emotion | choice | 1000 | 55.7% | 0.204 | 0.035 | 0.646 | 96 | 138 | 87 |
| banking77 | choice | 1000 | 14.7% | 0.315 | 0.111 | 1.066 | 98 | 134 | 910 |
| toxicchat | noul | 1000 | 39.1% | 0.274 | 0.133 | 0.286 | 94 | 128 | 108 |
| prompt_injection | noul | 116 | 52.6% | 0.150 | 0.227 | 0.475 | 94 | 130 | 87 |
| sst5 | score | 1000 | 24.4% | 0.380 | 0.119 | 0.943 | 94 | 128 | 98 |
| belebele_eng_Latn | choice | 900 | 57.0% | 0.074 | 0.113 | 0.540 | 130 | 132 | 205 |
| belebele_deu_Latn | choice | 900 | 51.2% | 0.084 | 0.066 | 0.603 | 126 | 128 | 253 |
| belebele_fra_Latn | choice | 900 | 51.6% | 0.105 | 0.069 | 0.608 | 111 | 131 | 254 |
| belebele_spa_Latn | choice | 900 | 51.1% | 0.100 | 0.069 | 0.626 | 128 | 768 | 243 |
| belebele_ita_Latn | choice | 900 | 48.9% | 0.124 | 0.069 | 0.645 | 97 | 133 | 241 |
| belebele_por_Latn | choice | 900 | 50.4% | 0.108 | 0.089 | 0.617 | 95 | 130 | 236 |
| belebele_rus_Cyrl | choice | 900 | 50.0% | 0.101 | 0.082 | 0.620 | 94 | 128 | 258 |
| belebele_ukr_Cyrl | choice | 900 | 47.2% | 0.099 | 0.087 | 0.649 | 96 | 133 | 301 |
| belebele_arb_Arab | choice | 900 | 45.2% | 0.110 | 0.040 | 0.671 | 95 | 130 | 241 |
| belebele_hin_Deva | choice | 900 | 37.6% | 0.155 | 0.064 | 0.746 | 97 | 136 | 343 |
| belebele_zho_Hans | choice | 900 | 54.3% | 0.071 | 0.091 | 0.573 | 94 | 143 | 201 |
| belebele_jpn_Jpan | choice | 900 | 48.7% | 0.090 | 0.044 | 0.642 | 95 | 129 | 229 |
| **belebele (mean of 12)** | | | 49.4% | | | | | | |
| **Tier 1 mean** | | | 57.2% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.080 | 0.053 |
| noul | 2116 | 0.147 | 0.047 |
| score | 1000 | 0.380 | 0.119 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 0.441 |
| choice:2 | 0.885 |
| choice:3-5 | 1.598 |
| choice:6-26 | 1.800 |
| choice:27-255 | 1.957 |
| score:2 | 1.000 |
| score:3-5 | 3.200 |
| score:6-10 | 1.000 |
