# anchored-step-250-n300

checkpoint `/releases/mixv2/pilots/anchored/snapshots/step-250/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 93.0% | 0.131 | 0.025 | 0.144 | 127 | 145 | 72 |
| imdb | choice | 300 | 90.3% | 0.086 | 0.058 | 0.152 | 141 | 145 | 329 |
| hellaswag | choice | 300 | 79.7% | 0.108 | 0.056 | 0.296 | 132 | 147 | 127 |
| arc_easy | choice | 300 | 97.3% | 0.038 | 0.023 | 0.042 | 135 | 139 | 85 |
| arc_challenge | choice | 300 | 91.0% | 0.042 | 0.037 | 0.135 | 133 | 138 | 119 |
| boolq | noul | 300 | 85.7% | 0.019 | 0.125 | 0.202 | 134 | 143 | 201 |
| agnews | choice | 300 | 82.3% | 0.105 | 0.139 | 0.287 | 128 | 141 | 157 |
| emotion | choice | 300 | 62.7% | 0.076 | 0.072 | 0.527 | 96 | 128 | 84 |
| banking77 | choice | 300 | 76.0% | 0.070 | 0.074 | 0.333 | 140 | 177 | 1052 |
| toxicchat | noul | 300 | 49.3% | 0.066 | 0.059 | 0.169 | 128 | 130 | 77 |
| prompt_injection | noul | 116 | 68.1% | 0.234 | 0.201 | 0.491 | 131 | 138 | 90 |
| sst5 | score | 300 | 47.3% | 0.171 | 0.096 | 0.687 | 128 | 132 | 92 |
| belebele_eng_Latn | choice | 300 | 92.0% | 0.080 | 0.044 | 0.123 | 136 | 150 | 211 |
| belebele_deu_Latn | choice | 300 | 90.0% | 0.073 | 0.045 | 0.154 | 134 | 145 | 240 |
| belebele_fra_Latn | choice | 300 | 89.7% | 0.068 | 0.030 | 0.162 | 131 | 133 | 263 |
| belebele_spa_Latn | choice | 300 | 88.0% | 0.068 | 0.058 | 0.192 | 133 | 134 | 239 |
| belebele_ita_Latn | choice | 300 | 89.3% | 0.071 | 0.028 | 0.177 | 128 | 132 | 255 |
| belebele_por_Latn | choice | 300 | 89.3% | 0.071 | 0.039 | 0.163 | 128 | 130 | 237 |
| belebele_rus_Cyrl | choice | 300 | 90.7% | 0.087 | 0.048 | 0.151 | 130 | 137 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 88.0% | 0.066 | 0.071 | 0.193 | 128 | 131 | 287 |
| belebele_arb_Arab | choice | 300 | 88.3% | 0.096 | 0.057 | 0.182 | 108 | 128 | 270 |
| belebele_hin_Deva | choice | 300 | 81.7% | 0.082 | 0.065 | 0.283 | 129 | 132 | 265 |
| belebele_zho_Hans | choice | 300 | 90.3% | 0.080 | 0.050 | 0.170 | 129 | 132 | 213 |
| belebele_jpn_Jpan | choice | 300 | 85.7% | 0.060 | 0.036 | 0.221 | 97 | 129 | 231 |
| **belebele (mean of 12)** | | | 88.6% | | | | | | |
| **Tier 1 mean** | | | 77.8% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.054 | 0.019 |
| noul | 716 | 0.058 | 0.026 |
| score | 300 | 0.171 | 0.096 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.784 |
| choice:2 | 0.387 |
| choice:3-5 | 0.748 |
| choice:6-26 | 0.921 |
| choice:27-255 | 1.285 |
| score:2 | 1.000 |
| score:3-5 | 1.619 |
| score:6-10 | 1.000 |
