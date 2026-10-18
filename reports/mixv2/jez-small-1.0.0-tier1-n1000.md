# jez-small-1.0.0-n1000

checkpoint `/releases/mixv2/jez-small-1.0.0/weights` · backend torch on cuda · 1000 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 81.3% | 0.069 | 0.114 | 0.267 | 199 | 211 | 79 |
| imdb | choice | 1000 | 86.9% | 0.016 | 0.034 | 0.187 | 211 | 219 | 312 |
| hellaswag | choice | 1000 | 46.2% | 0.067 | 0.101 | 0.655 | 201 | 216 | 147 |
| arc_easy | choice | 1000 | 79.5% | 0.060 | 0.100 | 0.298 | 150 | 1005 | 89 |
| arc_challenge | choice | 1000 | 61.1% | 0.041 | 0.030 | 0.513 | 175 | 205 | 114 |
| boolq | noul | 1000 | 73.7% | 0.042 | 0.047 | 0.362 | 199 | 211 | 170 |
| agnews | choice | 1000 | 75.4% | 0.088 | 0.052 | 0.375 | 200 | 211 | 171 |
| emotion | choice | 1000 | 54.4% | 0.158 | 0.136 | 0.634 | 150 | 207 | 87 |
| banking77 | choice | 1000 | 60.6% | 0.086 | 0.065 | 0.544 | 156 | 217 | 910 |
| toxicchat | noul | 1000 | 49.0% | 0.036 | 0.026 | 0.117 | 154 | 208 | 108 |
| prompt_injection | noul | 116 | 49.1% | 0.398 | 0.406 | 0.739 | 153 | 210 | 87 |
| sst5 | score | 1000 | 34.5% | 0.105 | 0.065 | 0.786 | 157 | 208 | 98 |
| belebele_eng_Latn | choice | 900 | 70.7% | 0.068 | 0.109 | 0.414 | 205 | 221 | 205 |
| belebele_deu_Latn | choice | 900 | 61.6% | 0.045 | 0.095 | 0.508 | 202 | 213 | 253 |
| belebele_fra_Latn | choice | 900 | 64.0% | 0.044 | 0.073 | 0.501 | 182 | 209 | 254 |
| belebele_spa_Latn | choice | 900 | 60.7% | 0.047 | 0.064 | 0.523 | 203 | 205 | 243 |
| belebele_ita_Latn | choice | 900 | 60.9% | 0.036 | 0.068 | 0.521 | 158 | 212 | 241 |
| belebele_por_Latn | choice | 900 | 65.1% | 0.064 | 0.126 | 0.477 | 153 | 207 | 236 |
| belebele_rus_Cyrl | choice | 900 | 64.6% | 0.055 | 0.106 | 0.498 | 155 | 209 | 258 |
| belebele_ukr_Cyrl | choice | 900 | 55.7% | 0.041 | 0.075 | 0.584 | 159 | 214 | 301 |
| belebele_arb_Arab | choice | 900 | 52.8% | 0.023 | 0.050 | 0.603 | 155 | 204 | 241 |
| belebele_hin_Deva | choice | 900 | 42.6% | 0.053 | 0.054 | 0.696 | 161 | 215 | 343 |
| belebele_zho_Hans | choice | 900 | 67.9% | 0.065 | 0.107 | 0.446 | 156 | 210 | 201 |
| belebele_jpn_Jpan | choice | 900 | 55.2% | 0.021 | 0.059 | 0.586 | 154 | 206 | 229 |
| **belebele (mean of 12)** | | | 60.1% | | | | | | |
| **Tier 1 mean** | | | 62.5% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.025 | 0.048 |
| noul | 2116 | 0.058 | 0.053 |
| score | 1000 | 0.105 | 0.065 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.094 |
| choice:2 | 0.736 |
| choice:3-5 | 1.162 |
| choice:6-26 | 1.082 |
| choice:27-255 | 0.999 |
| score:2 | 1.000 |
| score:3-5 | 1.159 |
| score:6-10 | 1.254 |
