# gentle-step-500-n300

checkpoint `/releases/mixv2/pilots/gentle/snapshots/step-500/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 89.7% | 0.079 | 0.030 | 0.151 | 146 | 165 | 72 |
| imdb | choice | 300 | 85.7% | 0.025 | 0.102 | 0.192 | 158 | 159 | 329 |
| hellaswag | choice | 300 | 72.3% | 0.224 | 0.178 | 0.461 | 157 | 165 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.055 | 0.028 | 0.057 | 149 | 150 | 85 |
| arc_challenge | choice | 300 | 90.7% | 0.071 | 0.043 | 0.149 | 155 | 171 | 119 |
| boolq | noul | 300 | 88.3% | 0.030 | 0.156 | 0.166 | 159 | 165 | 201 |
| agnews | choice | 300 | 83.0% | 0.101 | 0.156 | 0.281 | 154 | 156 | 157 |
| emotion | choice | 300 | 64.3% | 0.081 | 0.047 | 0.512 | 117 | 159 | 84 |
| banking77 | choice | 300 | 71.7% | 0.059 | 0.096 | 0.404 | 155 | 203 | 1052 |
| toxicchat | noul | 300 | 27.1% | 0.122 | 0.100 | 0.250 | 149 | 155 | 77 |
| prompt_injection | noul | 116 | 58.6% | 0.354 | 0.236 | 0.692 | 150 | 155 | 90 |
| sst5 | score | 300 | 43.7% | 0.197 | 0.058 | 0.722 | 154 | 157 | 92 |
| belebele_eng_Latn | choice | 300 | 90.7% | 0.099 | 0.034 | 0.173 | 161 | 165 | 211 |
| belebele_deu_Latn | choice | 300 | 90.0% | 0.101 | 0.050 | 0.177 | 160 | 179 | 240 |
| belebele_fra_Latn | choice | 300 | 90.0% | 0.097 | 0.047 | 0.190 | 154 | 159 | 263 |
| belebele_spa_Latn | choice | 300 | 86.3% | 0.077 | 0.066 | 0.226 | 159 | 171 | 239 |
| belebele_ita_Latn | choice | 300 | 86.0% | 0.080 | 0.046 | 0.210 | 157 | 163 | 255 |
| belebele_por_Latn | choice | 300 | 87.7% | 0.085 | 0.024 | 0.204 | 148 | 153 | 237 |
| belebele_rus_Cyrl | choice | 300 | 89.7% | 0.098 | 0.056 | 0.181 | 149 | 162 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 86.0% | 0.087 | 0.048 | 0.224 | 150 | 152 | 287 |
| belebele_arb_Arab | choice | 300 | 85.7% | 0.082 | 0.049 | 0.224 | 118 | 155 | 270 |
| belebele_hin_Deva | choice | 300 | 82.3% | 0.111 | 0.071 | 0.303 | 158 | 160 | 265 |
| belebele_zho_Hans | choice | 300 | 86.0% | 0.070 | 0.048 | 0.210 | 154 | 156 | 213 |
| belebele_jpn_Jpan | choice | 300 | 85.7% | 0.074 | 0.045 | 0.234 | 118 | 159 | 231 |
| **belebele (mean of 12)** | | | 87.2% | | | | | | |
| **Tier 1 mean** | | | 73.8% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.068 | 0.029 |
| noul | 716 | 0.112 | 0.048 |
| score | 300 | 0.197 | 0.058 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 2.709 |
| choice:2 | 0.632 |
| choice:3-5 | 0.702 |
| choice:6-26 | 0.990 |
| choice:27-255 | 1.166 |
| score:2 | 1.000 |
| score:3-5 | 1.649 |
| score:6-10 | 1.000 |
