# gentle-step-750-n300

checkpoint `/releases/mixv2/pilots/gentle/snapshots/step-750/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 91.0% | 0.080 | 0.027 | 0.140 | 150 | 169 | 72 |
| imdb | choice | 300 | 88.7% | 0.025 | 0.079 | 0.160 | 163 | 167 | 329 |
| hellaswag | choice | 300 | 71.7% | 0.158 | 0.132 | 0.419 | 151 | 166 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.035 | 0.028 | 0.051 | 149 | 152 | 85 |
| arc_challenge | choice | 300 | 90.7% | 0.063 | 0.050 | 0.130 | 149 | 157 | 119 |
| boolq | noul | 300 | 87.0% | 0.013 | 0.147 | 0.177 | 153 | 166 | 201 |
| agnews | choice | 300 | 83.0% | 0.104 | 0.132 | 0.271 | 149 | 152 | 157 |
| emotion | choice | 300 | 59.3% | 0.101 | 0.065 | 0.564 | 111 | 150 | 84 |
| banking77 | choice | 300 | 70.7% | 0.078 | 0.074 | 0.414 | 155 | 198 | 1052 |
| toxicchat | noul | 300 | 27.6% | 0.120 | 0.088 | 0.242 | 147 | 151 | 77 |
| prompt_injection | noul | 116 | 62.1% | 0.299 | 0.202 | 0.580 | 154 | 162 | 90 |
| sst5 | score | 300 | 41.3% | 0.244 | 0.037 | 0.768 | 152 | 157 | 92 |
| belebele_eng_Latn | choice | 300 | 88.0% | 0.069 | 0.042 | 0.196 | 161 | 163 | 211 |
| belebele_deu_Latn | choice | 300 | 85.0% | 0.082 | 0.050 | 0.215 | 158 | 166 | 240 |
| belebele_fra_Latn | choice | 300 | 86.7% | 0.080 | 0.056 | 0.213 | 151 | 156 | 263 |
| belebele_spa_Latn | choice | 300 | 84.0% | 0.067 | 0.047 | 0.246 | 156 | 158 | 239 |
| belebele_ita_Latn | choice | 300 | 84.7% | 0.066 | 0.061 | 0.230 | 154 | 156 | 255 |
| belebele_por_Latn | choice | 300 | 84.0% | 0.075 | 0.071 | 0.222 | 151 | 157 | 237 |
| belebele_rus_Cyrl | choice | 300 | 87.7% | 0.082 | 0.055 | 0.193 | 153 | 157 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 84.0% | 0.065 | 0.056 | 0.243 | 151 | 155 | 287 |
| belebele_arb_Arab | choice | 300 | 86.3% | 0.094 | 0.056 | 0.233 | 118 | 156 | 270 |
| belebele_hin_Deva | choice | 300 | 78.3% | 0.058 | 0.050 | 0.317 | 155 | 162 | 265 |
| belebele_zho_Hans | choice | 300 | 84.3% | 0.056 | 0.050 | 0.218 | 153 | 160 | 213 |
| belebele_jpn_Jpan | choice | 300 | 84.0% | 0.056 | 0.055 | 0.238 | 118 | 158 | 231 |
| **belebele (mean of 12)** | | | 84.8% | | | | | | |
| **Tier 1 mean** | | | 73.5% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.048 | 0.026 |
| noul | 716 | 0.096 | 0.051 |
| score | 300 | 0.244 | 0.037 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 2.573 |
| choice:2 | 0.562 |
| choice:3-5 | 0.829 |
| choice:6-26 | 0.899 |
| choice:27-255 | 1.270 |
| score:2 | 1.000 |
| score:3-5 | 1.974 |
| score:6-10 | 1.000 |
