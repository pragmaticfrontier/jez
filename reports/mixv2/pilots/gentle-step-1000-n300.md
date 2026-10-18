# gentle-step-1000-n300

checkpoint `/releases/mixv2/pilots/gentle/snapshots/step-1000/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 91.0% | 0.086 | 0.028 | 0.136 | 144 | 163 | 72 |
| imdb | choice | 300 | 89.3% | 0.019 | 0.079 | 0.149 | 159 | 164 | 329 |
| hellaswag | choice | 300 | 70.3% | 0.175 | 0.145 | 0.448 | 153 | 168 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.040 | 0.031 | 0.052 | 149 | 175 | 85 |
| arc_challenge | choice | 300 | 90.7% | 0.071 | 0.066 | 0.132 | 155 | 162 | 119 |
| boolq | noul | 300 | 87.3% | 0.022 | 0.155 | 0.182 | 159 | 166 | 201 |
| agnews | choice | 300 | 83.3% | 0.103 | 0.141 | 0.270 | 152 | 153 | 157 |
| emotion | choice | 300 | 58.3% | 0.097 | 0.067 | 0.566 | 115 | 157 | 84 |
| banking77 | choice | 300 | 71.0% | 0.082 | 0.094 | 0.417 | 157 | 202 | 1052 |
| toxicchat | noul | 300 | 27.6% | 0.122 | 0.089 | 0.239 | 143 | 149 | 77 |
| prompt_injection | noul | 116 | 62.1% | 0.303 | 0.205 | 0.597 | 150 | 857 | 90 |
| sst5 | score | 300 | 40.7% | 0.235 | 0.058 | 0.758 | 147 | 168 | 92 |
| belebele_eng_Latn | choice | 300 | 85.0% | 0.067 | 0.063 | 0.220 | 161 | 164 | 211 |
| belebele_deu_Latn | choice | 300 | 83.7% | 0.070 | 0.050 | 0.239 | 153 | 164 | 240 |
| belebele_fra_Latn | choice | 300 | 85.7% | 0.080 | 0.057 | 0.237 | 155 | 159 | 263 |
| belebele_spa_Latn | choice | 300 | 82.0% | 0.077 | 0.065 | 0.268 | 153 | 156 | 239 |
| belebele_ita_Latn | choice | 300 | 82.7% | 0.069 | 0.073 | 0.254 | 152 | 159 | 255 |
| belebele_por_Latn | choice | 300 | 83.3% | 0.088 | 0.048 | 0.245 | 155 | 157 | 237 |
| belebele_rus_Cyrl | choice | 300 | 84.7% | 0.088 | 0.046 | 0.219 | 156 | 159 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 82.0% | 0.070 | 0.059 | 0.269 | 153 | 159 | 287 |
| belebele_arb_Arab | choice | 300 | 84.3% | 0.100 | 0.081 | 0.258 | 118 | 156 | 270 |
| belebele_hin_Deva | choice | 300 | 77.7% | 0.066 | 0.067 | 0.331 | 155 | 159 | 265 |
| belebele_zho_Hans | choice | 300 | 83.0% | 0.068 | 0.053 | 0.237 | 154 | 161 | 213 |
| belebele_jpn_Jpan | choice | 300 | 82.7% | 0.066 | 0.062 | 0.255 | 113 | 155 | 231 |
| **belebele (mean of 12)** | | | 83.1% | | | | | | |
| **Tier 1 mean** | | | 73.3% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.054 | 0.024 |
| noul | 716 | 0.094 | 0.048 |
| score | 300 | 0.235 | 0.058 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 2.521 |
| choice:2 | 0.552 |
| choice:3-5 | 0.827 |
| choice:6-26 | 0.842 |
| choice:27-255 | 1.249 |
| score:2 | 1.000 |
| score:3-5 | 1.893 |
| score:6-10 | 1.000 |
