# anchored-step-1000-n300

checkpoint `/releases/mixv2/pilots/anchored/snapshots/step-1000/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 93.7% | 0.081 | 0.031 | 0.116 | 171 | 193 | 72 |
| imdb | choice | 300 | 91.7% | 0.036 | 0.040 | 0.128 | 187 | 188 | 329 |
| hellaswag | choice | 300 | 80.7% | 0.164 | 0.113 | 0.328 | 181 | 193 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.033 | 0.018 | 0.040 | 173 | 180 | 85 |
| arc_challenge | choice | 300 | 91.7% | 0.045 | 0.026 | 0.125 | 179 | 185 | 119 |
| boolq | noul | 300 | 85.7% | 0.021 | 0.127 | 0.192 | 180 | 189 | 201 |
| agnews | choice | 300 | 83.0% | 0.103 | 0.164 | 0.286 | 172 | 174 | 157 |
| emotion | choice | 300 | 62.7% | 0.131 | 0.094 | 0.543 | 143 | 185 | 84 |
| banking77 | choice | 300 | 76.3% | 0.070 | 0.078 | 0.350 | 164 | 212 | 1052 |
| toxicchat | noul | 300 | 48.5% | 0.089 | 0.065 | 0.192 | 167 | 170 | 77 |
| prompt_injection | noul | 116 | 66.4% | 0.247 | 0.176 | 0.528 | 169 | 994 | 90 |
| sst5 | score | 300 | 47.0% | 0.133 | 0.074 | 0.667 | 166 | 169 | 92 |
| belebele_eng_Latn | choice | 300 | 93.7% | 0.093 | 0.044 | 0.135 | 179 | 182 | 211 |
| belebele_deu_Latn | choice | 300 | 89.7% | 0.076 | 0.048 | 0.160 | 187 | 204 | 240 |
| belebele_fra_Latn | choice | 300 | 90.7% | 0.076 | 0.059 | 0.171 | 181 | 182 | 263 |
| belebele_spa_Latn | choice | 300 | 90.3% | 0.087 | 0.087 | 0.196 | 173 | 176 | 239 |
| belebele_ita_Latn | choice | 300 | 89.7% | 0.080 | 0.053 | 0.180 | 174 | 181 | 255 |
| belebele_por_Latn | choice | 300 | 91.3% | 0.086 | 0.062 | 0.170 | 171 | 177 | 237 |
| belebele_rus_Cyrl | choice | 300 | 92.7% | 0.102 | 0.073 | 0.152 | 133 | 141 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 89.0% | 0.081 | 0.048 | 0.189 | 134 | 135 | 287 |
| belebele_arb_Arab | choice | 300 | 90.3% | 0.101 | 0.045 | 0.175 | 105 | 134 | 270 |
| belebele_hin_Deva | choice | 300 | 82.7% | 0.078 | 0.064 | 0.285 | 133 | 134 | 265 |
| belebele_zho_Hans | choice | 300 | 91.7% | 0.093 | 0.069 | 0.170 | 135 | 139 | 213 |
| belebele_jpn_Jpan | choice | 300 | 86.3% | 0.054 | 0.047 | 0.220 | 102 | 137 | 231 |
| **belebele (mean of 12)** | | | 89.8% | | | | | | |
| **Tier 1 mean** | | | 78.1% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.067 | 0.047 |
| noul | 716 | 0.075 | 0.031 |
| score | 300 | 0.133 | 0.074 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 2.188 |
| choice:2 | 0.587 |
| choice:3-5 | 0.727 |
| choice:6-26 | 0.746 |
| choice:27-255 | 1.280 |
| score:2 | 1.000 |
| score:3-5 | 1.633 |
| score:6-10 | 1.000 |
