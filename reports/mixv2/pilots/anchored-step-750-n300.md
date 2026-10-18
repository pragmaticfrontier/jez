# anchored-step-750-n300

checkpoint `/releases/mixv2/pilots/anchored/snapshots/step-750/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 93.7% | 0.103 | 0.035 | 0.126 | 128 | 144 | 72 |
| imdb | choice | 300 | 91.7% | 0.054 | 0.038 | 0.135 | 140 | 143 | 329 |
| hellaswag | choice | 300 | 79.3% | 0.169 | 0.136 | 0.342 | 131 | 142 | 127 |
| arc_easy | choice | 300 | 98.0% | 0.036 | 0.017 | 0.040 | 134 | 136 | 85 |
| arc_challenge | choice | 300 | 91.7% | 0.042 | 0.040 | 0.127 | 130 | 141 | 119 |
| boolq | noul | 300 | 86.0% | 0.029 | 0.134 | 0.190 | 134 | 147 | 201 |
| agnews | choice | 300 | 82.7% | 0.108 | 0.147 | 0.284 | 127 | 128 | 157 |
| emotion | choice | 300 | 62.7% | 0.174 | 0.106 | 0.563 | 94 | 126 | 84 |
| banking77 | choice | 300 | 77.3% | 0.073 | 0.075 | 0.341 | 140 | 177 | 1052 |
| toxicchat | noul | 300 | 41.3% | 0.083 | 0.048 | 0.193 | 126 | 128 | 77 |
| prompt_injection | noul | 116 | 66.4% | 0.244 | 0.167 | 0.518 | 131 | 143 | 90 |
| sst5 | score | 300 | 47.3% | 0.112 | 0.078 | 0.666 | 127 | 132 | 92 |
| belebele_eng_Latn | choice | 300 | 93.0% | 0.099 | 0.042 | 0.145 | 130 | 132 | 211 |
| belebele_deu_Latn | choice | 300 | 90.3% | 0.094 | 0.053 | 0.168 | 127 | 136 | 240 |
| belebele_fra_Latn | choice | 300 | 90.7% | 0.085 | 0.063 | 0.180 | 126 | 128 | 263 |
| belebele_spa_Latn | choice | 300 | 90.7% | 0.098 | 0.102 | 0.202 | 129 | 131 | 239 |
| belebele_ita_Latn | choice | 300 | 90.3% | 0.096 | 0.067 | 0.189 | 127 | 130 | 255 |
| belebele_por_Latn | choice | 300 | 91.3% | 0.101 | 0.083 | 0.179 | 126 | 129 | 237 |
| belebele_rus_Cyrl | choice | 300 | 93.0% | 0.110 | 0.086 | 0.157 | 127 | 134 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 89.0% | 0.084 | 0.075 | 0.195 | 126 | 128 | 287 |
| belebele_arb_Arab | choice | 300 | 90.7% | 0.109 | 0.060 | 0.182 | 98 | 129 | 270 |
| belebele_hin_Deva | choice | 300 | 82.3% | 0.081 | 0.067 | 0.292 | 126 | 128 | 265 |
| belebele_zho_Hans | choice | 300 | 92.0% | 0.102 | 0.077 | 0.174 | 128 | 131 | 213 |
| belebele_jpn_Jpan | choice | 300 | 86.3% | 0.062 | 0.070 | 0.224 | 101 | 135 | 231 |
| **belebele (mean of 12)** | | | 90.0% | | | | | | |
| **Tier 1 mean** | | | 77.5% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.077 | 0.055 |
| noul | 716 | 0.073 | 0.031 |
| score | 300 | 0.112 | 0.078 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 2.208 |
| choice:2 | 0.506 |
| choice:3-5 | 0.713 |
| choice:6-26 | 0.653 |
| choice:27-255 | 1.313 |
| score:2 | 1.000 |
| score:3-5 | 1.587 |
| score:6-10 | 1.000 |
