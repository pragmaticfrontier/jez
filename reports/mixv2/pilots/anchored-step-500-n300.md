# anchored-step-500-n300

checkpoint `/releases/mixv2/pilots/anchored/snapshots/step-500/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 93.3% | 0.101 | 0.039 | 0.130 | 127 | 143 | 72 |
| imdb | choice | 300 | 91.3% | 0.048 | 0.036 | 0.136 | 142 | 150 | 329 |
| hellaswag | choice | 300 | 79.7% | 0.192 | 0.121 | 0.352 | 132 | 143 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.051 | 0.024 | 0.047 | 128 | 133 | 85 |
| arc_challenge | choice | 300 | 92.0% | 0.073 | 0.024 | 0.131 | 132 | 136 | 119 |
| boolq | noul | 300 | 87.3% | 0.024 | 0.142 | 0.186 | 139 | 141 | 201 |
| agnews | choice | 300 | 82.7% | 0.106 | 0.157 | 0.286 | 128 | 129 | 157 |
| emotion | choice | 300 | 62.3% | 0.112 | 0.063 | 0.534 | 96 | 130 | 84 |
| banking77 | choice | 300 | 77.3% | 0.080 | 0.078 | 0.332 | 142 | 180 | 1052 |
| toxicchat | noul | 300 | 43.8% | 0.076 | 0.053 | 0.187 | 129 | 129 | 77 |
| prompt_injection | noul | 116 | 65.5% | 0.249 | 0.192 | 0.522 | 127 | 136 | 90 |
| sst5 | score | 300 | 47.3% | 0.127 | 0.091 | 0.665 | 128 | 130 | 92 |
| belebele_eng_Latn | choice | 300 | 92.3% | 0.099 | 0.041 | 0.137 | 133 | 136 | 211 |
| belebele_deu_Latn | choice | 300 | 91.0% | 0.098 | 0.037 | 0.162 | 141 | 152 | 240 |
| belebele_fra_Latn | choice | 300 | 90.0% | 0.086 | 0.050 | 0.171 | 137 | 140 | 263 |
| belebele_spa_Latn | choice | 300 | 89.3% | 0.087 | 0.060 | 0.198 | 131 | 144 | 239 |
| belebele_ita_Latn | choice | 300 | 90.0% | 0.099 | 0.042 | 0.184 | 127 | 130 | 255 |
| belebele_por_Latn | choice | 300 | 90.7% | 0.096 | 0.061 | 0.175 | 128 | 131 | 237 |
| belebele_rus_Cyrl | choice | 300 | 92.7% | 0.118 | 0.078 | 0.161 | 131 | 144 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 88.7% | 0.089 | 0.075 | 0.199 | 128 | 129 | 287 |
| belebele_arb_Arab | choice | 300 | 89.7% | 0.108 | 0.045 | 0.187 | 101 | 129 | 270 |
| belebele_hin_Deva | choice | 300 | 82.3% | 0.103 | 0.078 | 0.293 | 127 | 139 | 265 |
| belebele_zho_Hans | choice | 300 | 91.0% | 0.103 | 0.076 | 0.177 | 138 | 139 | 213 |
| belebele_jpn_Jpan | choice | 300 | 86.3% | 0.071 | 0.035 | 0.228 | 95 | 128 | 231 |
| **belebele (mean of 12)** | | | 89.5% | | | | | | |
| **Tier 1 mean** | | | 77.7% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.080 | 0.044 |
| noul | 716 | 0.064 | 0.037 |
| score | 300 | 0.127 | 0.091 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.987 |
| choice:2 | 0.513 |
| choice:3-5 | 0.665 |
| choice:6-26 | 0.799 |
| choice:27-255 | 1.212 |
| score:2 | 1.000 |
| score:3-5 | 1.576 |
| score:6-10 | 1.000 |
