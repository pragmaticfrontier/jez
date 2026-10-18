# gentle-step-250-n300

checkpoint `/releases/mixv2/pilots/gentle/snapshots/step-250/adapter` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 92.0% | 0.109 | 0.028 | 0.146 | 150 | 170 | 72 |
| imdb | choice | 300 | 85.7% | 0.041 | 0.077 | 0.192 | 161 | 164 | 329 |
| hellaswag | choice | 300 | 81.0% | 0.222 | 0.143 | 0.357 | 155 | 166 | 127 |
| arc_easy | choice | 300 | 97.3% | 0.046 | 0.019 | 0.055 | 149 | 163 | 85 |
| arc_challenge | choice | 300 | 88.0% | 0.069 | 0.044 | 0.153 | 154 | 155 | 119 |
| boolq | noul | 300 | 88.3% | 0.036 | 0.132 | 0.169 | 161 | 163 | 201 |
| agnews | choice | 300 | 82.3% | 0.096 | 0.144 | 0.274 | 151 | 153 | 157 |
| emotion | choice | 300 | 61.7% | 0.077 | 0.072 | 0.538 | 110 | 155 | 84 |
| banking77 | choice | 300 | 73.7% | 0.130 | 0.076 | 0.388 | 155 | 208 | 1052 |
| toxicchat | noul | 300 | 50.0% | 0.084 | 0.078 | 0.185 | 150 | 154 | 77 |
| prompt_injection | noul | 116 | 66.4% | 0.254 | 0.219 | 0.529 | 154 | 163 | 90 |
| sst5 | score | 300 | 47.0% | 0.265 | 0.062 | 0.791 | 152 | 186 | 92 |
| belebele_eng_Latn | choice | 300 | 91.3% | 0.123 | 0.041 | 0.165 | 153 | 157 | 211 |
| belebele_deu_Latn | choice | 300 | 89.0% | 0.122 | 0.062 | 0.187 | 157 | 163 | 240 |
| belebele_fra_Latn | choice | 300 | 89.0% | 0.113 | 0.040 | 0.200 | 159 | 160 | 263 |
| belebele_spa_Latn | choice | 300 | 87.7% | 0.110 | 0.064 | 0.228 | 156 | 160 | 239 |
| belebele_ita_Latn | choice | 300 | 87.7% | 0.117 | 0.056 | 0.219 | 154 | 157 | 255 |
| belebele_por_Latn | choice | 300 | 89.0% | 0.121 | 0.038 | 0.197 | 150 | 153 | 237 |
| belebele_rus_Cyrl | choice | 300 | 91.0% | 0.130 | 0.052 | 0.185 | 149 | 157 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 86.0% | 0.104 | 0.064 | 0.230 | 145 | 151 | 287 |
| belebele_arb_Arab | choice | 300 | 86.7% | 0.120 | 0.067 | 0.222 | 116 | 152 | 270 |
| belebele_hin_Deva | choice | 300 | 82.7% | 0.147 | 0.084 | 0.312 | 151 | 152 | 265 |
| belebele_zho_Hans | choice | 300 | 89.7% | 0.135 | 0.065 | 0.201 | 150 | 152 | 213 |
| belebele_jpn_Jpan | choice | 300 | 86.3% | 0.123 | 0.044 | 0.244 | 114 | 152 | 231 |
| **belebele (mean of 12)** | | | 88.0% | | | | | | |
| **Tier 1 mean** | | | 77.0% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.086 | 0.028 |
| noul | 716 | 0.061 | 0.019 |
| score | 300 | 0.265 | 0.062 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.882 |
| choice:2 | 0.525 |
| choice:3-5 | 0.635 |
| choice:6-26 | 0.874 |
| choice:27-255 | 1.396 |
| score:2 | 1.000 |
| score:3-5 | 2.534 |
| score:6-10 | 1.000 |
