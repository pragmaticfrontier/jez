# gemma-4-E4B-it-zeroshot-n300

checkpoint `google/gemma-4-E4B-it` · backend torch on cuda · 300 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 300 | 94.3% | 0.044 | 0.040 | 0.097 | 164 | 185 | 72 |
| imdb | choice | 300 | 93.7% | 0.062 | 0.035 | 0.124 | 179 | 1039 | 329 |
| hellaswag | choice | 300 | 82.3% | 0.122 | 0.051 | 0.287 | 172 | 189 | 127 |
| arc_easy | choice | 300 | 97.7% | 0.018 | 0.027 | 0.036 | 168 | 173 | 85 |
| arc_challenge | choice | 300 | 90.7% | 0.061 | 0.041 | 0.144 | 173 | 181 | 119 |
| boolq | noul | 300 | 81.7% | 0.157 | 0.064 | 0.321 | 182 | 187 | 201 |
| agnews | choice | 300 | 81.7% | 0.176 | 0.174 | 0.354 | 169 | 170 | 157 |
| emotion | choice | 300 | 63.3% | 0.338 | 0.080 | 0.684 | 127 | 174 | 84 |
| banking77 | choice | 300 | 74.3% | 0.190 | 0.074 | 0.418 | 164 | 217 | 1052 |
| toxicchat | noul | 300 | 53.3% | 0.104 | 0.070 | 0.211 | 171 | 186 | 77 |
| prompt_injection | noul | 116 | 73.3% | 0.247 | 0.098 | 0.472 | 167 | 1016 | 90 |
| sst5 | score | 300 | 46.3% | 0.416 | 0.097 | 0.892 | 163 | 166 | 92 |
| belebele_eng_Latn | choice | 300 | 91.7% | 0.060 | 0.038 | 0.139 | 173 | 174 | 211 |
| belebele_deu_Latn | choice | 300 | 89.3% | 0.074 | 0.052 | 0.175 | 174 | 183 | 240 |
| belebele_fra_Latn | choice | 300 | 88.3% | 0.093 | 0.033 | 0.205 | 170 | 172 | 263 |
| belebele_spa_Latn | choice | 300 | 88.0% | 0.086 | 0.049 | 0.206 | 171 | 196 | 239 |
| belebele_ita_Latn | choice | 300 | 90.0% | 0.076 | 0.036 | 0.179 | 170 | 174 | 255 |
| belebele_por_Latn | choice | 300 | 89.0% | 0.080 | 0.054 | 0.179 | 170 | 173 | 237 |
| belebele_rus_Cyrl | choice | 300 | 89.3% | 0.077 | 0.050 | 0.167 | 171 | 181 | 263 |
| belebele_ukr_Cyrl | choice | 300 | 86.0% | 0.100 | 0.059 | 0.229 | 170 | 173 | 287 |
| belebele_arb_Arab | choice | 300 | 87.0% | 0.096 | 0.052 | 0.208 | 131 | 193 | 270 |
| belebele_hin_Deva | choice | 300 | 78.7% | 0.151 | 0.035 | 0.353 | 169 | 175 | 265 |
| belebele_zho_Hans | choice | 300 | 88.0% | 0.085 | 0.055 | 0.188 | 171 | 176 | 213 |
| belebele_jpn_Jpan | choice | 300 | 85.3% | 0.108 | 0.066 | 0.262 | 128 | 173 | 231 |
| **belebele (mean of 12)** | | | 87.6% | | | | | | |
| **Tier 1 mean** | | | 78.5% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 6000 | 0.101 | 0.015 |
| noul | 716 | 0.148 | 0.042 |
| score | 300 | 0.416 | 0.097 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 5.974 |
| choice:2 | 3.471 |
| choice:3-5 | 2.931 |
| choice:6-26 | 5.485 |
| choice:27-255 | 3.016 |
| score:2 | 1.000 |
| score:3-5 | 6.180 |
| score:6-10 | 1.000 |
