# jez-large-1.0.0-anchored-n1000-tier1cal

checkpoint `/releases/mixv2/jez-large-1.0.0-anchored/weights` · backend torch on cuda · 1000 items per task · re-reported from `jez-large-1.0.0-anchored-n1000.json` with temperatures fitted on the even-indexed items of this run

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 94.2% | 0.089 | 0.019 | 0.114 | 169 | 1071 | 78 |
| imdb | choice | 1000 | 95.3% | 0.040 | 0.028 | 0.078 | 176 | 210 | 315 |
| hellaswag | choice | 1000 | 83.9% | 0.191 | 0.135 | 0.293 | 187 | 233 | 145 |
| arc_easy | choice | 1000 | 97.7% | 0.033 | 0.019 | 0.039 | 167 | 219 | 88 |
| arc_challenge | choice | 1000 | 89.5% | 0.023 | 0.019 | 0.151 | 150 | 195 | 112 |
| boolq | noul | 1000 | 84.5% | 0.030 | 0.056 | 0.209 | 164 | 185 | 167 |
| agnews | choice | 1000 | 84.5% | 0.081 | 0.110 | 0.246 | 160 | 222 | 171 |
| emotion | choice | 1000 | 58.9% | 0.124 | 0.039 | 0.571 | 126 | 932 | 86 |
| banking77 | choice | 1000 | 75.8% | 0.075 | 0.040 | 0.366 | 180 | 237 | 1053 |
| toxicchat | noul | 1000 | 53.5% | 0.041 | 0.027 | 0.095 | 152 | 230 | 107 |
| prompt_injection | noul | 116 | 66.4% | 0.247 | 0.206 | 0.528 | 150 | 209 | 88 |
| sst5 | score | 1000 | 47.1% | 0.116 | 0.037 | 0.660 | 153 | 214 | 96 |
| belebele_eng_Latn | choice | 900 | 93.8% | 0.092 | 0.058 | 0.130 | 205 | 222 | 201 |
| belebele_deu_Latn | choice | 900 | 92.0% | 0.096 | 0.041 | 0.154 | 184 | 212 | 254 |
| belebele_fra_Latn | choice | 900 | 92.4% | 0.095 | 0.055 | 0.150 | 182 | 194 | 252 |
| belebele_spa_Latn | choice | 900 | 90.3% | 0.087 | 0.052 | 0.181 | 208 | 233 | 237 |
| belebele_ita_Latn | choice | 900 | 90.1% | 0.076 | 0.043 | 0.174 | 161 | 215 | 244 |
| belebele_por_Latn | choice | 900 | 90.7% | 0.078 | 0.040 | 0.166 | 138 | 201 | 235 |
| belebele_rus_Cyrl | choice | 900 | 91.2% | 0.087 | 0.043 | 0.169 | 131 | 182 | 253 |
| belebele_ukr_Cyrl | choice | 900 | 89.4% | 0.072 | 0.032 | 0.177 | 137 | 177 | 278 |
| belebele_arb_Arab | choice | 900 | 90.6% | 0.096 | 0.053 | 0.174 | 146 | 189 | 258 |
| belebele_hin_Deva | choice | 900 | 81.6% | 0.077 | 0.046 | 0.298 | 137 | 164 | 245 |
| belebele_zho_Hans | choice | 900 | 91.9% | 0.093 | 0.057 | 0.160 | 148 | 193 | 215 |
| belebele_jpn_Jpan | choice | 900 | 87.3% | 0.071 | 0.044 | 0.217 | 132 | 187 | 229 |
| **belebele (mean of 12)** | | | 90.1% | | | | | | |
| **Tier 1 mean** | | | 78.6% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.067 | 0.035 |
| noul | 2116 | 0.045 | 0.014 |
| score | 1000 | 0.116 | 0.037 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.668 |
| choice:2 | 0.472 |
| choice:3-5 | 0.761 |
| choice:6-26 | 0.690 |
| choice:27-255 | 1.256 |
| score:2 | 1.000 |
| score:3-5 | 1.563 |
| score:6-10 | 1.000 |
