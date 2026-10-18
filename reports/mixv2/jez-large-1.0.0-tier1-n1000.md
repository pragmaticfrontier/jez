# jez-large-1.0.0-n1000

checkpoint `/releases/mixv2/jez-large-1.0.0/weights` · backend torch on cuda · 1000 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 94.2% | 0.024 | 0.024 | 0.095 | 170 | 211 | 78 |
| imdb | choice | 1000 | 94.8% | 0.005 | 0.042 | 0.078 | 178 | 280 | 315 |
| hellaswag | choice | 1000 | 80.3% | 0.026 | 0.035 | 0.286 | 174 | 192 | 145 |
| arc_easy | choice | 1000 | 95.7% | 0.021 | 0.025 | 0.063 | 128 | 188 | 88 |
| arc_challenge | choice | 1000 | 86.4% | 0.032 | 0.021 | 0.201 | 136 | 179 | 112 |
| boolq | noul | 1000 | 85.2% | 0.023 | 0.033 | 0.212 | 172 | 192 | 167 |
| agnews | choice | 1000 | 82.3% | 0.107 | 0.103 | 0.285 | 152 | 182 | 171 |
| emotion | choice | 1000 | 57.7% | 0.170 | 0.146 | 0.607 | 125 | 998 | 86 |
| banking77 | choice | 1000 | 70.8% | 0.076 | 0.049 | 0.408 | 167 | 224 | 1053 |
| toxicchat | noul | 1000 | 52.6% | 0.042 | 0.032 | 0.102 | 131 | 178 | 107 |
| prompt_injection | noul | 116 | 68.1% | 0.219 | 0.247 | 0.470 | 130 | 180 | 88 |
| sst5 | score | 1000 | 44.2% | 0.176 | 0.184 | 0.748 | 130 | 184 | 96 |
| belebele_eng_Latn | choice | 900 | 86.9% | 0.035 | 0.058 | 0.192 | 170 | 176 | 201 |
| belebele_deu_Latn | choice | 900 | 83.4% | 0.030 | 0.045 | 0.236 | 171 | 176 | 254 |
| belebele_fra_Latn | choice | 900 | 85.2% | 0.035 | 0.039 | 0.223 | 171 | 183 | 252 |
| belebele_spa_Latn | choice | 900 | 83.0% | 0.032 | 0.047 | 0.240 | 177 | 203 | 237 |
| belebele_ita_Latn | choice | 900 | 83.0% | 0.024 | 0.045 | 0.242 | 135 | 178 | 244 |
| belebele_por_Latn | choice | 900 | 83.7% | 0.022 | 0.037 | 0.230 | 138 | 1074 | 235 |
| belebele_rus_Cyrl | choice | 900 | 85.1% | 0.021 | 0.042 | 0.225 | 135 | 179 | 253 |
| belebele_ukr_Cyrl | choice | 900 | 83.7% | 0.019 | 0.025 | 0.235 | 139 | 203 | 278 |
| belebele_arb_Arab | choice | 900 | 82.7% | 0.030 | 0.032 | 0.249 | 139 | 193 | 258 |
| belebele_hin_Deva | choice | 900 | 73.6% | 0.048 | 0.052 | 0.378 | 136 | 184 | 245 |
| belebele_zho_Hans | choice | 900 | 85.2% | 0.029 | 0.023 | 0.216 | 148 | 203 | 215 |
| belebele_jpn_Jpan | choice | 900 | 81.8% | 0.028 | 0.026 | 0.276 | 137 | 185 | 229 |
| **belebele (mean of 12)** | | | 83.1% | | | | | | |
| **Tier 1 mean** | | | 76.6% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.019 | 0.016 |
| noul | 2116 | 0.036 | 0.022 |
| score | 1000 | 0.176 | 0.184 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.102 |
| choice:2 | 0.575 |
| choice:3-5 | 1.094 |
| choice:6-26 | 1.093 |
| choice:27-255 | 1.082 |
| score:2 | 1.000 |
| score:3-5 | 1.020 |
| score:6-10 | 1.185 |
