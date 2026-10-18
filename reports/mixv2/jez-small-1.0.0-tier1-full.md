# jez-small-1.0.0-full

checkpoint `/releases/mixv2/jez-small-1.0.0/weights` · backend torch on cuda · full splits · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 81.3% | 0.069 | 0.114 | 0.267 | 136 | 859 | 79 |
| imdb | choice | 25000 | 88.1% | 0.015 | 0.022 | 0.173 | 146 | 168 | 312 |
| hellaswag | choice | 10042 | 46.9% | 0.069 | 0.091 | 0.655 | 144 | 156 | 147 |
| arc_easy | choice | 2376 | 77.7% | 0.036 | 0.070 | 0.322 | 104 | 851 | 89 |
| arc_challenge | choice | 1172 | 61.5% | 0.033 | 0.027 | 0.510 | 120 | 144 | 114 |
| boolq | noul | 3270 | 72.2% | 0.059 | 0.045 | 0.381 | 142 | 159 | 170 |
| agnews | choice | 7600 | 73.5% | 0.111 | 0.078 | 0.402 | 139 | 154 | 171 |
| emotion | choice | 2000 | 54.4% | 0.155 | 0.132 | 0.640 | 103 | 367 | 87 |
| banking77 | choice | 3080 | 60.1% | 0.095 | 0.099 | 0.570 | 108 | 152 | 910 |
| toxicchat | noul | 5083 | 55.4% | 0.024 | 0.020 | 0.093 | 104 | 158 | 108 |
| prompt_injection | noul | 116 | 49.1% | 0.398 | 0.406 | 0.739 | 104 | 145 | 87 |
| sst5 | score | 2210 | 32.5% | 0.121 | 0.092 | 0.793 | 107 | 145 | 98 |
| belebele_eng_Latn | choice | 900 | 70.7% | 0.068 | 0.109 | 0.414 | 141 | 148 | 205 |
| belebele_deu_Latn | choice | 900 | 61.6% | 0.045 | 0.095 | 0.508 | 139 | 146 | 253 |
| belebele_fra_Latn | choice | 900 | 64.0% | 0.044 | 0.073 | 0.501 | 135 | 168 | 254 |
| belebele_spa_Latn | choice | 900 | 60.7% | 0.047 | 0.064 | 0.523 | 161 | 167 | 243 |
| belebele_ita_Latn | choice | 900 | 60.9% | 0.036 | 0.068 | 0.521 | 119 | 162 | 241 |
| belebele_por_Latn | choice | 900 | 65.1% | 0.064 | 0.126 | 0.477 | 111 | 158 | 236 |
| belebele_rus_Cyrl | choice | 900 | 64.6% | 0.055 | 0.106 | 0.498 | 113 | 157 | 258 |
| belebele_ukr_Cyrl | choice | 900 | 55.7% | 0.041 | 0.075 | 0.584 | 113 | 169 | 301 |
| belebele_arb_Arab | choice | 900 | 52.8% | 0.023 | 0.050 | 0.603 | 104 | 144 | 241 |
| belebele_hin_Deva | choice | 900 | 42.6% | 0.053 | 0.054 | 0.696 | 115 | 151 | 343 |
| belebele_zho_Hans | choice | 900 | 67.9% | 0.065 | 0.107 | 0.446 | 105 | 158 | 201 |
| belebele_jpn_Jpan | choice | 900 | 55.2% | 0.021 | 0.059 | 0.586 | 105 | 150 | 229 |
| **belebele (mean of 12)** | | | 60.1% | | | | | | |
| **Tier 1 mean** | | | 62.5% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 62942 | 0.023 | 0.038 |
| noul | 8469 | 0.042 | 0.032 |
| score | 2210 | 0.121 | 0.092 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.094 |
| choice:2 | 0.736 |
| choice:3-5 | 1.162 |
| choice:6-26 | 1.082 |
| choice:27-255 | 0.999 |
| score:2 | 1.000 |
| score:3-5 | 1.159 |
| score:6-10 | 1.254 |
