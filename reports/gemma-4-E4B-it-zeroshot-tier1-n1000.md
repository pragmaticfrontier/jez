# gemma-4-E4B-it-zeroshot-n1000

checkpoint `google/gemma-4-E4B-it` · backend torch on cuda · 1000 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 93.6% | 0.052 | 0.022 | 0.115 | 193 | 1088 | 78 |
| imdb | choice | 1000 | 95.7% | 0.038 | 0.027 | 0.083 | 202 | 214 | 315 |
| hellaswag | choice | 1000 | 83.9% | 0.094 | 0.073 | 0.257 | 196 | 218 | 145 |
| arc_easy | choice | 1000 | 97.8% | 0.012 | 0.043 | 0.034 | 148 | 201 | 88 |
| arc_challenge | choice | 1000 | 88.8% | 0.077 | 0.041 | 0.180 | 152 | 200 | 112 |
| boolq | noul | 1000 | 83.1% | 0.152 | 0.036 | 0.316 | 193 | 216 | 167 |
| agnews | choice | 1000 | 83.7% | 0.151 | 0.113 | 0.313 | 174 | 204 | 171 |
| emotion | choice | 1000 | 59.2% | 0.362 | 0.059 | 0.755 | 146 | 996 | 86 |
| banking77 | choice | 1000 | 75.4% | 0.199 | 0.059 | 0.430 | 178 | 233 | 1053 |
| toxicchat | noul | 1000 | 59.7% | 0.050 | 0.042 | 0.104 | 159 | 213 | 107 |
| prompt_injection | noul | 116 | 73.3% | 0.247 | 0.131 | 0.472 | 145 | 206 | 88 |
| sst5 | score | 1000 | 48.7% | 0.406 | 0.038 | 0.876 | 155 | 209 | 96 |
| belebele_eng_Latn | choice | 900 | 92.2% | 0.057 | 0.037 | 0.128 | 188 | 206 | 201 |
| belebele_deu_Latn | choice | 900 | 91.1% | 0.058 | 0.031 | 0.150 | 193 | 198 | 254 |
| belebele_fra_Latn | choice | 900 | 90.2% | 0.075 | 0.036 | 0.166 | 197 | 206 | 252 |
| belebele_spa_Latn | choice | 900 | 89.0% | 0.076 | 0.029 | 0.188 | 191 | 206 | 237 |
| belebele_ita_Latn | choice | 900 | 89.1% | 0.084 | 0.021 | 0.185 | 150 | 195 | 244 |
| belebele_por_Latn | choice | 900 | 89.9% | 0.074 | 0.022 | 0.169 | 156 | 210 | 235 |
| belebele_rus_Cyrl | choice | 900 | 88.1% | 0.088 | 0.013 | 0.192 | 157 | 208 | 253 |
| belebele_ukr_Cyrl | choice | 900 | 87.9% | 0.092 | 0.029 | 0.207 | 162 | 215 | 278 |
| belebele_arb_Arab | choice | 900 | 88.1% | 0.080 | 0.025 | 0.195 | 172 | 236 | 258 |
| belebele_hin_Deva | choice | 900 | 78.0% | 0.165 | 0.042 | 0.369 | 160 | 223 | 245 |
| belebele_zho_Hans | choice | 900 | 88.9% | 0.076 | 0.058 | 0.178 | 159 | 217 | 215 |
| belebele_jpn_Jpan | choice | 900 | 85.8% | 0.109 | 0.041 | 0.247 | 157 | 201 | 229 |
| **belebele (mean of 12)** | | | 88.2% | | | | | | |
| **Tier 1 mean** | | | 79.3% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.101 | 0.010 |
| noul | 2116 | 0.106 | 0.018 |
| score | 1000 | 0.406 | 0.038 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 4.942 |
| choice:2 | 3.052 |
| choice:3-5 | 3.046 |
| choice:6-26 | 5.450 |
| choice:27-255 | 3.042 |
| score:2 | 1.000 |
| score:3-5 | 5.846 |
| score:6-10 | 1.000 |
