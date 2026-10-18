# qwen3.5-0.8b-zeroshot-full

checkpoint `Qwen/Qwen3.5-0.8B` · backend torch on cuda · full splits · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 83.3% | 0.019 | 0.038 | 0.241 | 175 | 1138 | 79 |
| imdb | choice | 25000 | 89.2% | 0.019 | 0.004 | 0.160 | 192 | 216 | 312 |
| hellaswag | choice | 10042 | 43.6% | 0.026 | 0.039 | 0.678 | 217 | 229 | 147 |
| arc_easy | choice | 2376 | 80.0% | 0.122 | 0.204 | 0.307 | 154 | 1053 | 89 |
| arc_challenge | choice | 1172 | 59.8% | 0.044 | 0.114 | 0.520 | 164 | 204 | 114 |
| boolq | noul | 3270 | 74.8% | 0.043 | 0.088 | 0.342 | 218 | 231 | 170 |
| agnews | choice | 7600 | 73.9% | 0.105 | 0.045 | 0.374 | 191 | 222 | 171 |
| emotion | choice | 2000 | 54.9% | 0.211 | 0.017 | 0.661 | 147 | 218 | 87 |
| banking77 | choice | 3080 | 12.4% | 0.333 | 0.028 | 1.117 | 137 | 190 | 910 |
| toxicchat | noul | 5083 | 37.9% | 0.283 | 0.120 | 0.274 | 142 | 194 | 108 |
| prompt_injection | noul | 116 | 52.6% | 0.150 | 0.262 | 0.475 | 159 | 272 | 87 |
| sst5 | score | 2210 | 25.2% | 0.372 | 0.100 | 0.940 | 136 | 196 | 98 |
| belebele_eng_Latn | choice | 900 | 57.0% | 0.074 | 0.100 | 0.540 | 189 | 203 | 205 |
| belebele_deu_Latn | choice | 900 | 51.2% | 0.084 | 0.054 | 0.603 | 209 | 215 | 253 |
| belebele_fra_Latn | choice | 900 | 51.6% | 0.105 | 0.075 | 0.608 | 161 | 203 | 254 |
| belebele_spa_Latn | choice | 900 | 51.1% | 0.100 | 0.062 | 0.626 | 177 | 198 | 243 |
| belebele_ita_Latn | choice | 900 | 48.9% | 0.124 | 0.065 | 0.645 | 158 | 210 | 241 |
| belebele_por_Latn | choice | 900 | 50.4% | 0.108 | 0.071 | 0.617 | 164 | 329 | 236 |
| belebele_rus_Cyrl | choice | 900 | 50.0% | 0.101 | 0.096 | 0.620 | 157 | 357 | 258 |
| belebele_ukr_Cyrl | choice | 900 | 47.2% | 0.099 | 0.081 | 0.649 | 151 | 205 | 301 |
| belebele_arb_Arab | choice | 900 | 45.2% | 0.110 | 0.041 | 0.671 | 150 | 206 | 241 |
| belebele_hin_Deva | choice | 900 | 37.6% | 0.155 | 0.068 | 0.746 | 159 | 218 | 343 |
| belebele_zho_Hans | choice | 900 | 54.3% | 0.071 | 0.091 | 0.573 | 148 | 241 | 201 |
| belebele_jpn_Jpan | choice | 900 | 48.7% | 0.090 | 0.033 | 0.642 | 156 | 224 | 229 |
| **belebele (mean of 12)** | | | 49.4% | | | | | | |
| **Tier 1 mean** | | | 56.7% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 62942 | 0.044 | 0.019 |
| noul | 8469 | 0.185 | 0.057 |
| score | 2210 | 0.372 | 0.100 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 0.379 |
| choice:2 | 0.880 |
| choice:3-5 | 1.442 |
| choice:6-26 | 1.868 |
| choice:27-255 | 2.302 |
| score:2 | 1.000 |
| score:3-5 | 3.172 |
| score:6-10 | 1.000 |
