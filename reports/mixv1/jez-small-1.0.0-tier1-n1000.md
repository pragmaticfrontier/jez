# jez-small-1.0.0-n1000

checkpoint `/releases/jez-small-1.0.0/weights` · backend torch on cuda · 1000 items per task · debias True

| task | type | n | accuracy/metric | ECE raw | ECE calibrated (odd half) | Brier | p50 ms | p99 ms | tokens |
|---|---|---|---|---|---|---|---|---|---|
| sst2 | choice | 872 | 85.6% | 0.012 | 0.052 | 0.219 | 136 | 818 | 79 |
| imdb | choice | 1000 | 86.1% | 0.021 | 0.036 | 0.192 | 146 | 222 | 312 |
| hellaswag | choice | 1000 | 44.0% | 0.104 | 0.120 | 0.687 | 140 | 155 | 147 |
| arc_easy | choice | 1000 | 77.3% | 0.043 | 0.061 | 0.321 | 101 | 838 | 89 |
| arc_challenge | choice | 1000 | 60.4% | 0.035 | 0.056 | 0.533 | 135 | 158 | 114 |
| boolq | noul | 1000 | 66.7% | 0.098 | 0.066 | 0.427 | 135 | 176 | 170 |
| agnews | choice | 1000 | 76.7% | 0.060 | 0.044 | 0.349 | 128 | 225 | 171 |
| emotion | choice | 1000 | 58.1% | 0.116 | 0.058 | 0.584 | 101 | 146 | 87 |
| banking77 | choice | 1000 | 59.5% | 0.131 | 0.214 | 0.572 | 108 | 145 | 910 |
| toxicchat | noul | 1000 | 17.5% | 0.069 | 0.058 | 0.139 | 118 | 159 | 108 |
| prompt_injection | noul | 116 | 48.3% | 0.504 | 0.494 | 0.990 | 106 | 152 | 87 |
| sst5 | score | 1000 | 30.4% | 0.142 | 0.112 | 0.806 | 102 | 140 | 98 |
| belebele_eng_Latn | choice | 900 | 73.1% | 0.121 | 0.130 | 0.424 | 139 | 145 | 205 |
| belebele_deu_Latn | choice | 900 | 58.7% | 0.057 | 0.068 | 0.548 | 133 | 197 | 253 |
| belebele_fra_Latn | choice | 900 | 65.3% | 0.101 | 0.103 | 0.506 | 139 | 204 | 254 |
| belebele_spa_Latn | choice | 900 | 60.8% | 0.077 | 0.065 | 0.540 | 137 | 851 | 243 |
| belebele_ita_Latn | choice | 900 | 61.0% | 0.066 | 0.059 | 0.533 | 115 | 152 | 241 |
| belebele_por_Latn | choice | 900 | 64.2% | 0.088 | 0.084 | 0.504 | 111 | 149 | 236 |
| belebele_rus_Cyrl | choice | 900 | 59.9% | 0.049 | 0.072 | 0.541 | 109 | 166 | 258 |
| belebele_ukr_Cyrl | choice | 900 | 54.0% | 0.046 | 0.046 | 0.605 | 110 | 188 | 301 |
| belebele_arb_Arab | choice | 900 | 53.0% | 0.048 | 0.054 | 0.607 | 102 | 143 | 241 |
| belebele_hin_Deva | choice | 900 | 42.4% | 0.047 | 0.061 | 0.706 | 129 | 215 | 343 |
| belebele_zho_Hans | choice | 900 | 67.1% | 0.080 | 0.089 | 0.466 | 100 | 133 | 201 |
| belebele_jpn_Jpan | choice | 900 | 53.7% | 0.045 | 0.068 | 0.594 | 105 | 191 | 229 |
| **belebele (mean of 12)** | | | 59.4% | | | | | | |
| **Tier 1 mean** | | | 59.2% | | | | | | |

## Pooled ECE by question type (Tier 3 gate: choice ≤ 0.05)

| type | answers | ECE raw | ECE calibrated (odd half) |
|---|---|---|---|
| choice | 18672 | 0.045 | 0.051 |
| noul | 2116 | 0.107 | 0.086 |
| score | 1000 | 0.142 | 0.112 |

## Temperatures (fitted on even-indexed items)

| bucket | T |
|---|---|
| noul | 1.260 |
| choice:2 | 0.761 |
| choice:3-5 | 1.066 |
| choice:6-26 | 1.194 |
| choice:27-255 | 1.177 |
| score:2 | 1.000 |
| score:3-5 | 1.222 |
| score:6-10 | 1.386 |
