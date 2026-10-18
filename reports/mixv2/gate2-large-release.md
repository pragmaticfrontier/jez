## Gate 2 against `gemma-4-E4B-it-zeroshot-n1000` (1000 items per task)

| task | gemma-4-E4B-it-zeroshot-n1000 | jez-large-1.0.0-anchored-n1000 |
|---|---|---|
| sst2 | 93.6 | 94.2 (+0.6) |
| imdb | 95.7 | 95.3 (-0.4) |
| hellaswag | 83.9 | 83.9 (+0.0) |
| arc_easy | 97.8 | 97.7 (-0.1) |
| arc_challenge | 88.8 | 89.5 (+0.7) |
| boolq | 83.1 | 84.5 (+1.4) |
| belebele | 88.2 | 90.1 (+1.9) |
| agnews | 83.7 | 84.5 (+0.8) |
| emotion | 59.2 | 58.9 (-0.3) |
| banking77 | 75.4 | 75.8 (+0.4) |
| toxicchat | 0.597 | 0.535 (-0.062) |
| prompt_injection | 73.3 | 66.4 (-6.9) |
| sst5 | 48.7 | 47.1 (-1.6) |
| **mean delta** | | **-0.3** |
| **worst task** | | **-6.9** |
| **gate 2** | | FAIL |

### jez-large-1.0.0-anchored-n1000: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | +0.6 | [-0.6, +1.7] |  |
| imdb | -0.4 | [-1.2, +0.3] |  |
| hellaswag | +0.0 | [-1.4, +1.6] |  |
| arc_easy | -0.1 | [-0.8, +0.6] |  |
| arc_challenge | +0.7 | [-0.5, +1.9] |  |
| boolq | +1.4 | [-0.1, +2.9] |  |
| agnews | +0.8 | [+0.0, +1.6] |  |
| emotion | -0.3 | [-1.6, +1.1] |  |
| banking77 | +0.4 | [-1.2, +1.9] |  |
| toxicchat | -0.1 | [-0.9, +0.8] |  |
| prompt_injection | -6.9 | [-12.1, -2.6] | significant |
| sst5 | -1.6 | [-5.0, +1.7] |  |
| belebele_eng_Latn | +1.6 | [+0.4, +2.7] | significant |
| belebele_deu_Latn | +0.9 | [-0.3, +2.1] |  |
| belebele_fra_Latn | +2.2 | [+1.1, +3.4] | significant |
| belebele_spa_Latn | +1.3 | [+0.1, +2.7] | significant |
| belebele_ita_Latn | +1.0 | [-0.2, +2.3] |  |
| belebele_por_Latn | +0.8 | [-0.3, +1.9] |  |
| belebele_rus_Cyrl | +3.1 | [+1.8, +4.6] | significant |
| belebele_ukr_Cyrl | +1.6 | [+0.3, +2.8] | significant |
| belebele_arb_Arab | +2.4 | [+1.0, +4.0] | significant |
| belebele_hin_Deva | +3.6 | [+2.1, +5.2] | significant |
| belebele_zho_Hans | +3.0 | [+1.6, +4.6] | significant |
| belebele_jpn_Jpan | +1.6 | [+0.2, +3.0] | significant |
