## Gate 2 against `gemma-4-E4B-it-zeroshot-n300` (300 items per task)

| task | gemma-4-E4B-it-zeroshot-n300 | anchored-step-250-n300 | anchored-step-500-n300 | anchored-step-750-n300 | anchored-step-1000-n300 | gentle-step-250-n300 | gentle-step-1000-n300 |
|---|---|---|---|---|---|---|---|
| sst2 | 94.3 | 93.0 (-1.3) | 93.3 (-1.0) | 93.7 (-0.7) | 93.7 (-0.7) | 92.0 (-2.3) | 91.0 (-3.3) |
| imdb | 93.7 | 90.3 (-3.3) | 91.3 (-2.3) | 91.7 (-2.0) | 91.7 (-2.0) | 85.7 (-8.0) | 89.3 (-4.3) |
| hellaswag | 82.3 | 79.7 (-2.7) | 79.7 (-2.7) | 79.3 (-3.0) | 80.7 (-1.7) | 81.0 (-1.3) | 70.3 (-12.0) |
| arc_easy | 97.7 | 97.3 (-0.3) | 97.7 (+0.0) | 98.0 (+0.3) | 97.7 (+0.0) | 97.3 (-0.3) | 97.7 (+0.0) |
| arc_challenge | 90.7 | 91.0 (+0.3) | 92.0 (+1.3) | 91.7 (+1.0) | 91.7 (+1.0) | 88.0 (-2.7) | 90.7 (+0.0) |
| boolq | 81.7 | 85.7 (+4.0) | 87.3 (+5.7) | 86.0 (+4.3) | 85.7 (+4.0) | 88.3 (+6.7) | 87.3 (+5.7) |
| belebele | 87.6 | 88.6 (+1.0) | 89.5 (+1.9) | 90.0 (+2.4) | 89.8 (+2.3) | 88.0 (+0.4) | 83.1 (-4.5) |
| agnews | 81.7 | 82.3 (+0.7) | 82.7 (+1.0) | 82.7 (+1.0) | 83.0 (+1.3) | 82.3 (+0.7) | 83.3 (+1.7) |
| emotion | 63.3 | 62.7 (-0.7) | 62.3 (-1.0) | 62.7 (-0.7) | 62.7 (-0.7) | 61.7 (-1.7) | 58.3 (-5.0) |
| banking77 | 74.3 | 76.0 (+1.7) | 77.3 (+3.0) | 77.3 (+3.0) | 76.3 (+2.0) | 73.7 (-0.7) | 71.0 (-3.3) |
| toxicchat | 0.533 | 0.493 (-0.041) | 0.438 (-0.096) | 0.413 (-0.121) | 0.485 (-0.048) | 0.500 (-0.033) | 0.276 (-0.257) |
| prompt_injection | 73.3 | 68.1 (-5.2) | 65.5 (-7.8) | 66.4 (-6.9) | 66.4 (-6.9) | 66.4 (-6.9) | 62.1 (-11.2) |
| sst5 | 46.3 | 47.3 (+1.0) | 47.3 (+1.0) | 47.3 (+1.0) | 47.0 (+0.7) | 47.0 (+0.7) | 40.7 (-5.7) |
| **mean delta** | | **-0.4** | **-0.1** | **-0.0** | **-0.1** | **-1.3** | **-3.5** |
| **worst task** | | **-5.2** | **-7.8** | **-6.9** | **-6.9** | **-8.0** | **-12.0** |
| **gate 2** | | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

### anchored-step-250-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -1.3 | [-3.3, +0.7] |  |
| imdb | -3.3 | [-5.7, -1.3] | significant |
| hellaswag | -2.7 | [-5.7, +0.0] |  |
| arc_easy | -0.3 | [-1.0, +0.0] |  |
| arc_challenge | +0.3 | [-1.7, +2.7] |  |
| boolq | +4.0 | [+1.3, +7.0] | significant |
| agnews | +0.7 | [-1.3, +2.7] |  |
| emotion | -0.7 | [-2.7, +1.3] |  |
| banking77 | +1.7 | [-2.0, +5.3] |  |
| toxicchat | +0.0 | [-2.0, +2.0] |  |
| prompt_injection | -5.2 | [-9.5, -1.7] | significant |
| sst5 | +1.0 | [-5.3, +8.0] |  |
| belebele_eng_Latn | +0.3 | [-1.7, +2.3] |  |
| belebele_deu_Latn | +0.7 | [-1.7, +3.0] |  |
| belebele_fra_Latn | +1.3 | [-0.3, +3.3] |  |
| belebele_spa_Latn | +0.0 | [-2.3, +2.3] |  |
| belebele_ita_Latn | -0.7 | [-2.7, +1.7] |  |
| belebele_por_Latn | +0.3 | [-1.7, +2.3] |  |
| belebele_rus_Cyrl | +1.3 | [-1.0, +3.7] |  |
| belebele_ukr_Cyrl | +2.0 | [-0.3, +4.3] |  |
| belebele_arb_Arab | +1.3 | [-1.3, +4.0] |  |
| belebele_hin_Deva | +3.0 | [+0.3, +5.7] | significant |
| belebele_zho_Hans | +2.3 | [-0.3, +5.0] |  |
| belebele_jpn_Jpan | +0.3 | [-2.0, +2.7] |  |

### anchored-step-500-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -1.0 | [-3.0, +1.3] |  |
| imdb | -2.3 | [-4.7, +0.0] |  |
| hellaswag | -2.7 | [-5.3, -0.3] | significant |
| arc_easy | +0.0 | [-1.0, +1.0] |  |
| arc_challenge | +1.3 | [-0.7, +3.3] |  |
| boolq | +5.7 | [+2.7, +9.0] | significant |
| agnews | +1.0 | [-0.7, +2.7] |  |
| emotion | -1.0 | [-3.3, +1.7] |  |
| banking77 | +3.0 | [-0.3, +6.3] |  |
| toxicchat | -0.3 | [-2.7, +2.3] |  |
| prompt_injection | -7.8 | [-12.9, -2.6] | significant |
| sst5 | +1.0 | [-6.0, +8.0] |  |
| belebele_eng_Latn | +0.7 | [-1.7, +3.0] |  |
| belebele_deu_Latn | +1.7 | [-0.3, +3.7] |  |
| belebele_fra_Latn | +1.7 | [-0.3, +3.7] |  |
| belebele_spa_Latn | +1.3 | [-1.3, +4.0] |  |
| belebele_ita_Latn | +0.0 | [-2.3, +2.3] |  |
| belebele_por_Latn | +1.7 | [-0.3, +4.0] |  |
| belebele_rus_Cyrl | +3.3 | [+1.3, +5.7] | significant |
| belebele_ukr_Cyrl | +2.7 | [+0.3, +5.0] | significant |
| belebele_arb_Arab | +2.7 | [-0.3, +5.7] |  |
| belebele_hin_Deva | +3.7 | [+0.7, +6.7] | significant |
| belebele_zho_Hans | +3.0 | [+0.3, +5.7] | significant |
| belebele_jpn_Jpan | +1.0 | [-1.3, +3.7] |  |

### anchored-step-750-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -0.7 | [-2.7, +1.3] |  |
| imdb | -2.0 | [-4.3, +0.3] |  |
| hellaswag | -3.0 | [-5.7, -0.3] | significant |
| arc_easy | +0.3 | [-0.7, +1.7] |  |
| arc_challenge | +1.0 | [-1.0, +3.3] |  |
| boolq | +4.3 | [+1.0, +7.7] | significant |
| agnews | +1.0 | [-0.7, +2.7] |  |
| emotion | -0.7 | [-2.7, +1.3] |  |
| banking77 | +3.0 | [-0.3, +6.3] |  |
| toxicchat | -0.7 | [-3.0, +2.0] |  |
| prompt_injection | -6.9 | [-12.1, -2.6] | significant |
| sst5 | +1.0 | [-5.7, +8.0] |  |
| belebele_eng_Latn | +1.3 | [-1.0, +3.3] |  |
| belebele_deu_Latn | +1.0 | [-1.3, +3.3] |  |
| belebele_fra_Latn | +2.3 | [+0.3, +4.3] | significant |
| belebele_spa_Latn | +2.7 | [+0.0, +5.3] |  |
| belebele_ita_Latn | +0.3 | [-2.0, +3.0] |  |
| belebele_por_Latn | +2.3 | [+0.3, +4.7] | significant |
| belebele_rus_Cyrl | +3.7 | [+1.7, +6.0] | significant |
| belebele_ukr_Cyrl | +3.0 | [+0.3, +5.7] | significant |
| belebele_arb_Arab | +3.7 | [+0.7, +6.7] | significant |
| belebele_hin_Deva | +3.7 | [+0.7, +7.0] | significant |
| belebele_zho_Hans | +4.0 | [+1.3, +6.7] | significant |
| belebele_jpn_Jpan | +1.0 | [-1.3, +3.7] |  |

### anchored-step-1000-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -0.7 | [-2.7, +1.3] |  |
| imdb | -2.0 | [-4.3, +0.3] |  |
| hellaswag | -1.7 | [-4.7, +1.0] |  |
| arc_easy | +0.0 | [-1.0, +1.0] |  |
| arc_challenge | +1.0 | [-1.0, +3.3] |  |
| boolq | +4.0 | [+0.7, +7.3] | significant |
| agnews | +1.3 | [-0.3, +3.0] |  |
| emotion | -0.7 | [-2.7, +1.3] |  |
| banking77 | +2.0 | [-1.3, +5.3] |  |
| toxicchat | +0.3 | [-2.0, +2.7] |  |
| prompt_injection | -6.9 | [-12.1, -2.6] | significant |
| sst5 | +0.7 | [-6.0, +7.7] |  |
| belebele_eng_Latn | +2.0 | [+0.0, +4.3] |  |
| belebele_deu_Latn | +0.3 | [-2.0, +2.7] |  |
| belebele_fra_Latn | +2.3 | [+0.3, +4.3] | significant |
| belebele_spa_Latn | +2.3 | [+0.0, +5.0] |  |
| belebele_ita_Latn | -0.3 | [-2.7, +2.0] |  |
| belebele_por_Latn | +2.3 | [+0.0, +4.7] |  |
| belebele_rus_Cyrl | +3.3 | [+1.3, +5.7] | significant |
| belebele_ukr_Cyrl | +3.0 | [+0.3, +5.7] | significant |
| belebele_arb_Arab | +3.3 | [+0.3, +6.3] | significant |
| belebele_hin_Deva | +4.0 | [+1.0, +7.3] | significant |
| belebele_zho_Hans | +3.7 | [+1.0, +6.3] | significant |
| belebele_jpn_Jpan | +1.0 | [-1.3, +3.3] |  |

### gentle-step-250-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -2.3 | [-5.0, +0.7] |  |
| imdb | -8.0 | [-11.3, -5.0] | significant |
| hellaswag | -1.3 | [-4.7, +2.0] |  |
| arc_easy | -0.3 | [-2.0, +1.0] |  |
| arc_challenge | -2.7 | [-5.3, +0.0] |  |
| boolq | +6.7 | [+2.7, +11.0] | significant |
| agnews | +0.7 | [-1.3, +2.7] |  |
| emotion | -1.7 | [-5.0, +2.0] |  |
| banking77 | -0.7 | [-4.7, +3.3] |  |
| toxicchat | -0.3 | [-2.7, +2.0] |  |
| prompt_injection | -6.9 | [-11.2, -2.6] | significant |
| sst5 | +0.7 | [-6.7, +8.3] |  |
| belebele_eng_Latn | -0.3 | [-3.0, +2.3] |  |
| belebele_deu_Latn | -0.3 | [-3.3, +2.7] |  |
| belebele_fra_Latn | +0.7 | [-2.0, +3.0] |  |
| belebele_spa_Latn | -0.3 | [-3.0, +2.7] |  |
| belebele_ita_Latn | -2.3 | [-5.0, +0.0] |  |
| belebele_por_Latn | +0.0 | [-2.7, +2.7] |  |
| belebele_rus_Cyrl | +1.7 | [-0.7, +4.0] |  |
| belebele_ukr_Cyrl | +0.0 | [-3.0, +3.0] |  |
| belebele_arb_Arab | -0.3 | [-3.7, +2.7] |  |
| belebele_hin_Deva | +4.0 | [+1.0, +7.3] | significant |
| belebele_zho_Hans | +1.7 | [-0.7, +4.3] |  |
| belebele_jpn_Jpan | +1.0 | [-2.0, +4.0] |  |

### gentle-step-1000-n300: paired 95% intervals of the accuracy delta

| task | delta | 95% interval | |
|---|---|---|---|
| sst2 | -3.3 | [-6.3, -0.3] | significant |
| imdb | -4.3 | [-7.0, -1.7] | significant |
| hellaswag | -12.0 | [-16.7, -7.3] | significant |
| arc_easy | +0.0 | [-1.7, +1.7] |  |
| arc_challenge | +0.0 | [-2.3, +2.3] |  |
| boolq | +5.7 | [+1.7, +10.0] | significant |
| agnews | +1.7 | [-1.0, +4.3] |  |
| emotion | -5.0 | [-9.7, -0.3] | significant |
| banking77 | -3.3 | [-7.3, +1.0] |  |
| toxicchat | -2.3 | [-5.3, +0.7] |  |
| prompt_injection | -11.2 | [-17.2, -5.2] | significant |
| sst5 | -5.7 | [-12.7, +2.0] |  |
| belebele_eng_Latn | -6.7 | [-10.7, -2.3] | significant |
| belebele_deu_Latn | -5.7 | [-10.0, -1.3] | significant |
| belebele_fra_Latn | -2.7 | [-6.3, +0.7] |  |
| belebele_spa_Latn | -6.0 | [-10.0, -2.0] | significant |
| belebele_ita_Latn | -7.3 | [-11.3, -3.3] | significant |
| belebele_por_Latn | -5.7 | [-9.3, -2.0] | significant |
| belebele_rus_Cyrl | -4.7 | [-8.7, -0.7] | significant |
| belebele_ukr_Cyrl | -4.0 | [-8.3, +0.3] |  |
| belebele_arb_Arab | -2.7 | [-6.7, +1.0] |  |
| belebele_hin_Deva | -1.0 | [-5.7, +3.7] |  |
| belebele_zho_Hans | -5.0 | [-9.0, -1.0] | significant |
| belebele_jpn_Jpan | -2.7 | [-6.3, +1.0] |  |
