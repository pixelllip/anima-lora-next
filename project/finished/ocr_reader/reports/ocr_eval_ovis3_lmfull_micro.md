# OCR eval — `ovis3_lmfull_micro` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 141 s for 5117 crops (36.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1849 | 72.3 | 0.872 | 79.0 % | 27 |
| speech | 2559 | 2137 | 83.5 | 0.980 | 97.3 % | 7 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 61.4 | 0.832 |
| square | 486 | 73.0 | 0.851 |
| vertical | 1448 | 76.7 | 0.897 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 69.3 | 0.720 |
| 2 | 1017 | 81.9 | 0.891 |
| 3 | 730 | 71.5 | 0.884 |
| 4 | 315 | 61.6 | 0.849 |
| 5 | 167 | 53.9 | 0.854 |
| 6 | 135 | 74.8 | 0.907 |
| 7 | 48 | 60.4 | 0.829 |
| 8+ | 58 | 32.8 | 0.756 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 003 10006d6f | うーん | ラーン | 0.00 |
| SaladDays_vol18 085 1000b822 | リリ | シシ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | おお | 0.00 |
| MAD_STONE 050 10007050 | グワシィ | ブタッ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ん | 0.00 |
| SaladDays_vol18 085 1000b823 | リ | ツ | 0.00 |
| LoveHina_vol14 051 10006ed3 | ゴ | グ | 0.00 |
| LoveHina_vol14 046 10006eb9 | そ・・・ | と… | 0.00 |
| SaladDays_vol18 084 1000b810 | リリ | ツッ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バン | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 085 1000b82b | リリ | ツツ | 0.00 |
| ParaisoRoad 043 1000a239 | だっ | ギィ | 0.00 |
| ParaisoRoad 043 1000a234 | げ~ | ぐ~ | 0.00 |
| ParaisoRoad 035 1000a202 | ゴォォォォォォ | ドオオオオオ | 0.00 |
| ParaisoRoad 035 1000a201 | す~ | ず | 0.00 |
| ParaisoRoad 046 1000a24b | プシュゥ | ペクッ | 0.00 |
| ParaisoRoad 035 1000a1ff | パッ | バリ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ブー | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | トッ | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | ぐ~ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
