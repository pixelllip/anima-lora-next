# OCR eval — `ovis3_lora_micro` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 164 s for 5117 crops (31.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1850 | 72.3 | 0.868 | 78.9 % | 36 |
| speech | 2559 | 2189 | 85.5 | 0.981 | 97.3 % | 13 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 58.7 | 0.814 |
| square | 486 | 76.7 | 0.865 |
| vertical | 1448 | 76.7 | 0.892 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 69.3 | 0.708 |
| 2 | 1017 | 80.6 | 0.889 |
| 3 | 730 | 73.4 | 0.879 |
| 4 | 315 | 63.2 | 0.862 |
| 5 | 167 | 52.7 | 0.834 |
| 6 | 135 | 69.6 | 0.876 |
| 7 | 48 | 60.4 | 0.834 |
| 8+ | 58 | 39.7 | 0.727 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 037 1000a218 | じ~ん | う~ | 0.00 |
| ParaisoRoad 038 1000a219 | ガイイィィィィィィィィィィ | ザザザザザザザ | 0.00 |
| ParaisoRoad 037 1000a216 | がくっ | ガ | 0.00 |
| ParaisoRoad 057 1000a26a | ど | とっ! | 0.00 |
| ParaisoRoad 033 1000a1f1 | あ | ぱ | 0.00 |
| ParaisoRoad 040 1000a223 | ギ | キ | 0.00 |
| ParaisoRoad 043 1000a239 | だっ | ギィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィィ | 0.00 |
| ParaisoRoad 034 1000a1fc | キ | オ | 0.00 |
| ParaisoRoad 046 1000a24b | プシュゥ | パクォ | 0.00 |
| LoveHina_vol14 051 10006ed3 | ゴ | バ | 0.00 |
| ParaisoRoad 052 1000a25a | バリ | ぐり | 0.00 |
| ParaisoRoad 059 1000a276 | さ | ガ | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | トツ | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | バグ | 0.00 |
| ParaisoRoad 033 1000a1e9 | げっ | ガフ | 0.00 |
| ParaisoRoad 034 1000a1f9 | イイイイイイ | アアアアア | 0.00 |
| ParaisoRoad 035 1000a202 | ゴォォォォォォ | オオオオオ | 0.00 |
| ParaisoRoad 059 1000a27a | さわ | ガ | 0.00 |
| ParaisoRoad 055 1000a261 | う | ク | 0.00 |
| ParaisoRoad 059 1000a277 | あ | オ | 0.00 |
| SaladDays_vol18 084 1000b80b | リ | ク | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | ばあ | 0.00 |
| SaladDays_vol18 085 1000b823 | リ | ツ | 0.00 |
