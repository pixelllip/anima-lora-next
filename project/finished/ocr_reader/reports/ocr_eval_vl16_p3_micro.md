# OCR eval — `vl16_p3_micro` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 133 s for 5117 crops (38.6 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1924 | 75.2 | 0.884 | 80.5 % | 20 |
| speech | 2559 | 2237 | 87.4 | 0.984 | 97.9 % | 16 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 63.8 | 0.841 |
| square | 486 | 75.9 | 0.864 |
| vertical | 1448 | 79.9 | 0.908 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 67.0 | 0.686 |
| 2 | 1017 | 84.6 | 0.908 |
| 3 | 730 | 74.8 | 0.895 |
| 4 | 315 | 66.3 | 0.868 |
| 5 | 167 | 60.5 | 0.867 |
| 6 | 135 | 77.0 | 0.903 |
| 7 | 48 | 54.2 | 0.851 |
| 8+ | 58 | 32.8 | 0.723 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SyabondamaKieta 092 1000c358 | ごしごし | じゃじゃじゃ | 0.00 |
| LoveHina_vol14 003 10006d70 | んにゃー | ヒげー | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | ドド | 0.00 |
| SyabondamaKieta 055 1000c2e7 | ァァ | アア | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | ゴー | 0.00 |
| MAD_STONE 050 10007050 | グワシィ | バァン | 0.00 |
| MAD_STONE 084 100070a4 | グイ | ブブ | 0.00 |
| MAD_STONE 086 100070ae | グッ | ブツ | 0.00 |
| LoveHina_vol14 066 10006f60 | みゅ? | あ゙? | 0.00 |
| LoveHina_vol14 063 10006f4b | イイ・・・ | ブブ… | 0.00 |
| SaladDays_vol18 085 1000b823 | リ | ツ | 0.00 |
| SaladDays_vol18 084 1000b810 | リリ | ツーツ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア… | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | イ | 0.00 |
| MukoukizuNoChonbo 049 100090d8 | は | ほ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | ガ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | バオオオ | 0.00 |
| LoveHina_vol14 059 10006f21 | カツーン | ガガッ | 0.00 |
| MukoukizuNoChonbo 041 10009084 | リー | ソー | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 050 1000b771 | ぐぅぅぅ | バラララ | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | やっ | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ト | 0.00 |
