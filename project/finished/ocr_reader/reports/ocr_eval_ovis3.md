# OCR eval — `ovis3` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 161 s for 5117 crops (31.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 224 | 8.8 | 0.230 | 13.6 % | 42 |
| speech | 2559 | 1171 | 45.8 | 0.836 | 74.3 % | 12 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 10.9 | 0.292 |
| square | 486 | 5.1 | 0.137 |
| vertical | 1448 | 9.0 | 0.234 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 13.6 | 0.148 |
| 2 | 1017 | 12.1 | 0.212 |
| 3 | 730 | 5.5 | 0.202 |
| 4 | 315 | 5.7 | 0.270 |
| 5 | 167 | 4.2 | 0.238 |
| 6 | 135 | 11.9 | 0.346 |
| 7 | 48 | 4.2 | 0.296 |
| 8+ | 58 | 10.3 | 0.449 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 009 1000b6a8 | ギャー | あの二人：
えー、
おー | 0.00 |
| SaladDays_vol18 008 1000b69a | ゴ | フッ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コッ | 0.00 |
| SaladDays_vol18 008 1000b69c | ゴゴ | フッ | 0.00 |
| SaladDays_vol18 008 1000b69d | ザワッ | H | 0.00 |
| SaladDays_vol18 008 1000b69e | ゴゴ | マッ | 0.00 |
| SaladDays_vol18 008 1000b69f | ゴッ | フン | 0.00 |
| SaladDays_vol18 008 1000b6a0 | ピクッ | Kun | 0.00 |
| SaladDays_vol18 008 1000b6a1 | ゴ | フッ | 0.00 |
| SaladDays_vol18 009 1000b6a4 | バキ | いい | 0.00 |
| SaladDays_vol18 009 1000b6a5 | バキ | パッ | 0.00 |
| SaladDays_vol18 009 1000b6a6 | ギャァァアアア | あんなのせい
だからねっ！！

——
トコトン
悪いよなあ、
ちっ
人：
たく、 | 0.00 |
| MukoukizuNoChonbo 042 10009095+10009096 | パチーン! | 10X-1! | 0.00 |
| SaladDays_vol18 009 1000b6a7 | ギャー | お | 0.00 |
| SaladDays_vol18 007 1000b699 | ザワ | 告 | 0.00 |
| MukoukizuNoChonbo 042 10009092 | パチ | 110 | 0.00 |
| SaladDays_vol18 009 1000b6a9 | ギャー | お | 0.00 |
| MukoukizuNoChonbo 042 10009090 | ぴょこ | うん | 0.00 |
| SaladDays_vol18 009 1000b6aa | ギャー | おー | 0.00 |
| MukoukizuNoChonbo 042 1000908e | ぴょこ | S | 0.00 |
| SaladDays_vol18 009 1000b6ab | バキ | はっ | 0.00 |
| MukoukizuNoChonbo 042 1000908b | ぴょこ | 0561 | 0.00 |
| MukoukizuNoChonbo 042 1000908a | ぴょこっ | 5000 | 0.00 |
| MukoukizuNoChonbo 042 10009089 | ぴょこ | ཀུ་མཱ | 0.00 |
| SaladDays_vol18 009 1000b6ac+1000b6ad | ギラッ | ひっ！？ | 0.00 |
