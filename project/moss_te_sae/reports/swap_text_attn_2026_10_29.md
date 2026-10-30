# The swap_text LoRA's string read

`src/swap_text_attn.py`, run 20261029-2324. Results in
`output/tests/swap_text_attn/20261029-2324/` (`crops.png`, columns reference / lora /
no_string_keys / heads_off; per-arm PNGs, latents, `summary.json`, `acc_<page>.pt`).

**Result.** The swap_text LoRA reads each item's string the way Base reads a quoted
string in t2i ([quote_attn](quote_attn_2026_10_29.md)). Image queries in a box attend to
their own item's string 9× more per token than to the other items' strings (16–19× on
blocks 19–25), and the read peaks on blocks 19–21. The four t2i string heads rank near the
top again. Cutting every string key loses the words on all 11 items and keeps the place,
the JA removal, the script and the line layout. Cutting only the four t2i heads changes
almost nothing, so the read is spread over more heads than those four.

## Setup

The four held-out pages of `output/tests/moss_i2i_swap_text_region/` (en 2, ko 2; 11 text
items), with their staged prompts and JA references (full size, isolated,
`ref_timestep 0`). The LoRA is `moss_i2i_swap_text_en200_ko800_rw2_mod16_ema5e5`, rendered
with `scripts/moss/generate.py`'s defaults (ER-SDE 50 steps, Spectrum, CFG 3, uncond skip 22,
seed 0, INT8 TE), eager. The `lora` arm reproduces the shipped renders at px 0.6–1.2
(0–255). Every arm changes the cond branch only.

- `lora`: on every real forward, the queries in each item's box and 1024 queries outside
  every box are scored against every key: image, template, the rest of the caption, each
  item's string (its `"text"` value) and bbox list, the cached reference, and the
  reference token at the query's own cell (`copy`).
- `no_string_keys`: image queries cannot attend to any item's string tokens.
- `heads_off`: only b13 h20, b21 h20, b24 h11 and b26 h12 lose the string keys.

A negative coordinate merges with the key's `":[` into one token, which then counts as
coordinate.

## Where a box reads (lora)

Mean over the 4 pages and the items, box queries:

| blocks | own string | own / other string, per token | own coord | reference | template |
|---|---|---|---|---|---|
| 0–7 | 0.0002 | 1.2× | 0.004 | **0.29** | 0.20 |
| 8–13 | 0.005 | 5.7× | 0.003 | 0.13 | 0.30 |
| 14–18 | 0.006 | 3.3× | 0.014 | 0.09 | 0.27 |
| **19–21** | **0.042** | **15.8×** | 0.009 | 0.03 | 0.42 |
| 22–25 | 0.019 | 18.6× | 0.006 | 0.02 | 0.32 |
| 26–35 | 0.022 | 10.5× | 0.019 | 0.03 | 0.35 |
| 36–40 | 0.016 | 4.8× | 0.018 | 0.12 | 0.17 |

- **Each box is bound to its own string.** The other items' strings get 0.002–0.006 on blocks 8–40
  (all items together). Queries outside every box give the strings 0.0002–0.002. The box also
  reads its own coordinates more than any other item's (0.003–0.019 against
  0.0003–0.009 per item).
- **The blocks split the work.** Blocks 0–7 read the reference (0.29 of the mass, 0.015 on
  the own cell). Blocks 19–35 read the strings while the reference drops to 0.02–0.03.
  Blocks 36–40 go back to the reference (0.12, 0.03 on the own cell).
- **The t2i heads come back.** Ranked by own-string mass per page (top 8), on all 4 pages:
  b21 h20, b24 h11, b19 h6, b28 h0, b21 h3. Also b26 h12 and b19 h19 on 3, b13 h20 on 2.
- The template still takes 0.28 of the box queries' mass.

## Causal arms

Read glyph by glyph (✓ = as captioned):

| item | lora | no_string_keys | heads_off |
|---|---|---|---|
| Uwah… | ✓ | AWAN NU | ✓ |
| I'M GOING TO MOVE, OKAY? | ✓ | GIVN SOINING TO S: ROLA SKRE. | ✓ |
| UGH… OH SAKURA-CHAN… YOU'RE THE BEST… MMM… | ✓ | INNIA'S WHEP MENGHA IMAIY BHASN ♥ | `THE` dropped |
| YOU REALLY DRANK MY ESSENCE, HUH? | ✓ | UKT MRENY THE IMUY BOITT I ME PLUS! | ✓ |
| 어머? 처음 오시는 분인가요? | 분**이가**요 | 녀믜을 일기가는 쩜다요♡? | **처 처음** … 분**이나**요 |
| 저기… 이건 뭔가요? | ✓ | 벌가… 얼권 젤마요요? | 이**전** |
| 벌거 주기? (OCR misread) | 벌**겨** | 벌거 **수**기♡ | 벌**겨** |
| 싸… 싼다 | ✓ | 나… 만가 | ✓ |
| 좀…더 | ✓ | **출**…더 | ✓ |
| … 아타고씨 | ✓ | 장써 멋로3념 늑레리♥ | ✓ |
| 굉장해… 질 속이 정액으로 가득해… | ✓ | 췌슥레♥ 핫징우를 께븍레♥ | ✓ |

- **Without the string keys the words go and the rest stays.** Every balloon is
  re-lettered in place, the Japanese is gone, the script matches the caption's language,
  the line count and the final `?` mostly survive. What the strings carry is what to
  write. The place, the script and the removal come from the rest of the prompt and the
  reference.
- **Short strings leak**, as in quote_attn: `벌거 수기♡`, `출…더`, `SOINING TO`. The rows
  after a string read it in the causal TE.
- **Hearts come back from the reference.** Where the JA lettering had `♡`, the cut arm
  draws `♥` / `♡`, which the caption does not have.
- **The four t2i heads are not needed.** With them cut, the errors go from 2 to 4: `THE`
  dropped, `이건` → `이전` (an error the earlier swap_text runs made too, `docs/moss/i2i.md`),
  and a doubled `처`. With one seed that is not separable from noise.

## Limits

One seed, four pages, the LoRA only: no Base forward was scored on the same trajectory,
so this does not show whether the LoRA built the read or reuses Base's (Base reads t2i
strings on the same blocks and heads, quote_attn; Base binds a layout item's content to
its box on blocks 26–35, `docs/moss/findings/bbox_grounding.md`). Attention mass is not
influence, and the cut arms cut every block at once.
