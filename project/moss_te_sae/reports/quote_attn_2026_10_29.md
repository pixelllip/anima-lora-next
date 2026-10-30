# A quoted string in the DiT

`src/quote_attn.py`, run 20261029-2236. Results in
`output/tests/quote_attn/20261029-2236/` (`grid.jpg`, columns base / swap_suffix /
no_quote_keys / quote_only / heat; per-arm PNGs, `heat_p*.png`, latents, `summary.json`,
`raw.pt`).

**Result.** A quoted string goes the way the tags go: the template gates, the caption rows
carry, and the suffix carries nothing. Unlike a tag, the string is read locally. Image
queries inside the lettering attend to the string's rows 7–20× more than queries outside
it, through the same few heads on every prompt (b13 h20, b21 h20, b24 h11, b26 h12), with
the most mass on blocks 19–21. Cutting the string's rows keeps the balloon, its place and
its script and loses the words. Keeping only the string's rows and the template loses the
character and keeps the words (3 of 4). The string's rows carry what to write. The rest of
the caption carries that a balloon exists and in which script.

## Setup

As in [template_attn](template_attn_2026_10_29.md) (Base, Euler 30 steps, CFG 3, seed 1,
1024², layout JSON, INT8 TE, caststation trigger), on two of the A/B prompts (Miku, Neru)
with `, speech bubble. She is saying "<s>".` appended, for `MY COMPUTER` (2 tokens) and
`지승현의 컴퓨터` (5 tokens: `지 승 현 의 ␣컴퓨터`). The layout converter keeps the
sentence in `summary.caption`. The string's tokens are their own key segment.

- `base`: every image query (4096) is scored on every block and step.
- `swap_suffix`: the 7 suffix rows come from the same character's prompt with the other
  string.
- `no_quote_keys`: image queries cannot attend to the string's rows.
- `quote_only`: image queries cannot attend to the other caption rows. The template and
  the string are kept.

The text region is the AnimeText box on the base render (21–128 image tokens, one box per
prompt).

## Where the image reads the string (base)

Mass on the string's rows, region queries against the rest, mean over the 4 prompts, all
steps:

| blocks | region | rest | ratio | top head (region) |
|---|---|---|---|---|
| 0–7 | 0.0001 | 0.0001 | ~1.5 | ≤ 0.006 |
| 8–13 | 0.0032 | 0.0002 | 11–16 | b13 h20, 0.17–0.35 |
| 14–18 | 0.0032 | 0.0006 | 4–6 | b16 h14 (b17 h13 on Neru EN), 0.05–0.09 |
| **19–21** | **0.020** | 0.0017 | 8–14 | **b21 h20, 0.19–0.39** |
| 22–25 | 0.0070 | 0.0008 | 4–11 | b24 h11, 0.13–0.28 |
| 26–35 | 0.0056 | 0.0004 | 8–21 | b26 h12, 0.07–0.22 |
| 36–40 | 0.0031 | 0.0001 | 6–47 | ≤ 0.022 |

- **The template gate does not depend on place.** The template takes 0.29 of region
  queries and 0.29 of the rest.
- **Per token, the string outweighs the other caption rows** inside the region: 3.3–7.3×
  (English 4.6–7.3, Korean 3.3–3.4).
- **The heads are fixed.** The same four heads top the region read on all four prompts,
  both scripts. b21 h20 puts up to 0.39 of a region query's mass on the 2 English tokens.
- `heat_p*.png` (blocks 8–41): the string mass sits on the balloon's lettering, with a
  faint speckle over the background elsewhere.

## Causal arms

Read glyph by glyph from the renders (bold = differs from the string):

| prompt | base | swap_suffix | no_quote_keys | quote_only |
|---|---|---|---|---|
| Miku EN | MY COMPUTER | MY COMPUTER | **NOW MANGIECH?** | MY COMPUTER, default figure |
| Miku KO | 지승현의 컴**푸더** | 지승현의 컴퓨터 | **비켠 칠치 가 / (쟁지 컴퓌므…)** | 지승현의 컴**뮤**터, default figure |
| Neru EN | MY COMPUTER | MY COMPUTER | MY COMPUTER**!** | MY COMPUTER, default figure |
| Neru KO | 지승현의 컴**뮤더** | 지승현의 컴**무**터 | **윈항에을 클엄택에 다미으** | scribble, no Hangul; a hooded figure |

Mean absolute pixel difference against base (0–255), text box / rest: swap_suffix 1.8–38 /
3.8–20, no_quote_keys 25–50 / 3.6–14, quote_only 40–52 / 63–79.

- **The suffix carries no string.** A Korean prompt with the English prompt's suffix writes
  Korean. The jamo that change (`퓨` ↔ `푸` / `뮤` / `무`, `터` ↔ `더`) change between base
  renders too. Base gets the Korean string at 2 wrong jamo per render.
- **Without the string's rows the words go and the rest stays.** The balloon stays where
  it was, in the same script, beside the same character (rest px 3.6–14): English
  pseudo-words for English, Hangul nonsense for Korean.
- **The cut leaks.** Neru EN still writes `MY COMPUTER!`, and Miku KO's small line has a
  `컴`. The rows after the string (`\"}}`, the suffix) read it in the causal TE, and the
  DiT takes some of that.
- **The string's rows alone write the words.** With the rest of the caption cut, the
  character becomes the default figure of [template_causal](template_causal_2026_10_29.md)'s
  `no_caption_keys`, and the balloon carries the string in 3 of 4 prompts. The two
  `quote_only` Korean renders differ (Miku: a near-exact line on the default figure; Neru:
  a hooded figure and a scribble), so the string's rows also carry some of the preceding
  caption. The two English ones are nearly the same image.

## Against bbox grounding

[`docs/moss/findings/bbox_grounding.md`](../../../docs/moss/findings/bbox_grounding.md)
found a layout item's content following its box mostly in blocks 26–35 (b14 h4 at every
σ), with the location built by the text refiner from the coordinate tokens. Here there is
no box: the DiT places the balloon itself, and the string is read most on blocks 19–21
through b21 h20. Whether a boxed text item is read through the same heads is not checked.

## Limits

One seed, four prompts, one detector box per prompt. Attention mass is not influence, and
the `no_*` arms cut every block at once. The leak on Neru EN means the string's rows are
the main route for the words, not the only one.
