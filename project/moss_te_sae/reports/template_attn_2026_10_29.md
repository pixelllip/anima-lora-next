# Template rows in the DiT, Base and the channel LoRA

`src/template_attn.py` (capture), `src/template_attn_cmp.py` (arm comparison, CPU).
Runs 20261029-2131 (Base) and 20261029-2140 (Base + `caststation_ema097_mod16`, value rows
saved). Results in `output/tests/template_attn/{20261029-2131,lora_cmp_20261029}/`.

**Result.** This reproduces `docs/moss/findings/attention_sink_chat_template.md`
(2026-10-18) on the layout-JSON A/B eval prompts. The 10 template rows take about 95 % of
the image tokens' text attention, and caption rows get 1–3 %. The new read is per row
across prompts at the same latent: the four heaviest rows (`<|im_start|>` prefix, `user`,
`<|im_end|>`, `assistant`) carry the same value for every prompt. That settles the
finding's "the prefix is a pure register" hypothesis and extends it to the heavy suffix
rows. The channel LoRA (mod16) leaves attention where it was (within 0.003) and changes
values the same way for every prompt.

## Setup

The 7 A/B eval prompts (`bench/moss_ab/lora_ab.py` PROMPTS + `project/finished/moss_lora/composition_prompts.txt`,
caststation trigger, layout JSON, INT8 TE), Base DiT, Euler 30 steps, CFG 3, no Spectrum,
seed 1, 1024². Cond branch, every block and step; 1024 sampled image queries and every
caption query. A prompt is 3 prefix rows (`<|im_start|>user\n`), 39–69 caption rows and 7
suffix rows (`<|im_end|>\n<|im_start|>assistant\n<think>\n`). The finding's run used two
plain-text prompts × 4 seeds; this one uses layout JSON.

## Where the image reads the text (Base)

Image queries, mean over blocks (early / mid / late thirds of the steps):

| keys | mass | value contribution `‖Σ_seg p·v‖ / ‖Σ p·v‖` |
|---|---|---|
| image | 0.70 / 0.72 / 0.69 | 0.83 / 0.84 / 0.81 |
| prefix (3 rows) | 0.16 / 0.16 / 0.18 | 0.14 / 0.15 / 0.17 |
| caption body (~60 rows) | **0.026 / 0.013 / 0.010** | 0.08 / 0.05 / 0.04 |
| suffix (7 rows) | 0.12 / 0.11 / 0.13 | 0.15 / 0.14 / 0.15 |

- Template mass 0.27–0.30, inside the finding's 0.24–0.34. Per row, about 0.05 for a
  prefix row against 0.0003 for a caption row.
- Per template token (image queries): `user` 0.076, `assistant` 0.073, `<|im_start|>`
  (prefix) 0.067, `<|im_end|>` 0.034, prefix `\n` 0.019. The other five suffix rows get
  0.0005–0.007 each.
- The caption is read directly only by single heads in a few blocks: the max over heads
  is 0.17 in blocks 8 / 19 and 0.12–0.13 in blocks 18 / 35 / 37 / 40. The block mean is at
  most 0.04.
- Caption queries put 0.15–0.18 on the prefix, 0.14–0.15 on the suffix and 0.25 on the
  caption.
- The value contribution here is uncentred, so the shared offset inflates it. The
  finding's centred write (0.15–0.51, 1.13–2.1× the mass share) is the better measure of
  how much the template writes.

## What the template rows carry across prompts

Cross-prompt spread of each row's value, mean over blocks, `mean_p ‖v_p − v̄‖ / ‖v̄‖`. At
step 0 every prompt starts from the same x_T and σ, so the spread is the prompt alone:

| row | `<\|im_start\|>` | `user` | `\n` | `<\|im_end\|>` | `\n` | `<\|im_start\|>` | `assistant` | `\n` | `<think>` | `\n` | caption mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mass (image q.) | 0.067 | 0.076 | 0.019 | 0.034 | 0.0005 | 0.0013 | 0.073 | 0.0035 | 0.0007 | 0.0073 | ~0.02 total |
| spread, step 0 | 0.03 | 0.03 | 0.01 | 0.03 | 0.70 | 0.39 | 0.03 | 0.21 | 0.11 | 0.02 | 0.33 |
| spread, step 29 | 0.02 | 0.02 | 0.01 | 0.02 | 0.75 | 0.28 | 0.02 | 0.13 | 0.20 | 0.03 | 0.27 |

- **The heavy rows are prompt-constant.** In absolute terms (all saved steps) the four
  heavy rows vary by 1–2 (‖v̄‖ 59–98), and the caption-mean row by 33 (‖v̄‖ 101). They are
  a fixed per-head bias, not a channel for the caption.
- **The caption content in the suffix sits on the rows the image ignores.** The `\n`
  after `<|im_end|>` varies most (0.70), at 0.0005 of the mass.
- So the caption reaches the image tokens almost only through the 1–3 % of their mass
  on the caption rows.

## Channel LoRA (`caststation_ema097_mod16`) against Base

Same prompts, seed and sampler.

- **Attention does not move.** Every segment's mass and contribution stays within 0.003
  of Base (all three step thirds, image and caption queries), and so does every template
  token's mass. In the finding, lr1e4 / lr5e5 moved step-0 template mass 5.4 % / 3.0 %
  relative; mod16 is about 1 % (0.003 on ~0.28).
- The only change: the heads that read the caption directly in late blocks weaken. The
  max head in block 37 drops 0.126 → 0.075, block 35 0.125 → 0.095, block 36 0.069 →
  0.057. The block means stay put.
- **At step 0 the value change is small and the same for every prompt.** Relative
  change: 0.01–0.07 on template rows, 0.05 on the caption mean. The pairwise cosine of
  the per-prompt deltas is 0.60–0.77 on the heavy rows and 0.70 on the caption mean. This
  is the prompt-independent shift [lora_use](lora_use_2026_10_27.md) found from the
  output side, seen here on the values.
- mod16 makes the late suffix rows more alike across prompts (absolute spread of `\n` /
  `<think>` / `\n` 38 / 69 / 20 → 28 / 30 / 8), but they get little mass.

## Limits

One seed, seven prompts. Attention mass is not gradient influence. The causal check is
[template_causal](template_causal_2026_10_29.md).
