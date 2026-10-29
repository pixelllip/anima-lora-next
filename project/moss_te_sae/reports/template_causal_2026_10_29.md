# Causal check on the template rows

`src/template_causal.py`, run 20261029-2202. Results in
`output/tests/template_causal/20261029-2202/` (`grid.jpg`, columns base / swap_suffix /
no_caption_keys / no_template_keys; per-arm PNGs, latents, `summary.json`).

**Result.** The prompt reaches the image through the caption rows; the template rows
only gate. Another prompt's suffix rows change nothing visible and bring none of the
donor's tags. Cutting the image off from the caption rows (1–3 % of its mass) loses the
prompt: every prompt renders much the same default character. Cutting it off from the
template rows collapses every render into texture. This confirms
[template_attn](template_attn_2026_10_29.md): the template is a load-bearing,
prompt-constant gate, not a content channel.

## Setup

As in [template_attn](template_attn_2026_10_29.md): the 7 A/B eval prompts (layout JSON,
INT8 TE), Base, Euler 30 steps, CFG 3, seed 1, 1024². The reference arm is
template_attn's Base latent (same eager path). Every arm changes the cond branch only;
uncond stays the shipped empty prompt.

- `swap_suffix`: the 7 suffix rows of the TE output come from prompt `(i + 3) % 7`. The
  3 prefix rows are bit-identical across prompts (causal TE; max |Δ| 0).
- `no_caption_keys`: image queries cannot attend to the caption rows, in every block.
  Caption queries still see everything.
- `no_template_keys`: image queries cannot attend to the 10 template rows.

Anima Tagger scores: `own` = recall of the prompt's tags (6 tag prompts), `donor` =
recall of the donor's tags that the prompt lacks, `J` = Jaccard of the kept tag set
against the base render, `px` = mean absolute pixel difference against it (0–255). Scale
for `px`: two different prompts' base renders differ by 60–85.

| arm | own | donor | J | px |
|---|---|---|---|---|
| base | 1.00 (all 6) | 0.00–0.55 | 1 | 0 |
| swap_suffix | 1.00 (all 6) | **= base on every prompt** | 0.77–1.00 | 0.4–13.5 |
| no_caption_keys | 0.09–0.38 | 0.00–0.22 | 0.08–0.26 | 52–71 |
| no_template_keys | 0.00–0.15 | 0.00–0.14 | 0.00–0.07 | 64–102 |

## Reading

- **Suffix rows carry no usable content.** Swapping all 7 leaves composition, pose and
  colours in place. The largest change is Neru's sleeve (p1, px 13.5), and no donor tag
  appears that the base render did not already have. This holds although the suffix TE
  rows read the whole caption: the DiT does not take caption content from them.
- **Without the caption rows the prompt is gone.** The renders become one default
  figure: a girl in a blue jacket with arms spread, often a speech-bubble sketch beside
  her. Some attributes leak (black bob on p5, grey hair on p0 / p2), so the template rows
  carry a little prompt signal. Their cross-prompt spread was 0.01–0.03, and CFG 3 scales
  the cond − uncond difference that remains. Recall 0.09–0.38 is mostly generic tags
  (`1girl`, `looking at viewer`).
- **Without the template rows the model breaks.** All seven turn into texture noise, presumably because the
  mass that parked on the template now lands on the image and caption keys. This is
  the off-distribution failure `docs/moss/findings/attention_sink_chat_template.md`
  predicted for editing the template, shown here directly.

## What it means for LoRA triggers

The DiT reads a prompt through a narrow path, the 1–3 % of image attention on the
caption rows. The rest goes to a constant gate. A LoRA fitting a style can move the
image without using that path, which matches [lora_use](lora_use_2026_10_27.md)
(prompt-independent shift) and the mod16 comparison in
[template_attn](template_attn_2026_10_29.md). To make a trigger bind, the training signal
has to go through the caption-row reads. Untested ways to push it there: raise the
reading heads' caption mass, or contrast trigger-present against trigger-absent
prompts. A template-side or global adapter would not do it.

## Limits

One seed, seven prompts, Base only. The `no_*` arms cut every block at once. A per-block
cut would say which blocks carry the caption read (template_attn points to single heads
in blocks 8 / 19 / 35 / 37).
