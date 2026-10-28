# `mod_rank` 16 on i2i: better preservation outside the edit, edit strength unchanged

Measured 2026-10-28 on the uncensor i2i task, MOSS-Image Base. Follows the t2i check in
`docs/moss/README.md` (`--mod_rank`: loss unchanged at r16 on caststation and ama_mitsuki).
Runs: `output/ckpt_moss_i2i/moss_i2i_uncensor_combined_muon_lr5e5_rw10to1{,_mod16}`;
renders: `output/tests/moss_i2i_uncensor/ab_mod16/{rw10to1,rw10to1_mod16}`, sheet
`compare_input_base_mod16.jpg` beside them.

## Setting

66 pairs (`post_image_dataset/moss/i2i_uncensor_combined`), fixed caption, full-size
isolated reference at `ref_timestep 0`, rank 32 / alpha 128, Muon lr 5e-5, 10 epochs = 660
updates, `region_weight` 10 → 1, `ema_window` 0.1, seed 0, swap 38 / budget 0.175. The arm
adds `--mod_rank 16` (`mod_alpha` default 90.5) and changes nothing else; the σ draws match
on all 660 steps, so the losses pair step by step. Cost: 3.59 → 3.78 s/step, +0.46 % file
size (~344k parameters), peak allocated 12.96 → 13.11 GB.

## Measured

Paired training loss, mod16 against the baseline (share of steps where mod16 is lower):

| steps | loss | edit region | rest (copied) |
|---|---|---|---|
| 1–110 | −1.5 % (87 %) | −0.7 % (71 %) | −2.7 % (86 %) |
| 111–220 | −6.8 % (95 %) | −4.1 % (84 %) | −17.1 % (99 %) |
| 221–330 | −3.5 % (95 %) | −2.2 % (85 %) | −9.1 % (97 %) |
| 331–440 | −2.4 % (94 %) | −1.4 % (75 %) | −8.2 % (98 %) |
| 441–550 | −1.5 % (86 %) | −0.0 % (70 %) | −7.7 % (95 %) |
| 551–660 | −2.6 % (89 %) | −0.8 % (71 %) | −5.5 % (95 %) |

Unlike t2i, the loss drops, and almost all of it on the copied tokens.

The modulation adapter: entropy rank 15.3 of 16, ‖ΔW‖ 2.6 % of the base
`shared_block_modulation` weight (block 40's q adapter: 1.05 %). Per chunk: FFN shift
12.9 %, attention gate 4.5 %, the other four 0.6–1.9 %. The block adapters come out as in
the baseline (mean ‖ΔW‖ 1.378 in both).

## Images

Six held-out pages, seed 0, ER-SDE 50 steps, CFG 3 (`output/tests/moss_i2i_uncensor/inputs`).

- **Preservation outside the edit: mod16 is visibly better, text most of all** (read by eye
  on the sheet).
- **Edit: a tie.** The edited area (ΔE > 8 against the input) is 0–1.8 % of the page in
  both arms, and inside it the arms differ by median ΔE 0.5–1.7.
- Page-level numbers miss the preservation gain: PSNR outside the edit moves ±0.5 dB either
  way per page, and ΔL / Δb outside it follow the same pattern in both arms. Small glyphs are
  too few pixels to move a page mean.

## Hypothesis (untested)

With `ref_timestep 0` the reference tokens take their modulation from the same
`shared_block_modulation` at t = 0 (`MOSSImageTransformer.ref_modulation`), so the adapter
can set the FFN shift for clean reference tokens apart from the noisy ones. That is a
handle the block adapters do not have, and it fits a gain that lands on the copied tokens.
An arm with `ref_timestep` equal to the image's would separate the two.

## Open

- A preservation metric that sees text: ΔE or OCR on text crops outside the edit, not a
  page mean.
- The same A/B on sanitize or swap_text, where the copied area holds most of the page's
  text.
