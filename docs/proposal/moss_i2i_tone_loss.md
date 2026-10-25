# MOSS i2i: a tone term on the unedited tokens

Status: proposal (2026-10-25). Nothing trained or implemented yet.

The sanitize LoRA `moss_i2i_sanitize_wm` (e6) brightens, cools and slightly desaturates
the whole page, including far from any text it removes. The training pairs do not teach
this, so the shift is something the run produced on its own. This proposal adds a loss
term that compares the coarse tone of the unedited tokens' x̂₀ with the target's, so
"keep the page's tone" becomes an explicit constraint rather than a by-product of MSE.

## Measurements

Five held-out pages (`output/moss_results/{hard_to_sanitize1-3,usual_case1-2}.png`), all
rendered with the same CLI settings: `scripts/moss/generate.py`, the sanitize
`generate.toml`, `--sampler er_sde --cns auto --spectrum`, seed 42. Each output is
compared with its input in Lab after a σ=3 px blur, on one shared mask per image: the
pixels **neither** model changed by ΔE > 8, dilated by 20 px (26-71 % of the page).

| image | 200p ΔL / Δb / sat× | e6 ΔL / Δb / sat× |
|---|---|---|
| hard_to_sanitize1 | +0.24 / −0.23 / 1.07 | **+1.95 / −0.63** / 0.99 |
| hard_to_sanitize2 | −0.45 / −0.14 / 0.97 | **+0.90 / −0.68** / 1.01 |
| hard_to_sanitize3 | −0.32 / +0.10 / 0.92 | **+1.98 / −1.06 / 0.80** |
| usual_case1 | −0.28 / +0.07 / 1.04 | +0.57 / −0.23 / 0.92 |
| usual_case2 | +0.19 / +0.33 / 1.02 | **+2.28 / −0.52** / 0.97 |

- 200p (`moss_i2i_sanitize_200p`: 200 pairs, AdamW + `dc_isolate`) moves lightness by at
  most ±0.5, in no consistent direction.
- e6 (`moss_i2i_sanitize_wm`: 280 pairs incl. 80 `wm_*`, Muon, `dc_isolate` off, 6 ep) is
  brighter on all five (+0.6 to +2.3) and bluer on all five (Δb −0.2 to −1.1). A median
  ΔE ≈ 2.5 is about where people start to notice.
- The renderer matters too: the same 200p LoRA moved +0.6 to +1.0 in L from ComfyUI to the
  CLI. Compare runs only within one renderer.

**The data is tone-neutral.** Target − condition over the unedited pixels of every staged
pair (same blur, ΔE > 8 mask, 1024 px long side):

| pair kind | n | ΔL mean | Δb mean | sat× median |
|---|---|---|---|---|
| textless | 160 | −0.00 | −0.00 | 1.000 |
| mined twin | 39 | −0.00 | −0.01 | 1.000 |
| wm | 80 | −0.05 | +0.02 | 1.005 |

**The set's overall tone points the other way.** Absolute means (Lab, HSV saturation, 512 px
thumbnails):

| group | n | L | b | chroma | sat | L > 90 |
|---|---|---|---|---|---|---|
| staged targets | 200 | 62.0 | +6.22 | 13.0 | 0.212 | 14.4 % |
| `wm_*` sources (counted twice) | 80 | 63.2 | +6.48 | 13.0 | 0.212 | 17.6 % |
| the five test pages | 5 | 63.9 | +1.46 | 10.8 | 0.169 | 26.2 % |
| corpus sample (`retrieved/`) | 400 | 68.8 | +4.01 | 12.5 | 0.195 | 29.8 % |

The targets are darker, yellower and more saturated than the corpus. A pull toward the
set's mean would darken and warm the output; e6 brightens and cools it. The duplicated
`wm_*` sources sit within 1.2 L of the rest.

The lossless-WebP change does not bear on this: the converted PNGs were already lossless,
so no training pixel changed.

## Why plain MSE lets it through (hypothesis)

The flow-matching loss is the per-token MSE on v. A tone offset on the copied area is
one more residual in that mean, with no separate weight. At ΔL ≈ 2 it is small next to
the per-step loss (0.09 average in the last epoch of e6), and the LoRA's capacity goes
to the edit. `rest_weight > 1` (the region loss already in `train.py`) does not fix
this: it scales every frequency on the rest tokens equally, so the offset stays the same
share of the loss.

Two other causes are not ruled out, and Phase 0 checks the cheap one:
- the shift accumulates with training (e3 vs e6);
- Muon / `dc_isolate` off vs AdamW, or the `wm_*` pairs. Only a 2×2 of runs separates
  these, so it is left out unless Phase 1 fails.

## The term

With `x_t = (1−σ)x₀ + σε` and `v = ε − x₀`, the prediction's clean estimate is
`x̂₀ = x_t − σ·v̂`, so `x̂₀ − x₀ = σ(v − v̂)`. On the rest tokens `R` (the complement of
`edit_region`, grown by an extra margin), pool the error coarsely per channel and
penalise the pooled values:

```
e      = (x̂₀ − x₀) ⊙ 1_R                                  # (1, 128, h, w), packed latent
pooled = avgpool_k(e) / avgpool_k(1_R)                     # masked mean per k×k cell
L      = L_mse + tone_weight · mean over cells with coverage ≥ ½ of pooled²
```

- `tone_pool = 0` means one cell (the per-channel mean over all of `R`); `k > 0` pools
  k×k tokens (k = 8 is 128 px at 16 px per token), which also catches tone drifting
  between regions.
- The term reads x̂₀, not v, because tone is a property of the image. Its σ factor
  down-weights high σ, where x̂₀ is still mostly a guess.
- The mask comes from `edit_region(latent, ref, …)` as the region loss computes it: it
  needs a full-size reference (`ref_scale 1`), which sanitize and swap_text have, and
  pixel-aligned pairs. The staged mined twins pass (unedited share 82 %, ΔL 0.00).
  `tone_dilate` grows the rest mask's margin beyond `region_dilate`, so seams stay out
  of `R`.
- Logged separately as `tone` in progress events, like the region / rest parts.

New `TrainRequest` fields (advanced): `tone_weight` (0 = off, the default),
`tone_pool` (0 | k), `tone_dilate` (tokens, default 2).

## Plan

**Phase 0: does the shift grow with training? (about 10 min, no code)**
Render `moss_i2i_sanitize_wm_e3` with the settings above and run the tone script. If e3
is already near e6, the shift sets in early and the term needs to act from the start; if
e3 is clean, stopping earlier is a second fix to keep in mind.

**Phase 1: implement + invariant tests**
- `library/moss/train.py`: `tone_loss(pred, target, latent, noisy, sigma, rest_mask, pool)`
  next to `region_mse`; precompute the rest masks once per sample like `regions`.
- Tests (`tests/test_moss_train.py`):
  - `tone_weight = 0` gives a loss bit-identical to today's.
  - A constant per-channel offset on x̂₀ over `R` gives exactly offset² at `tone_pool = 0`.
  - Errors inside the edit region contribute nothing.
  - Cells under half coverage are skipped.
- Bench (CONTRIBUTING Tier 2): promote the tone script to `bench/moss_i2i_tone/run_bench.py`
  on `bench/_common.py`. Input: a LoRA plus a page folder. Output: per-image ΔL / Δb /
  sat× / ΔE on the shared-mask method, in the `result.json` envelope.

**Phase 2: arms (e6 recipe, 280 pairs, Muon, local `blocks_to_swap 28` or a pod)**
Micro arms first, `max_steps 560` (2 epochs), compared with an equal-step control:

| arm | tone_weight | tone_pool |
|---|---|---|
| control | 0 | — |
| DC | λ | 0 |
| pooled | λ | 8 |

Pick λ from the logged magnitudes of the control (`tone` logged but not added), so the
term starts at about 10 % of `L_mse`. Extend the winner to 6 epochs.

**Gates (on the bench page set; widen to ~20 held-out sanitize pages before shipping)**
- |ΔL| ≤ 0.5 and |Δb| ≤ 0.3 on average, sat× within 0.97–1.03, with no page past
  |ΔL| 1.0.
- Edit quality no worse than the control by eye on hard_to_sanitize2 (red caption boxes,
  dense SFX) and usual_case2 (faint tiled watermark).
- The edit-region part of the loss (logged `region`) no worse than the control at equal
  steps.

## Out of scope

- **A correction at render time.** Fit a per-channel tone shift on the unedited region
  (Lab or latent) against the full-size reference and apply it to the output. It works on
  LoRAs already trained and stays available as a fallback, but it hides the cause.
- The other i2i tasks. Colorize changes tone on purpose, so the term does not apply;
  swap_text could take it unchanged once sanitize shows it works.
