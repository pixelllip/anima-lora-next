# The text refiner turns a layout bbox into the location MOSS draws at

Measured 2026-10-28 on MOSS-Image Base. Probe, ablation and per-condition tables:
`bench/moss_bbox_grounding/` (`probe.py`, `ablate.py`, `README.md`).

## Setting

Teacher-forced, no sampling. The 36 cached swap_text targets with a single `text` item
(`post_image_dataset/moss/i2i_swap_text`: a page whose box holds pasted lettering, the
caption's item gives the line and its `abs_centralized` `yxyx` box), INT8 TE, fixed noise per
page, one cond forward per caption variant: the staged `bbox`, the box moved off itself
(`bbox_moved`), no box (`none`), and a 3×3 position word for the true or the moved box
(`pos`, `pos_moved`). Regions: the true box `T`, the moved box `M`, the rest.

## Measured

**The box matters at high σ.** In-box flow-matching MSE at σ 0.9: `bbox` 0.620, `none`
0.723, `bbox_moved` 0.723, `pos` 0.698, `pos_moved` 0.725. The moved box also raises the
loss in `M` (0.550 → 0.565): lettering is pushed where there is none. At σ 0.6 the visible
lettering carries the location and the box adds ~2 %; at 0.3, nothing. A position word gives
about a quarter of the box's gain, on the same heads, measured against a tight box it can
only name to a ninth of the canvas.

**Causal: the text refiner builds the location.** Masking the coordinate tokens as keys for
every query (σ 0.9; share of the box's in-box gain lost, 1 = gone; bootstrap 95 % CI):

| masked | gain lost | moved-box push left |
|---|---|---|
| in the two `text_embedder` blocks only (the DiT still sees the tokens) | 0.93 [0.55, 1.35] | 0.02 |
| in all 42 DiT blocks, refiner intact | 0.83 [0.46, 1.24] | 0.14 |
| refiner + all blocks, also the JSON tail and template suffix | 1.00 | 0.02 |
| DiT blocks 0–13 | 0.01 | 0.58 |
| DiT blocks 26–35 (best single range) | 0.39 | 0.40 |
| DiT blocks 9–41 | 0.60 | 0.42 |

- The digits have to read each other in the refiner to become a usable location. The
  refiner is unmodulated, so the location code is σ-independent, and the MOSS adapter
  targets (`layers.N.proj.*`) do not touch it: every LoRA trained so far left it intact.
- The DiT reads the refined coordinate tokens directly, spread over blocks ~14–35; no
  single range carries it. Blocks 0–13 do not need them, although the hidden-state
  difference between `bbox` and `bbox_moved` already concentrates in the two boxes from
  block 7.
- The tokens after the box carry it causally from the TE, but that route adds ≤ 0.04.
- Correlational read (`probe.py`): item-content attention follows the box mostly in blocks
  26–35, over many heads (the content map moves to the moved box on 70–90 % of pages).
  b14 h4 follows the box at every σ.

## Hypothesis (untested)

An i2i task can hand the location to this path instead of learning it: put the edit region
in the layout prompt as an item with a box (from OCR, a detector or a small predictor on the
reference) and let the LoRA learn only what to draw there. The location is decided at high σ,
which is where an i2i LoRA otherwise has to find the region from the reference alone.

## Open

- Labels other than `text` (characters, objects): same path or not.
- Whether the sanitize LoRA, trained without boxes, localises the text through a path of
  its own (reference attention) or reuses this one.
- A training A/B: an i2i task with and without box items at small data.
