# moss_bbox_grounding — where MOSS-Image Base reads a layout item's location

`probe.py`: teacher-forced, no sampling. Cached swap_text targets with one `text` item
(`post_image_dataset/moss/i2i_swap_text`, 36 of the 40 one-item pages; the other 4 have no
box move that changes both cells and position word), fixed noise per page, σ 0.9 / 0.6 /
0.3, one cond forward per caption variant: `bbox` (staged), `bbox_moved` (point
reflection, else a flip), `none`, `pos` / `pos_moved` (3×3 position word instead of the
box). Regions: `T` true box, `M` moved box, `R` rest. Metric definitions: the module
docstring.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_bbox_grounding/probe.py --n 0"

## Run 20261028-2229-base_swaptext (Base, INT8 TE, 36 pages, ~15 min)

Flow-matching MSE per region (T / M):

| σ | bbox | bbox_moved | none | pos | pos_moved |
|---|---|---|---|---|---|
| 0.9 | **0.620** / 0.550 | 0.723 / **0.565** | 0.723 / 0.549 | 0.698 / 0.549 | 0.725 / 0.559 |
| 0.6 | 0.299 / 0.316 | 0.304 / 0.316 | 0.305 / 0.316 | 0.302 / 0.316 | 0.305 / 0.316 |
| 0.3 | 0.444 / 0.437 | 0.444 / 0.438 | 0.444 / 0.437 | 0.443 / 0.437 | 0.446 / 0.437 |

- The box is used at high σ: at 0.9 it cuts the in-box loss 14 % against no box, and a
  moved box raises the loss in the moved region (lettering drawn where there is none). By
  0.6 the visible lettering carries the location and the box adds ~2 %.
- The position word gives about a quarter of the box's gain at σ 0.9 (0.723 → 0.698); a
  3×3 word names a ninth of the canvas, so this measure (a tight box) undersells it.
- Location enters the image stream early: ‖h(bbox_moved) − h(bbox)‖ concentrates in T ∪ M
  from block 7 (T∪M / R 2.3–4.8 on blocks 0–6, 8.2 on 7, ~7.5–8 through 22, then down to 3.2).
  Top readers of the coordinate tokens (in-box queries over the rest): b20 h13, b23 h12,
  b7 h5, b17 h21, b17 h1, b18 h9, b14 h2, b8 h6, b5 h11.
- Item-content attention follows the box mostly in blocks 26–35 (follow up to ~10 at σ 0.9;
  b26 h6, b30 h19, b27 h1, b32 h20, b28 h21; the content map moves to the moved box on
  70–90 % of pages). It is spread over many heads in those blocks, not a few. b14 h4 is the
  one head that follows the box at every σ (flip on 94 / 58 / 47 % of pages).
- Position words drive the same heads (b26 h6, b14 h4, b8 h6, b22 h0), weaker.

Correlational only: content-attention following in 26–35 can be downstream of the block 5–8
write. `ablate.py` below is the causal check.

## Run 20261028-2254-ablate_s09 (`ablate.py`, σ 0.9, 36 pages, 21 min)

Keys masked for every query. `recovery`: share of the box's in-box gain lost (1 = the box
no longer helps); `moved`: share of the moved box's push into M left (1 = intact). Bootstrap
95 % CI over pages; the per-page gain is noisy, so the CIs are wide.

| condition | recovery | moved |
|---|---|---|
| coord, text refiner only (`coord@r`) | **0.93** [0.55, 1.35] | **0.02** [−0.08, 0.11] |
| coord, all 42 blocks, refiner intact | 0.83 [0.46, 1.24] | 0.14 [−0.06, 0.35] |
| coord, refiner + all blocks | 0.96 | 0.05 |
| + tail / + tail + suffix (refiner + all) | 1.00 / 1.00 | 0.01 / 0.02 |
| coord, blocks 0–4 / 5–8 / 9–13 | 0.07 / −0.05 / 0.01 | 0.73 / 0.77 / 1.00 |
| coord, blocks 14–20 / 21–25 / 26–35 / 36–41 | 0.10 / 0.22 / 0.39 / 0.06 | 0.26 / 0.54 / 0.40 / 0.93 |
| coord, blocks 0–8 / 0–13 / 0–20 / 0–25 | 0.00 / 0.01 / 0.38 / 0.42 | 0.56 / 0.58 / 0.24 / 0.23 |
| coord, blocks 9–41 / 21–41 | 0.60 / 0.46 | 0.42 / 0.34 |

- **The text refiner builds the location.** Masking the coordinate tokens in the two
  `text_embedder` blocks alone (the DiT still sees them) removes the box's effect, so the
  digits have to attend to each other there to become usable. The refiner is unmodulated
  (no timestep), so this code is σ-independent, and the MOSS LoRA targets (`layers.N.proj.*`)
  leave it untouched.
- **The DiT reads the refined coordinate tokens directly,** spread over blocks ~14–35 with
  no single range carrying it (best single range 26–35, 0.39; all blocks 0.83). Masking them
  in blocks 0–13 costs nothing, so the block-7 `hdiff` jump is not needed for the effect
  (later blocks re-read the tokens).
- **The TE route is negligible:** the tokens after the box (JSON tail, template suffix)
  carry the box causally from the TE, but masking them on top adds ≤ 0.04.
