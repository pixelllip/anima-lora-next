# bench/moss_mod_guidance

Modulation guidance (Starodubcev et al., ICLR 2026, arXiv 2602.09268) on MOSS-Image: a
pooled-text head distilled into the timestep embedding, then steered with a
quality `p₊ − p₋` delta. A one-afternoon try (2026-10-28), not a line. Anima's
version and its verdict: `docs/anima/inference/mod-guidance.md`,
`docs/anima/findings/mod_guidance_quality_tag_axis.md`.

**Verdict: no quality lever on MOSS t2i.** The pooled TE vector does not move more for
quality tags than for content tags at the same slot; a 400-step micro distill never
beat the uncond floor; steering at w 3 / 6 barely changes the image. Parked; the open
caveat is the step count (below).

## Why it applies at all

MOSS's AdaLN path is text-blind like Anima's original: `time_conditioning` →
`shared_block_modulation` (one `Linear(dim, 6·dim)` shared by all 42 blocks) and
`video_head_modulation`, timestep only. Both are affine in `time`, so a head offset on
`time` enters every block as `W_mod · offset`; `distill.py::attach_head` patches
`time_conditioning` that way and leaves `model.py` untouched. Two differences from
Anima: the modulation also hits the text tokens (single stream), and there is no
CLIP — the paper's argument for the pooled channel is CLIP's contrastive geometry,
here the pool is Qwen3.5-9B hidden states.

## 1. TE probe — `probe_pool.py` (CPU-light, < 1 min GPU)

`rel_dpool = ‖pool(base+tag) − pool(base)‖ / ‖pool(base)‖`, Anima's upstream
selectivity metric (Anima: quality 0.059 / content 0.031, ratio 1.89; quality-axis
cross-content cos 0.814). 12 resized-tree captions × 6 quality + 6 content tags, three
formats (`layout` default prompt, `plain` tags, `nl` English with phrases appended), two
spaces (`te` raw 4096, `emb` after the DiT's `text_embedder`), three pools.

| run | quality / content ratio |
|---|---|
| quality after the rating, content at the tail (`20261028-1914-pool-probe`) | te 1.7–3.45, emb 1.05–1.18, nl 0.78–1.03 |
| `--content_front`: content after the rating too (`20261028-1915-pool-probe-front`) | **0.84–1.03 everywhere** |

The first run's `te` selectivity is position: the TE is causal, so a tag inserted at
the front rewrites every later row. With both at the front (the official slot for
quality tags), `score_9` moves the pool as much as `holding a sword`. The `nl` format
appends both and never showed selectivity. The pos − neg direction is consistent
across content (mean-pool cos 0.81–0.88), so a steering delta exists; it is just not
more quality-specific than any other front-slot edit.

Anima's own probe spliced quality at the front and content at the tail with a causal
Qwen3 TE, so its 1.89 may carry the same position effect.

## 2. Micro distill — `distill.py` (~20 min)

Head: LayerNorm(4096, no affine) → Linear(4096, 3072) → SiLU → Linear(3072, 3072),
output zero-init, input the mean of the caption's content-token TE rows (template span
3 / 7 tokens). Teacher: frozen Base on the cached caption, no head. Student: Base on
the empty prompt + the head. MSE between velocities. 400 samples from the five
`*_layout` caches at ~1 MP, one step each, AdamW lr 3e-5 (10 % warmup, cosine to 10 %).
Phase 1 runs every teacher forward (and the uncond forward = the floor) with the DiT
resident; phase 2 trains under the LoRA trainer's swapper (36 blocks, compile,
budget 0.5). Phase 1 ~10 min, phase 2 ~10 min.

- lr 2e-4 diverged in the smoke: Adam moves the zero-init 3072² output coherently,
  `|off|` 3.4 after 2 steps against a `|time|` of 2.8–4.6.
- At 3e-5, loss / floor was 1.27 (first 20) → **1.35 (last 50)**; it never went below 1
  and got worse as `|off|` grew (0.5–1.2). The floor is measured on a different path
  (eager, resident) than the student (compiled, swapped): at step 0, head exactly 0,
  the ratio is 1.2, so small-floor samples are inflated by bf16 path noise — but
  samples with floor ≥ 0.03 stay above 1 too.

Head: `output/ckpt_moss/mod_guidance/pooled_head.safetensors` (not shipped).

## 3. Render — `eval.py` (~30 min)

3 prompts (2 layout tag prompts, 1 NL) × 2 seeds at 1024², Euler / CFG 3 / skip 22 /
50 steps. Arms: `off` (no head), `w0` (head on, each branch its own pool's offset),
`w3` / `w6` (steering on blocks 12–41, both branches), and on tag prompts `pq` / `pc`
(cond offset from the prompt with `score_9` vs `red ribbon` spliced after the rating,
text unchanged). Results: `results/20261028-1945-eval-400/` (grids per prompt).

- **off → w0 changes the image a lot** (brighter, lower contrast, pink / purple cast,
  composition and outfits move). Not the head improving anything: the uncond branch
  gets `head(pool(""))`, a pool never seen in training, as large as the cond offset and
  anti-aligned with it (cos −0.17 … −0.53). CFG multiplies the branch difference by 3:
  mod-space rms 0.03–0.15, the size of the shift / gate chunks themselves (0.1–0.16).
- **Steering is small**: `‖delta‖` 0.06–0.08 (~8 % of the head output), mod rms 0.002;
  w6 ≈ 0.012. w3 / w6 look like w0, except a coat drifting beige → pink → red on one
  seed (colour, not quality).
- **pq vs pc**: latent rms to w0 0.28 vs 0.24, inside the trajectory-divergence floor
  (any perturbation here gives 0.2–0.3); no visible quality difference.

## Open

- 400 × batch 1 cannot tell "MOSS can't" from "too few, too noisy steps" (paper: 1K–4K
  iterations, batched; Anima: 5000). A retry would measure the floor on the student's
  own path (head 0), use grad accumulation, and add pools of the empty prompt to the
  training set so the uncond offset is trained.
- A `w0_shared` arm (both branches get the cond offset) separates the head's common
  push from the CFG-amplified branch mismatch; not run.
- `eval.py`'s per-layer wrapper closes over the block index, so dynamo recompiles per
  block, hits `recompile_limit` 16 and falls back to eager (~1 min / render); it also
  renders without Spectrum. Fix both before reusing it.
- i2i edit strength (the paper's FLUX Kontext use, `p₊` = the edit prompt, `p₋` = "") is
  a different question this did not test.
