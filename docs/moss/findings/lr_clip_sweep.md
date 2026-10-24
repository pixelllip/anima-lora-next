# LR × grad clip × grad accum on a small character set: 5e-5 / clip 1 stays best

Measured 2026-10-24 on MOSS-Image Base, judged by eye. Runs: `output/ckpt_moss/caststation_lay08p02_at*`;
side-by-side columns: `output/tests/ab_caststation_clip_sweep/cols/`.

## Setting

One dataset, `channel_(caststation)_layout` (layout-JSON captions, `plain_caption_rate`
0.2), 20 epochs = 600 samples, seed 0, LoRA rank 32 on q/k/v/o/ffn with `weight_svd`
down init, channel scaling α 0.5 and `dc_isolate` — `configs/moss/lora.toml` as shipped
apart from the three knobs below. `max_grad_norm = 1e9` is "no clip". With `grad_accum`
the sample count stays 600 and the optimizer takes 600 / `grad_accum` updates (warmup
counts updates).

Each LoRA was sampled by `project/moss_lora/src/lora_ab.py --arms er_sde:1.0` (3 prompts
× seeds 1, 2 × Base and SFT, 50 steps, CFG 3) and laid side by side with
`project/moss_lora/src/ab_cols.py`.

| column | run | lr | max_grad_norm | grad_accum | updates |
|---|---|---|---|---|---|
| clip1_lr5e5 | `caststation_lay08p02_at` | 5e-5 | 1 | 1 | 600 |
| clip7_lr5e5 | `caststation_lay08p02_at_clip7` | 5e-5 | 7 | 1 | 600 |
| noclip_lr2e5 | `caststation_lay08p02_at_lr2e5_noclip` | 2e-5 | off | 1 | 600 |
| ga2_noclip_lr1e4 | `caststation_lay08p02_at_ga2_lr1e4_noclip` | 1e-4 | off | 2 | 300 |
| ga2_noclip_lr5e5 | `caststation_lay08p02_at_ga2_lr5e5_noclip` | 5e-5 | off | 2 | 300 |

## Measured

- **clip1_lr5e5 — the shipped default — is the best of the five.** Neither loosening the
  clip (7, off) nor trading lr against it beat it.
- **lr 1e-4 without a clip distorts, even with `grad_accum` 2.** Halving the update count
  and averaging two samples per update does not make up for removing the clip at 1e-4.
- **At 5e-5, `grad_accum` 2 without a clip is a toss-up: no worse than clip7_lr5e5, not
  better than clip 1.** It also departs most from the other columns in content (e.g. on
  the bench prompt, base seed 1: red neckerchief where the other four draw blue, a hedge
  in place of most of the blossoms).
- **The loss curve cannot tell these runs apart.** Per-epoch loss of ga2_noclip_lr1e4,
  ga2_noclip_lr5e5 and noclip_lr2e5 agree to ~0.005 at every epoch (final 0.232 / 0.231 /
  0.231) and share the same bumps (epoch 11: 0.347 / 0.349 / 0.345): with one seed the σ
  draws dominate the curve. Judge these knobs on samples.

## Not tested

- lr 1e-4 **with** clip 1 at `grad_accum` 2. `caststation_ga4_lr1e4_s{0,1001}` (1e-4,
  clip 1, `grad_accum` 4, 150 updates) were trained but are not in this comparison.
- Whether the ranking holds on a larger set: this is one 30-image character set and one
  training seed.
