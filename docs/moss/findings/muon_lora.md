# Muon for the MOSS LoRA: same step size, ~3× the rank directions, no `dc_isolate`

Measured 2026-10-24 on MOSS-Image Base, judged by eye. Runs: `output/ckpt_moss/caststation_lay08p02_at{,_muon,_muon_nodc}`;
side-by-side columns: `output/tests/ab_caststation_dc_isolate/cols/` (none / adam / muon /
muon_noDC). Basis: arXiv 2605.10468 (Qu, Huang, Horvath, ICML 2026). Its claim is that
fine-tuning with a different optimizer from the one used in pretraining disrupts pretrained
knowledge, and LoRA mitigates it. On their Muon-pretrained model, LoRA-Muon beat LoRA-Adam.
MOSS was pretrained with Muon–AdamW (`project/moss_lora/report_notes.md`).

## Setting

As [lr_clip_sweep.md](lr_clip_sweep.md): `channel_(caststation)_layout`, 20 epochs = 600
samples, seed 0, rank 32 / alpha 128 on q/k/v/o/ffn, `weight_svd`, channel scaling α 0.5,
lr 5e-5, clip 1. The arms change only the optimizer and `dc_isolate`. Muon is
`torch.optim.Muon` with `adjust_lr_fn="match_rms_adamw"` (Moonlight scaling
0.2·√max(m, n)) and weight decay 0.01 (torch AdamW's default). The no-LoRA column is the
Muon LoRA at multiplier 0. Sampled by `lora_ab.py --arms er_sde:1.0` (3 prompts × seeds 1, 2 ×
Base and SFT, 50 steps, CFG 3).

| column | run | optimizer | `dc_isolate` | min |
|---|---|---|---|---|
| adam | `caststation_lay08p02_at` | AdamW | on | 19.1 |
| muon | `caststation_lay08p02_at_muon` | Muon | on | 20.4 |
| muon_noDC | `caststation_lay08p02_at_muon_nodc` | Muon | off | 22.9 |

## Measured

Weight space, mean over the 252 adapted linears (ΔW = scale · up · down; DC = the
calibrated token mean x̄ from `library/moss/calibration/channel_mean.safetensors`):

| | ‖ΔW‖_F | stable rank (of 32) | top direction's energy | energy on DC input | ‖ΔW x̄‖ |
|---|---|---|---|---|---|
| adam | 1.156 | 6.5 | 17.7 % | 1.04 % | 1.98 |
| muon | 1.161 | 18.5 | 5.5 % | 1.27 % | 2.57 |
| muon_noDC | 1.146 | 18.7 | 5.4 % | 0.45 % | 1.24 |

- **Muon at the AdamW lr moves the adapter by the same amount** (‖ΔW‖_F within 0.5 %, on
  every target), so 5e-5 carries over. It spreads that over ~3× the rank directions, the
  same direction as the paper's rank-8 LLM result (6–7 vs 3–5).
- **Under Muon, `dc_isolate` works as a bias channel.** The DC row stays on the DC direction
  (cos 0.98; AdamW lets it drift to 0.79), and the token-mean output shift is 2.57 with it
  and 1.24 without. AdamW + `dc_isolate` gives 1.98. Muon's orthogonalised step already
  caps the near-rank-1 bias part of the `up` gradient at one direction's share, which is
  what `dc_isolate` was built to do for AdamW.
- AdamW's top direction is not the DC row (|cos| 0.08 with x̄).
- **Images**: the three LoRAs carry the set's style to the same degree (flat pale colour,
  thin clean line, small eye highlights) and none adds artifacts. On composition the
  columns differ cell by cell, and the user judged Muon better than AdamW. All three write
  the trigger as text on Miku's name plate, so that is not optimizer-related.
- **Loss**: paired by epoch (same seed, so same σ draws), muon − adam is −0.0006 to
  +0.0017, and muon_noDC − muon is −0.0004 to −0.0040 at every epoch.

## Not tested

- A second seed or dataset: every arm is one 30-image set and one training seed.
- Muon without channel scaling (α 0.5 is also an AdamW per-channel step fix). α 0.75 and
  a kaiming init followed: [muon_channel_init.md](muon_channel_init.md).
- Muon at lr 1e-4: the spread update may hold where AdamW + `dc_isolate` over-fits.
- `lokr` / `loha` under Muon: the trainer keeps them on AdamW.
