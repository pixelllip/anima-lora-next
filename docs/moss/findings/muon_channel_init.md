# Under Muon: channel scaling α 0.75 and a kaiming `down` init

Measured 2026-10-25 on MOSS-Image Base, judged by eye. Follows [muon_lora.md](muon_lora.md),
whose shipped arm (Muon, `dc_isolate` off, α 0.5, `weight_svd`) is the baseline here. Runs:
`output/ckpt_moss/caststation_lay08p02_at_muon_nodc{,_cs075,_cs075_kaiming}`; side-by-side
columns: `output/tests/ab_caststation_muon_alpha/cols/` (none / svd α 0.5 / svd α 0.75 /
kaiming α 0.75). Weight stats: `_archive/bench/moss_channel_stats/lora_weight_stats.py`.

New default: **Muon, lr 5e-5, `channel_scaling_alpha` 0.75, `down_init` kaiming,
`dc_isolate` off.**

## Setting

As [muon_lora.md](muon_lora.md): `channel_(caststation)_layout`, 20 epochs = 600 samples,
seed 0, rank 32 / alpha 128 on q/k/v/o/ffn, lr 5e-5, clip 1, Muon, `dc_isolate` off.
The arms change only `channel_scaling_alpha` and `down_init`. Sampled by `lora_ab.py --arms
er_sde:1.0` (3 prompts × seeds 1, 2 × Base and SFT, 50 steps, CFG 3).

| column | run | α | `down_init` | min |
|---|---|---|---|---|
| svd α 0.5 | `caststation_lay08p02_at_muon_nodc` | 0.5 | `weight_svd` | 22.9 |
| svd α 0.75 | `caststation_lay08p02_at_muon_nodc_cs075` | 0.75 | `weight_svd` | 20.6 |
| kaiming α 0.75 | `caststation_lay08p02_at_muon_nodc_cs075_kaiming` | 0.75 | kaiming | 20.3 |

## Why α goes up under Muon

Channel scaling was tuned for Adam, whose per-element step makes a column's output leverage
`|x[c]|`; α 0.5 cuts it to `|x[c]|^0.5`. Muon orthogonalises the `(32, in)` `down` update as
`(GGᵀ)^(-1/2) G`, a left multiply by a 32 × 32 matrix: it flattens the rank side and keeps
the gradient's column (input-channel) proportions, so on the input axis it steps like SGD
and the loud channel's lead is closer to `|x[c]|²` than `|x[c]|`. α 0.5 under Muon lands
near what Adam gives with no scaling.

Update of `down` alone (saved `down` minus the `weight_svd` init rebuilt from the base DiT),
loud channel's output leverage over the mean channel, every third block 11–41:

| target | svd α 0.5 | svd α 0.75 |
|---|---|---|
| ffn_up_gate | 6.9× | 1.6× |
| ffn_down | 4.2× | 1.3× |
| q | 3.2× | 1.3× |
| o | 2.1× | 1.1× |

AdamW at α 0.5 (`channel_scaling.md`: Δdown on ch 42 at 0.13–0.25× an average column, ch 42
at 11–26× the median) comes to roughly 2–5× at the FFN input. α 0.75 under Muon is flatter
than that; ‖Δdown‖ / ‖init‖ grows (FFN 0.141 → 0.150, o 0.141 → 0.202) as the step moves
to other channels.

## Why `weight_svd` goes

The base weight's top right singular vectors lean on the massive channel and the input DC.
Against a norm-matched kaiming init, every third block 11–41:

| target | ch 42 column / mean | ‖D₀ x̄‖ / ‖K x̄‖ | ch 42 share of `h` energy |
|---|---|---|---|
| ffn_up_gate | 3.2× | 4.1× | 34 % |
| q | 1.3× | 6.3× | 2.6 % |
| ffn_down | 1.6× | 1.9× | 3.4 % |
| o | 1.3× | 1.2× | 0.6 % |

So `h = down · x` starts with 4–6× more DC at the FFN input and q, which is the bias part
of the `up` gradient that `dc_isolate` was built to contain (`channel_scaling.md`, "The `up`
gradient"). It also dominates ΔW's input profile: ΔW's loud-channel share sits at 27.8 % for
both α under `weight_svd` (α only reaches the update of `down`, which stays 14–20 % of the
init), and falls to 6.8 % with kaiming.

## Measured

Weight space, mean over the 252 adapted linears (as in [muon_lora.md](muon_lora.md)):

| | ‖ΔW‖_F | stable rank (of 32) | top direction's energy | energy on DC input | ‖ΔW x̄‖ | ch 42 share, FFN input |
|---|---|---|---|---|---|---|
| svd α 0.5 | 1.146 | 18.7 | 5.4 % | 0.45 % | 1.24 | 27.8 % |
| svd α 0.75 | 1.151 | 18.5 | 5.5 % | 0.45 % | 1.24 | 27.8 % |
| kaiming α 0.75 | 1.151 | 18.5 | 5.5 % | 0.03 % | **0.43** | **6.8 %** |

- Kaiming keeps the step size and the rank spread and cuts the output DC shift to a third
  (AdamW + `dc_isolate`: 1.98).
- **Loss**: paired by epoch, svd α 0.75 − svd α 0.5 and kaiming − svd α 0.75 both stay
  within ±0.001.
- **Images**: the three LoRAs carry the set's style to the same degree (flat pale colour,
  thin clean line) and none adds artifacts. The user judged α 0.75 better than α 0.5; it
  more often keeps the no-LoRA pose (Base p0 / p1). Kaiming is close to svd α 0.75, with
  slightly darker colours in places, and on SFT p2 turns the no-LoRA red scarf blue. The
  init was chosen on the weight-space result, not by eye. All three still write the trigger
  as text on Miku's name plate.

## Not tested

- A second seed or dataset: every arm is one 30-image set and one training seed.
- α 1.0 under Muon (leverage flat on the `down` side).
- Muon without channel scaling, now with kaiming.
- `svd_slice`: it needs `down_init weight_svd`, so slice LoRAs keep the DC-heavy basis.
- A `weight_svd` init with x̄ projected out: tested in
  [dcfree_svd_init.md](dcfree_svd_init.md).
