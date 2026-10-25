# MOSS LoRA: a `weight_svd` init with the input DC projected out

Status: proposal (2026-10-25). Init statistics measured; nothing implemented or trained.

`weight_svd` seeds `down` with the base weight's top-r right singular vectors. On MOSS those
vectors lean on the input DC (the token-mean x̄, carried mostly by massive channel 42), so
`h = down · x` starts with 4–6× more DC than a kaiming init at the FFN input and q. Under
Muon that DC feeds the bias part of the `up` gradient, and the shipped default moved to
kaiming for it ([muon_channel_init.md](../moss/findings/muon_channel_init.md): output DC
shift ‖ΔW x̄‖ 1.24 → 0.43 at the same ‖ΔW‖ and stable rank). Kaiming pays for that with
the subspace: it captures 8–31 % of the base weight's energy that the SVD basis does.
This proposal keeps the basis and drops only the DC direction.

## The init

With `û = x̄ / ‖x̄‖` per target (`library/moss/calibration/channel_mean.safetensors`, the
file `dc_isolate` reads):

```
down = top-r right singular vectors of W(I − ûûᵀ), / √3
```

Every row is orthogonal to x̄, so `down · x̄ = 0` at init. Unlike `dc_isolate` there is no
dedicated DC row (the row that became a bias channel under Muon) and no re-projection after
each step: only the starting basis changes.

## Measured (init only)

Every third block 11–41; DC ratio against a norm-matched kaiming init; capture =
‖W Q‖²_F over the same for the `weight_svd` basis (Q an orthonormal basis of the rows):

| target | init | DC into `h` (× kaiming) | capture |
|---|---|---|---|
| ffn_up_gate | `weight_svd` | 4.10 | 100 % |
| | **⊥ DC** | **0** | **99.4 %** |
| | kaiming | 1.00 | 30.6 % |
| q | `weight_svd` | 6.25 | 100 % |
| | **⊥ DC** | **0** | **98.5 %** |
| | kaiming | 1.00 | 14.2 % |
| ffn_down | `weight_svd` | 1.83 | 100 % |
| | **⊥ DC** | **0** | **100 %** |
| | kaiming | 1.00 | 8.5 % |
| o | `weight_svd` | 1.18 | 100 % |
| | **⊥ DC** | **0** | **100 %** |
| | kaiming | 1.00 | 16.3 % |

Ch 42 itself is not removed, only its mean: the column stays 2.35× the mean column at the
FFN input (`weight_svd` 3.15×, kaiming 0.99×). What it feeds into `h` is channel 42's
token-to-token variation, which is small (peak-to-mean 2.7, `channel_scaling.md`).

## Plan

**Phase 1: code (CPU, tests)**
- `LoRAAdapter.init_down_weight_svd(..., dc_dir=None)` (`library/lines/adapters.py`):
  with `dc_dir`, take the singular vectors of `W − (W û)ûᵀ`. `svd_slice` /
  `svd_slice_count` compose unchanged.
- A third `down_init` choice, `weight_svd_nodc` (`library/moss/requests.py`). The trainer
  loads `channel_mean` when `dc_isolate` or this init needs it; stamped as `down_init`.
- Tests (`tests/test_moss_lora.py`): `down · x̄ ≈ 0` at init; the result equals
  `weight_svd` on a weight already orthogonal to x̄; a plain-LoRA round trip.

**Phase 2: one arm**
The caststation recipe of the findings doc (Muon, lr 5e-5, α 0.75, `dc_isolate` off,
20 epochs, seed 0), `--down_init weight_svd_nodc`, against
`caststation_lay08p02_at_muon_nodc_cs075{,_kaiming}`. About 20 min plus sampling.
Read ‖ΔW x̄‖ and the ch 42 share with
`_archive/bench/moss_channel_stats/lora_weight_stats.py`, the epoch-paired loss, and the
four-column grid in `output/tests/ab_caststation_muon_alpha/cols/`.

**Decision**: it replaces kaiming as the default if ‖ΔW x̄‖ stays near kaiming's 0.43 and
the images hold the style at least as well by eye. If the DC shift climbs back toward
`weight_svd`'s 1.24, the gradient refills the DC direction within training and an init
cannot help; kaiming stays.

## Side check: the sanitize tone shift

[moss_i2i_tone_loss.md](moss_i2i_tone_loss.md) reports a whole-page brightening in
`moss_i2i_sanitize_wm` (Muon, `weight_svd`, α 0.5, `dc_isolate` off) that the AdamW +
`dc_isolate` 200p LoRA does not show. A global shift looks like a DC effect, but the
output DC shift does not separate them: ‖ΔW x̄‖ is 4.49 for 200p and 4.16 for e6 (3.09 at
e3). So this init is not expected to fix it, and the tone proposal stands on its own.

## Out of scope

- A covariance-weighted SVD (CorDA-style, the top directions of `W Σ^{1/2}` with the
  centred input covariance Σ). It needs a per-target covariance calibration, and the
  projection already removes the DC.
- Re-projecting during training (the `dc_isolate` mechanism without its DC row).
