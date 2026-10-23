# MOSS-Image — is the loss gap a high-frequency residual, and would VR-loss help with it?

Status: **closed 2026-10-23.** Phase A passes: 72 % of the residual gap sits above
0.5 Nyquist (§ Phase A result). Phase B fails: VR's control variate is uncorrelated with
the MOSS residual (ρ² ≈ 0.01), so at the λ the trainer would learn, VR is a no-op
(§ Phase B result). Do not port VR-loss to MOSS. Step 2 of 2 after
`docs/proposal/moss_latent_hf_budget.md`, whose gate passed: the FLUX.2 latent carries
2.3–2.5× Anima's centred HF share, and about 0.16 more energy per element above 0.5
Nyquist. That is roughly the size of the unexplained loss gap. No training; Phase A runs
forwards only, Phase B adds adapter gradients at a fixed checkpoint.

## Why this is not the Anima headroom rig re-run

Step 1's gate table says to run "VR headroom ρ² at the Anima rig's settings". Anima's own
record says that number does not decide anything:

- **Loss-level ρ² overstated the case on Anima.** The 0.999 headline came from a
  near-converged merged T-LoRA against base (`_archive/bench/fm_vr_headroom/README.md`,
  2026-05-14 errata). Live training put `vr/lambda_ema` near −0.75, not −1. The re-probe
  on the v1.0 quarter LoRA read **MARGINAL** (high-band mid-σ ρ² median 0.23, null gap
  0.21; `results/20260524-1716-v1.0-quarter-vs-base`).
- **There is no valid Anima gradient-level reading.** `_archive/bench/vr_grad_diagnostics/`
  (2026-05-15) reported a VR gradient 6 % of the standard one in norm, cos 0.03–0.04 to
  it, 4.0–4.5× its relative variance and a direction lift of −0.18. But that bench
  passes `t · 1000` as the DiT timestep (`_forward_velocity`), and Anima's DiT takes σ in
  [0, 1] (`train.py`: "timesteps are [0,1]-scaled"). `fm_vr_headroom` fixed the same bug
  on 2026-05-24 (its `tfix` runs); the gradient bench was never re-run. Its numbers were
  measured far outside the trained timestep range and say nothing about VR. No shipped
  config sets `vr_loss_weight`.

So a MOSS port needs two things the Anima record never established: that the MOSS loss
gap really is high-frequency residual (Phase A), and that VR's gradient is usable on
MOSS (Phase B). A goes first because it is cheap, and it settles the step-1 question
whichever way B lands.

## Phase A — split the base model's residual by frequency

Forwards of the **base** DiTs only (no LoRA), on content both lines have cached: stems
present in both `post_image_dataset/lora/` and `post_image_dataset/moss/<dir>/`
(`ama_mitsuki_layout` and `channel_(caststation)_layout` have Anima counterparts). Each
model reads its own cached latent and TE, so content matches but crops do not. That is
fine here because the quantity is per element.

- **σ grid** 0.1, 0.2, …, 0.9 for both (as run: midpoints 0.05…0.95, see the result),
  each model's own timestep convention (MOSS `1000·σ`). Same ε seed per (stem, σ, draw)
  within a model.
- **Size:** 24 stems × 9 σ × 2 draws = 432 forwards per model (as run: 10 σ, 480). MOSS at ~1 MP with block
  swap is the cost, roughly 15–25 min through the daemon.
- **Per forward:** the residual `y = u_pred − (ε − x_0)` on the /8 grid (MOSS:
  `unpatchify` of the packed residual, as in step 1's M1). Record `E[y²]` and its energy
  above 0.25 and above 0.5 Nyquist (same radial FFT as step 1).
- **Self-check:** σ-weight `E[y²]` by each model's own training density (MOSS
  `sigma_mu −1.09, s 0.95, cutoff 0.95`; Anima `sigmoid(N(0,1))`). The result should land
  near each line's early-step logged loss. If it does not, the probe is not measuring the
  training loss and nothing below is read.

| Reading (both weighted to Anima's σ density, as step 1 did) | Next |
|---|---|
| The > 0.5 Nyquist band carries ≥ 50 % of the MOSS − Anima residual gap | the loss gap is an HF floor: Phase B |
| 25–50 % | partial: Phase B, with the expectation written down first |
| < 25 % | the gap sits at low frequency; step 1's HF surplus is predictable. Close the VR line for MOSS and record the gap as model/data |

Per-σ curves are reported as well. An HF residual that stays high down to small σ is the
"unpredictable texture" reading; one that falls with σ is detail the model can predict.

### Phase A result (2026-10-23)

`bench/moss_vr_headroom/residual_bands.py`, run `results/20261023-2340/` (gitignored):
12 stems each from `ama_mitsuki` and `channel_(caststation)`, σ at the midpoints of ten
equal cells over [0, 1], 2 ε draws, Base DiTs, ~9 min in total. Each cell is weighted by
its probability mass under the density, read off the CDF. That replaces the 0.1…0.9
grid above, which stops short of σ > 0.95, where MOSS puts 9 % of its training mass
after the cutoff rescale (Anima puts 0.2 %).

Residual energy per element `E[y²]`, and the part above 0.5 Nyquist:

| σ | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| Anima `E[y²]` | 0.205 | 0.090 | 0.065 | 0.056 | 0.053 | 0.054 | 0.059 | 0.068 | 0.088 | 0.147 |
| Anima > 0.5 Nyq | 0.146 | 0.058 | 0.040 | 0.033 | 0.031 | 0.030 | 0.030 | 0.031 | 0.031 | 0.028 |
| MOSS `E[y²]` | 0.621 | 0.369 | 0.262 | 0.211 | 0.186 | 0.177 | 0.183 | 0.208 | 0.266 | 0.436 |
| MOSS > 0.5 Nyq | 0.481 | 0.276 | 0.191 | 0.150 | 0.129 | 0.119 | 0.117 | 0.122 | 0.133 | 0.143 |

| Weighted | Anima | MOSS | gap | above 0.25 Nyq | above 0.5 Nyq |
|---|---|---|---|---|---|
| Anima's σ density | 0.066 | 0.227 | **0.161** | 0.146 (91 %) | **0.115 (72 %)** |
| each line's own density | 0.066 | 0.265 | 0.199 | | |

- **Self-check passes.** Unmasked Anima on caststation logged 0.070 and 0.071 in its
  first epoch (`output/logs/caststation_vp_stock_20260928-2104`,
  `caststation_jp_v1_20261008-2047`); the probe gives 0.069 on that artist. MOSS logged
  0.259 (caststation, `caststation_ga4_lr1e4_s0`) and 0.290 (`moss_ama_mitsuki_ga4_lr1e4`)
  in the first epoch; the probe gives 0.251 and 0.279.
- **Correction to step 1's table.** The Anima "~0.09" there came from soup shards trained
  with `masked_loss = true` on 14 artists. Unmasked, like for like with MOSS, Anima sits
  at ~0.07, so the gap is larger than that table implied: 0.16 at Anima's density, 0.20
  at each line's own.
- **The gap is an HF floor: Phase B is open.** 72 % of the matched-density gap is above
  0.5 Nyquist and 91 % above 0.25. The 0.161 gap also matches step 1's
  0.16-per-element HF energy surplus, though that agreement may be partly coincidental.
- **The texture reading holds.** MOSS's HF residual is largest at small σ (0.48 at
  σ = 0.05, against Anima's 0.15). As σ → 0, a band's residual approaches the share of
  its Fourier dimensions the model cannot predict beyond the noisy observation. White ε
  puts 0.80 of its energy above 0.5 Nyquist, so that is roughly 60 % of the band for MOSS
  and 18 % for Anima. At σ = 0.95 MOSS's HF residual (0.143) is about the clean latent's
  own HF energy (0.132): the caption predicts none of it, as expected.

Phase A says the floor is there; it does not say VR's gradient makes use of it.

## Phase B — VR-loss gradients on a trained MOSS LoRA

A port of `_archive/bench/vr_grad_diagnostics/run_bench.py` onto `library/moss/`, at fixed
θ = one shipped t2i LoRA (e.g. `output/ckpt_moss/caststation_ga4_lr1e4_s0`) on its own
cache:

- `x_0^L` = Gaussian blur of the normalised latent on the /8 grid,
  `σ_low = min(H/8, W/8) / 4` (`vr_fei_sigma_low_div` default), re-patchified. Paired
  `x_t^L` uses the same ε.
- Reference forward: the same transformer with every adapter's `multiplier = 0`
  (`library/lines/adapters.py` returns the base output untouched). No second DiT.
- K = 24 minibatches of 1, σ drawn from MOSS's training density. Per batch: `g_std` and
  `g_vr = g_std + ∂(2λ⟨z, y⟩)`, λ online as in `_flow_matching_vr_loss`, gradients kept on
  CPU (K × 2 × adapter size; check against the 64 GB box before choosing rank).
- Loss-level side readout at the Anima rig's settings: global and high-band ρ², with a
  decorrelated-ε null.

**Gate** (`vr_grad_diagnostics` criteria, all three). The third column is the archived
Anima reading, void because of the timestep bug above:

| Metric | Pass | Anima (void) |
|---|---|---|
| `gradient_magnitude_ratio_vr_over_std` | 0.5–2 | 0.06 |
| `cov_sq_ratio_vr_over_std` | < 0.7 | 4.0 |
| `direction_lift_vr_minus_std` | > 0.05 | −0.18 |

A pass justifies wiring VR into `library/moss/train.py` and a wall-clock-matched A/B
(VR costs one extra no-grad forward per step, more under block swap). A fail closes the
line, with Phase A's split kept as the explanation of the loss gap.

### Phase B result (2026-10-23)

`bench/moss_vr_headroom/grad_diagnostics.py`, run `results/20261023-2351/` (gitignored):
`caststation_ga4_lr1e4_s0` reloaded as fp32 adapters (74.3 M params) on the bf16 Base,
24 of its 30 training stems, one batch each, σ from MOSS's training density, ~3 min with
no block swap. How it differs from the plan:

- **λ comes from the 24 batches, not an online EMA.** `g_vr` is linear in λ, so the
  bench keeps `s_k = ∂mean(y²)` and `c_k = ∂mean(z·y)` and reads every metric for any λ
  off their Gram matrices. The rows below are the pooled `λ = −Σcov(y,z)/Σvar(z)`, which
  an EMA converges to, and the fixed −1 of `vr_loss.md`'s open question.
- **Each gradient takes its own forward** rather than `retain_graph` through the block
  swapper.
- **The direction cosine leaves batch k out of its reference.** The archived bench kept
  it in, which inflates the cosine; that version is reported as `incl`.
- **The ρ²-with-null readout was not run.** The per-batch correlation below already
  settles it.

| λ | `‖g_vr‖ / ‖g_std‖` | cov² ratio | lift (LOO) | lift (incl) | cos(mean g_std, mean g_vr) |
|---|---|---|---|---|---|
| pooled −0.19 | 1.01 ✓ | 0.99 ✗ | +0.011 ✗ | +0.009 | 0.993 |
| fixed −1 | 1.19 ✓ | 0.80 ✗ | +0.177 ✓ | +0.165 | 0.844 |

- **The control variate is dead on MOSS.** Per batch, `ρ²(y, z)` has median 0.014 and
  max 0.044. At the pooled λ the loss falls by 1.3 %, and `g_vr` is `g_std` (cos 0.993,
  norm ratio 1.01). VR as the trainer runs it, with λ learned online, would be a no-op
  that costs one extra forward per step.
- **Phase A explains why.** `x_0^L` is a heavy low-pass (σ_low 21–32 px on the /8 grid, median 30; Anima's default is the same `min/4`),
  so `z` carries the base model's low-frequency error. The MOSS residual is 72 %
  HF texture that is not in `z` at all, so nothing can cancel it. A low-pass control
  variate can only remove shared low-frequency error, and MOSS has little of it.
- **λ = −1 is a different loss, not variance reduction.** The loss rises by 33 % on
  average (`‖y − z‖² > ‖y‖²`), and the mean gradient turns 32° away from the standard
  one. Its single-batch gradients agree with each other better (lift +0.18), but the cov²
  ratio is still 0.80, above the 0.7 line. Whether training on that HF-directed objective
  improves samples would take a training A/B. Nothing measured here argues for one.
- **Gradient norms are σ-skewed.** 7 of the 24 batches (σ ≥ 0.85) carry 57 % of
  `Σ‖g_std‖²`, and the standard-gradient LOO cosine is only +0.006. At batch 1, the mean
  gradient over 24 batches is not a well-resolved direction, so the lift numbers are
  noisy. That does not touch the pooled-λ verdict, which rests on ρ² and cos 0.993.

**Verdict: close.** Phase A stands as the account of the MOSS − Anima loss gap: it is an
HF texture floor the base cannot predict. VR-loss with a low-pass pair cannot remove it.

## Not in scope

- Wiring VR into `library/moss/train.py` (Phase B closed it).
- LPIPS or other perceptual correction (`vr_loss.md` open question).
- Per-step σ logging in Anima's `train.py`. Phase A's self-check uses early logged loss
  instead.
