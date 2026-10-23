# MOSS-Image — is the loss gap a high-frequency residual, and would VR-loss help with it?

Status: proposed 2026-10-23, unmeasured. Step 2 of 2 after
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
- **At the gradient level VR failed every criterion** (`_archive/bench/vr_grad_diagnostics/`,
  2026-05-15, from a T-LoRA and from a VR-trained LoRA). The VR gradient was 6 % of the
  standard one in norm, cos 0.03–0.04 to it, with 4.0–4.5× its relative variance and a
  direction lift of −0.18 to −0.19. VR redirects the adapter toward `x_0^H`; it does not
  denoise the standard gradient. No shipped config sets `vr_loss_weight`.

So a MOSS port needs two things the Anima record never established: that the MOSS loss
gap really is high-frequency residual (Phase A), and that VR's gradient is usable on
MOSS where it was not on Anima (Phase B). The prior for B is a fail. A goes first because
it is cheap, and it settles the step-1 question whichever way B lands.

## Phase A — split the base model's residual by frequency

Forwards of the **base** DiTs only (no LoRA), on content both lines have cached: stems
present in both `post_image_dataset/lora/` and `post_image_dataset/moss/<dir>/`
(`ama_mitsuki_layout` and `channel_(caststation)_layout` have Anima counterparts). Each
model reads its own cached latent and TE, so content matches but crops do not. That is
fine here because the quantity is per element.

- **σ grid** 0.1, 0.2, …, 0.9 for both, each model's own timestep convention (MOSS
  `1000·σ`). Same ε seed per (stem, σ, draw) within a model.
- **Size:** 24 stems × 9 σ × 2 draws = 432 forwards per model. MOSS at ~1 MP with block
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

**Gate** (`vr_grad_diagnostics` criteria, all three):

| Metric | Pass | Anima (T-LoRA) |
|---|---|---|
| `gradient_magnitude_ratio_vr_over_std` | 0.5–2 | 0.06 |
| `cov_sq_ratio_vr_over_std` | < 0.7 | 4.0 |
| `direction_lift_vr_minus_std` | > 0.05 | −0.18 |

A pass justifies wiring VR into `library/moss/train.py` and a wall-clock-matched A/B
(VR costs one extra no-grad forward per step, more under block swap). A fail closes the
line, with Phase A's split kept as the explanation of the loss gap.

## Not in scope

- Wiring VR into `library/moss/train.py`; that waits for Phase B.
- LPIPS or other perceptual correction (`vr_loss.md` open question).
- Per-step σ logging in Anima's `train.py`. Phase A's self-check uses early logged loss
  instead.
