# MOSS-Image — how much of the FLUX.2 latent is high frequency?

Status: proposed 2026-10-23, unmeasured. Step 1 of 2. Step 2, a VR-loss headroom probe
on MOSS (`docs/experimental/vr_loss.md`), is written only if this step's gate passes.
VAE encodes only, no DiT, no training.

## Why

MOSS's flow-matching loss sits well above Anima's, and nothing measured so far explains
most of the gap:

| | Anima (soup shards) | MOSS t2i (two runs) | MOSS sanitize i2i |
|---|---|---|---|
| run-mean loss | ~0.09 (`loss/average`) | 0.27–0.29 | 0.19 |
| median σ trained | 0.50 | 0.77–0.78 | 0.78 |
| loss reweighted to Anima's σ density | — | 0.22–0.27 | 0.29 |
| cached latent E[x²] (20 files each) | 0.47 | 0.75 | 0.75 |

- **σ density is ruled out.** Reweighting MOSS's per-σ-bin loss to Anima's
  `sigmoid(N(0,1))` density leaves the t2i loss at 0.22–0.27. Anima's logs carry no
  per-step σ, so only MOSS could be rebinned.
- **Latent scale explains about 1.2×.** The predict-zero loss of `u = ε − x_0` is
  `1 + E[x_0²]`, which is 1.47 for Anima and 1.75 for MOSS. That leaves roughly 2× of the
  gap unexplained.
- **The DiT is not the bottleneck either.** By AsymFlow's criterion
  (`docs/findings/asymflow_parameterization.md`), the per-patch noise dimension
  over the DiT width is 128 / 3072 = 1/24 for MOSS and 64 / 2048 = 1/32 for Anima. Both
  are in the latent regime.

**Hypothesis H.** The FLUX.2 latent packs twice the dimensions per pixel
(32 ch at /8 against Anima's 16 ch at /8), and each channel is standardised to unit
scale. So a larger share of every latent element is high-frequency texture that no
context predicts. That share sets an irreducible floor on the per-element MSE.

H matters because a high-frequency floor is exactly what VR-loss removes from the loss
and the gradient. If H fails, the higher MOSS loss is not a reason to port VR-loss.

## What is already known

- **Anima latent radial spectrum: β = 2.26, R² = 0.9994** over 200 originals; per
  artist 2.26 ± 0.08 (`_archive/spd/bench/plan.md`, method in
  `_archive/spd/bench/measure_latent_spectrum.py`). The SPD paper reports β ≈ 1.92 for
  FLUX.1 latents, but that number comes from different data.
- **No per-token PCA of either latent exists.** The "participation ratio ≈ 6.2" that
  `asymflow_parameterization.md` attributes to Anima latents was measured on
  PE-Spatial features (`_archive/bench/ip_adapter/analysis.md`). Fix that line when
  this lands.

## Measurements

All four run on **matched pixels**, so the two VAEs differ in nothing but the encoder:

- **Images:** about 200 originals from `image_dataset/`, drawn round-robin by artist.
  These are originals, not the resized tree, because resizing is a low-pass that
  steepens the tail.
- **Crop:** each image is fitted once with MOSS's `target_size` (area ≤ 1024², sides
  multiples of 16). The same tensor goes to both encoders, so both latents land on the
  same /8 grid.
- **Normalisation:** each latent is standardised as its DiT sees it. Anima uses the
  per-channel `latents_mean` / `latents_std` (`encode_pixels_to_latents`). MOSS uses
  `library/moss/model.py::encode`, which applies the BatchNorm buffers over the packed
  128 channels, and is then unpacked to 32 ch at /8 (`unpack_latents`) for M1.

| # | Quantity | Per model | Why |
|---|---|---|---|
| M0 | E[x²] and per-channel variance spread on the matched crops | scalar + 16 / 32 values | redoes the 0.47 / 0.75 cache numbers on matched pixels |
| M1 | radial power spectrum per channel on the /8 grid: β over the SPD script's fit band, plus the **HF energy share** (energy above 0.25 and above 0.5 of Nyquist ÷ total) | β, two shares | the direct test of H; β alone does not compare models with different channel counts, the shares do |
| M2 | **in-token Haar split.** The DiT token is a 2×2 latent patch (Anima 64 dims, MOSS 128). Split each patch into its 2×2 mean (LL) and the three detail bands; report the detail share of token energy | one share | the HF the patch embedding must carry *within* a token, where the noise prediction lives |
| M3 | **token PCA.** Covariance of centred token vectors pooled over all images: participation ratio `(Σλ)² / Σλ²`, rank at 90 / 95 / 99 % energy, and that rank over D | PR, three ranks | how many of the 64 / 128 token dimensions are used at all |

Output: one `result.json` envelope from `bench/moss_latent_budget/run_bench.py` (new;
`bench/_common.py`), plus per-bin CSVs and a log-log plot of both spectra on one axis.
Submit through the daemon, since it loads two VAEs on the GPU; expect a few minutes.

## Gate

The deciding number is the HF energy share ratio, MOSS ÷ Anima. Read M1's share above
0.5 Nyquist and M2's detail share together.

| Ratio | Reading | Next |
|---|---|---|
| ≥ 1.5× in both M1 and M2 | H holds: the FLUX.2 latent carries clearly more HF per element, enough to cover a large part of the unexplained ~2× | step 2: VR headroom (ρ² between the LoRA-off forward on the low-passed `x_t` and the full forward) on MOSS, at the Anima rig's settings |
| 1.2–1.5× | H is partial | step 2 is still cheap, but record the lowered expectation before running it |
| < 1.2× | H fails: HF share does not explain the loss gap | do not port VR-loss on this rationale; note the loss gap as a model/data difference |

M3 does not gate. A low PR with a high HF share would mean the extra dimensions are spent
on unstructured texture, which strengthens H. A high PR would mean the extra dimensions
carry structure, which weakens it.

## Not in scope

- Changing, compressing or distilling the VAE. The DiT is trained on this latent, so
  replacing it is a base-model change (Tier 3).
- Any DiT forward. The energy shares are a necessary condition only; whether the DiT
  can predict that energy is what step 2's ρ² measures.
- A per-σ loss curve for Anima. It would need σ logged per step in `train.py`, which is
  a separate change.
