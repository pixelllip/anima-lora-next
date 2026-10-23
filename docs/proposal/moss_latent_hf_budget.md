# MOSS-Image — how much of the FLUX.2 latent is high frequency?

Status: **measured 2026-10-23, gate passes (H holds, min ratio 2.33×)** — § Results.
Step 1 of 2. Step 2 is `docs/proposal/moss_vr_headroom.md`. VAE encodes only, no DiT,
no training.

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
- **No per-token PCA of either latent existed before this step.** The "participation
  ratio ≈ 6.2" that `asymflow_parameterization.md` attributed to Anima latents was
  measured on PE-Spatial features (`_archive/bench/ip_adapter/analysis.md`); that line
  now cites M3's 4.1 / 64.

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
  128 channels; M1 reads that normalised latent un-patchified to 32 ch at /8
  (`unpatchify`, not `unpack_latents`, which also undoes the BN). The BN stats of one
  channel differ across its four 2×2 phases by ≤ 1.1 % (`bn_phase_spread` in the
  envelope), so the un-patchified grid carries no phase pattern at Nyquist.

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
0.5 Nyquist and M2's detail share together, both **centred**: each channel (M1) or token
dim (M2) has its pooled mean taken out before the share is formed. A constant offset is
predictable, and Anima's channel means are far enough from zero (45 % of its E[x²]) that
the uncentred shares overstate the ratio.

| Ratio | Reading | Next |
|---|---|---|
| ≥ 1.5× in both M1 and M2 | H holds: the FLUX.2 latent carries clearly more HF per element, enough to cover a large part of the unexplained ~2× | step 2: VR headroom (ρ² between the LoRA-off forward on the low-passed `x_t` and the full forward) on MOSS, at the Anima rig's settings |
| 1.2–1.5× | H is partial | step 2 is still cheap, but record the lowered expectation before running it |
| < 1.2× | H fails: HF share does not explain the loss gap | do not port VR-loss on this rationale; note the loss gap as a model/data difference |

M3 does not gate. A low PR with a high HF share would mean the extra dimensions are spent
on unstructured texture, which strengthens H. A high PR would mean the extra dimensions
carry structure, which weakens it.

## Results (2026-10-23)

`make daemon-run ARGS="bench/moss_latent_budget/run_bench.py"`, run
`bench/moss_latent_budget/results/20261023-2321/` (gitignored; the numbers below are the
record). 200 originals from 91 artist folders, seed 0, `target_size` at 1024, Anima VAE
bf16 + 2D fold (shipped preprocess), FLUX.2 VAE fp32 (shipped MOSS cache), 809,680
tokens per model. A rerun gives the same numbers to the last digit.

| | Anima | MOSS | MOSS ÷ Anima |
|---|---|---|---|
| **M0** E[x²] | 0.426 | 0.769 | 1.81× |
| M0 predict-zero loss `1 + E[x²]` | 1.43 | 1.77 | 1.24× |
| M0 per-channel variance, min–max | 0.10–0.42 (16 ch) | 0.46–1.23 (32 ch) | |
| M0 share of E[x²] in the channel means | 0.45 | 0.09 | |
| **M1** β over 0.06–0.5 Nyquist (R²) | 2.22 (0.999) | 1.71 (0.994) | |
| M1 share above 0.25 Nyquist, centred (raw) | 0.201 (0.110) | 0.466 (0.422) | 2.32× (3.82×) |
| **M1 share above 0.5 Nyquist, centred** (raw) | 0.104 (0.057) | 0.264 (0.239) | **2.55×** (4.20×) |
| M1 energy above 0.5 Nyquist per element | 0.024 | 0.184 | 7.6× |
| **M2 Haar detail share, centred** (raw) | 0.122 (0.067) | 0.285 (0.258) | **2.33×** (3.85×) |
| **M3** participation ratio / D | 4.1 / 64 | 23.3 / 128 | 5.7× (2.9× over D) |
| M3 rank at 90 / 95 / 99 % | 10 / 18 / 39 | 65 / 90 / 119 | |

- **The gate passes: H holds.** Both centred ratios are above 1.5×, and the raw ones are
  higher still.
- **M0 reproduces the cache numbers** (0.47 / 0.75) on matched pixels, so the
  1.2× latent-scale part of the gap stands.
- **The HF energy gap is about the size of the loss gap.** Energy above 0.5 Nyquist is
  0.184 − 0.024 ≈ 0.16 per element higher on MOSS, close to the ~0.13–0.18 left between
  Anima's ~0.09 and MOSS's σ-reweighted 0.22–0.27. This is circumstantial: it holds only
  if that band is mostly unpredictable, which no VAE-only number can show.
- **β:** Anima's 2.22 agrees with the SPD bench's 2.26 (that bench used longest side
  1536). The FLUX.2 latent is flatter at 1.71, below the FLUX.1 1.92 the SPD paper reports.
- **M3 weakens H, read as the gate table says.** MOSS uses nearly all 128 token dims
  (rank 119 at 99 %), Anima about 40 of 64. A participation ratio cannot tell
  structure from noise-like texture, though, and the HF numbers above favour texture.
  Step 2 decides which.

## Not in scope

- Changing, compressing or distilling the VAE. The DiT is trained on this latent, so
  replacing it is a base-model change (Tier 3).
- Any DiT forward. The energy shares are a necessary condition only; whether the DiT
  can predict that energy is what step 2's ρ² measures.
- A per-σ loss curve for Anima. It would need σ logged per step in `train.py`, which is
  a separate change.
