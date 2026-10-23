#!/usr/bin/env python3
"""How much of the FLUX.2 (MOSS) latent is high frequency, against Anima's Qwen-Image latent?

Step 1 of ``docs/proposal/moss_latent_hf_budget.md``: VAE encodes only, no DiT. Each
original from ``image_dataset/`` (round-robin by artist folder) is fitted once with
MOSS's ``target_size`` (area ≤ resolution², sides multiples of 16) and the same pixels
go through both encoders, each latent standardised as its DiT sees it:

  * Anima: ``encode_pixels_to_latents`` (per-channel ``latents_mean`` / ``latents_std``),
    ``(16, H/8, W/8)``; DiT token = 2×2 patch, 64 dims.
  * MOSS: ``library.moss.model.encode`` (BatchNorm over the packed 128 channels),
    ``(128, H/16, W/16)``; DiT token = the packed vector. The /8 grid for M1 is
    ``unpatchify`` of that normalised latent (the BN stats differ across the 2×2
    phases by < 1 %, logged as ``bn_phase_spread``, so no phase pattern is injected).

  M0  E[x²], per-channel mean / variance and its spread
  M1  radial power spectrum per channel: β over the SPD fit band and the HF energy share
      above 0.25 / 0.5 of Nyquist (Parseval: shares of Σx², DC included)
  M2  in-token Haar split: share of token energy in the three 2×2 detail bands,
      raw and after centring each token dim on its pooled mean
  M3  token PCA: participation ratio, rank at 90 / 95 / 99 % energy, rank / D

Gate: MOSS ÷ Anima of M1's and M2's centred shares (share above 0.5 Nyquist; detail share).

    make daemon-run ARGS="bench/moss_latent_budget/run_bench.py"
    make daemon-run ARGS="bench/moss_latent_budget/run_bench.py --num_images 50 --label smoke"

Writes ``result.json``, ``per_image.csv``, ``radial_profile.csv``,
``channel_stats.csv``, ``pca_spectrum.csv`` and ``radial_spectrum.png``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch

from bench._anima import DEFAULT_VAE
from bench._common import make_run_dir, start_heartbeat, write_result
from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.env import resolve_under_home

log = logging.getLogger("bench.moss_latent_budget")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
HF_CUTS = (0.25, 0.5)


def discover_images(root: Path, n: int, seed: int) -> list[Path]:
    """Round-robin across artist folders (``_archive/spd/bench`` recipe)."""
    exts = {e.lower() for e in IMAGE_EXTENSIONS}
    by_folder: dict[Path, list[Path]] = {}
    for p in root.rglob("*", recurse_symlinks=True):
        if p.suffix.lower() in exts:
            by_folder.setdefault(p.parent, []).append(p)
    rng = random.Random(seed)
    for paths in by_folder.values():
        paths.sort()
        rng.shuffle(paths)
    folders = sorted(by_folder)
    rng.shuffle(folders)
    picks: list[Path] = []
    i = 0
    while len(picks) < n and any(by_folder.values()):
        f = folders[i % len(folders)]
        if by_folder[f]:
            picks.append(by_folder[f].pop())
        i += 1
    return picks


class Budget:
    """Streaming M0–M3 accumulators for one model's normalised latent."""

    def __init__(self, name: str, channels: int, n_bins: int):
        self.name, self.c, self.n_bins = name, channels, n_bins
        self.d = channels * 4
        self.n_el = 0
        self.ch_sum = np.zeros(channels)
        self.ch_sq = np.zeros(channels)
        self.tok_sum = np.zeros(self.d)
        self.tok_outer = np.zeros((self.d, self.d))
        self.n_tok = 0
        self.band_e = np.zeros(4)  # LL, LH, HL, HH energy summed over tokens
        self.power_total = 0.0
        self.power_hf = np.zeros(len(HF_CUTS))
        self.sum_logp = np.zeros((channels, n_bins))
        self.cnt_logp = np.zeros((channels, n_bins))
        self.sum_p = np.zeros(n_bins)  # pooled per-element power per bin
        self.rows: list[dict] = []

    def add(self, grid: np.ndarray) -> dict:
        """``grid``: ``(C, h, w)`` float64 on the /8 latent grid, h and w even."""
        c, h, w = grid.shape
        n = h * w
        self.n_el += c * n
        self.ch_sum += grid.sum(axis=(1, 2))
        self.ch_sq += (grid**2).sum(axis=(1, 2))
        ex2 = float((grid**2).mean())

        # M1: per-element power spectral density, sum over bins = Σx² / n.
        power = np.abs(np.fft.fftshift(np.fft.fft2(grid), axes=(1, 2))) ** 2 / n
        ky = (np.arange(h) - h // 2) / (h / 2.0)
        kx = (np.arange(w) - w // 2) / (w / 2.0)
        kr = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).ravel()
        tot = power.sum()
        hf = np.array([power.reshape(c, -1)[:, kr > cut].sum() for cut in HF_CUTS])
        self.power_total += tot
        self.power_hf += hf
        idx = np.minimum(
            (kr * self.n_bins).astype(int), self.n_bins
        )  # n_bins = past Nyquist
        for ch in range(c):
            s = np.bincount(idx, weights=power[ch].ravel(), minlength=self.n_bins + 1)
            k = np.bincount(idx, minlength=self.n_bins + 1)
            prof = s[: self.n_bins] / np.maximum(k[: self.n_bins], 1)
            good = (k[: self.n_bins] > 0) & (prof > 0)
            self.sum_logp[ch, good] += np.log10(prof[good])
            self.cnt_logp[ch, good] += 1
            self.sum_p += s[: self.n_bins]

        # M2 / M3: tokens are 2×2 patches, (C, 2, 2) per token.
        t = grid.reshape(c, h // 2, 2, w // 2, 2).transpose(1, 3, 0, 2, 4)
        t = t.reshape(-1, c, 2, 2)
        a, b, cc, d = t[..., 0, 0], t[..., 0, 1], t[..., 1, 0], t[..., 1, 1]
        bands = np.stack(
            [
                (a + b + cc + d) / 2,
                (a - b + cc - d) / 2,
                (a + b - cc - d) / 2,
                (a - b - cc + d) / 2,
            ]
        )  # orthonormal Haar: energies add up to the token's
        be = (bands**2).sum(axis=(1, 2))
        self.band_e += be
        flat = t.reshape(t.shape[0], -1)
        self.tok_sum += flat.sum(axis=0)
        self.tok_outer += flat.T @ flat
        self.n_tok += flat.shape[0]

        row = {
            f"{self.name}_ex2": ex2,
            **{f"{self.name}_hf{cut}": float(e / tot) for cut, e in zip(HF_CUTS, hf)},
            f"{self.name}_detail": float(be[1:].sum() / be.sum()),
        }
        self.rows.append(row)
        return row

    def k_centres(self) -> np.ndarray:
        return (np.arange(self.n_bins) + 0.5) / self.n_bins

    def logp(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sum_logp / np.where(self.cnt_logp > 0, self.cnt_logp, np.nan)

    def metrics(self, band: tuple[float, float]) -> dict:
        mean = self.ch_sum / (self.n_el / self.c)
        ex2_ch = self.ch_sq / (self.n_el / self.c)
        var = ex2_ch - mean**2
        m0 = {
            "ex2": float(self.ch_sq.sum() / self.n_el),
            "ex2_per_image_mean": float(
                np.mean([r[f"{self.name}_ex2"] for r in self.rows])
            ),
            "predict_zero_loss": float(1 + self.ch_sq.sum() / self.n_el),
            "mean_sq_share": float((mean**2).sum() / ex2_ch.sum()),
            "var_min": float(var.min()),
            "var_max": float(var.max()),
            "var_max_over_min": float(var.max() / var.min()),
            "var_cv": float(var.std() / var.mean()),
            "per_channel_mean": mean.tolist(),
            "per_channel_var": var.tolist(),
        }

        k = self.k_centres()
        logp = self.logp()
        logp_mean = np.nanmean(logp, axis=0)
        beta, r2, n_pts = fit_slope(k, logp_mean, *band)
        ch_beta = [fit_slope(k, logp[ch], *band)[0] for ch in range(self.c)]
        ch_beta = [b for b in ch_beta if b is not None]
        # Subtracting each channel's pooled mean only moves the DC bins, so the centred
        # shares keep the HF energy and drop Σ mean² from the total.
        centred_total = self.power_total - (self.n_el / self.c) * (mean**2).sum()
        m1 = {
            "beta": beta,
            "beta_r2": r2,
            "beta_fit_points": n_pts,
            "beta_per_channel_mean": float(np.mean(ch_beta)),
            "beta_per_channel_std": float(np.std(ch_beta)),
            **{
                f"hf_share_above_{cut}": float(e / self.power_total)
                for cut, e in zip(HF_CUTS, self.power_hf)
            },
            **{
                f"hf_share_above_{cut}_centred": float(e / centred_total)
                for cut, e in zip(HF_CUTS, self.power_hf)
            },
            **{
                f"hf_energy_above_{cut}_per_element": float(e / self.n_el)
                for cut, e in zip(HF_CUTS, self.power_hf)
            },
            **{
                f"hf_share_above_{cut}_per_image_mean": float(
                    np.mean([r[f"{self.name}_hf{cut}"] for r in self.rows])
                )
                for cut in HF_CUTS
            },
        }

        mu = self.tok_sum / self.n_tok
        second = self.tok_outer / self.n_tok
        cov = second - np.outer(mu, mu)
        # Centred Haar bands: subtract each token dim's pooled mean first. The token
        # layout is (C, 2, 2) row-major, so the mean patch reshapes the same way.
        mp = mu.reshape(self.c, 2, 2)
        a, b, cc, d = mp[:, 0, 0], mp[:, 0, 1], mp[:, 1, 0], mp[:, 1, 1]
        mean_bands = np.stack(
            [
                (a + b + cc + d) / 2,
                (a - b + cc - d) / 2,
                (a + b - cc - d) / 2,
                (a - b - cc + d) / 2,
            ]
        )
        centred = self.band_e / self.n_tok - (mean_bands**2).sum(axis=1)
        m2 = {
            "detail_share": float(self.band_e[1:].sum() / self.band_e.sum()),
            "detail_share_centred": float(centred[1:].sum() / centred.sum()),
            "band_energy_per_token": dict(
                zip(("LL", "LH", "HL", "HH"), (self.band_e / self.n_tok).tolist())
            ),
            "detail_share_per_image_mean": float(
                np.mean([r[f"{self.name}_detail"] for r in self.rows])
            ),
        }

        lam = np.clip(np.linalg.eigvalsh(cov)[::-1], 0, None)
        cum = np.cumsum(lam) / lam.sum()
        ranks = {
            f"rank_{int(q * 100)}": int(np.searchsorted(cum, q) + 1)
            for q in (0.9, 0.95, 0.99)
        }
        m3 = {
            "dim": self.d,
            "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
            "pr_over_dim": float(lam.sum() ** 2 / (lam**2).sum() / self.d),
            **ranks,
            **{f"{k}_over_dim": v / self.d for k, v in ranks.items()},
            "n_tokens": self.n_tok,
        }
        self.eigvals = lam
        return {"m0": m0, "m1": m1, "m2": m2, "m3": m3}


def fit_slope(k: np.ndarray, logp: np.ndarray, lo: float, hi: float):
    """``log10 P = a − β·log10 k`` over ``k ∈ [lo, hi]`` (SPD bench's fit)."""
    band = (k >= lo) & (k <= hi) & np.isfinite(logp)
    if band.sum() < 4:
        return None, None, int(band.sum())
    x = np.log10(k[band])
    y = logp[band]
    A = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    ss_res = float(((y - A @ coef) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(-coef[0]), 1.0 - ss_res / ss_tot, int(band.sum())


def gate(ratio_m1: float, ratio_m2: float) -> str:
    lo = min(ratio_m1, ratio_m2)
    if lo >= 1.5:
        return "H holds (both ratios >= 1.5): run step 2, the VR headroom probe on MOSS"
    if lo >= 1.2:
        return "H partial (min ratio 1.2-1.5): step 2 still cheap; record the lowered expectation first"
    return "H fails (a ratio < 1.2): do not port VR-loss on this rationale"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image_dir", default="image_dataset")
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resolution", type=int, default=1024, help="MOSS target_size area cap edge"
    )
    ap.add_argument("--anima_vae", default=DEFAULT_VAE)
    ap.add_argument(
        "--anima_dtype", default="bf16", choices=DTYPES, help="shipped preprocess: bf16"
    )
    ap.add_argument("--moss_model_dir", default=None)
    ap.add_argument(
        "--moss_dtype", default="fp32", choices=DTYPES, help="shipped MOSS cache: fp32"
    )
    ap.add_argument("--n_bins", type=int, default=96)
    ap.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=(0.06, 0.5),
        help="β fit band, fraction of Nyquist",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from anima_lora import load_vae as load_anima_vae
    from library.moss.cache import fit, load_rgb, target_size, to_pixels
    from library.moss.loader import load_vae as load_moss_vae
    from library.moss.model import encode as moss_encode
    from library.moss.model import unpatchify
    from library.moss.requests import resolve_model_dir

    start_heartbeat(label="moss_latent_budget")
    picks = discover_images(
        resolve_under_home(args.image_dir), args.num_images, args.seed
    )
    if not picks:
        raise SystemExit(f"no images under {args.image_dir}")
    log.info(f"{len(picks)} images from {len({p.parent for p in picks})} folders")

    anima_vae = load_anima_vae(
        args.anima_vae,
        device=args.device,
        dtype=DTYPES[args.anima_dtype],
        eval=True,
        vae_2d=True,
    )
    anima_vae.requires_grad_(False)
    moss_dir = resolve_model_dir(args.moss_model_dir)
    moss_vae = load_moss_vae(
        moss_dir, dtype=DTYPES[args.moss_dtype], device=args.device
    ).eval()
    moss_vae.requires_grad_(False)

    bn_mean = moss_vae.bn.running_mean.float().view(32, 4)
    bn_sd = (
        (moss_vae.bn.running_var.float() + moss_vae.config.batch_norm_eps)
        .sqrt()
        .view(32, 4)
    )
    bn_phase_spread = {
        "mean_std_over_sd_max": float((bn_mean.std(1) / bn_sd.mean(1)).max()),
        "sd_rel_std_max": float((bn_sd.std(1) / bn_sd.mean(1)).max()),
    }

    anima = Budget("anima", 16, args.n_bins)
    moss = Budget("moss", 32, args.n_bins)
    rows = []
    for i, path in enumerate(picks):
        try:
            image = load_rgb(path)
        except Exception as e:
            log.warning(f"skip {path}: {e}")
            continue
        w, h = target_size(*image.size, args.resolution)
        image = fit(image, w, h)
        with torch.no_grad():
            px = (
                IMAGE_TRANSFORMS(np.asarray(image))
                .unsqueeze(0)
                .to(args.device, DTYPES[args.anima_dtype])
            )
            za = anima_vae.encode_pixels_to_latents(px)[0].double().cpu().numpy()
            zm = (
                unpatchify(moss_encode(to_pixels(image), moss_vae))[0]
                .double()
                .cpu()
                .numpy()
            )
        assert za.shape[1:] == zm.shape[1:] == (h // 8, w // 8), (
            za.shape,
            zm.shape,
            (w, h),
        )
        row = {"path": str(path), "width": w, "height": h}
        row.update(anima.add(za))
        row.update(moss.add(zm))
        rows.append(row)
        if (i + 1) % 25 == 0:
            log.info(
                f"  {i + 1}/{len(picks)}  anima E[x²] {row['anima_ex2']:.3f}  moss {row['moss_ex2']:.3f}"
            )

    band = tuple(args.band)
    res = {"anima": anima.metrics(band), "moss": moss.metrics(band)}
    ratio = {
        "m0_ex2": res["moss"]["m0"]["ex2"] / res["anima"]["m0"]["ex2"],
        "m0_predict_zero_loss": res["moss"]["m0"]["predict_zero_loss"]
        / res["anima"]["m0"]["predict_zero_loss"],
        **{
            f"m1_hf_share_above_{cut}": res["moss"]["m1"][f"hf_share_above_{cut}"]
            / res["anima"]["m1"][f"hf_share_above_{cut}"]
            for cut in HF_CUTS
        },
        **{
            f"m1_hf_share_above_{cut}_centred": res["moss"]["m1"][
                f"hf_share_above_{cut}_centred"
            ]
            / res["anima"]["m1"][f"hf_share_above_{cut}_centred"]
            for cut in HF_CUTS
        },
        "m2_detail_share": res["moss"]["m2"]["detail_share"]
        / res["anima"]["m2"]["detail_share"],
        "m2_detail_share_centred": res["moss"]["m2"]["detail_share_centred"]
        / res["anima"]["m2"]["detail_share_centred"],
        "m3_participation_ratio": res["moss"]["m3"]["participation_ratio"]
        / res["anima"]["m3"]["participation_ratio"],
        "m3_pr_over_dim": res["moss"]["m3"]["pr_over_dim"]
        / res["anima"]["m3"]["pr_over_dim"],
    }
    verdict = gate(
        ratio["m1_hf_share_above_0.5_centred"], ratio["m2_detail_share_centred"]
    )

    run_dir = make_run_dir("moss_latent_budget", label=args.label)
    with open(run_dir / "per_image.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    k = anima.k_centres()
    lp_a, lp_m = anima.logp(), moss.logp()
    with open(run_dir / "radial_profile.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "k_norm",
                "anima_logP_mean",
                "moss_logP_mean",
                "anima_pooled_power",
                "moss_pooled_power",
            ]
            + [f"anima_ch{c}" for c in range(16)]
            + [f"moss_ch{c}" for c in range(32)]
        )
        pa, pm = np.nanmean(lp_a, axis=0), np.nanmean(lp_m, axis=0)
        for b in range(args.n_bins):
            wr.writerow(
                [
                    f"{k[b]:.5f}",
                    f"{pa[b]:.5f}",
                    f"{pm[b]:.5f}",
                    f"{anima.sum_p[b]:.6g}",
                    f"{moss.sum_p[b]:.6g}",
                ]
                + [f"{v:.5f}" for v in lp_a[:, b]]
                + [f"{v:.5f}" for v in lp_m[:, b]]
            )

    with open(run_dir / "channel_stats.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "channel", "mean", "var"])
        for name in ("anima", "moss"):
            m0 = res[name]["m0"]
            for c, (mu, var) in enumerate(
                zip(m0["per_channel_mean"], m0["per_channel_var"])
            ):
                wr.writerow([name, c, f"{mu:.6f}", f"{var:.6f}"])

    with open(run_dir / "pca_spectrum.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "index", "eigval", "cum_share"])
        for name, bud in (("anima", anima), ("moss", moss)):
            cum = np.cumsum(bud.eigvals) / bud.eigvals.sum()
            for j, (lam, cs) in enumerate(zip(bud.eigvals, cum)):
                wr.writerow([name, j, f"{lam:.6g}", f"{cs:.6f}"])

    artifacts = [
        "per_image.csv",
        "radial_profile.csv",
        "channel_stats.csv",
        "pca_spectrum.csv",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for name, lp, colour in (
            ("Anima (16 ch)", pa, "#1f4e8c"),
            ("MOSS / FLUX.2 (32 ch)", pm, "#c0392b"),
        ):
            key = "anima" if name.startswith("Anima") else "moss"
            beta = res[key]["m1"]["beta"]
            ax.plot(k, lp, "o-", ms=2.5, color=colour, label=f"{name}, β = {beta:.2f}")
        for lo_hi in band:
            ax.axvline(lo_hi, color="0.6", lw=0.8, ls=":")
        for cut in HF_CUTS:
            ax.axvline(cut, color="0.3", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("radial frequency / Nyquist")
        ax.set_ylabel("log10 power per element (mean over channels, images)")
        ax.set_title(f"Matched crops, {len(rows)} images")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "radial_spectrum.png", dpi=130)
        artifacts.append("radial_spectrum.png")
    except ImportError:
        log.warning("matplotlib missing: no plot")

    metrics = {
        **res,
        "ratio_moss_over_anima": ratio,
        "verdict": verdict,
        "n_images": len(rows),
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=args.device,
        extra={"bn_phase_spread": bn_phase_spread, "moss_model_dir": str(moss_dir)},
    )

    log.info(f"\n=== {len(rows)} images → {run_dir}")
    for name in ("anima", "moss"):
        r = res[name]
        log.info(
            f"{name:5s}  M0 E[x²] {r['m0']['ex2']:.3f} (var spread {r['m0']['var_min']:.2f}–{r['m0']['var_max']:.2f})"
            f"  M1 β {r['m1']['beta']:.2f} HF>0.25 {r['m1']['hf_share_above_0.25']:.4f}"
            f" HF>0.5 {r['m1']['hf_share_above_0.5']:.4f} (centred {r['m1']['hf_share_above_0.5_centred']:.4f})"
            f"  M2 detail {r['m2']['detail_share']:.4f} (centred {r['m2']['detail_share_centred']:.4f})"
            f"  M3 PR {r['m3']['participation_ratio']:.1f}/{r['m3']['dim']}"
            f" r90/95/99 {r['m3']['rank_90']}/{r['m3']['rank_95']}/{r['m3']['rank_99']}"
        )
    log.info("ratio MOSS/Anima: " + "  ".join(f"{k} {v:.3f}" for k, v in ratio.items()))
    log.info(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
