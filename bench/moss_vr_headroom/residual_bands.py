#!/usr/bin/env python3
"""Phase A of ``docs/proposal/moss_vr_headroom.md``: is the MOSS − Anima loss gap HF residual?

Base DiTs only, no adapter, forwards under ``no_grad``. Content is matched, crops are
not: the stems cached by both lines (``post_image_dataset/lora/<artist>/`` and
``post_image_dataset/moss/<artist>_layout/``), each model reading its own cached latent
and main caption embedding.

Per (stem, σ, draw), ``x_t = (1 − σ)·x_0 + σ·ε`` and the FM residual
``y = u_pred − (ε − x_0)``, per element:

  * ``E[y²]``, the training loss at that σ;
  * its energy above 0.25 and 0.5 of Nyquist on the /8 latent grid (radial FFT as in
    ``bench/moss_latent_budget``; MOSS's packed residual is ``unpatchify``-ed first).

σ sits at the midpoints of ten equal cells over [0, 1]. Each cell's weight is that
cell's probability mass under a training density, read off its CDF: Anima's
``sigmoid(N(0, 1))`` and MOSS's ``1 − sigmoid(N(−1.09, 0.95))`` truncated at 0.95,
rescaled to [0, 1], with a 1 % uniform mixture (``library/moss/train.py::sample_sigma``).

Gate (both weighted to Anima's density): share of the MOSS − Anima residual gap carried
above 0.5 Nyquist. ≥ 0.5 → Phase B; 0.25–0.5 → Phase B with a lowered expectation;
< 0.25 → close the VR line for MOSS.

    make daemon-run ARGS="bench/moss_vr_headroom/residual_bands.py"
    make daemon-run ARGS="bench/moss_vr_headroom/residual_bands.py --per_artist 2 --draws 1 --label smoke"

Writes ``result.json``, ``per_forward.csv``, ``per_sigma.csv`` and ``residual_by_sigma.png``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import zlib

import numpy as np
import torch

from bench._anima import add_common_args, add_model_args, build_anima
from bench._common import make_run_dir, start_heartbeat, write_result
from library.env import resolve_under_home

log = logging.getLogger("bench.moss_vr_headroom.residual_bands")
logging.basicConfig(level=logging.INFO, format="%(message)s")

HF_CUTS = (0.25, 0.5)
SIGMAS = tuple((k + 0.5) / 10 for k in range(10))
# library/moss/requests.py TrainRequest defaults
MOSS_MU, MOSS_S, MOSS_CUTOFF, MOSS_UNIFORM = -1.09, 0.95, 0.95, 0.01


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _logit(s: float) -> float:
    s = min(max(s, 1e-12), 1 - 1e-12)
    return math.log(s / (1 - s))


def anima_cdf(s: float) -> float:
    return _phi(_logit(s))


def moss_cdf(s: float) -> float:
    """σ = sigmoid(−Z), so logit σ ~ N(−mu, s²); truncate σ ≤ cutoff, rescale by it."""

    def raw(u: float) -> float:
        return _phi((_logit(u) + MOSS_MU) / MOSS_S)

    trunc = raw(MOSS_CUTOFF * s) / raw(MOSS_CUTOFF)
    return (1 - MOSS_UNIFORM) * trunc + MOSS_UNIFORM * s


def cell_weights(cdf) -> np.ndarray:
    edges = [k / 10 for k in range(11)]
    w = np.array([cdf(edges[k + 1]) - cdf(edges[k]) for k in range(10)])
    return w / w.sum()


def band_energies(r: torch.Tensor) -> tuple[float, ...]:
    """``r (C, h, w)`` fp32 on the /8 grid → per-element ``(total, >0.25, >0.5)``."""
    c, h, w = r.shape
    power = torch.fft.fft2(r).abs().square() / (h * w)  # sums to Σ r²
    ky = torch.fft.fftfreq(h, device=r.device) * 2  # Nyquist = 1
    kx = torch.fft.fftfreq(w, device=r.device) * 2
    kr = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    n = c * h * w
    out = [float(power.sum()) / n]
    for cut in HF_CUTS:
        out.append(float(power[:, kr > cut].sum()) / n)
    return tuple(out)


def stems_by_artist(artists: list[str], per_artist: int, seed: int) -> dict[str, dict]:
    """``{stem: {artist, anima: (npz, te), moss: CacheItem}}`` from both caches."""
    from bench.soft_tokens_contrastive.reward_premise_probe import discover_pairs
    from library.moss.cache import list_items

    rng = np.random.default_rng(seed)
    picks = {}
    for artist in artists:
        anima = discover_pairs(
            str(resolve_under_home(f"post_image_dataset/lora/{artist}"))
        )
        moss = {
            it.stem: it
            for it in list_items(
                resolve_under_home(f"post_image_dataset/moss/{artist}_layout")
            )
        }
        both = sorted(set(anima) & set(moss))
        take = [
            both[i]
            for i in rng.choice(len(both), min(per_artist, len(both)), replace=False)
        ]
        log.info(f"{artist}: {len(both)} stems in both caches, {len(take)} taken")
        for stem in take:
            picks[stem] = {"artist": artist, "anima": anima[stem], "moss": moss[stem]}
    return picks


def noise_like(shape, stem: str, si: int, draw: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(
        (zlib.crc32(stem.encode()) + 7919 * si + 104729 * draw) & 0x7FFFFFFF
    )
    return torch.randn(shape, generator=g)


def run_anima(args, picks: dict, rows: list[dict]) -> None:
    from library.io.cache import load_cached_crossattn_emb, load_cached_latents
    from library.runtime.device import clean_memory_on_device

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device, dtype = bundle.anima, bundle.device, bundle.dtype
    if hasattr(anima, "prepare_block_swap_before_forward"):
        anima.prepare_block_swap_before_forward()
    try:
        for n, (stem, p) in enumerate(picks.items()):
            npz, te = p["anima"]
            lat, *_ = load_cached_latents(npz)
            x0 = lat.float().to(device).unsqueeze(0)  # (1, 16, h, w)
            emb = load_cached_crossattn_emb(te).to(device, dtype).unsqueeze(0)
            pad = torch.zeros(1, 1, *x0.shape[-2:], dtype=dtype, device=device)
            x0_e = band_energies(x0[0])
            for si, sigma in enumerate(SIGMAS):
                t_b = torch.full((1,), sigma, device=device, dtype=dtype)
                for d in range(args.draws):
                    eps = noise_like(x0.shape, stem, si, d).to(device)
                    x_t = ((1 - sigma) * x0 + sigma * eps).to(dtype)
                    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                        v = anima.forward_mini_train_dit(
                            x_t.unsqueeze(2),
                            t_b,
                            emb,
                            padding_mask=pad,
                            skip_pooled_text_proj=True,
                        )
                    y = v.squeeze(2).float() - (eps - x0)
                    rows.append(
                        _row(
                            "anima",
                            stem,
                            p["artist"],
                            sigma,
                            d,
                            x0.shape,
                            band_energies(y[0]),
                            x0_e,
                        )
                    )
            if (n + 1) % 6 == 0:
                log.info(f"  anima {n + 1}/{len(picks)}")
    finally:
        del bundle, anima
        clean_memory_on_device(torch.device(args.device))


def run_moss(args, picks: dict, rows: list[dict]) -> None:
    from library.moss.cache import load_item
    from library.moss.loader import load_transformer
    from library.moss.model import unpatchify
    from library.moss.requests import resolve_model_dir

    device = torch.device(args.device)
    log.info(
        f"moss: {torch.cuda.memory_allocated() / 1024**3:.2f} GB still allocated before load"
    )
    transformer = load_transformer(
        resolve_model_dir(args.moss_model_dir), args.moss_model, device=device
    )
    transformer.requires_grad_(False)
    try:
        for n, (stem, p) in enumerate(picks.items()):
            latent, embeds = load_item(p["moss"])
            x0 = latent.unsqueeze(0).to(device, torch.float32)  # (1, 128, h, w)
            embeds = embeds.unsqueeze(0).to(device)
            x0_e = band_energies(unpatchify(x0)[0])
            for si, sigma in enumerate(SIGMAS):
                timestep = torch.tensor([1000 * sigma], device=device)
                for d in range(args.draws):
                    eps = noise_like(x0.shape, stem, si, d).to(device)
                    x_t = (1 - sigma) * x0 + sigma * eps
                    with torch.no_grad():
                        pred = transformer(x_t, timestep, embeds, return_dict=False)[0]
                    y = unpatchify(pred.float() - (eps - x0))
                    rows.append(
                        _row(
                            "moss",
                            stem,
                            p["artist"],
                            sigma,
                            d,
                            unpatchify(x0).shape,
                            band_energies(y[0]),
                            x0_e,
                        )
                    )
            if (n + 1) % 6 == 0:
                log.info(f"  moss {n + 1}/{len(picks)}")
    finally:
        del transformer
        torch.cuda.empty_cache()


def _row(model, stem, artist, sigma, draw, shape, y_e, x0_e) -> dict:
    return {
        "model": model,
        "stem": stem,
        "artist": artist,
        "sigma": sigma,
        "draw": draw,
        "grid_h": shape[-2],
        "grid_w": shape[-1],
        "y2": y_e[0],
        **{f"y2_hf{cut}": e for cut, e in zip(HF_CUTS, y_e[1:])},
        "x0_2": x0_e[0],
        **{f"x0_2_hf{cut}": e for cut, e in zip(HF_CUTS, x0_e[1:])},
    }


def per_sigma(rows: list[dict], model: str) -> dict[str, np.ndarray]:
    keys = ["y2", *(f"y2_hf{c}" for c in HF_CUTS)]
    out = {k: np.zeros(len(SIGMAS)) for k in keys}
    for si, sigma in enumerate(SIGMAS):
        sel = [r for r in rows if r["model"] == model and r["sigma"] == sigma]
        for k in keys:
            out[k][si] = np.mean([r[k] for r in sel])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap, vae=False, text_encoder=False)
    ap.add_argument("--artists", default="ama_mitsuki,channel_(caststation)")
    ap.add_argument("--per_artist", type=int, default=12)
    ap.add_argument("--draws", type=int, default=2, help="ε draws per (stem, σ)")
    ap.add_argument("--moss_model_dir", default=None)
    ap.add_argument("--moss_model", default="base", choices=("base", "sft"))
    add_common_args(ap)
    args = ap.parse_args()

    start_heartbeat(label="residual_bands")
    picks = stems_by_artist(
        [a for a in args.artists.split(",") if a], args.per_artist, args.seed
    )
    if not picks:
        raise SystemExit("no stems cached by both lines")
    rows: list[dict] = []
    log.info(f"{len(picks)} stems × {len(SIGMAS)} σ × {args.draws} draws per model")
    run_anima(args, picks, rows)
    run_moss(args, picks, rows)

    w_anima, w_moss = cell_weights(anima_cdf), cell_weights(moss_cdf)
    curves = {m: per_sigma(rows, m) for m in ("anima", "moss")}
    weighted = {
        m: {
            density: {k: float(v @ w) for k, v in curves[m].items()}
            for density, w in (("anima_density", w_anima), ("moss_density", w_moss))
        }
        for m in curves
    }
    a, m = weighted["anima"]["anima_density"], weighted["moss"]["anima_density"]
    gap = m["y2"] - a["y2"]
    gap_share = {
        f"hf{c}": (m[f"y2_hf{c}"] - a[f"y2_hf{c}"]) / gap if gap > 0 else None
        for c in HF_CUTS
    }
    share = gap_share["hf0.5"]
    if share is None:
        verdict = (
            "no residual gap at Anima's σ density: nothing for the HF band to explain"
        )
    elif share >= 0.5:
        verdict = "the loss gap is an HF floor (>= 50 % above 0.5 Nyquist): Phase B"
    elif share >= 0.25:
        verdict = (
            "partial (25-50 % above 0.5 Nyquist): Phase B, expectation recorded first"
        )
    else:
        verdict = "the gap sits below 0.5 Nyquist (< 25 %): close the VR line for MOSS"
    x0 = {
        model: {
            k: float(
                np.mean(
                    [
                        r[k]
                        for r in rows
                        if r["model"] == model
                        and r["sigma"] == SIGMAS[0]
                        and r["draw"] == 0
                    ]
                )
            )
            for k in ("x0_2", *(f"x0_2_hf{c}" for c in HF_CUTS))
        }
        for model in curves
    }

    run_dir = make_run_dir("moss_vr_headroom", label=args.label)
    with open(run_dir / "per_forward.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(run_dir / "per_sigma.csv", "w", newline="") as f:
        wr = csv.writer(f)
        keys = list(curves["anima"])
        wr.writerow(
            [
                "sigma",
                "w_anima",
                "w_moss",
                *(f"{mm}_{k}" for mm in curves for k in keys),
            ]
        )
        for si, sigma in enumerate(SIGMAS):
            wr.writerow(
                [sigma, f"{w_anima[si]:.5f}", f"{w_moss[si]:.5f}"]
                + [f"{curves[mm][k][si]:.6f}" for mm in curves for k in keys]
            )
    artifacts = ["per_forward.csv", "per_sigma.csv"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for model, colour in (("anima", "#1f4e8c"), ("moss", "#c0392b")):
            ax.plot(
                SIGMAS, curves[model]["y2"], "o-", color=colour, label=f"{model} E[y²]"
            )
            ax.plot(
                SIGMAS,
                curves[model]["y2_hf0.5"],
                "s--",
                color=colour,
                label=f"{model} > 0.5 Nyquist",
            )
        ax.set_xlabel("σ")
        ax.set_ylabel("residual energy per element")
        ax.set_title(f"Base FM residual, {len(picks)} stems × {args.draws} draws")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "residual_by_sigma.png", dpi=130)
        artifacts.append("residual_by_sigma.png")
    except ImportError:
        log.warning("matplotlib missing: no plot")

    metrics = {
        "per_sigma": {
            mm: {k: v.tolist() for k, v in curves[mm].items()} for mm in curves
        },
        "sigmas": list(SIGMAS),
        "weights": {"anima_density": w_anima.tolist(), "moss_density": w_moss.tolist()},
        "weighted": weighted,
        "gap_anima_density": gap,
        "gap_share": gap_share,
        "x0_energy": x0,
        "verdict": verdict,
        "n_stems": len(picks),
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=args.device,
        extra={"stems": {s: p["artist"] for s, p in picks.items()}},
    )

    log.info(f"\n=== {len(picks)} stems → {run_dir}")
    log.info("σ      " + "  ".join(f"{s:.2f}" for s in SIGMAS))
    for mm in curves:
        for k in ("y2", "y2_hf0.5"):
            log.info(f"{mm:5s} {k:9s} " + "  ".join(f"{v:.3f}" for v in curves[mm][k]))
    for mm in weighted:
        for density, vals in weighted[mm].items():
            log.info(
                f"{mm:5s} @ {density:13s} "
                + "  ".join(f"{k} {v:.4f}" for k, v in vals.items())
            )
    log.info(
        f"gap at Anima density {gap:.4f}; share above 0.25 / 0.5 Nyquist: {gap_share}"
    )
    log.info(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
