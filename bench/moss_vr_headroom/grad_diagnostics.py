#!/usr/bin/env python3
"""Phase B of ``docs/proposal/moss_vr_headroom.md``: would VR-loss's gradient help on MOSS?

At a fixed θ (a trained MOSS LoRA, fp32 adapters on the bf16 Base), K minibatches of
one cached sample each, σ drawn from MOSS's training density. Per batch:

    y = u_pred(x_t; θ) − (ε − x_0)                    # standard FM residual
    z = u_pred(x_t^L; multiplier 0) − (ε − x_0^L)     # control variate, no grad
    s_k = ∂ mean(y²) / ∂θ                              # standard gradient
    c_k = ∂ mean(z·y) / ∂θ                             # cross term
    g_vr_k(λ) = s_k + 2λ·c_k                           # ∂ mean((y + λz)²) / ∂θ

``x_0^L`` is the Gaussian low-pass of the normalised latent on the /8 grid,
``σ_low = min(H/8, W/8) / 4`` (``vr_fei_sigma_low_div`` default), re-patchified; ``x_t^L``
uses the same ε. Each gradient takes its own forward (no ``retain_graph`` through the
block swapper). Because ``g_vr`` is linear in λ, the 24×24 Gram matrices of s and c
give every metric for any λ: the pooled ``λ = −Σcov(y,z) / Σvar(z)`` and the fixed −1.

Metrics per λ (``_archive/bench/vr_grad_diagnostics/README.md`` definitions):

  * ``magnitude_ratio``  ‖mean g_vr‖ / ‖mean s‖
  * ``cov_sq_ratio``     (Var g_vr / ‖mean g_vr‖²) / (Var s / ‖mean s‖²)
  * ``direction_lift``   mean_k cos(g_vr_k, ref) − mean_k cos(s_k, ref); ``ref`` leaves
    batch k out (``_loo``, gated) or keeps it in (``_incl``, as the archived bench did)

Gate (pooled λ): magnitude ratio in [0.5, 2], cov_sq ratio < 0.7, LOO lift > 0.05.

    make daemon-run ARGS="bench/moss_vr_headroom/grad_diagnostics.py"
    make daemon-run ARGS="bench/moss_vr_headroom/grad_diagnostics.py --num_batches 3 --label smoke"

Writes ``result.json``, ``per_batch.csv`` and ``gram.npz`` (the three Gram matrices).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time

import numpy as np
import torch

from bench._common import make_run_dir, start_heartbeat, write_result
from bench.moss_vr_headroom.residual_bands import (
    MOSS_CUTOFF,
    MOSS_MU,
    MOSS_S,
    MOSS_UNIFORM,
)
from library.env import resolve_under_home

log = logging.getLogger("bench.moss_vr_headroom.grad_diagnostics")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_LORA = "output/ckpt_moss/caststation_ga4_lr1e4_s0.safetensors"
DEFAULT_CACHE = "post_image_dataset/moss/channel_(caststation)_layout"


def low_pass(x0: torch.Tensor, div: float) -> torch.Tensor:
    """Packed ``(1, 128, h, w)`` → its Gaussian low-pass on the /8 grid, re-packed."""
    from library.moss.model import patchify, unpatchify
    from library.runtime.fei import fei_sigma_low, gaussian_blur_2d

    grid = unpatchify(x0)
    return patchify(gaussian_blur_2d(grid, fei_sigma_low(*grid.shape[-2:], div)))


def flat_grads(params: list[torch.Tensor]) -> torch.Tensor:
    out = torch.cat([p.grad.detach().float().reshape(-1).cpu() for p in params])
    for p in params:
        p.grad = None
    return out


def gram(a: torch.Tensor, b: torch.Tensor, chunk: int = 1 << 24) -> np.ndarray:
    """``a (K, P)`` · ``b (K, P)ᵀ`` in float64, chunked over P."""
    out = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float64)
    for i in range(0, a.shape[1], chunk):
        out += a[:, i : i + chunk].double() @ b[:, i : i + chunk].double().T
    return out.numpy()


def metrics_for(lam: float, gss: np.ndarray, gsc: np.ndarray, gcc: np.ndarray) -> dict:
    """Every Phase B metric at ``λ``, from the Gram matrices of s and c."""
    a = 2.0 * lam
    gvv = gss + a * (gsc + gsc.T) + a * a * gcc  # ⟨g_vr_i, g_vr_j⟩
    k = gss.shape[0]

    def stats(g: np.ndarray) -> dict:
        norms = np.sqrt(np.diag(g))
        ref_sq = g.sum() / k**2  # ‖mean g‖²
        var = float((norms**2).mean() - ref_sq)
        incl = g.sum(axis=1) / k / (norms * np.sqrt(ref_sq))
        loo_dot = (g.sum(axis=1) - np.diag(g)) / (k - 1)
        loo_sq = (g.sum() - 2 * g.sum(axis=1) + np.diag(g)) / (k - 1) ** 2
        loo = loo_dot / (norms * np.sqrt(loo_sq))
        return {
            "ref_norm": float(np.sqrt(ref_sq)),
            "var": var,
            "cov_sq": var / ref_sq,
            "mean_cos_incl": float(incl.mean()),
            "mean_cos_loo": float(loo.mean()),
            "norms": norms,
        }

    s, v = stats(gss), stats(gvv)
    # cos(mean s, mean g_vr) and the cross-cosine of each g_vr_k to mean s
    gsv = gss + a * gsc  # ⟨s_i, g_vr_j⟩
    cos_refs = float(gsv.sum() / k**2 / (s["ref_norm"] * v["ref_norm"]))
    cross = gsv.T.sum(axis=1) / k / (v["norms"] * s["ref_norm"])
    return {
        "lambda": lam,
        "magnitude_ratio": v["ref_norm"] / s["ref_norm"],
        "cov_sq_std": s["cov_sq"],
        "cov_sq_vr": v["cov_sq"],
        "cov_sq_ratio": v["cov_sq"] / s["cov_sq"],
        "var_ratio_raw": v["var"] / s["var"],
        "cos_ref_std_ref_vr": cos_refs,
        "mean_cos_std_ref_incl": s["mean_cos_incl"],
        "mean_cos_vr_ref_incl": v["mean_cos_incl"],
        "direction_lift_incl": v["mean_cos_incl"] - s["mean_cos_incl"],
        "mean_cos_std_ref_loo": s["mean_cos_loo"],
        "mean_cos_vr_ref_loo": v["mean_cos_loo"],
        "direction_lift_loo": v["mean_cos_loo"] - s["mean_cos_loo"],
        "mean_cos_vr_k_std_ref": float(cross.mean()),
    }


def gate(m: dict) -> tuple[bool, str]:
    checks = {
        "magnitude_ratio in [0.5, 2]": 0.5 <= m["magnitude_ratio"] <= 2.0,
        "cov_sq_ratio < 0.7": m["cov_sq_ratio"] < 0.7,
        "direction_lift_loo > 0.05": m["direction_lift_loo"] > 0.05,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if not failed:
        return (
            True,
            "PASS: VR's gradient is usable on MOSS; wire it into library/moss/train.py and A/B",
        )
    return False, "FAIL (" + ", ".join(failed) + "): close the VR line for MOSS"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--num_batches", type=int, default=24)
    ap.add_argument("--sigma_low_div", type=float, default=4.0)
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 sizes it from free VRAM"
    )
    ap.add_argument("--moss_model_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from library.moss.lora import LoRANetwork
    from library.moss.loader import empty_cache, load_transformer
    from library.moss.requests import resolve_model_dir
    from library.moss.train import (
        NUM_TRAIN_TIMESTEPS,
        checkpoint_blocks,
        place,
        sample_sigma,
        samples_from_cache,
        step_reserve_gb,
    )

    start_heartbeat(label="grad_diagnostics")
    device = torch.device("cuda")
    samples = samples_from_cache(resolve_under_home(args.cache))
    rng = np.random.default_rng(args.seed)
    picks = [
        samples[i]
        for i in rng.choice(
            len(samples), min(args.num_batches, len(samples)), replace=False
        )
    ]
    if len(picks) < 3:
        raise SystemExit(f"need >= 3 batches, have {len(picks)}")
    joint = max(s.n_img + s.n_text for s in picks)

    transformer = load_transformer(resolve_model_dir(args.moss_model_dir), "base")
    transformer.requires_grad_(False)
    network, meta = LoRANetwork.from_file(
        transformer, resolve_under_home(args.lora), dtype=torch.float32
    )
    network.apply_to()
    network.to(device)
    network.requires_grad_(True)
    attached = place(
        transformer,
        device,
        args.blocks_to_swap,
        step_reserve_gb(joint, grad_checkpointing=True),
    )
    checkpoint_blocks(transformer.layers)
    empty_cache()
    params = [p for p in network.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    log.info(
        f"LoRA {args.lora}: {meta.get('algo', 'lora')} rank {meta['rank']}, {n_params / 1e6:.1f}M params"
    )
    log.info(
        f"{len(picks)} batches, joint tokens up to {joint}; grads on CPU ~{2 * len(picks) * n_params * 4 / 1024**3:.1f} GB"
    )

    gen = torch.Generator().manual_seed(args.seed)
    s_rows, c_rows, rows = [], [], []
    t0 = time.time()
    for k, sample in enumerate(picks):
        latent, embeds = sample.load()
        x0 = latent.unsqueeze(0).to(device, torch.float32)
        embeds = embeds.unsqueeze(0).to(device)
        sigma = sample_sigma(gen, MOSS_MU, MOSS_S, MOSS_CUTOFF, MOSS_UNIFORM)
        eps = torch.randn(
            x0.shape, generator=torch.Generator().manual_seed(args.seed * 1000 + k)
        ).to(device)
        x0_l = low_pass(x0, args.sigma_low_div)
        timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)

        def residual(x_t, target, timestep=timestep, embeds=embeds):
            return (
                transformer(x_t, timestep, embeds, return_dict=False)[0].float()
                - target
            )

        network.set_multiplier(0.0)
        with torch.no_grad():
            z = residual((1 - sigma) * x0_l + sigma * eps, eps - x0_l)
        network.set_multiplier(1.0)
        x_t = (1 - sigma) * x0 + sigma * eps

        y = residual(x_t, eps - x0)
        y_d = y.detach().double()
        cov, var_z, y2 = (
            float((y_d * z.double()).sum()),
            float(z.double().square().sum()),
            float(y_d.square().sum()),
        )
        y.square().mean().backward()
        s_rows.append(flat_grads(params))
        del y

        (residual(x_t, eps - x0) * z).mean().backward()
        c_rows.append(flat_grads(params))

        rows.append(
            {
                "k": k,
                "stem": sample.stem,
                "sigma": sigma,
                "tokens": sample.n_img,
                "y2_mean": y2 / x0.numel(),
                "z2_mean": var_z / x0.numel(),
                "cov_yz_mean": cov / x0.numel(),
                "lambda_batch": -cov / max(var_z, 1e-30),
                "s_norm": float(s_rows[-1].norm()),
                "c_norm": float(c_rows[-1].norm()),
            }
        )
        log.info(
            f"  {k + 1}/{len(picks)} σ {sigma:.3f}  y² {rows[-1]['y2_mean']:.4f}  z² {rows[-1]['z2_mean']:.4f}"
            f"  λ_b {rows[-1]['lambda_batch']:+.3f}  ‖s‖ {rows[-1]['s_norm']:.3e}  ({time.time() - t0:.0f}s)"
        )
        del residual, z, x0, x0_l, eps, embeds
    if attached is not None:
        attached.detach()

    S, C = torch.stack(s_rows), torch.stack(c_rows)
    del s_rows, c_rows
    gss, gsc, gcc = gram(S, S), gram(S, C), gram(C, C)
    del S, C

    lam_pooled = -sum(r["cov_yz_mean"] for r in rows) / sum(r["z2_mean"] for r in rows)
    arms = {
        "pooled": metrics_for(lam_pooled, gss, gsc, gcc),
        "fixed_-1": metrics_for(-1.0, gss, gsc, gcc),
    }
    for r in rows:
        r["loss_reduction_pooled"] = (
            1
            - (
                r["y2_mean"]
                + 2 * lam_pooled * r["cov_yz_mean"]
                + lam_pooled**2 * r["z2_mean"]
            )
            / r["y2_mean"]
        )
    passed, verdict = gate(arms["pooled"])

    run_dir = make_run_dir("moss_vr_headroom", label=args.label)
    with open(run_dir / "per_batch.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(
        run_dir / "gram.npz",
        gss=gss,
        gsc=gsc,
        gcc=gcc,
        sigma=np.array([r["sigma"] for r in rows]),
    )

    metrics = {
        "arms": arms,
        "lambda_pooled": lam_pooled,
        "lambda_batch_median": float(np.median([r["lambda_batch"] for r in rows])),
        "loss_reduction_pooled_mean": float(
            np.mean([r["loss_reduction_pooled"] for r in rows])
        ),
        "n_batches": len(rows),
        "n_params": n_params,
        "gate_pass": passed,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["per_batch.csv", "gram.npz"],
        device="cuda",
        extra={
            "lora_meta": {k: meta[k] for k in ("rank", "alpha", "targets") if k in meta}
        },
    )

    log.info(f"\n=== {len(rows)} batches → {run_dir}")
    log.info(
        f"λ pooled {lam_pooled:+.4f} (batch median {metrics['lambda_batch_median']:+.4f}); "
        f"loss-level reduction at pooled λ {metrics['loss_reduction_pooled_mean']:.4f}"
    )
    for name, m in arms.items():
        log.info(
            f"[{name:8s}] λ {m['lambda']:+.3f}  |g_vr|/|g_std| {m['magnitude_ratio']:.4f}  "
            f"cov² ratio {m['cov_sq_ratio']:.3f} ({m['cov_sq_vr']:.3f}/{m['cov_sq_std']:.3f})  "
            f"cos(refs) {m['cos_ref_std_ref_vr']:+.3f}  lift loo {m['direction_lift_loo']:+.3f} "
            f"(std {m['mean_cos_std_ref_loo']:+.3f}, vr {m['mean_cos_vr_ref_loo']:+.3f})  "
            f"incl {m['direction_lift_incl']:+.3f}"
        )
    log.info(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
