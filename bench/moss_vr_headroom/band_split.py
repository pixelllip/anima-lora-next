#!/usr/bin/env python3
"""Is the HF part of the MOSS FM loss mostly gradient noise? (follow-up to Phase B)

``docs/proposal/moss_vr_headroom.md`` closed VR-loss: the MOSS loss gap is an HF
texture floor that a low-pass control variate cannot cancel. A floor the model cannot
predict leaves the optimum alone and only adds gradient variance, so the remaining
lever is loss-side: weight the band above 0.5 Nyquist by ``w < 1``. The DiT's input is
untouched. This bench measures what that buys, at a fixed θ (a trained MOSS LoRA).

Per batch the residual ``y = u_pred − (ε − x_0)`` is split on the /8 grid by a radial
FFT mask, ``y = y_LF + y_HF`` (orthogonal, so ``mean y² = mean y_LF² + mean y_HF²``):

    l_k = ∂ mean(y_LF²) / ∂θ,   h_k = ∂ mean(y_HF²) / ∂θ,   g_k(w) = l_k + w·h_k

``w = 1`` is standard FM. ``g(w)`` is linear in w, so the Gram matrices of l and h give
every metric for any w (``grad_diagnostics.metrics_for`` with ``s = l + h``, ``c = h``,
``λ = (w − 1) / 2``). Each gradient takes its own forward. Gradients are kept in bf16
on CPU.

Reading (pre-registered, at w = 0.25): the HF band is mostly noise, and a training A/B
of an HF-weighted loss is worth running, if the cov² ratio against w = 1 is ≤ 0.7 **and**
the mean gradient keeps its direction (cos ≥ 0.9 to w = 1's). A lower cos means the
HF band carries the mean direction, i.e. the style signal. Close.

    make daemon-run ARGS="bench/moss_vr_headroom/band_split.py"
    make daemon-run ARGS="bench/moss_vr_headroom/band_split.py --num_batches 3 --label smoke"

Writes ``result.json``, ``per_batch.csv`` and ``gram.npz``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time

import numpy as np
import torch

from bench._common import make_run_dir, start_heartbeat, write_result
from bench.moss_vr_headroom.grad_diagnostics import (
    DEFAULT_CACHE,
    DEFAULT_LORA,
    gram,
    metrics_for,
)
from bench.moss_vr_headroom.residual_bands import (
    MOSS_CUTOFF,
    MOSS_MU,
    MOSS_S,
    MOSS_UNIFORM,
)
from library.env import resolve_under_home

log = logging.getLogger("bench.moss_vr_headroom.band_split")
logging.basicConfig(level=logging.INFO, format="%(message)s")

WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
GATE_W, GATE_COV_SQ, GATE_COS = 0.25, 0.7, 0.9


def hf_part(y: torch.Tensor, cut: float) -> torch.Tensor:
    """Packed ``(1, 128, h, w)`` → its component above ``cut`` × Nyquist on the /8 grid,
    re-packed. Differentiable; the radial mask is Hermitian-symmetric, so the result
    is real."""
    from library.moss.model import patchify, unpatchify

    grid = unpatchify(y)
    h, w = grid.shape[-2:]
    ky = torch.fft.fftfreq(h, device=y.device) * 2
    kx = torch.fft.fftfreq(w, device=y.device) * 2
    mask = (ky[:, None] ** 2 + kx[None, :] ** 2) > cut**2
    return patchify(torch.fft.ifft2(torch.fft.fft2(grid) * mask).real)


def flat_grads_bf16(params: list[torch.Tensor]) -> torch.Tensor:
    out = torch.cat(
        [p.grad.detach().reshape(-1).to("cpu", torch.bfloat16) for p in params]
    )
    for p in params:
        p.grad = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lora", default=DEFAULT_LORA)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument(
        "--num_batches",
        type=int,
        default=48,
        help="stems cycle when this exceeds the cache",
    )
    ap.add_argument(
        "--cut", type=float, default=0.5, help="band edge, fraction of Nyquist"
    )
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 sizes it from free VRAM"
    )
    ap.add_argument("--moss_model_dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from library.moss.loader import empty_cache, load_transformer
    from library.moss.lora import LoRANetwork
    from library.moss.requests import resolve_model_dir
    from library.moss.train import (
        NUM_TRAIN_TIMESTEPS,
        checkpoint_blocks,
        place,
        sample_sigma,
        samples_from_cache,
        step_reserve_gb,
    )

    start_heartbeat(label="band_split")
    device = torch.device("cuda")
    samples = samples_from_cache(resolve_under_home(args.cache))
    rng = np.random.default_rng(args.seed)
    order = []
    while len(order) < args.num_batches:
        order.extend(rng.permutation(len(samples)).tolist())
    picks = [samples[i] for i in order[: args.num_batches]]
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
    log.info(f"LoRA {args.lora}: rank {meta['rank']}, {n_params / 1e6:.1f}M params")
    log.info(
        f"{len(picks)} batches over {len({s.stem for s in picks})} stems, joint tokens up to {joint}; "
        f"grads on CPU ~{2 * len(picks) * n_params * 2 / 1024**3:.1f} GB (bf16)"
    )

    gen = torch.Generator().manual_seed(args.seed)
    l_rows, h_rows, rows = [], [], []
    t0 = time.time()
    for k, sample in enumerate(picks):
        latent, embeds = sample.load()
        x0 = latent.unsqueeze(0).to(device, torch.float32)
        embeds = embeds.unsqueeze(0).to(device)
        sigma = sample_sigma(gen, MOSS_MU, MOSS_S, MOSS_CUTOFF, MOSS_UNIFORM)
        eps = torch.randn(
            x0.shape, generator=torch.Generator().manual_seed(args.seed * 1000 + k)
        ).to(device)
        x_t = (1 - sigma) * x0 + sigma * eps
        timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)

        def bands(x_t=x_t, target=eps - x0, timestep=timestep, embeds=embeds):
            y = (
                transformer(x_t, timestep, embeds, return_dict=False)[0].float()
                - target
            )
            y_hf = hf_part(y, args.cut)
            return y - y_hf, y_hf

        y_lf, y_hf = bands()
        lf2, hf2 = (
            float(y_lf.detach().square().mean()),
            float(y_hf.detach().square().mean()),
        )
        y_lf.square().mean().backward()
        l_rows.append(flat_grads_bf16(params))
        del y_lf, y_hf

        _, y_hf = bands()
        y_hf.square().mean().backward()
        h_rows.append(flat_grads_bf16(params))
        del y_hf

        ln, hn = float(l_rows[-1].float().norm()), float(h_rows[-1].float().norm())
        cos_lh = float(l_rows[-1].float() @ h_rows[-1].float()) / max(ln * hn, 1e-30)
        rows.append(
            {
                "k": k,
                "stem": sample.stem,
                "sigma": sigma,
                "y2_lf": lf2,
                "y2_hf": hf2,
                "l_norm": ln,
                "h_norm": hn,
                "cos_l_h": cos_lh,
            }
        )
        log.info(
            f"  {k + 1}/{len(picks)} σ {sigma:.3f}  y² LF {lf2:.4f} HF {hf2:.4f}  ‖l‖ {ln:.3e} ‖h‖ {hn:.3e}"
            f"  cos {cos_lh:+.3f}  ({time.time() - t0:.0f}s)"
        )
        del x0, eps, embeds, x_t, bands
    if attached is not None:
        attached.detach()

    L, H = torch.stack(l_rows), torch.stack(h_rows)
    del l_rows, h_rows
    gll, glh, ghh = gram(L, L), gram(L, H), gram(H, H)
    del L, H
    # s = l + h (standard FM), c = h: g(w) = s + (w − 1)·h = s + 2λ·c with λ = (w − 1)/2
    gss = gll + glh + glh.T + ghh
    gsc = glh + ghh
    arms = {f"w={w:g}": metrics_for((w - 1) / 2, gss, gsc, ghh) for w in WEIGHTS}
    for name, m in arms.items():
        m["w"] = float(name.removeprefix("w="))
        m.pop("lambda")

    def band(g: np.ndarray) -> dict:
        k = g.shape[0]
        ref_sq = g.sum() / k**2
        return {
            "ref_norm": float(np.sqrt(ref_sq)),
            "cov_sq": float(np.diag(g).mean() / ref_sq - 1),
        }

    lf, hf = band(gll), band(ghh)
    s_ref_sq = gss.sum() / gss.shape[0] ** 2
    bands_summary = {
        "lf": lf,
        "hf": hf,
        # share of ‖mean g_std‖² along each band's mean gradient
        "hf_share_of_mean_grad": float(
            (glh.sum() + ghh.sum()) / gss.shape[0] ** 2 / s_ref_sq
        ),
        "lf_share_of_mean_grad": float(
            (gll.sum() + glh.sum()) / gss.shape[0] ** 2 / s_ref_sq
        ),
        "cos_mean_l_mean_h": float(glh.sum() / np.sqrt(gll.sum() * ghh.sum())),
        "loss_hf_share": float(
            np.mean([r["y2_hf"] / (r["y2_lf"] + r["y2_hf"]) for r in rows])
        ),
    }
    g = arms[f"w={GATE_W:g}"]
    noise = g["cov_sq_ratio"] <= GATE_COV_SQ and g["cos_ref_std_ref_vr"] >= GATE_COS
    verdict = (
        f"HF band is mostly noise (w={GATE_W:g}: cov² ratio {g['cov_sq_ratio']:.2f}, cos {g['cos_ref_std_ref_vr']:.3f}): "
        "an HF-weighted training A/B is worth running"
        if noise
        else f"no (w={GATE_W:g}: cov² ratio {g['cov_sq_ratio']:.2f}, cos {g['cos_ref_std_ref_vr']:.3f}): "
        "close the HF-weighting lever"
    )

    run_dir = make_run_dir("moss_vr_headroom", label=args.label or "band_split")
    with open(run_dir / "per_batch.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(
        run_dir / "gram.npz",
        gll=gll,
        glh=glh,
        ghh=ghh,
        sigma=np.array([r["sigma"] for r in rows]),
    )
    metrics = {
        "arms": arms,
        "bands": bands_summary,
        "cut": args.cut,
        "n_batches": len(rows),
        "n_params": n_params,
        "gate": {
            "w": GATE_W,
            "cov_sq_ratio_max": GATE_COV_SQ,
            "cos_min": GATE_COS,
            "pass": noise,
        },
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
    )

    log.info(f"\n=== {len(rows)} batches → {run_dir}")
    log.info(
        f"loss HF share {bands_summary['loss_hf_share']:.3f}; mean-gradient share LF "
        f"{bands_summary['lf_share_of_mean_grad']:.3f} / HF {bands_summary['hf_share_of_mean_grad']:.3f}; "
        f"cos(mean l, mean h) {bands_summary['cos_mean_l_mean_h']:+.3f}; "
        f"cov² LF {lf['cov_sq']:.2f} HF {hf['cov_sq']:.2f}"
    )
    for name, m in arms.items():
        log.info(
            f"[{name:6s}] |g|/|g_std| {m['magnitude_ratio']:.3f}  cov² ratio {m['cov_sq_ratio']:.3f}  "
            f"cos to w=1 {m['cos_ref_std_ref_vr']:+.3f}  lift loo {m['direction_lift_loo']:+.3f} "
            f"(w=1 {m['mean_cos_std_ref_loo']:+.3f})"
        )
    log.info(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
