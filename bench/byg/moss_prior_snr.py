#!/usr/bin/env python3
"""BYG Phase-0 gate step 1 on MOSS-Image: how much of σ carries the DDS prior.

BYG's training signal is the frozen base's caption-pair velocity difference
``v_tgt − v_src`` (``bench/byg/README.md``). This measures its size across σ on
real cached latents, against two references at the same noisy input:

  * ``null``: the source caption with its ``summary.caption`` tags shuffled — a
    semantically empty prompt change, the floor an edit signal has to clear
  * ``cfg``: the shipped uncond branch (``""``, skip block 22), so
    ``‖v_src − v_unc‖/‖v_src‖`` is the guidance-delta curve of
    ``docs/proposal/moss_anima_negatives_revisit.md`` Tier 3 §4

The edit is BYG v1's tag swap, restricted to one ``<color> hair`` (else
``<color> eyes``) tag in ``summary.caption`` so the change is one local
attribute. Each latent is noised as ``(1 − σ)·x0 + σ·ε`` (one ε per sample,
shared over σ and arms) on every ``--sigma_stride``-th σ of the sampler grid;
one per-sample reference forward per arm, CFG 1, no LoRA. Per σ, pooled over
samples:

  * ``r_edit`` / ``r_null`` / ``r_cfg``: ``‖Δv‖ / ‖v_src‖``, full latent and
    4×4 average-pooled (``_low``: layout / colour blocks)
  * ``snr``: ``r_edit / r_null`` (and ``snr_low``)
  * ``top10_edit`` / ``top10_null``: share of ``‖Δv‖²`` in the top 10 % of
    tokens (a local edit concentrates, 0.1 = uniform)

``signal_window`` reads the smallest σ where ``snr_low`` still clears each
``--snr_levels`` and the share of BYG's ``t ~ U(0,1)`` and of the 50-step
sampler grid above it. ``x0_pred`` under src and edit is decoded for
``--grid_n`` samples at the grid σ nearest each ``--grid_sigmas``
(``grid_<stem>.png``: row 1 src, row 2 edit, row 3 |diff| ×4).

    make daemon-run ARGS="bench/byg/moss_prior_snr.py --label calib"

Writes ``prior_snr.npz``, ``prior_snr.png``, the grids and ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bench._common import make_run_dir, start_heartbeat, write_result

POOL = 4
# contrasting hair / eye colours: no synonym pairs (grey/gray, blonde/yellow)
TARGETS = ["red", "blue", "green", "pink", "purple", "black", "white", "blonde"]
SOURCE_COLORS = (
    "light blue|dark blue|light green|dark green|light brown|red|blue|green|yellow"
    "|purple|pink|orange|black|white|brown|grey|gray|silver|blonde|aqua"
)
_EDIT_RE = {
    part: re.compile(rf"\b({SOURCE_COLORS}) {part}\b") for part in ("hair", "eyes")
}
_SAME = {"grey": "white", "gray": "white", "silver": "white", "yellow": "blonde"}
ARMS = ("src", "edit", "null", "unc")


def _dump(layout: dict) -> str:
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def build_prompts(caption: str, rng: random.Random) -> dict | None:
    """``{src, edit, null, swap}`` layout-JSON strings, or None without a hair /
    eye colour tag. All three are re-serialised the same way so they differ in
    content only."""
    layout = json.loads(caption)
    bag = layout.get("summary", {}).get("caption", "")
    for part, rx in _EDIT_RE.items():
        m = rx.search(bag)
        if m:
            break
    else:
        return None
    old = m.group(1)
    new = rng.choice(
        [c for c in TARGETS if c not in (old, _SAME.get(old)) and c not in old]
    )
    src = _dump(layout)
    layout["summary"]["caption"] = f"{bag[: m.start(1)]}{new}{bag[m.end(1) :]}"
    edit = _dump(layout)
    tags = bag.split(", ")  # summary.caption is a flat tag bag (no clauses)
    for _ in range(8):
        rng.shuffle(tags)
        if ", ".join(tags) != bag:
            break
    layout["summary"]["caption"] = ", ".join(tags)
    null = _dump(layout)
    return {"src": src, "edit": edit, "null": null, "swap": f"{old} {part} -> {new}"}


def window(sigmas: np.ndarray, snr: np.ndarray, level: float) -> float | None:
    """Smallest σ above which ``snr`` stays ≥ ``level`` (σ descending in)."""
    below = np.nonzero(snr < level)[0]
    if len(below) == 0:
        return float(sigmas[-1])
    i = below[0]
    return None if i == 0 else float(sigmas[i - 1])


def top_share(delta: torch.Tensor, frac: float = 0.1) -> float:
    """Share of ``‖delta‖²`` in the top ``frac`` of tokens (channel-summed)."""
    e = delta.square().sum(1).flatten()
    k = max(1, int(round(frac * e.numel())))
    return float(e.topk(k).values.sum() / e.sum().clamp_min(1e-12))


def plot(path, sigmas, m):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for key, style in (("r_edit", "-"), ("r_null", "--"), ("r_cfg", ":")):
        axes[0].plot(sigmas, m[key], ls=style, marker=".", label=key)
        axes[0].plot(sigmas, m[f"{key}_low"], ls=style, label=f"{key} (4×4 pool)")
    axes[0].set(ylabel="‖Δv‖ / ‖v_src‖", title="velocity delta vs src", yscale="log")
    axes[1].plot(sigmas, m["snr"], marker=".", label="edit / null")
    axes[1].plot(sigmas, m["snr_low"], marker=".", label="edit / null (4×4 pool)")
    axes[1].axhline(1, color="grey", lw=1)
    axes[1].set(ylabel="r_edit / r_null", title="edit signal over the null floor")
    axes[2].plot(sigmas, m["top10_edit"], marker=".", label="edit")
    axes[2].plot(sigmas, m["top10_null"], marker=".", label="null")
    axes[2].axhline(0.1, color="grey", lw=1)
    axes[2].set(ylabel="share of ‖Δv‖² in top 10 % tokens", title="locality")
    for ax in axes:
        ax.set(xlabel="σ", xlim=(1, 0))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_grid(path, rows, labels, tile=192):
    from PIL import Image, ImageDraw

    w0, h0 = rows[0][0].size
    th = round(tile * h0 / w0)
    cols = len(rows[0])
    sheet = Image.new("RGB", (cols * tile, len(rows) * th + 18), "white")
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(labels):
        draw.text((c * tile + 4, 3), name, fill="black")
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            sheet.paste(im.resize((tile, th)), (c * tile, 18 + r * th))
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="post_image_dataset/moss/_calib_mixed")
    ap.add_argument("--n", type=int, default=16, help="samples with a hair/eye tag")
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--text_encoder", default=None, help="default: the shipped TE")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--shift", type=float, default=None)
    ap.add_argument("--sigma_stride", type=int, default=2)
    ap.add_argument("--snr_levels", default="3,2,1.5")
    ap.add_argument("--grid_n", type=int, default=4)
    ap.add_argument("--grid_sigmas", default="0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.3")
    ap.add_argument(
        "--blocks_to_swap", type=int, default=-1, help="-1 = size from free VRAM"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from library.env import resolve_under_home
    from library.moss import cache as moss_cache
    from library.moss.generate import cached_embeds
    from library.moss.loader import empty_cache, load_transformer, load_vae
    from library.moss.model import decode, snapped_sigmas
    from library.moss.requests import (
        DEFAULT_SHIFT,
        DEFAULT_STEPS,
        DEFAULT_TEXT_ENCODER,
        DEFAULT_UNCOND_SKIP,
        resolve_model_dir,
    )
    from library.runtime.blockswap import place
    from safetensors import safe_open

    device = torch.device("cuda")
    model_dir = resolve_model_dir(args.model_dir)
    precision = args.text_encoder or DEFAULT_TEXT_ENCODER
    steps, shift = args.steps or DEFAULT_STEPS, args.shift or DEFAULT_SHIFT
    full_grid = snapped_sigmas(steps, shift).numpy().astype(np.float64)
    sigmas = full_grid[:: args.sigma_stride]

    rng = random.Random(args.seed)
    picked = []
    for item in moss_cache.list_items(resolve_under_home(args.cache)):
        with safe_open(str(item.te_path), "pt") as f:
            meta = f.metadata() or {}
        if meta.get("caption_format") != "layout_json":
            continue
        prompts = build_prompts(meta["caption"], rng)
        if prompts is not None:
            picked.append((item, prompts))
    if not picked:
        raise SystemExit(f"no layout-JSON caption with a hair/eye colour in {args.cache}")
    pick = np.linspace(0, len(picked) - 1, min(args.n, len(picked))).round()
    picked = [picked[i] for i in sorted(set(pick.astype(int)))]
    for item, p in picked:
        print(f"  {item.stem} {item.width}x{item.height}: {p['swap']}", flush=True)

    texts = [p[a] for _, p in picked for a in ("src", "edit", "null")] + [""]
    embeds = cached_embeds(model_dir, texts, Path(args.te_cache), precision)
    unc = embeds[""].unsqueeze(0).to(device)

    transformer = load_transformer(model_dir, args.model)
    transformer.requires_grad_(False).eval()
    attached = place(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if args.blocks_to_swap < 0 else args.blocks_to_swap,
        activation_reserve_gb=4.0,
        label="transformer",
    )
    empty_cache()

    grid_items = {it.stem for it, _ in picked[:: max(1, len(picked) // args.grid_n)]}
    grid_items = set(list(grid_items)[: args.grid_n])
    want = [float(s) for s in args.grid_sigmas.split(",")]
    grid_idx = sorted({int(np.abs(sigmas - s).argmin()) for s in want})
    print(
        f"{len(picked)} samples × {len(sigmas)} σ × {len(ARMS)} arms "
        f"(steps {steps}, shift {shift}, stride {args.sigma_stride}), TE {precision}",
        flush=True,
    )

    # per σ, sums over samples: ‖Δ‖² for edit/null/cfg, full + pooled, and ‖v_src‖²
    keys = ("edit", "null", "cfg")
    acc = {f"{k}{s}": np.zeros(len(sigmas)) for k in (*keys, "src") for s in ("", "_low")}
    top = {k: np.zeros(len(sigmas)) for k in ("edit", "null")}
    per_sample = []
    grids: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    gen = torch.Generator(device=device)
    start_heartbeat()
    t0 = time.time()
    done, total = 0, len(picked) * len(sigmas)
    with torch.no_grad():
        for k, (item, p) in enumerate(picked):
            latent, _ = moss_cache.load_item(item)
            x0 = latent.unsqueeze(0).to(device, torch.float32)
            gen.manual_seed(args.seed * 100003 + k)
            noise = torch.randn(x0.shape, generator=gen, device=device)
            cond = {a: embeds[p[a]].unsqueeze(0).to(device) for a in ("src", "edit", "null")}
            row = []
            for s, sigma in enumerate(sigmas):
                noisy = (1.0 - sigma) * x0 + sigma * noise
                timestep = torch.tensor([1000.0 * sigma], device=device)
                v = {
                    a: transformer(noisy, timestep, cond[a], return_dict=False)[0].float()
                    for a in cond
                }
                v["unc"] = transformer(
                    noisy, timestep, unc, return_dict=False, skip_layers=DEFAULT_UNCOND_SKIP
                )[0].float()
                deltas = {
                    "edit": v["edit"] - v["src"],
                    "null": v["null"] - v["src"],
                    "cfg": v["src"] - v["unc"],
                    "src": v["src"],
                }
                sq = {}
                for name, d in deltas.items():
                    sq[name] = d.square().sum().item()
                    acc[name][s] += sq[name]
                    acc[f"{name}_low"][s] += F.avg_pool2d(d, POOL).square().sum().item()
                for name in top:
                    top[name][s] += top_share(deltas[name])
                row.append([sq["edit"], sq["null"], sq["cfg"], sq["src"]])
                if item.stem in grid_items and s in grid_idx:
                    grids.setdefault(item.stem, []).append(
                        (
                            (noisy - sigma * v["src"]).cpu(),
                            (noisy - sigma * v["edit"]).cpu(),
                        )
                    )
                done += 1
                print(
                    f"  progress {done}/{total}  {item.stem} σ={sigma:.3f} "
                    f"edit {np.sqrt(sq['edit'] / sq['src']):.4f} "
                    f"null {np.sqrt(sq['null'] / sq['src']):.4f} "
                    f"cfg {np.sqrt(sq['cfg'] / sq['src']):.4f}  "
                    f"{(time.time() - t0) / done:.1f}s/σ",
                    flush=True,
                )
            per_sample.append(row)
    if attached is not None:
        attached.detach()
    transformer.to("cpu")
    del transformer
    empty_cache()

    m = {}
    for name in keys:
        for s in ("", "_low"):
            m[f"r_{name}{s}"] = np.sqrt(acc[f"{name}{s}"] / acc[f"src{s}"])
    m["snr"] = m["r_edit"] / m["r_null"]
    m["snr_low"] = m["r_edit_low"] / m["r_null_low"]
    m["top10_edit"] = top["edit"] / len(picked)
    m["top10_null"] = top["null"] / len(picked)

    run_dir = make_run_dir("byg", args.label)
    np.savez(
        run_dir / "prior_snr.npz",
        sigmas=sigmas,
        per_sample_sq=np.array(per_sample),  # (N, σ, [edit, null, cfg, src])
        **m,
    )
    plot(run_dir / "prior_snr.png", sigmas, m)

    # decode with the DiT off the card
    from diffusers.image_processor import VaeImageProcessor
    from PIL import ImageChops

    vae = load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    labels = [f"s={sigmas[i]:.3f}" for i in grid_idx]  # PIL default font has no σ
    grid_files = []
    for stem, pairs in grids.items():
        src_row, edit_row, diff_row = [], [], []
        for zs, ze in pairs:
            a, b = (
                processor.postprocess(decode(z.to(device), vae), output_type="pil")[0]
                for z in (zs, ze)
            )
            src_row.append(a)
            edit_row.append(b)
            diff_row.append(ImageChops.difference(a, b).point(lambda x: min(255, 4 * x)))
        name = f"grid_{stem}.png"
        save_grid(run_dir / name, [src_row, edit_row, diff_row], labels)
        grid_files.append(name)
        print(f"[vae] {name}", flush=True)
    del vae
    empty_cache()

    def r4(x):
        return None if x is None else round(float(x), 4)

    signal_window = {}
    for level in (float(x) for x in args.snr_levels.split(",")):
        for kind in ("snr", "snr_low"):
            sk = window(sigmas, m[kind], level)
            signal_window[f"{kind}>={level:g}"] = {
                "sigma": r4(sk),
                "u01_share_above": None if sk is None else r4(1.0 - sk),
                "grid_steps_above": None if sk is None else int((full_grid >= sk).sum()),
            }
    print(json.dumps(signal_window, indent=1), flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "signal_window": signal_window,
            "by_sigma": {
                f"{s:.3f}": {k: r4(v[i]) for k, v in m.items()}
                for i, s in enumerate(sigmas)
            },
        },
        label=args.label,
        artifacts=["prior_snr.npz", "prior_snr.png", *grid_files],
        device=device,
        extra={
            "text_encoder": precision,
            "sampler": {"steps": steps, "shift": shift},
            "sigmas": [r4(s) for s in sigmas],
            "samples": [
                {"stem": it.stem, "size": f"{it.width}x{it.height}", "swap": p["swap"]}
                for it, p in picked
            ],
            "anima_reference": "cfg delta / ‖v_cond‖ peaks at σ=1, ~0.02 floor "
            "below σ≈0.85 (docs/findings/crossattn_self_attn_dominance.md)",
        },
    )
    print(f"wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
