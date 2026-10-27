"""Single-step tone bias of sanitize LoRAs on the unedited tokens (no training).

Does the x̂₀ the training loss sees already carry the page-wide brightening /
cooling that e6 renders show (docs/moss/proposal/i2i_tone_loss.md)? Per LoRA,
per σ, with the same noise for every LoRA:

- training pairs: ``x̂₀ = x_t − σ·v̂`` against the cached target on the rest tokens
  (complement of ``edit_region``, grown by ``--tone_dilate``);
- held-out pages (``--pages``): the target on the rest is the reference itself;
  the rest is the pixels no render changed (``tone_map.py``'s mask), per token.

Logged in latent space: the proposal's tone term at pool 0 and 8 next to the rest
v-MSE, and the per-channel DC error's consistency across samples. Then, with the
DiT off the card, x̂₀ and x₀ of the first ``--decode`` pairs (and every page) at
the lowest σs are decoded and compared in Lab on the rest pixels, the same metric
as the renders.

    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/tone_probe.py"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_transformer, load_vae
from library.moss.model import decode, encode
from library.moss.train import NUM_TRAIN_TIMESTEPS, edit_region

CKPT = "output/ckpt_moss_i2i"
LORAS = {
    "base": None,
    "200p": f"{CKPT}/moss_i2i_sanitize_200p.safetensors",
    "e3": f"{CKPT}/moss_i2i_sanitize_wm/moss_i2i_sanitize_wm_e3.safetensors",
    "e6": f"{CKPT}/moss_i2i_sanitize_wm.safetensors",
    "cs0": f"{CKPT}/moss_i2i_sanitize_cs0_lr1e4_600.safetensors",
}
PAGES = [
    "hard_to_sanitize1",
    "hard_to_sanitize2",
    "hard_to_sanitize3",
    "usual_case1",
    "usual_case2",
]
RENDERS = [
    "output/tests/moss_i2i_sanitize_200p_cli",
    "output/tests/moss_i2i_sanitize_wm_e6",
    "output/tests/moss_i2i_sanitize_cs0_lr1e4_600",
]


def pick_pairs(cache_dir: Path, n: int) -> list:
    items = moss_cache.list_items(cache_dir)
    wm = [i for i in items if i.stem.startswith("wm_")]
    rest = [i for i in items if not i.stem.startswith("wm_")]
    k_wm = max(1, round(n * len(wm) / len(items))) if wm else 0

    def spread(xs, k):
        return (
            [xs[round(j * (len(xs) - 1) / max(1, k - 1))] for j in range(k)]
            if k
            else []
        )

    return spread(rest, n - k_wm) + spread(wm, k_wm)


def page_mask(page: str, size: tuple[int, int], src: str) -> np.ndarray:
    """Pixels no render changed by ΔE > 8, dilated 20 px (``tone_map.py``)."""

    from tone_map import delta_e, lab

    src_lab, _ = lab(
        moss_cache.fit(
            moss_cache.load_rgb(resolve_under_home(f"{src}/{page}.png")), *size
        )
    )
    changed = np.zeros(src_lab.shape[:2], bool)
    for d in RENDERS:
        out_lab, _ = lab(Image.open(resolve_under_home(f"{d}/000_{page}_base_s42.png")))
        changed |= delta_e(out_lab, src_lab) > 8
    return ~ndimage.binary_dilation(changed, iterations=20)


def tone_term(err: torch.Tensor, rest: torch.Tensor, pool: int) -> float:
    """The proposal's term: masked mean of ``err`` (1,C,h,w) per pool×pool cell
    (pool 0 = one cell), squared, over cells with coverage ≥ ½."""
    m = rest[None, None].float()
    e = err * m
    if pool == 0:
        cov = m.mean()
        return float(((e.mean((2, 3)) / cov) ** 2).mean()) if cov > 0 else float("nan")
    num = F.avg_pool2d(e, pool, ceil_mode=True)
    cov = F.avg_pool2d(m, pool, ceil_mode=True)
    ok = cov[0, 0] >= 0.5
    if not ok.any():
        return float("nan")
    pooled = num / cov.clamp(min=1e-6)
    return float((pooled[0][:, ok] ** 2).mean())


def lab_delta(x_hat: torch.Tensor, x0: torch.Tensor, rest_px: np.ndarray, vae) -> dict:
    from tone_map import lab

    def img(z):
        px = decode(z.to(vae.device), vae)[0].float().cpu()
        return Image.fromarray(
            ((px.permute(1, 2, 0).numpy() + 1) * 127.5)
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

    a_lab, a_sat = lab(img(x_hat))
    b_lab, b_sat = lab(img(x0))
    d = (a_lab - b_lab)[rest_px]
    return {
        "dL": float(d[:, 0].mean()),
        "db": float(d[:, 2].mean()),
        "sat": float(a_sat[rest_px].mean() / max(b_sat[rest_px].mean(), 1e-6)),
    }


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="post_image_dataset/moss/i2i_sanitize")
    p.add_argument(
        "--pages",
        default="output/moss_i2i_eval/sanitize/inputs",
        help="'' skips the held-out pages",
    )
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--sigmas", default="0.1,0.3,0.6")
    p.add_argument(
        "--decode", type=int, default=6, help="pairs decoded per LoRA at the decode σs"
    )
    p.add_argument("--decode_sigmas", default="0.1,0.3")
    p.add_argument(
        "--loras", default=",".join(LORAS), help="names from LORAS or name=path"
    )
    p.add_argument("--tone_dilate", type=int, default=2)
    p.add_argument("--out", default="project/moss_lora/space/tone_probe.json")
    args = p.parse_args()
    device = torch.device("cuda")
    sigmas = [float(s) for s in args.sigmas.split(",")]
    dec_sigmas = [float(s) for s in args.decode_sigmas.split(",")]
    paths = dict(LORAS)
    names = []
    for tok in args.loras.split(","):
        name, _, path = tok.partition("=")
        if path:
            paths[name] = path
        names.append(name)

    # ---- samples: (name, x0, ref, embeds, rest tokens, rest pixels or None)
    items = pick_pairs(resolve_under_home(args.cache), args.pairs)
    samples = []
    for it in items:
        x0, emb = moss_cache.load_item(it)
        ref = moss_cache.load_ref(it)
        region = edit_region(x0, ref, 0.3, 5, 1 + args.tone_dilate)
        samples.append(
            dict(name=it.stem, x0=x0[None], ref=ref[None], emb=emb[None], rest=~region)
        )
    fixed_emb = samples[0]["emb"]
    print(
        f"{len(samples)} pairs ({sum(s['name'].startswith('wm_') for s in samples)} wm), "
        f"rest share median {np.median([s['rest'].float().mean().item() for s in samples]):.0%}",
        flush=True,
    )
    if args.pages:
        vae = load_vae(None, device=device)
        for page in PAGES:
            src = moss_cache.load_rgb(resolve_under_home(f"{args.pages}/{page}.png"))
            w, h = moss_cache.target_size(*src.size, 1024)
            ref = (
                encode(moss_cache.to_pixels(moss_cache.fit(src, w, h)), vae)
                .float()
                .cpu()
            )
            px = page_mask(page, (w, h), args.pages)
            tok = px.reshape(h // 16, 16, w // 16, 16).all((1, 3))
            samples.append(
                dict(
                    name=page,
                    x0=ref,
                    ref=ref,
                    emb=fixed_emb,
                    rest=torch.from_numpy(tok),
                    rest_px=px,
                    page=True,
                )
            )
            print(f"[page] {page} {w}x{h} rest {tok.mean():.0%}", flush=True)
        del vae
        empty_cache()

    # ---- fixed noise per (sample, σ), shared by every LoRA
    noise = {}
    for i, s in enumerate(samples):
        g = torch.Generator().manual_seed(1000 + i)
        noise[i] = torch.randn(s["x0"].shape, generator=g)

    transformer = load_transformer(None, "base", device=device)
    from library.moss.lora import load_for_inference

    rows, keep = [], {}
    for name in names:
        net = None
        transformer.ref_timestep, transformer.ref_isolate = None, False
        if paths[name]:
            net = load_for_inference(transformer, resolve_under_home(paths[name]))
        else:
            # The base model never saw a reference; run it as the LoRAs run.
            transformer.ref_timestep, transformer.ref_isolate = 0.0, True
        t0 = time.time()
        for i, s in enumerate(samples):
            x0, eps = s["x0"].to(device), noise[i].to(device)
            for sigma in sigmas:
                x_t = (1 - sigma) * x0 + sigma * eps
                ts = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
                v_hat = transformer(
                    x_t,
                    ts,
                    s["emb"].to(device),
                    return_dict=False,
                    ref=s["ref"].to(device),
                )[0].float()
                x_hat = x_t - sigma * v_hat
                err = x_hat - x0
                rest = s["rest"].to(device)
                v_err = (v_hat - (eps - x0)).pow(2).mean(1)[0]
                dc = (err * rest).sum((2, 3))[0] / rest.sum().clamp(min=1)
                rows.append(
                    dict(
                        lora=name,
                        sample=s["name"],
                        page=bool(s.get("page")),
                        sigma=sigma,
                        rest_vmse=float(v_err[rest].mean()),
                        region_vmse=float(v_err[~rest].mean())
                        if (~rest).any()
                        else None,
                        tone0=tone_term(err, rest, 0),
                        tone8=tone_term(err, rest, 8),
                        dc=dc.cpu().tolist(),
                    )
                )
                decode_me = s.get("page") or i < args.decode
                if sigma in dec_sigmas and decode_me:
                    keep[(name, i, sigma)] = x_hat.cpu()
        print(
            f"[{name}] {len(samples)} samples x {len(sigmas)} σ in {time.time() - t0:.0f}s",
            flush=True,
        )
        if net is not None:
            net.restore()
            del net
    transformer.to("cpu")
    del transformer
    empty_cache()

    # ---- decode with the DiT off the card
    vae = load_vae(None, device=device)
    decoded = []
    for (name, i, sigma), x_hat in keep.items():
        s = samples[i]
        px = s.get("rest_px")
        if px is None:
            h, w = s["rest"].shape
            px = s["rest"].numpy().repeat(16, 0).repeat(16, 1)
        decoded.append(
            dict(
                lora=name,
                sample=s["name"],
                page=bool(s.get("page")),
                sigma=sigma,
                **lab_delta(x_hat, s["x0"], px, vae),
            )
        )
    del vae
    empty_cache()

    out = resolve_under_home(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(args), rows=rows, decoded=decoded)))
    summarize(rows, decoded)
    print(f"-> {out}", flush=True)


def summarize(rows: list[dict], decoded: list[dict]) -> None:
    import collections

    for page in (False, True):
        print(f"\n=== {'held-out pages' if page else 'training pairs'} (latent) ===")
        print(
            f"{'lora':>5} {'σ':>4} | {'rest vMSE':>9} {'tone0':>8} {'tone8':>8} {'t8/vMSE':>7} | "
            f"{'‖mean dc‖':>9} {'mean‖dc‖':>8} {'consist':>7}"
        )
        groups = collections.defaultdict(list)
        for r in rows:
            if r["page"] == page:
                groups[(r["lora"], r["sigma"])].append(r)
        for (lora, sigma), rs in groups.items():
            dc = np.array([r["dc"] for r in rs])
            mean_dc = np.linalg.norm(dc.mean(0))
            norm_dc = np.linalg.norm(dc, axis=1).mean()
            vm = np.nanmean([r["rest_vmse"] for r in rs])
            t0 = np.nanmean([r["tone0"] for r in rs])
            t8 = np.nanmean([r["tone8"] for r in rs])
            print(
                f"{lora:>5} {sigma:>4} | {vm:9.4f} {t0:8.5f} {t8:8.5f} {t8 / vm:7.3f} | "
                f"{mean_dc:9.4f} {norm_dc:8.4f} {mean_dc / norm_dc:7.2f}"
            )
    print("\n=== decoded x̂₀ vs x₀, Lab on the rest pixels ===")
    groups = collections.defaultdict(list)
    for d in decoded:
        groups[(d["page"], d["lora"], d["sigma"])].append(d)
    for (page, lora, sigma), ds in sorted(groups.items()):
        print(
            f"{'page' if page else 'pair':>4} {lora:>5} σ {sigma}: "
            f"ΔL {np.nanmean([d['dL'] for d in ds]):+.2f} (min {np.nanmin([d['dL'] for d in ds]):+.2f} max {np.nanmax([d['dL'] for d in ds]):+.2f}) "
            f"Δb {np.nanmean([d['db'] for d in ds]):+.2f} sat× {np.nanmean([d['sat'] for d in ds]):.3f}  n={len(ds)}"
        )


if __name__ == "__main__":
    main()
