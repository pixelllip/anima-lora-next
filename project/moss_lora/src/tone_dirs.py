"""Tone directions in the packed DiT latent, and the probe's DC error projected on them.

A page shifted in Lab by ±δ (uniform L, a shadow lift ``δ·(1 − L/100)``, a, b),
VAE-encoded and packed: the per-channel latent mean's response per unit shift is
that tone's direction (128-d). ``tone_probe.py``'s per-sample DC error on the rest
tokens is then fitted on the directions (least squares), giving a predicted ΔL / Δb
per sample to compare between LoRAs and against the decoded Lab measurement.

    make daemon-run ARGS="project/moss_lora/src/tone_dirs.py"      # directions (GPU) + analysis
    .venv/bin/python project/moss_lora/src/tone_dirs.py --analyze   # analysis only
"""

from __future__ import annotations

import argparse
import collections
import json

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache

PAGES = [
    "hard_to_sanitize1",
    "hard_to_sanitize2",
    "hard_to_sanitize3",
    "usual_case1",
    "usual_case2",
]
DELTA = 3.0
TONES = ("L", "shadow", "a", "b")


def shifted(img: Image.Image, tone: str, delta: float) -> Image.Image:
    rgb = np.asarray(img, np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    if tone == "L":
        lab[..., 0] += delta
    elif tone == "shadow":
        lab[..., 0] += delta * (1 - lab[..., 0] / 100)
    else:
        lab[..., 1 if tone == "a" else 2] += delta
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB).clip(0, 1)
    return Image.fromarray((out * 255).round().astype(np.uint8))


def images(n_pairs: int) -> list[tuple[str, Image.Image]]:
    out = []
    for page in PAGES:
        im = moss_cache.load_rgb(resolve_under_home(f"output/moss_results/{page}.png"))
        out.append((page, moss_cache.fit(im, *moss_cache.target_size(*im.size, 1024))))
    root = resolve_under_home("post_image_dataset/i2i/sanitize/staging")
    rows = [json.loads(line) for line in open(root / "pairs.jsonl")]
    for row in rows[:: max(1, len(rows) // n_pairs)][:n_pairs]:
        im = moss_cache.load_rgb(root / row["target"])
        out.append(
            (row["stem"], moss_cache.fit(im, *moss_cache.target_size(*im.size, 1024)))
        )
    return out


@torch.no_grad()
def directions(n_pairs: int, path) -> None:
    from library.moss.loader import load_vae
    from library.moss.model import encode

    vae = load_vae(None, device=torch.device("cuda"))
    per = {t: [] for t in TONES}
    for name, img in images(n_pairs):
        for tone in TONES:
            plus, minus = (
                encode(moss_cache.to_pixels(shifted(img, tone, s * DELTA)), vae)[
                    0
                ].mean((1, 2))
                for s in (1, -1)
            )
            per[tone].append(((plus - minus) / (2 * DELTA)).cpu())
        print(f"[dir] {name} {img.size}", flush=True)
    stacked = {t: torch.stack(v) for t, v in per.items()}
    save_file({t: v.contiguous() for t, v in stacked.items()}, str(path))


def analyze(path, probe) -> None:
    per = load_file(str(path))
    print(
        "=== direction stability: cos of each image's direction to the mean; |d| per unit ==="
    )
    dirs = {}
    for t, v in per.items():
        v = v.double().numpy()
        m = v.mean(0)
        cos = v @ m / np.linalg.norm(v, axis=1) / np.linalg.norm(m)
        dirs[t] = m
        print(
            f"  {t:>6}: cos min {cos.min():.2f} median {np.median(cos):.2f}  |d| {np.linalg.norm(m):.4f}"
        )
    names = list(dirs)
    D = np.stack([dirs[t] for t in names], 1)
    gram = D.T @ D / np.outer(np.linalg.norm(D, axis=0), np.linalg.norm(D, axis=0))
    print(
        "  cos between directions:\n"
        + "\n".join(
            f"    {a:>6} " + " ".join(f"{gram[i, j]:+.2f}" for j in range(len(names)))
            for i, a in enumerate(names)
        )
    )

    d = json.load(open(probe))
    rows = [
        r for r in d["rows"] if r["rest_vmse"] == r["rest_vmse"]
    ]  # empty rest → nan
    # Fit L / a / b only (shadow is near-collinear with L); report shadow alone too.
    fit_names = ["L", "a", "b"]
    F = np.stack([dirs[t] for t in fit_names], 1)
    pinv = np.linalg.pinv(F)
    u_shadow = dirs["shadow"] / np.linalg.norm(dirs["shadow"])
    for r in rows:
        dc = np.asarray(r["dc"])
        c = pinv @ dc
        r["pL"], r["pa"], r["pb"] = c
        r["pshadow"] = dc @ u_shadow / np.linalg.norm(dirs["shadow"])
        resid = dc - F @ c
        r["tone_share"] = 1 - (resid @ resid) / max(dc @ dc, 1e-12)

    print(
        "\n=== DC error fitted on the tone directions (predicted Lab shift, single step) ==="
    )
    print(
        f"{'set':>5} {'lora':>5} {'σ':>4} | {'pL':>6} {'±sd':>5} {'pb':>6} {'±sd':>5} {'pshad':>6} | "
        f"{'tone share of |dc|²':>19} | {'tone² (pL²+pb²)':>15}"
    )
    g = collections.defaultdict(list)
    for r in rows:
        g[("page" if r["page"] else "pair", r["lora"], r["sigma"])].append(r)
    for (s, lora, sigma), rs in sorted(
        g.items(), key=lambda k: (k[0][0], k[0][2], k[0][1])
    ):
        pL = np.array([r["pL"] for r in rs])
        pb = np.array([r["pb"] for r in rs])
        print(
            f"{s:>5} {lora:>5} {sigma:>4} | {pL.mean():+6.2f} {pL.std():5.2f} {pb.mean():+6.2f} {pb.std():5.2f} "
            f"{np.mean([r['pshadow'] for r in rs]):+6.2f} | {np.mean([r['tone_share'] for r in rs]):19.2f} | "
            f"{np.mean(pL**2 + pb**2):15.3f}"
        )

    print("\n=== predicted vs decoded (same sample / LoRA / σ) ===")
    key = {(r["lora"], r["sample"], r["sigma"]): r for r in rows}
    pairs = [
        (key[(x["lora"], x["sample"], x["sigma"])], x)
        for x in d["decoded"]
        if (x["lora"], x["sample"], x["sigma"]) in key and x["dL"] == x["dL"]
    ]
    for comp, dec in (("pL", "dL"), ("pb", "db"), ("pshadow", "dL")):
        a = np.array([p[comp] for p, _ in pairs])
        b = np.array([x[dec] for _, x in pairs])
        slope = (a @ b) / (a @ a)
        print(
            f"  {comp:>7} vs decoded {dec}: r = {np.corrcoef(a, b)[0, 1]:+.2f}, decoded ≈ {slope:.2f}·pred  (n={len(a)})"
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true", help="skip the VAE, reuse --dirs")
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--dirs", default="project/moss_lora/space/tone_dirs.safetensors")
    p.add_argument("--probe", default="project/moss_lora/space/tone_probe.json")
    args = p.parse_args()
    path = resolve_under_home(args.dirs)
    if not args.analyze:
        directions(args.pairs, path)
    analyze(path, resolve_under_home(args.probe))


if __name__ == "__main__":
    main()
