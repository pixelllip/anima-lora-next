"""Tone drift of sanitize renders on the unedited pixels (docs/moss/proposal/i2i_tone_loss.md).

Lab after a σ=3 px blur; the mask is the pixels no compared run changed by ΔE > 8,
dilated 20 px. Per image and run:

- global ΔL / Δb / sat× (the proposal's table);
- ΔL / Δb per 128 px cell (8×8 tokens, ``tone_pool 8``): spread and sign agreement;
- ΔL binned by input L (an offset vs a tone curve);
- median ΔE before and after removing one global Lab offset, one offset per cell,
  and a per-L-bin curve.

    .venv/bin/python project/moss_lora/src/tone_map.py \
        --runs 200p=output/tests/moss_i2i_sanitize_200p_cli e6=output/tests/moss_i2i_sanitize_wm_e6
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from library.env import resolve_under_home
from library.moss.cache import fit

PAGES = [
    "hard_to_sanitize1",
    "hard_to_sanitize2",
    "hard_to_sanitize3",
    "usual_case1",
    "usual_case2",
]
CELL = 128
L_BINS = [0, 20, 40, 60, 80, 90, 101]


def lab(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(img.convert("RGB"), np.float32) / 255.0
    rgb = cv2.GaussianBlur(rgb, (0, 0), 3)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab), sat


def delta_e(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(-1))


def cells(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Masked mean of ``d`` (H, W, C) per CELL×CELL tile with ≥ ½ coverage → (n, C)."""
    h, w = mask.shape
    out = []
    for y in range(0, h - CELL + 1, CELL):
        for x in range(0, w - CELL + 1, CELL):
            m = mask[y : y + CELL, x : x + CELL]
            if m.mean() >= 0.5:
                out.append(d[y : y + CELL, x : x + CELL][m].mean(0))
    return np.array(out)


def per_cell_offset(mask: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``d`` with each cell's masked mean removed (cells under ½ coverage keep the global)."""
    res = d - d[mask].mean(0)
    h, w = mask.shape
    for y in range(0, h, CELL):
        for x in range(0, w, CELL):
            m = mask[y : y + CELL, x : x + CELL]
            if m.mean() >= 0.5:
                tile = d[y : y + CELL, x : x + CELL]
                res[y : y + CELL, x : x + CELL] = tile - tile[m].mean(0)
    return res


def per_bin_curve(mask: np.ndarray, d: np.ndarray, L_in: np.ndarray) -> np.ndarray:
    res = d - d[mask].mean(0)
    idx = np.digitize(L_in, L_BINS) - 1
    for k in range(len(L_BINS) - 1):
        sel = mask & (idx == k)
        if sel.sum() > 500:
            res[idx == k] = d[idx == k] - d[sel].mean(0)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="output/moss_results")
    p.add_argument("--runs", nargs="+", required=True, help="name=dir")
    p.add_argument("--out", default=None, help="write the numbers as JSON")
    args = p.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    report = {}
    for page in PAGES:
        outs = {
            name: Image.open(resolve_under_home(f"{d}/000_{page}_base_s42.png"))
            for name, d in runs.items()
        }
        size = next(iter(outs.values())).size
        src_lab, src_sat = lab(
            fit(Image.open(resolve_under_home(f"{args.src}/{page}.png")), *size)
        )
        labs = {name: lab(img) for name, img in outs.items()}
        changed = np.zeros(src_lab.shape[:2], bool)
        for o_lab, _ in labs.values():
            changed |= delta_e(o_lab, src_lab) > 8
        mask = ~ndimage.binary_dilation(changed, iterations=20)
        print(f"\n== {page} {size[0]}x{size[1]}  unedited {mask.mean():.0%}")
        report[page] = {}
        for name, (o_lab, o_sat) in labs.items():
            d = o_lab - src_lab
            dL, da, db = d[mask].mean(0)
            sat = o_sat[mask].mean() / max(src_sat[mask].mean(), 1e-6)
            c = cells(mask, d)
            bins = []
            idx = np.digitize(src_lab[..., 0], L_BINS) - 1
            for k in range(len(L_BINS) - 1):
                sel = mask & (idx == k)
                bins.append(float(d[sel, 0].mean()) if sel.sum() > 500 else None)
            de = {
                "raw": float(np.median(np.sqrt((d[mask] ** 2).sum(-1)))),
                "global": float(
                    np.median(np.sqrt(((d - d[mask].mean(0))[mask] ** 2).sum(-1)))
                ),
                "cell": float(
                    np.median(np.sqrt((per_cell_offset(mask, d)[mask] ** 2).sum(-1)))
                ),
                "Lcurve": float(
                    np.median(
                        np.sqrt(
                            (per_bin_curve(mask, d, src_lab[..., 0])[mask] ** 2).sum(-1)
                        )
                    )
                ),
            }
            row = {
                "dL": float(dL),
                "da": float(da),
                "db": float(db),
                "sat": float(sat),
                "cells": len(c),
                "cell_dL_mean": float(c[:, 0].mean()),
                "cell_dL_std": float(c[:, 0].std()),
                "cell_dL_p10_p90": [
                    float(np.percentile(c[:, 0], 10)),
                    float(np.percentile(c[:, 0], 90)),
                ],
                "cell_dL_same_sign": float((np.sign(c[:, 0]) == np.sign(dL)).mean()),
                "cell_db_mean": float(c[:, 2].mean()),
                "cell_db_std": float(c[:, 2].std()),
                "cell_db_same_sign": float((np.sign(c[:, 2]) == np.sign(db)).mean()),
                "dL_by_L": dict(
                    zip([f"{a}-{b}" for a, b in zip(L_BINS, L_BINS[1:])], bins)
                ),
                "medianDE": de,
            }
            report[page][name] = row
            print(
                f"  {name:>5}: ΔL {dL:+.2f} Δa {da:+.2f} Δb {db:+.2f} sat× {sat:.2f} | "
                f"cells {len(c)} ΔL {c[:, 0].mean():+.2f}±{c[:, 0].std():.2f} "
                f"(p10 {row['cell_dL_p10_p90'][0]:+.2f} p90 {row['cell_dL_p10_p90'][1]:+.2f}, "
                f"same sign {row['cell_dL_same_sign']:.0%}) Δb {c[:, 2].mean():+.2f}±{c[:, 2].std():.2f} "
                f"({row['cell_db_same_sign']:.0%})"
            )
            print(
                "         ΔL by input L: "
                + "  ".join(
                    f"{k}:{'—' if v is None else f'{v:+.2f}'}"
                    for k, v in row["dL_by_L"].items()
                )
            )
            print(
                f"         median ΔE raw {de['raw']:.2f} → −global {de['global']:.2f} "
                f"→ −cell {de['cell']:.2f} | −L curve {de['Lcurve']:.2f}"
            )
    if args.out:
        with open(resolve_under_home(args.out), "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
