#!/usr/bin/env python3
"""σ-schedule A/B: the pipeline's snapped grid against ComfyUI's schedulers.

An arm is ``<schedule>:<mode><steps>``: ``snap`` (``snapped_sigmas``, the
pipeline), or ComfyUI's ``normal`` / ``sgm_uniform`` / ``simple`` rebuilt for
MOSS's flow sampling (shift 5, multiplier 1); mode ``p`` runs every forward,
``s`` adds Spectrum at ``GenerateRequest`` defaults. Euler, CFG 3, uncond skip
22, empty negative, base DiT, default (INT8) TE, no LoRA — what ``KSampler (Spectrum for MOSS)``
runs. ComfyUI's ``normal`` is the pipeline grid before the snap, so
``snap`` vs ``normal`` isolates the snap and ``normal`` vs ``sgm_uniform`` /
``simple`` the schedule tail.

``--ref`` adds a fine-grid Euler solve per prompt (``snap:p200`` by default);
each arm's latent rel-L2 to it is printed and written to ``result.json``.

    make daemon-run ARGS="--stall-timeout 900 project/moss_lora/src/schedule_ab.py"
"""

import argparse
import json

import torch
from safetensors.torch import load_file

from _render import decoded, grid, render, save_latent, settings_tag, uncond_embeds
from bench._moss import PROMPTS
from library.env import resolve_under_home
from library.moss.loader import compile_blocks, empty_cache, load_transformer, load_vae
from library.moss.model import snapped_sigmas
from library.moss.requests import DEFAULT_SHIFT, DEFAULT_TEXT_ENCODER

TOTAL = 1000


def _shift(t: torch.Tensor, s: float = DEFAULT_SHIFT) -> torch.Tensor:
    return s * t / (1 + (s - 1) * t)


def comfy_sigmas(name: str, steps: int) -> torch.Tensor:
    """ComfyUI's scheduler on ``ModelSamplingDiscreteFlow(shift 5, multiplier 1)``,
    without the terminal 0 (``denoise`` appends it)."""
    sigma_min = float(_shift(torch.tensor(1 / TOTAL, dtype=torch.float64)))
    if name == "simple":
        # sigmas[-(1 + int(x * ss))] of the table shift(k / TOTAL), k = 1..TOTAL
        ss = TOTAL / steps
        t = torch.tensor(
            [(TOTAL - int(x * ss)) / TOTAL for x in range(steps)], dtype=torch.float64
        )
    elif name == "normal":
        t = torch.linspace(1.0, sigma_min, steps, dtype=torch.float64)
    elif name == "sgm_uniform":
        t = torch.linspace(1.0, sigma_min, steps + 1, dtype=torch.float64)[:-1]
    else:
        raise ValueError(f"unknown schedule {name!r}")
    return _shift(t).float()


def sigmas_for(schedule: str, steps: int) -> torch.Tensor:
    if schedule == "snap":
        return snapped_sigmas(steps)
    return comfy_sigmas(schedule, steps)


def parse_arm(text: str) -> tuple[str, bool, int]:
    schedule, run = text.split(":")
    return schedule, run[0] == "s", int(run[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="anime_sunset,zh_fridge,ja_manga")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument(
        "--arms",
        default="snap:p50,normal:p50,sgm_uniform:p50,simple:p50,"
        "snap:s50,normal:s50,sgm_uniform:s50,simple:s50",
        help="<schedule>:<p|s><steps>, comma-separated",
    )
    ap.add_argument("--ref", default="", help="reference arm, e.g. snap:p200")
    ap.add_argument("--out", default="output/tests/schedule_ab")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    names = args.prompts.split(",")
    arms = [parse_arm(a) for a in args.arms.split(",")]
    ref = parse_arm(args.ref) if args.ref else None
    out = resolve_under_home(args.out)
    lat_dir = out / "latents"
    lat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    embeds, uncond, uncond_mask = uncond_embeds(
        device, [PROMPTS[n] for n in names], precision=DEFAULT_TEXT_ENCODER
    )

    def stem(name, arm):
        schedule, spectrum, steps = arm
        tag = settings_tag(
            prompt=PROMPTS[name], seed=args.seed, size=args.size, sampler="euler"
        )
        return f"{name}_{schedule}_{'s' if spectrum else 'p'}{steps}_{tag}"

    runs = [(n, a) for n in names for a in arms + ([ref] if ref else [])]
    todo = [r for r in runs if not (lat_dir / f"{stem(*r)}.safetensors").exists()]
    if todo:
        tr = load_transformer("", "base", device=device)
        compile_blocks(tr.layers)
        for name, arm in todo:
            schedule, spectrum, steps = arm
            sigmas = sigmas_for(schedule, steps)
            z = render(
                tr,
                embeds[PROMPTS[name]],
                uncond,
                uncond_mask,
                sigmas,
                seed=args.seed,
                height=h,
                width=w,
                cfg=3.0,
                sampler="euler",
                spectrum=spectrum,
            )
            save_latent(z, lat_dir / f"{stem(name, arm)}.safetensors")
            print(
                f"{name} {schedule} {'s' if spectrum else 'p'}{steps}: "
                f"{len(sigmas)} σ, last {float(sigmas[-1]):.4f}",
                flush=True,
            )
        del tr
        empty_cache()

    metrics = {}
    if ref:
        for name in names:
            zr = load_file(str(lat_dir / f"{stem(name, ref)}.safetensors"))["latent"]
            for arm in arms:
                z = load_file(str(lat_dir / f"{stem(name, arm)}.safetensors"))["latent"]
                rel = float((z - zr).norm() / zr.norm())
                key = f"{arm[0]}:{'s' if arm[1] else 'p'}{arm[2]}"
                metrics.setdefault(key, {})[name] = round(rel, 4)
        for key, per in metrics.items():
            print(
                f"rel-L2 to {args.ref}  {key:>16}  "
                + "  ".join(f"{n} {v:.4f}" for n, v in per.items()),
                flush=True,
            )
        (out / "result.json").write_text(
            json.dumps({"ref": args.ref, "rel_l2": metrics}, indent=1)
        )

    vae = load_vae("", device=device)
    rows = []
    for name in names:
        cells = []
        for arm in arms + ([ref] if ref else []):
            png = out / f"{stem(name, arm)}.png"
            im = decoded(lat_dir / f"{stem(name, arm)}.safetensors", vae, device, png)
            cells.append((f"{arm[0]} {'s' if arm[1] else 'p'}{arm[2]}", im))
        rows.append((name, cells))
    sheet = out / "grid.jpg"
    grid(rows).save(sheet, quality=90)
    print(f"wrote {sheet}", flush=True)


if __name__ == "__main__":
    main()
