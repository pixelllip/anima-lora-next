"""The DC bias a MOSS LoRA learns, and an ablation that removes it (docs/proposal/moss_i2i_tone_loss.md).

Per adapted linear, b = s·up·(down·x̄), with x̄ the calibrated signed token mean of
its input (``library/moss/calibration/channel_mean.safetensors``): the shift every
token gets from the input's DC. Prints Σ‖b‖ per run and projection, its leverage
over a random input direction of the same norm, and the cosine of the stacked b
between runs.

    .venv/bin/python project/moss_lora/src/dc_bias.py \
        --runs 200p=output/ckpt_moss_i2i/moss_i2i_sanitize_200p.safetensors e6=…

``--ablate <lora> --out <file>`` writes the LoRA with x̄ projected out of ``down``
on ``--kinds`` (default v, o, ffn_up_gate, ffn_down: the paths that add to the
residual stream), to render against the original.
"""

from __future__ import annotations

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home

KINDS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]


def load_mean() -> dict[str, torch.Tensor]:
    path = resolve_under_home("library/moss/calibration/channel_mean.safetensors")
    return {k: v.float() for k, v in load_file(str(path)).items()}


def prefix(name: str) -> str:
    return "lora_unet_" + name.replace(".", "_")


def dc_bias(path: str, mean: dict) -> dict[str, tuple[torch.Tensor, float, float]]:
    """Per linear: (b, ‖ΔW‖_F, leverage of x̄ over a random direction)."""
    sd = {k: v.float() for k, v in load_file(str(resolve_under_home(path))).items()}
    out = {}
    for name, xbar in mean.items():
        p = prefix(name)
        up, down = sd[p + ".lora_up.weight"], sd[p + ".lora_down.weight"]
        s = sd[p + ".alpha"].item() / down.shape[0]
        b = s * up @ (down @ xbar)
        fro = s * torch.sqrt(torch.trace((up.T @ up) @ (down @ down.T)))
        lev = b.norm() / (fro * xbar.norm() / xbar.numel() ** 0.5)
        out[name] = (b, fro.item(), lev.item())
    return out


def report(runs: dict[str, str]) -> None:
    mean = load_mean()
    res = {n: dc_bias(p, mean) for n, p in runs.items()}
    print(
        f"{'run':10s} {'Σ‖b‖':>8s} {'Σ‖ΔW‖F':>8s} {'lev med':>8s}  "
        + "  ".join(f"{k:>11s}" for k in KINDS)
    )
    for n, r in res.items():
        per = [
            sum(v[0].norm().item() for k, v in r.items() if k.endswith("." + kind))
            for kind in KINDS
        ]
        lev = torch.tensor([v[2] for v in r.values()]).median().item()
        print(
            f"{n:10s} {sum(v[0].norm().item() for v in r.values()):8.2f} "
            f"{sum(v[1] for v in r.values()):8.2f} {lev:8.2f}  "
            + "  ".join(f"{x:11.2f}" for x in per)
        )
    stacked = {n: torch.cat([r[k][0] for k in mean]) for n, r in res.items()}
    print("\ncos of the stacked b")
    print(" " * 10 + "".join(f"{n:>10s}" for n in runs))
    for a in runs:
        print(
            f"{a:10s}"
            + "".join(
                f"{torch.nn.functional.cosine_similarity(stacked[a], stacked[b], 0):10.2f}"
                for b in runs
            )
        )


def ablate(src: str, out: str, kinds: list[str]) -> None:
    src_path = resolve_under_home(src)
    with safe_open(str(src_path), "pt") as f:
        meta = f.metadata()
    sd = load_file(str(src_path))
    for name, xbar in load_mean().items():
        if name.rsplit(".", 1)[1] not in kinds:
            continue
        key = prefix(name) + ".lora_down.weight"
        d, u = sd[key].float(), xbar / xbar.norm()
        sd[key] = (d - (d @ u)[:, None] * u[None, :]).to(sd[key].dtype)
    save_file(sd, str(resolve_under_home(out)), metadata=meta)
    print(f"wrote {out} (x̄ projected out of down on {', '.join(kinds)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", help="name=lora.safetensors")
    p.add_argument("--ablate", help="LoRA to write without its DC response")
    p.add_argument("--out", help="output path for --ablate")
    p.add_argument("--kinds", nargs="+", default=["v", "o", "ffn_up_gate", "ffn_down"])
    args = p.parse_args()
    if args.ablate:
        if not args.out:
            p.error("--ablate needs --out")
        ablate(args.ablate, args.out, args.kinds)
    if args.runs:
        report(dict(r.split("=", 1) for r in args.runs))


if __name__ == "__main__":
    main()
