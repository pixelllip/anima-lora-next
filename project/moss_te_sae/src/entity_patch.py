"""Phase 2 B: a per-character patch on the name's TE-output rows, distilled from the
tag-expanded caption through the frozen DiT.

``L = ‖v(x_t, σ, N + Δ) − v(x_t, σ, T)‖²`` with ``N`` / ``T`` the phase-0 name /
name + tags captions, ``x_t`` from teacher generations (``--teacher_dir``, other seeds
than the evaluation) and σ from the training distribution. ``Δ`` covers the name's
rows only. Arms: ``split`` starts from the untrained ``split`` injection of
``entity_add.py`` at ``--rho``, ``zero`` from zero.

Per arm: ``<out_root>/<arm>/patch.safetensors`` and the patched ``N`` written into that
folder's ``_te_cache`` under ``generate.py``'s key, so ``scripts/moss/generate.py
--out_dir <out_root>/<arm>`` with the plain ``N`` prompt renders with the patch. The
validation gap ``‖v_S − v_T‖² / ‖v_N − v_T‖²`` (fixed x_t, σ 0.9 / 0.6 / 0.3) is logged
every ``--eval_every`` steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from entity_add import added_tags, phase0
from entity_dit import log, span
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.train import (
    NUM_TRAIN_TIMESTEPS,
    checkpoint_blocks,
    place,
    step_reserve_gb,
)


def te_cache_path(cache_dir: Path, text: str, precision: str) -> Path:
    """``library.moss.generate.cached_embeds``'s file for ``text``."""
    key = f"{loader.MAX_SEQUENCE_LENGTH}\0{text}"
    if precision != "bf16":
        key = f"{precision}\0{key}"
    return cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.safetensors"


def split_init(n_emb, t_emb, tags, base_rows, name_rows, rho):
    n, t = n_emb.float(), t_emb.float()
    base = n[base_rows].mean(0)
    scale = n[name_rows].norm(dim=-1).mean()
    out = torch.zeros(len(name_rows), n.shape[1])
    for k in range(len(name_rows)):
        mine = [r for i, (_, rs) in enumerate(tags) if i % len(name_rows) == k for r in rs]
        d = t[mine].mean(0) - base
        out[k] = rho * scale * d / d.norm()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="malfoid")
    ap.add_argument("--folder", default="malfoid_phase0")
    ap.add_argument("--teacher_dir", default="output/tests/malfoid_patch/teacher")
    ap.add_argument("--out_root", default="output/tests/malfoid_patch")
    ap.add_argument("--arms", default="split,zero")
    ap.add_argument("--rho", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr_scale", type=float, default=0.02, help="× name-row element rms")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--sigma_mu", type=float, default=-1.09)
    ap.add_argument("--sigma_s", type=float, default=0.95)
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_root = resolve_under_home(args.out_root)

    tok = loader.load_tokenizer(default_moss_dir())
    n_text, t_text, _ = phase0(args.folder)
    tags, base_rows = added_tags(tok, n_text, t_text)
    a, b = span(tok, n_text, args.char, '"characters"')
    name_rows = list(range(a, b))
    enc = loader.encode_prompts(
        default_moss_dir(), [n_text, t_text], log=log, precision=args.text_encoder
    )
    n_emb, t_emb = enc[n_text], enc[t_text]
    teachers = sorted(resolve_under_home(args.teacher_dir).glob("latents/*.safetensors"))
    x0s = [load_file(p)["latent"] for p in teachers]
    log(f"{args.char}: name rows {name_rows}, {len(tags)} tags, {len(x0s)} teacher latents")

    device = torch.device("cuda")
    transformer = loader.load_transformer(default_moss_dir(), "base")
    transformer.requires_grad_(False)
    joint = x0s[0].shape[2] * x0s[0].shape[3] + max(n_emb.shape[0], t_emb.shape[0])
    place(transformer, device, -1, step_reserve_gb(joint, grad_checkpointing=True))
    checkpoint_blocks(transformer.layers)

    def velocity(noisy, sigma, emb):
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        return transformer(
            noisy, t, emb[None].to(device, torch.bfloat16), return_dict=False
        )[0].float()

    def patched(delta):
        e = n_emb.to(device).float()
        pad = torch.zeros_like(e)
        pad[name_rows] = delta
        return e + pad

    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    val = []
    for i, sigma in enumerate((0.9, 0.6, 0.3)):
        x0 = x0s[i % len(x0s)].to(device)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        noisy = (1 - sigma) * x0 + sigma * noise
        with torch.no_grad():
            vt = velocity(noisy, sigma, t_emb)
            vn = velocity(noisy, sigma, n_emb)
        val.append((noisy, sigma, vt, float((vn - vt).pow(2).sum())))

    def gap(delta):
        with torch.no_grad():
            e = patched(delta)
            return [
                float((velocity(noisy, s, e) - vt).pow(2).sum()) / base
                for noisy, s, vt, base in val
            ]

    rms = float(n_emb[name_rows].float().pow(2).mean().sqrt())
    report = {"char": args.char, "N": n_text, "T": t_text, "arms": {}}
    for arm in args.arms.split(","):
        rng = random.Random(args.seed)
        noise_gen = torch.Generator(device="cpu").manual_seed(args.seed)
        init = (
            split_init(n_emb, t_emb, tags, base_rows, name_rows, args.rho)
            if arm == "split"
            else torch.zeros(len(name_rows), n_emb.shape[1])
        )
        delta = init.to(device).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=args.lr_scale * rms)
        hist = [{"step": 0, "gap": gap(delta.detach())}]
        log(f"[{arm}] step 0 val gap {['%.3f' % g for g in hist[0]['gap']]}")
        t0, losses = time.time(), []
        for step in range(1, args.steps + 1):
            x0 = rng.choice(x0s).to(device)
            z = rng.gauss(args.sigma_mu, args.sigma_s)
            sigma = 1.0 - 1.0 / (1.0 + math.exp(-z))
            noise = torch.randn(x0.shape, generator=noise_gen).to(device)
            noisy = (1 - sigma) * x0 + sigma * noise
            with torch.no_grad():
                vt = velocity(noisy, sigma, t_emb)
            loss = (velocity(noisy, sigma, patched(delta)) - vt).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if step % args.eval_every == 0 or step == args.steps:
                g = gap(delta.detach())
                hist.append(
                    {
                        "step": step,
                        "gap": g,
                        "train_loss": sum(losses) / len(losses),
                        "delta_over_row": float(
                            delta.detach().norm(dim=-1).mean()
                            / n_emb[name_rows].float().norm(dim=-1).mean()
                        ),
                    }
                )
                losses = []
                log(
                    f"[{arm}] step {step} loss {hist[-1]['train_loss']:.4f} val gap "
                    f"{['%.3f' % x for x in g]} ‖Δ‖/‖row‖ {hist[-1]['delta_over_row']:.2f} "
                    f"({(time.time() - t0) / step:.1f}s/step)"
                )
        out_dir = out_root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        d = delta.detach().cpu()
        save_file(
            {"delta": d.contiguous()},
            out_dir / "patch.safetensors",
            metadata={
                "char": args.char,
                "rows": json.dumps(name_rows),
                "N": n_text,
                "text_encoder": args.text_encoder,
            },
        )
        pe = n_emb.clone()
        pe[name_rows] = (pe[name_rows].float() + d).to(pe.dtype)
        cache = te_cache_path(out_dir / "_te_cache", n_text, args.text_encoder)
        cache.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeds": pe.contiguous()}, cache)
        report["arms"][arm] = hist
        log(f"[{arm}] wrote {out_dir}/patch.safetensors and {cache.name}")
    (out_root / "patch_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
