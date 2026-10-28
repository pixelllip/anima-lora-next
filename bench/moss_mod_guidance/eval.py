"""Render the micro-distilled pooled head (``distill.py``) as modulation guidance.

Arms per (prompt, seed), all Euler / CFG 3 / uncond skip 22 / shift 5, 1024²:

- ``off``   the base model, no head
- ``w0``    the head on, training-consistent: each CFG branch gets the offset of
            its own prompt's pool (cond / the empty negative)
- ``w3``, ``w6``  plus ``w · W_mod · (head(pool(p₊)) − head(pool(p₋)))`` on blocks
            ``start_layer``…41 of both branches, ``p±`` = the prompt with
            best / worst quality tags spliced after the rating (NL: appended)
- ``pq`` / ``pc`` (tag prompts only)  the quality-tag separation check: text
            unchanged, the cond offset taken from the prompt with ``score_9``
            (``pq``) or a content tag (``pc``) spliced in at the same slot

Latent RMS distance to ``w0`` is logged per arm; grids land in the run dir.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_mod_guidance/eval.py"
"""

import argparse
import json

import torch
from safetensors.torch import load_file

from bench._common import make_run_dir, write_result
from bench.moss_mod_guidance.distill import (
    PooledHead,
    attach_head,
    content_pool,
    steer_mod,
)
from bench.moss_mod_guidance.probe_pool import AXIS_NEG, AXIS_POS, splice
from library.env import resolve_under_home

TAG_PROMPTS = {
    "classroom": "general, 1girl, solo, long hair, silver hair, red eyes, school uniform, "
    "sitting, classroom, window, sunset, cherry blossoms, looking at viewer, smile",
    "winter_city": "general, 1boy, 1girl, couple, holding hands, street, night, city lights, "
    "snow, coat, scarf, smile, from behind",
}
NL_PROMPTS = {
    "fox_shrine": "A watercolor-style illustration of a fox spirit girl with white hair and "
    "fox ears, wearing a red and white shrine maiden outfit, standing in a bamboo forest "
    "with fireflies at dusk.",
}
NL_POS = " Masterpiece, best quality, highly detailed."
NL_NEG = " Worst quality, low quality, blurry."
CONTENT_TAG = "red ribbon"


@torch.no_grad()
def render(transformer, noise, sigmas, cond, uncond, off_c, off_u, steer, cfg, skip):
    """Euler over ``sigmas`` with per-branch time offsets; ``steer`` is
    ``(6, D) shift, block set`` or None."""
    x = noise.clone()
    sig = torch.cat([sigmas, torch.zeros(1)]).to(x.device)
    transformer._block_steer = steer
    for i in range(len(sigmas)):
        t = (sig[i] * 1000).reshape(1)
        transformer._time_offset = off_c
        vc = transformer(x, t, cond[0], cond[1], return_dict=False)[0].float()
        transformer._time_offset = off_u
        vu = transformer(
            x, t, uncond[0], uncond[1], skip_layers=skip, return_dict=False
        )[0].float()
        x = x + (sig[i + 1] - sig[i]) * (vu + cfg * (vc - vu))
    transformer._time_offset = None
    transformer._block_steer = None
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--head", default="output/ckpt_moss/mod_guidance/pooled_head.safetensors"
    )
    p.add_argument("--seeds", default="42,43")
    p.add_argument("--w", default="3,6")
    p.add_argument("--start_layer", type=int, default=12)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--text_encoder", default="int8")
    p.add_argument("--model", default="base")
    p.add_argument("--model_dir", default="")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    from PIL import Image, ImageDraw

    from library.moss.cache import caption_converter
    from library.moss.generate import cached_embeds, initial_noise
    from library.moss.loader import (
        compile_blocks,
        empty_cache,
        load_transformer,
        load_vae,
        pad_embeds,
    )
    from library.moss.model import decode, snapped_sigmas
    from library.moss.requests import DEFAULT_CFG, DEFAULT_SHIFT, DEFAULT_UNCOND_SKIP

    device = torch.device("cuda")
    run_dir = make_run_dir("moss_mod_guidance", args.label or "eval")
    to_layout = caption_converter("layout_json")
    seeds = [int(s) for s in args.seeds.split(",")]
    ws = [float(w) for w in args.w.split(",")]

    # name -> main, p+, p-, and (tag prompts) the pq / pc pool texts
    prompts = {}
    for name, raw in TAG_PROMPTS.items():
        prompts[name] = dict(
            main=to_layout(raw),
            pos=to_layout(splice(raw, AXIS_POS, True)),
            neg=to_layout(splice(raw, AXIS_NEG, True)),
            pq=to_layout(splice(raw, ["score_9"], True)),
            pc=to_layout(splice(raw, [CONTENT_TAG], True)),
        )
    for name, raw in NL_PROMPTS.items():
        prompts[name] = dict(main=raw, pos=raw + NL_POS, neg=raw + NL_NEG)
    texts = [""] + [t for d in prompts.values() for t in d.values()]
    embeds = cached_embeds(
        args.model_dir, texts, run_dir.parent / "_te_cache", args.text_encoder
    )

    head_sd = load_file(str(resolve_under_home(args.head)))
    from safetensors import safe_open

    with safe_open(str(resolve_under_home(args.head)), "pt") as f:
        span = tuple(json.loads(f.metadata()["span"]))
    head = PooledHead().to(device).eval()
    head.load_state_dict(head_sd)

    transformer = load_transformer(args.model_dir, args.model, device=device)
    transformer.requires_grad_(False)
    attach_head(transformer)
    if not args.no_compile:
        compile_blocks(transformer.layers)

    def off(text):
        with torch.no_grad():
            return head(content_pool(embeds[text], span)[None].to(device))

    def ctx(text):
        return pad_embeds([embeds[text]], device=device)

    sigmas = snapped_sigmas(args.steps, DEFAULT_SHIFT)
    blocks = frozenset(range(args.start_layer, len(transformer.layers)))
    uncond, off_u = ctx(""), off("")
    records, latents, metrics = [], {}, {}
    for name, d in prompts.items():
        cond, off_c = ctx(d["main"]), off(d["main"])
        delta = (off(d["pos"]) - off(d["neg"]))[0]
        dmod = steer_mod(transformer, delta)
        metrics[f"{name}.off_norm"] = off_c.norm().item()
        metrics[f"{name}.delta_norm"] = delta.norm().item()
        metrics[f"{name}.dmod_rms"] = dmod.float().pow(2).mean().sqrt().item()
        arms = {"off": (None, None, None), "w0": (off_c, off_u, None)}
        for w in ws:
            arms[f"w{w:g}"] = (off_c, off_u, (w * dmod, blocks))
        if "pq" in d:
            arms["pq"] = (off(d["pq"]), off_u, None)
            arms["pc"] = (off(d["pc"]), off_u, None)
        for seed in seeds:
            noise = initial_noise(seed, args.size, args.size).to(device)
            for arm, (oc, ou, steer) in arms.items():
                z = render(
                    transformer,
                    noise,
                    sigmas,
                    cond,
                    uncond,
                    oc,
                    ou,
                    steer,
                    DEFAULT_CFG,
                    DEFAULT_UNCOND_SKIP,
                ).cpu()
                latents[(name, seed, arm)] = z
                print(f"[render] {name} seed {seed} {arm}", flush=True)
            base = latents[(name, seed, "w0")]
            for arm in arms:
                dist = (latents[(name, seed, arm)] - base).pow(2).mean().sqrt().item()
                metrics[f"{name}.s{seed}.{arm}.rms_vs_w0"] = dist
                records.append(dict(prompt=name, seed=seed, arm=arm, rms_vs_w0=dist))
    transformer.to("cpu")
    del transformer
    empty_cache()

    vae = load_vae(args.model_dir, device=device)
    thumb = 384
    for name in prompts:
        arms = [a for (n, s, a) in latents if n == name and s == seeds[0]]
        grid = Image.new("RGB", (thumb * len(arms), (thumb + 20) * len(seeds)), "white")
        draw = ImageDraw.Draw(grid)
        for r, seed in enumerate(seeds):
            for c, arm in enumerate(arms):
                with torch.no_grad():
                    img = decode(latents[(name, seed, arm)].to(device), vae)[0]
                pil = Image.fromarray(
                    ((img.float().cpu().permute(1, 2, 0) + 1) * 127.5)
                    .clamp(0, 255)
                    .byte()
                    .numpy()
                )
                pil.save(run_dir / f"{name}_s{seed}_{arm}.png")
                grid.paste(
                    pil.resize((thumb, thumb)), (c * thumb, r * (thumb + 20) + 20)
                )
                draw.text(
                    (c * thumb + 4, r * (thumb + 20) + 4),
                    f"{arm} s{seed}",
                    fill="black",
                )
        grid.save(run_dir / f"grid_{name}.png")
    (run_dir / "rows.json").write_text(json.dumps(records, indent=1))
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics, artifacts=["rows.json"]
    )
    for k, v in metrics.items():
        print(f"{k:40s} {v:.4f}")
    print(f"[eval] {run_dir}")


if __name__ == "__main__":
    main()
