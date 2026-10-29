"""Causal check on the chat-template rows: render the A/B eval prompts with the
image cut off from part of the text, or with another prompt's suffix rows.

Same setup as ``template_attn.py`` (Base, Euler 30 steps, CFG 3, seed 1, 1024²,
layout JSON, INT8 TE); its Base latents are the reference arm. Arms, cond branch
only (the uncond branch is the shipped one):

- ``swap_suffix``: the 7 suffix rows of the TE output come from prompt ``(i + 3) % 7``
  (the prefix rows are the same for every prompt, causal TE);
- ``no_caption_keys``: image queries cannot attend to the caption rows (every block);
- ``no_template_keys``: image queries cannot attend to the 10 template rows.

Scored with the Anima Tagger: recall of the prompt's own tags, recall of the swap
donor's tags that the prompt does not have, Jaccard of the kept tag set against the
reference render, and mean absolute pixel difference against it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

import library.moss.model as mm
from bench.moss_ab._render import uncond_embeds
from bench.moss_ab.lora_ab import PROMPTS
from library.env import resolve_under_home
from library.moss import loader
from library.moss.cache import caption_converter
from library.moss.generate import denoise, initial_noise
from library.moss.model import decode, snapped_sigmas

COMPOSITION = "project/finished/moss_lora/composition_prompts.txt"
REF_RAW = "output/tests/template_attn/lora_cmp_20261029/base/raw.pt"
ARMS = ("swap_suffix", "no_caption_keys", "no_template_keys")
N_PREFIX, N_SUFFIX = 3, 7
DROP_TAGS = {"safe", "sensitive", "{artist}", "original"}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


class KeyCut:
    """Patches ``mm._attention`` on the cond branch's joint length: image queries
    lose the keys in ``cut`` (bool over the joint sequence)."""

    def __init__(self, n_img, cut):
        L = cut.shape[0]
        self.L = L
        mask = torch.ones(L, L, dtype=torch.bool)
        mask[:n_img, cut] = False
        self.mask = mask[None, None].cuda()
        self._orig = mm._attention

    def __enter__(self):
        mm._attention = self._wrapped
        return self

    def __exit__(self, *exc):
        mm._attention = self._orig

    def _wrapped(self, q, k, v, rope, heads, key_mask=None, *args, **kw):
        if q.shape[1] != self.L or key_mask is not None:
            return self._orig(q, k, v, rope, heads, key_mask, *args, **kw)
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = mm.rotate(q.reshape(shape), rope).to(v.dtype)
        k = mm.rotate(k.reshape(shape), rope).to(v.dtype)
        return mm._sdpa(q, k, v.reshape(shape), self.mask).flatten(2)


def tag_list(template):
    if "," not in template:
        return []
    return [t.strip() for t in template.split(",") if t.strip() not in DROP_TAGS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--te_precision", default="int8")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tagger", default="models/captioners/anima-tagger-dbv4")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/template_causal/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, args.size.split("x"))
    n_img = (h // 16) * (w // 16)
    device = torch.device("cuda")

    templates = list(PROMPTS) + [
        line
        for line in resolve_under_home(COMPOSITION).read_text().splitlines()
        if line.strip()
    ]
    conv = caption_converter("layout_json")
    prompts = [conv(p.format(artist=args.artist)) for p in templates]
    embeds, uncond, uncond_mask = uncond_embeds(device, prompts, args.te_precision)
    rows = [embeds[p] for p in prompts]
    pre = torch.stack([r[:N_PREFIX].float() for r in rows])
    log(f"prefix rows across prompts: max |Δ| {float((pre - pre[0]).abs().max()):.2e}")

    ref = torch.load(resolve_under_home(REF_RAW), weights_only=False)["prompts"]
    assert [r["prompt"] for r in ref] == templates
    latents = {("base", i): r["latent"] for i, r in enumerate(ref)}
    donor = {i: (i + 3) % len(prompts) for i in range(len(prompts))}

    lat_dir = out / "latents"
    lat_dir.mkdir(exist_ok=True)
    tr = None
    sigmas = snapped_sigmas(args.steps)
    for arm in ARMS:
        for i, row in enumerate(rows):
            path = lat_dir / f"{arm}_p{i}.safetensors"
            if path.exists():
                latents[(arm, i)] = load_file(path)["latent"]
                continue
            if tr is None:
                tr = loader.load_transformer("", "base", device=device)
            S = row.shape[0]
            cond_row = row
            cut = torch.zeros(n_img + S, dtype=torch.bool)
            if arm == "swap_suffix":
                cond_row = torch.cat([row[:-N_SUFFIX], rows[donor[i]][-N_SUFFIX:]])
            elif arm == "no_caption_keys":
                cut[n_img + N_PREFIX : n_img + S - N_SUFFIX] = True
            else:
                cut[n_img : n_img + N_PREFIX] = True
                cut[n_img + S - N_SUFFIX :] = True
            cond, cm = loader.pad_embeds([cond_row], device)
            noise = initial_noise(args.seed, h, w).to(device)
            kw = dict(cfg=args.cfg, sampler="euler")
            if cut.any():
                with KeyCut(n_img, cut):
                    z = denoise(tr, noise, cond, cm, uncond, uncond_mask, sigmas, **kw)
            else:
                z = denoise(tr, noise, cond, cm, uncond, uncond_mask, sigmas, **kw)
            latents[(arm, i)] = z.cpu()
            save_file({"latent": z.cpu().contiguous()}, str(path))
            log(f"{arm} p{i} done")
    del tr
    loader.empty_cache()

    vae = loader.load_vae("", device=device)
    images = {}
    for key, z in latents.items():
        png = out / f"{key[0]}_p{key[1]}.png"
        if not png.exists():
            x = decode(z.to(device), vae)[0].float().add(1).mul(127.5).round()
            x = x.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
            Image.fromarray(x).save(png)
        images[key] = Image.open(png).convert("RGB")
    del vae
    loader.empty_cache()

    from anime_tools.tagger.tagger import AnimaTagger

    tagger = AnimaTagger(resolve_under_home(args.tagger), device="cuda")
    pred = {k: tagger.predict(im) for k, im in images.items()}

    def recall(kept, scores, tags):
        tags = [t for t in tags if t in scores]
        return (sum(t in kept for t in tags) / len(tags)) if tags else None

    def px(a, b):
        return float(
            np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean()
        )

    results = []
    for i, t in enumerate(templates):
        own = tag_list(t)
        d = donor[i]
        donor_only = [x for x in tag_list(templates[d]) if x not in own]
        base_kept = set(pred[("base", i)]["kept"])
        rec = {"prompt": t, "donor": d, "arms": {}}
        for arm in ("base", *ARMS):
            p = pred[(arm, i)]
            kept = set(p["kept"])
            rec["arms"][arm] = {
                "own_recall": recall(kept, p["scores"], own),
                "donor_only_recall": recall(kept, p["scores"], donor_only),
                "jaccard_vs_base": len(kept & base_kept)
                / max(len(kept | base_kept), 1),
                "px_vs_base": px(images[(arm, i)], images[("base", i)]),
                "top": sorted(p["kept"], key=lambda k: -p["kept"][k])[:15],
            }
        rec["px_base_vs_donor_base"] = px(images[("base", i)], images[("base", d)])
        results.append(rec)
        log(
            f"p{i}: "
            + "  ".join(
                f"{a} own={v['own_recall'] if v['own_recall'] is None else round(v['own_recall'], 2)} "
                f"donor={v['donor_only_recall'] if v['donor_only_recall'] is None else round(v['donor_only_recall'], 2)} "
                f"J={v['jaccard_vs_base']:.2f} px={v['px_vs_base']:.1f}"
                for a, v in rec["arms"].items()
            )
            + f"  | px base vs donor base {rec['px_base_vs_donor_base']:.1f}"
        )

    cw = 384
    grid = Image.new("RGB", (cw * 4, cw * len(templates)), "white")
    for i in range(len(templates)):
        for j, arm in enumerate(("base", *ARMS)):
            grid.paste(
                images[(arm, i)].resize((cw, cw), Image.LANCZOS), (j * cw, i * cw)
            )
    grid.save(out / "grid.jpg", quality=90)
    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    log(f"wrote {out}/summary.json, grid.jpg (columns: base, {', '.join(ARMS)})")


if __name__ == "__main__":
    main()
