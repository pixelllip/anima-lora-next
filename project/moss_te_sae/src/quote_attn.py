"""Where the MOSS DiT reads a quoted string, and whether the string's own rows carry it.

Two tag prompts of the A/B set (Miku, Neru) get ``, speech bubble. She is saying
"<s>".`` for an English and a Korean string (layout JSON, INT8 TE, caststation trigger).
Base, Euler 30 steps, CFG 3, seed 1, 1024², as ``template_attn.py``. The string's
tokens (between the quotes) are their own key segment, ``quote``.

Arms, cond branch only:

- ``base``: every image query is scored against every key on every block and step;
  per query, the mass on image / prefix / body / suffix / quote (head mean) and the
  quote mass per head, summed per step third;
- ``swap_suffix``: the 7 suffix rows come from the same character's prompt with the
  other string;
- ``no_quote_keys``: image queries cannot attend to the string's rows;
- ``quote_only``: image queries cannot attend to the caption rows other than the
  string's (template and string kept).

AnimeText boxes on the base render split the image queries into the text region and
the rest. ``heat_p{i}.png``: quote mass per image token (blocks 8–41, all steps) over
the base render.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

import library.moss.model as mm
from bench.moss_ab._render import uncond_embeds
from bench.moss_ab.lora_ab import PROMPTS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.cache import caption_converter
from library.moss.generate import denoise, initial_noise
from library.moss.model import decode, snapped_sigmas
from probe import tokenize
from template_causal import KeyCut, log

SEGS = ("image", "prefix", "body", "suffix", "quote")
STRINGS = ("MY COMPUTER", "지승현의 컴퓨터")
ARMS = ("base", "swap_suffix", "no_quote_keys", "quote_only")
N_SUFFIX = 7
CHUNK = 1024


class QueryCapture:
    """Per-image-query segment mass on the cond branch, summed per step third."""

    def __init__(self, n_img, key_seg, heads, n_blocks):
        self.n_img, self.L = n_img, key_seg.shape[0]
        self.onehot = torch.nn.functional.one_hot(key_seg.long(), len(SEGS))
        self.onehot = self.onehot.float().cuda()
        self.seg = torch.zeros(3, n_blocks, n_img, len(SEGS))
        self.quote = torch.zeros(3, n_blocks, heads, n_img)
        self.count = torch.zeros(3, n_blocks)
        self.block = self.step = 0
        self._orig = mm._attention

    def __enter__(self):
        mm._attention = self._wrapped
        return self

    def __exit__(self, *exc):
        mm._attention = self._orig

    def _wrapped(self, q, k, v, rope, heads, key_mask=None, *args, **kw):
        if q.shape[1] == self.L and key_mask is None:
            self._score(q, k, rope, heads)
        return self._orig(q, k, v, rope, heads, key_mask, *args, **kw)

    @torch.no_grad()
    def _score(self, q, k, rope, heads):
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = mm.rotate(q.reshape(shape), rope)[0].float()
        k = mm.rotate(k.reshape(shape), rope)[0].float()
        t = min(self.step // 10, 2)
        for c in range(0, self.n_img, CHUNK):
            p = torch.einsum("qhd,lhd->hql", q[c : c + CHUNK], k)
            p = p.mul_(shape[-1] ** -0.5).softmax(-1)
            m = p @ self.onehot  # (H, Q, S)
            self.seg[t, self.block, c : c + CHUNK] += m.mean(0).cpu()
            self.quote[t, self.block, :, c : c + CHUNK] += m[..., 4].cpu()
            del p, m
        self.count[t, self.block] += 1


def segments(tok, text, string):
    """Key segment per joint position: 0 image, then 1 prefix, 2 body, 3 suffix,
    4 the string's tokens."""
    ids, seg = tokenize(tok, text, True)
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tok(formatted, return_offsets_mapping=True)
    a0 = formatted.find(f'"{string}') + 1  # layout JSON escapes it as \"…\"
    assert a0 > 0, (string, formatted)
    a1 = a0 + len(string)
    seg = seg.long() + 1
    for i, (a, b) in enumerate(enc.offset_mapping):
        if a < a1 and b > a0:
            assert seg[i] == 2, "the string shares a token with the template"
            seg[i] = 4
    return ids, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--te_precision", default="int8")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/quote_attn/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, args.size.split("x"))
    gh, gw = h // 16, w // 16
    n_img = gh * gw
    device = torch.device("cuda")

    templates, strings = [], []
    for base in PROMPTS[:2]:
        for s in STRINGS:
            templates.append(f'{base}, speech bubble. She is saying "{s}".')
            strings.append(s)
    donor = {i: i ^ 1 for i in range(len(templates))}
    conv = caption_converter("layout_json")
    prompts = [conv(p.format(artist=args.artist)) for p in templates]
    embeds, uncond, uncond_mask = uncond_embeds(device, prompts, args.te_precision)
    rows = [embeds[p] for p in prompts]
    tok = loader.load_tokenizer(default_moss_dir())
    segs = []
    for i, (text, s) in enumerate(zip(prompts, strings)):
        ids, seg = segments(tok, text, s)
        assert rows[i].shape[0] == seg.shape[0], (rows[i].shape, seg.shape)
        segs.append(seg)
        q = [tok.decode([int(t)]) for t, g in zip(ids[0].tolist(), seg) if g == 4]
        log(f"p{i}: S={seg.shape[0]} quote tokens {q}")

    lat_dir = out / "latents"
    lat_dir.mkdir(exist_ok=True)
    latents, caps = {}, {}
    tr = None
    sigmas = snapped_sigmas(args.steps)
    for arm in ARMS:
        for i, row in enumerate(rows):
            path = lat_dir / f"{arm}_p{i}.safetensors"
            if path.exists() and arm != "base":
                latents[(arm, i)] = load_file(path)["latent"]
                continue
            if tr is None:
                tr = loader.load_transformer("", "base", device=device)
                block_ref = []
                for b, layer in enumerate(tr.layers):
                    layer.register_forward_pre_hook(
                        lambda m, a, b=b: (
                            setattr(block_ref[0], "block", b) if block_ref else None
                        )
                    )
            S = row.shape[0]
            seg = segs[i]
            cond_row = row
            cut = torch.zeros(n_img + S, dtype=torch.bool)
            if arm == "swap_suffix":
                cond_row = torch.cat([row[:-N_SUFFIX], rows[donor[i]][-N_SUFFIX:]])
            elif arm == "no_quote_keys":
                cut[n_img:] = seg == 4
            elif arm == "quote_only":
                cut[n_img:] = seg == 2
            cond, cm = loader.pad_embeds([cond_row], device)
            noise = initial_noise(args.seed, h, w).to(device)
            kw = dict(cfg=args.cfg, sampler="euler")
            if arm == "base":
                key_seg = torch.cat([torch.zeros(n_img, dtype=torch.long), seg])
                cap = QueryCapture(n_img, key_seg, tr.layers[0].heads, len(tr.layers))
                block_ref[:] = [cap]

                def on_step(j, _x0, cap=cap):
                    cap.step = j + 1

                with cap:
                    z = denoise(
                        tr,
                        noise,
                        cond,
                        cm,
                        uncond,
                        uncond_mask,
                        sigmas,
                        on_step=on_step,
                        **kw,
                    )
                block_ref.clear()
                caps[i] = cap
            elif cut.any():
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
        x = decode(z.to(device), vae)[0].float().add(1).mul(127.5).round()
        x = x.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        Image.fromarray(x).save(png)
        images[key] = Image.open(png).convert("RGB")
    del vae
    loader.empty_cache()

    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device="cuda")
    boxes = {
        k: [
            [int(v) for v in b]
            for b in det.detect(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        ]
        for k, im in images.items()
    }
    del det
    loader.empty_cache()

    cy = (torch.arange(gh) * 16 + 8)[:, None].expand(gh, gw).flatten()
    cx = (torch.arange(gw) * 16 + 8)[None, :].expand(gh, gw).flatten()

    def px(a, b, sel=None):
        d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean(-1)
        return float(d[sel].mean()) if sel is not None else float(d.mean())

    results = []
    raw = {}
    for i, t in enumerate(templates):
        cap = caps[i]
        n = cap.count[:, :, None, None]
        seg_m = cap.seg / n  # (3, B, Q, S)
        quote_m = cap.quote / n  # (3, B, H, Q)
        region = torch.zeros(n_img, dtype=torch.bool)
        for x0, y0, x1, y1 in boxes[("base", i)]:
            region |= (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        pix = np.zeros((h, w), dtype=bool)
        for x0, y0, x1, y1 in boxes[("base", i)]:
            pix[y0:y1, x0:x1] = True
        rec = {
            "prompt": t,
            "string": strings[i],
            "n_quote_tokens": int((segs[i] == 4).sum()),
            "n_body_tokens": int((segs[i] == 2).sum()),
            "boxes": {f"{a}": boxes[(a, i)] for a in ARMS},
            "n_region_tokens": int(region.sum()),
        }
        by_block = {}
        for name, sel in (("region", region), ("rest", ~region)):
            if not sel.any():
                continue
            seg_sel = seg_m[:, :, sel].mean(2)  # (3, B, S)
            q_head = quote_m[:, :, :, sel].mean(3)  # (3, B, H)
            by_block[name] = {
                "mass_by_block": seg_sel.mean(0).tolist(),  # (B, S)
                "mass_thirds": seg_sel.mean(1).tolist(),  # (3, S)
                "quote_maxhead_by_block": q_head.mean(0).max(-1).values.tolist(),
                "quote_argmaxhead_by_block": q_head.mean(0).argmax(-1).tolist(),
            }
        rec["by_block"] = by_block
        rec["px_vs_base"] = {
            a: {
                "all": px(images[(a, i)], images[("base", i)]),
                "text_region": px(images[(a, i)], images[("base", i)], pix)
                if pix.any()
                else None,
                "rest": px(images[(a, i)], images[("base", i)], ~pix),
            }
            for a in ARMS[1:]
        }
        results.append(rec)
        raw[i] = {"seg": seg_m.half(), "quote": quote_m.half(), "region": region}

        heat = seg_m[:, 8:, :, 4].mean((0, 1)).reshape(gh, gw).numpy()
        heat = heat / max(heat.max(), 1e-8)
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_NEAREST)
        col = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
        base_np = np.asarray(images[("base", i)], np.float32)
        over = (0.45 * base_np + 0.55 * col).astype(np.uint8)
        for x0, y0, x1, y1 in boxes[("base", i)]:
            cv2.rectangle(over, (x0, y0), (x1, y1), (0, 255, 0), 3)
        Image.fromarray(over).save(out / f"heat_p{i}.png")
        images[("heat", i)] = Image.fromarray(over)

        r = by_block.get("region")
        o = by_block["rest"]
        msg = f"p{i} {strings[i]!r}: boxes {len(boxes[('base', i)])} region {int(region.sum())} tok"
        if r:
            rq = np.array(r["mass_by_block"])[:, 4]
            oq = np.array(o["mass_by_block"])[:, 4]
            rb = np.array(r["mass_by_block"])[:, 2]
            msg += (
                f" | quote mass region {rq.mean():.4f} (max blk {rq.argmax()} {rq.max():.3f})"
                f" rest {oq.mean():.4f} | body-other region {rb.mean():.4f}"
            )
        log(msg)

    cw = 384
    cols = (*ARMS, "heat")
    grid = Image.new("RGB", (cw * len(cols), cw * len(templates)), "white")
    for i in range(len(templates)):
        for j, a in enumerate(cols):
            grid.paste(images[(a, i)].resize((cw, cw), Image.LANCZOS), (j * cw, i * cw))
    grid.save(out / "grid.jpg", quality=90)
    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    torch.save({"templates": templates, "raw": raw}, out / "raw.pt")
    log(f"wrote {out}/summary.json, grid.jpg (columns: {', '.join(cols)})")


if __name__ == "__main__":
    main()
