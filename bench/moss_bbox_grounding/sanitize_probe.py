#!/usr/bin/env python3
"""Where the sanitize LoRA finds and removes the text, against Base's bbox path.

Teacher-forced, with the full-size isolated reference the LoRA trained on (its
stamped ``ref_timestep`` / ``ref_isolate``) and the fixed caption. Two page sets:

  heldout  the eval pages (``output/moss_i2i_eval/sanitize/inputs``) rendered by the
           LoRA (``scripts/moss/generate.py`` latents): the target is the LoRA's own
           output, so the edit region is what the LoRA chose to remove
  train    cached training pairs (``i2i_sanitize_idt``, clean→clean pairs left out):
           the staged clean twin as target

Regions: ``E`` = ``library.moss.train.edit_region`` of target vs reference (RMS
threshold 0.6, the recipe's min tokens / dilate), ``K`` = the rest. Per page and σ:

  ablation  LoRA off in a block range; flow-matching MSE on E and K. ``recovery`` =
            share of the LoRA's gain over Base lost there (1 = the range carries it)
  attention (LoRA and Base forwards, explicit softmax for the image queries), per
            block and head, E / K means of the mass on: the reference token at the
            query's own cell (``copy``), all reference tokens (``ref``), the caption
            tokens (``caption``, the template's ends left out)
  mark      per block, ‖h_LoRA − h_Base‖ (massive channels out) on image rows and on
            reference rows: E / K means and the within-page AUC of E over K

    make daemon-run ARGS="--stall-timeout 900 bench/moss_bbox_grounding/sanitize_probe.py"
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from bench._common import make_run_dir, start_heartbeat, write_result

RANGES = {
    "0-4": range(0, 5),
    "5-8": range(5, 9),
    "9-13": range(9, 14),
    "14-20": range(14, 21),
    "21-25": range(21, 26),
    "26-35": range(26, 36),
    "36-41": range(36, 42),
    "0-13": range(0, 14),
    "14-35": range(14, 36),
    "14-41": range(14, 42),
    "21-41": range(21, 42),
}
STATS = ("copy", "ref", "caption")


def heldout_pages(gen_dir, model_dir, device):
    from library.moss.generate import load_latent, read_latent_record, ref_latents

    recs = []
    for p in sorted(glob.glob(f"{gen_dir}/latents/*.safetensors")):
        rec = read_latent_record(Path(p))
        recs.append((rec, load_latent(Path(p))))
    refs = ref_latents(model_dir, [r for r, _ in recs], device)
    return [
        dict(
            stem=r["key"],
            latent=lat.float().reshape(1, *lat.shape[-3:]),
            ref=refs[r["key"]],
            set="heldout",
        )  # fmt: skip
        for r, lat in recs
    ]


def train_pages(cache_dir, n, seed):
    rng = np.random.default_rng(seed)
    stems = sorted(
        Path(p).name[: -len(".moss_ref.safetensors")]
        for p in glob.glob(f"{cache_dir}/*.moss_ref.safetensors")
    )
    stems = [s for s in stems if not s.startswith("idt_")]
    out = []
    for i in rng.permutation(len(stems)):
        s = stems[i]
        (lat,) = glob.glob(f"{cache_dir}/{s}_*x*.moss_latent.safetensors")
        with safe_open(lat, "pt") as f:
            latent = f.get_tensor("latent")[None].float()
        with safe_open(f"{cache_dir}/{s}.moss_ref.safetensors", "pt") as f:
            ref = f.get_tensor("latent")[None].float()
        if ref.shape != latent.shape:
            continue
        out.append(dict(stem=s, latent=latent, ref=ref, set="train"))
        if len(out) >= n:
            break
    return out


def auc(score, label):
    """Mann–Whitney AUC of ``score`` for ``label`` True over False."""
    order = score.argsort()
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(
        1, len(score) + 1, device=score.device, dtype=score.dtype
    )
    n1 = label.sum()
    n0 = len(label) - n1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class Probe:
    def __init__(self, num_blocks, heads, chunk, keep):
        self.L, self.H, self.chunk, self.keep = num_blocks, heads, chunk, keep
        self.block = None
        self.mode = None  # None | "store" | "compare"

    def start(self, mode, n_img, n_text, cap, regw, label, base_h=None):
        self.mode, self.n_img, self.n_text = mode, n_img, n_text
        self.N = 2 * n_img + n_text
        self.cap, self.regw, self.label = cap, regw, label
        self.att = np.full((len(STATS), self.L, self.H, 2), np.nan, np.float32)
        self.base_h = [None] * self.L if mode == "store" else base_h
        self.mark = np.full((2, self.L, 3), np.nan, np.float32)  # img/ref × (E, K, auc)

    def attention(self, q, k, v, rope, heads):
        from library.moss.model import rotate

        if self.mode is None or self.block is None or q.shape[1] != self.N:
            return
        b, n, t = self.block, self.n_img, self.n_text
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = rotate(q.reshape(shape), rope).transpose(1, 2)[0]
        k = rotate(k.reshape(shape), rope).transpose(1, 2)[0]
        scale = 1.0 / math.sqrt(shape[-1])
        diag = torch.arange(n, device=q.device)
        for h0 in range(0, heads, self.chunk):
            hs = slice(h0, min(h0 + self.chunk, heads))
            p = torch.softmax(
                (q[hs, :n].float() @ k[hs].float().transpose(-1, -2)) * scale, -1
            )
            copy = p[:, diag, n + t + diag]  # (h, Q)
            ref = p[..., n + t :].sum(-1)
            cap = p[..., n + self.cap[0] : n + self.cap[1]].sum(-1)
            for i, s in enumerate((copy, ref, cap)):
                self.att[i, b, hs] = (s @ self.regw).cpu().numpy()
            del p

    def __call__(self, index, x, n_img):
        if self.mode is None:
            return x
        n, t = self.n_img, self.n_text
        h = x[0].float() * self.keep
        h = torch.cat([h[:n], h[n + t :]])  # image rows, reference rows
        if self.mode == "store":
            self.base_h[index] = h.to(torch.float16).cpu()
            return x
        d = (h - self.base_h[index].to(h.device).float()).norm(dim=-1)
        for r, part in enumerate((d[:n], d[n:])):
            self.mark[r, index, :2] = (part @ self.regw).cpu().numpy()
            self.mark[r, index, 2] = auc(part, self.label)
        return x


def boot(values, n=2000, seed=0):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return [float("nan")] * 2
    means = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [
        round(float(np.percentile(means, 2.5)), 3),
        round(float(np.percentile(means, 97.5)), 3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--lora",
        default="output/ckpt_moss_i2i/moss_i2i_sanitize_idt_w0003_late_lr5e5_e8.safetensors",
    )
    ap.add_argument("--gen_dir", default="output/tests/moss_sanitize_probe")
    ap.add_argument("--cache_dir", default="post_image_dataset/moss/i2i_sanitize_idt")
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--caption", default="no sfx, no dialogue, no watermark")
    ap.add_argument("--sigmas", default="0.9,0.6,0.3")
    ap.add_argument("--seed", type=int, default=0)
    # 0.6, not the training recipe's 0.3: a rendered target drifts 0.1–0.3 everywhere
    ap.add_argument("--region_threshold", type=float, default=0.6)
    ap.add_argument("--region_min_tokens", type=int, default=5)
    ap.add_argument("--region_dilate", type=int, default=1)
    ap.add_argument("--min_area", type=float, default=0.003)
    ap.add_argument("--model", default="base")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument(
        "--te_cache", default="/media/sorryhyun/data/moss_image/bench/_te_cache"
    )
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--head_chunk", type=int, default=3)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    start_heartbeat(label="moss_sanitize_probe")

    import library.moss.model as moss_model
    from bench.moss_sinks.probe_attention import template_split
    from library.env import resolve_under_home
    from library.moss.generate import cached_embeds
    from library.moss.loader import load_tokenizer, load_transformer
    from library.moss.lora import load_for_inference
    from library.moss.requests import resolve_model_dir
    from library.moss.train import edit_region

    sigmas = [float(s) for s in args.sigmas.split(",") if s]
    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device("cuda")
    pages = heldout_pages(resolve_under_home(args.gen_dir), model_dir, device)
    pages += train_pages(resolve_under_home(args.cache_dir), args.train_n, args.seed)
    kept = []
    for p in pages:
        region = edit_region(
            p["latent"][0], p["ref"][0], args.region_threshold,
            args.region_min_tokens, args.region_dilate,
        )  # fmt: skip
        area = region.float().mean().item()
        print(f"[page] {p['set']} {p['stem']} edit {100 * area:.2f} %", flush=True)
        if area >= args.min_area and not region.all():
            kept.append(dict(p, region=region.flatten(), area=area))
    pages = kept

    tok = load_tokenizer(model_dir)
    pre, cap_len, _ = template_split(tok, args.caption)
    del tok
    embeds = cached_embeds(
        model_dir, [args.caption], Path(args.te_cache), precision=args.text_encoder
    )[args.caption][None].to(device, torch.bfloat16)
    n_text = embeds.shape[1]

    transformer = load_transformer(model_dir, args.model, device=device)
    network = load_for_inference(transformer, resolve_under_home(args.lora))
    print(
        f"[lora] ref_timestep {transformer.ref_timestep} ref_isolate "
        f"{transformer.ref_isolate}; {len(pages)} pages",
        flush=True,
    )
    by_block = {}
    for name, adapter in network.adapters.items():
        by_block.setdefault(int(name.split("_")[1]), []).append(adapter)

    def lora_off(blocks):
        for b, ads in by_block.items():
            for a in ads:
                a.multiplier = 0.0 if b in blocks else 1.0

    num_blocks, heads = len(transformer.layers), transformer.config.num_heads
    massive = np.load(resolve_under_home(args.massive))["channels"]
    keep = torch.ones(transformer.config.dim, device=device)
    keep[torch.from_numpy(massive).to(device)] = 0
    probe = Probe(num_blocks, heads, args.head_chunk, keep)
    original = moss_model._attention

    def wrapped(q, k, v, rope, heads_, *rest, **kw):
        probe.attention(q, k, v, rope, heads_)
        return original(q, k, v, rope, heads_, *rest, **kw)

    def set_block(index):
        def hook(_m, _a):
            probe.block = index

        return hook

    handles = [transformer.text_embedder.register_forward_pre_hook(set_block(None))]
    for i, layer in enumerate(transformer.layers):
        handles.append(layer.register_forward_pre_hook(set_block(i)))
    moss_model._attention = wrapped
    transformer.layer_hook = probe

    P, Z = len(pages), len(sigmas)
    conds = ["lora", "base", *RANGES]
    loss = np.full((P, Z, len(conds), 2), np.nan, np.float32)
    att = np.full((P, Z, 2, len(STATS), num_blocks, heads, 2), np.nan, np.float32)
    mark = np.full((P, Z, 2, num_blocks, 3), np.nan, np.float32)
    try:
        with torch.inference_mode():
            for pi, page in enumerate(pages):
                latent = page["latent"].to(device)
                ref = page["ref"].to(device)
                label = page["region"].to(device)
                regw = torch.stack([label, ~label], 1).float()
                regw = regw / regw.sum(0, keepdim=True)
                n_img = latent.shape[-2] * latent.shape[-1]
                gen = torch.Generator(device="cpu").manual_seed(args.seed + pi)
                for zi, sigma in enumerate(sigmas):
                    noise = torch.randn(latent.shape, generator=gen).to(device)
                    noisy = (1.0 - sigma) * latent + sigma * noise
                    target = noise - latent
                    timestep = torch.tensor([1000.0 * sigma], device=device)

                    def run():
                        pred = transformer(
                            noisy, timestep, embeds, return_dict=False, ref=ref
                        )[0]
                        err = (pred.float() - target).pow(2).mean(1)[0].flatten()
                        return (err @ regw).cpu().numpy()

                    cap = (pre, pre + cap_len)
                    network.set_multiplier(0.0)
                    probe.start("store", n_img, n_text, cap, regw, label)
                    loss[pi, zi, 1] = run()
                    att[pi, zi, 1] = probe.att
                    base_h = probe.base_h
                    network.set_multiplier(1.0)
                    probe.start("compare", n_img, n_text, cap, regw, label, base_h)
                    loss[pi, zi, 0] = run()
                    att[pi, zi, 0] = probe.att
                    mark[pi, zi] = probe.mark
                    probe.mode = None
                    del base_h, probe.base_h
                    for ci, name in enumerate(RANGES, start=2):
                        lora_off(set(RANGES[name]))
                        loss[pi, zi, ci] = run()
                    network.set_multiplier(1.0)
                print(
                    f"[probe] {pi + 1}/{P} {page['set']} {page['stem']} "
                    f"E lora {loss[pi, :, 0, 0].round(3)} base {loss[pi, :, 1, 0].round(3)}",
                    flush=True,
                )
    finally:
        moss_model._attention = original
        transformer.layer_hook = None
        for h in handles:
            h.remove()

    # --- summary -------------------------------------------------------------------
    sets = {s: np.array([p["set"] == s for p in pages]) for s in ("heldout", "train")}
    sets["all"] = np.ones(P, bool)
    summary = {}
    for sname, sel in sets.items():
        if not sel.any():
            continue
        out = {"n": int(sel.sum())}
        for zi, sigma in enumerate(sigmas):
            L = loss[sel, zi]  # (p, cond, 2)
            gain = L[:, 1] - L[:, 0]  # base − lora, per region
            row = {
                "E": {"lora": round(float(L[:, 0, 0].mean()), 4),
                      "base": round(float(L[:, 1, 0].mean()), 4)},
                "K": {"lora": round(float(L[:, 0, 1].mean()), 4),
                      "base": round(float(L[:, 1, 1].mean()), 4)},
            }  # fmt: skip
            for ci, name in enumerate(RANGES, start=2):
                rec = {}
                for ri, reg in enumerate("EK"):
                    g = gain[:, ri].mean()
                    per = (L[:, ci, ri] - L[:, 0, ri]) / g
                    rec[reg] = round(float(per.mean()), 3)
                    rec[f"{reg}_ci"] = boot(per)
                row[name] = rec
            out[f"sigma_{sigma}"] = row
        summary[sname] = out

    def ratio(a):  # log E/K, mean over pages and σ → (L, H)
        return np.nanmean(np.log(a[..., 0] + 1e-8) - np.log(a[..., 1] + 1e-8), (0, 1))

    att_summary = {}
    for si, stat in enumerate(STATS):
        for ci, cname in enumerate(("lora", "base")):
            r = ratio(att[:, :, ci, si])
            att_summary[f"{stat}_{cname}_EK_log_block"] = [
                round(float(x), 3) for x in r.mean(-1)
            ]
            if cname == "lora":
                d = r - ratio(att[:, :, 1, si])
                order = np.argsort(np.abs(d), axis=None)[::-1][:12]
                att_summary[f"{stat}_lora_minus_base_top"] = [
                    dict(
                        block=int(f // heads),
                        head=int(f % heads),
                        d=round(float(d.flat[f]), 3),
                        lora=round(float(r.flat[f]), 3),
                    )  # fmt: skip
                    for f in order
                ]
    mark_summary = {}
    for ri, rows in enumerate(("image", "ref")):
        m = np.nanmean(mark[:, :, ri], (0, 1))  # (L, 3)
        mark_summary[rows] = dict(
            E_over_K=[round(float(e / max(k, 1e-8)), 3) for e, k in m[:, :2]],
            auc=[round(float(a), 3) for a in m[:, 2]],
        )
        for zi, sigma in enumerate(sigmas):
            mark_summary[f"{rows}_auc_sigma_{sigma}"] = [
                round(float(a), 3) for a in np.nanmean(mark[:, zi, ri, :, 2], 0)
            ]

    run_dir = make_run_dir("moss_bbox_grounding", args.label or "sanitize")
    np.savez_compressed(
        run_dir / "per_page.npz", loss=loss, att=att, mark=mark,
        stems=np.array([p["stem"] for p in pages]),
        sets=np.array([p["set"] for p in pages]),
        areas=np.array([p["area"] for p in pages]), sigmas=np.array(sigmas),
        conds=np.array(conds),
    )  # fmt: skip
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=dict(loss=summary, attention=att_summary, mark=mark_summary),
        label=args.label,
        artifacts=["per_page.npz"],
        device=device,
    )
    print(f"[probe] wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
