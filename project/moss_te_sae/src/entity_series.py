"""Is a character's series laid out in MOSS's name vectors, and does it transfer?

Base DiT, int8 TE, one job:

- ``geometry``: name vectors (mean over the name's tokens, mean over 3 contexts, minus
  the invented names' mean; per-dimension z-scored) in the TE output, the
  ``text_embedder`` output and blocks 4…41, at one ``x_t`` (σ 0.9, zero latent). Captions
  carry no ``series`` field. Per space: leave-one-out nearest-series-centroid accuracy
  (every series is a class, singletons included) against a label-permutation null, and
  per character ``align`` = cos(name − mean, own series' LOO centroid − mean).
- ``render``: every grouped character, name only (no series field), one seed. The Anima
  Tagger's probability of her series tag, her hair-colour tag and (where in the vocab)
  her character tag. Spearman of ``align`` against each, per space.
- ``inject``: an invented name with a series centroid (raw ``text_embedder`` output,
  members' mean minus the invented names' mean) added to every name row, scaled to
  ρ × the name-row norm. Against the invented name alone, the invented name with the
  series written in the caption, and a random direction. Read by the tagger's series
  probability and the series' shared outfit tags, and on the one-step ``x̂₀`` as
  ``r`` / cos against ``E = x̂₀(series written) − x̂₀(invented name)``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from entity_dit import CONTEXTS, FAKES, caption, log, span
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.generate import denoise, initial_noise
from library.moss.model import decode, snapped_sigmas
from library.moss.spectrum import MossSpectrum
from library.moss.train import NUM_TRAIN_TIMESTEPS

# name -> (series, hair colour). Series are Danbooru copyright tags; Teto is UTAU on
# Danbooru but goes with the Vocaloid family here. Singletons are distractor classes.
CHARS = {
    "hatsune miku": ("vocaloid", "aqua"),
    "akita neru": ("vocaloid", "blonde"),
    "kagamine rin": ("vocaloid", "blonde"),
    "kagamine len": ("vocaloid", "blonde"),
    "megurine luka": ("vocaloid", "pink"),
    "kasane teto": ("vocaloid", "red"),
    "meiko (vocaloid)": ("vocaloid", "brown"),
    "kaito (vocaloid)": ("vocaloid", "blue"),
    "gumi": ("vocaloid", "green"),
    "yowane haku": ("vocaloid", "grey"),
    "hakurei reimu": ("touhou", "brown"),
    "kirisame marisa": ("touhou", "blonde"),
    "flandre scarlet": ("touhou", "blonde"),
    "hong meiling": ("touhou", "red"),
    "komeiji satori": ("touhou", "pink"),
    "cirno": ("touhou", "blue"),
    "inubashiri momiji": ("touhou", "white"),
    "patchouli knowledge": ("touhou", "purple"),
    "komeiji koishi": ("touhou", "green"),
    "kochiya sanae": ("touhou", "green"),
    "kijin seija": ("touhou", "black"),
    "yorigami shion": ("touhou", "blue"),
    "toyosatomimi no miko": ("touhou", "brown"),
    "houshou marine": ("hololive", "red"),
    "inugami korone": ("hololive", "brown"),
    "shirakami fubuki": ("hololive", "white"),
    "nekomata okayu": ("hololive", "purple"),
    "uruha rushia": ("hololive", "green"),
    "nanashi mumei": ("hololive", "brown"),
    "ouro kronii": ("hololive", "blue"),
    "ijichi nijika": ("bocchi the rock!", "blonde"),
    "kita ikuyo": ("bocchi the rock!", "red"),
    "gotou hitori": ("bocchi the rock!", "pink"),
    "yamada ryo": ("bocchi the rock!", "blue"),
    "kaname madoka": ("mahou shoujo madoka magica", "pink"),
    "akemi homura": ("mahou shoujo madoka magica", "black"),
    "miki sayaka": ("mahou shoujo madoka magica", "blue"),
    "tomoe mami": ("mahou shoujo madoka magica", "blonde"),
    "ram (re:zero)": ("re:zero kara hajimeru isekai seikatsu", "pink"),
    "rem (re:zero)": ("re:zero kara hajimeru isekai seikatsu", "blue"),
    "emilia (re:zero)": ("re:zero kara hajimeru isekai seikatsu", "white"),
    "frieren": ("sousou no frieren", "white"),
    "fern (sousou no frieren)": ("sousou no frieren", "purple"),
    "rias gremory": ("high school dxd", "red"),
    "himejima akeno": ("high school dxd", "black"),
    "toujou koneko": ("high school dxd", "white"),
    "erza scarlet": ("fairy tail", "red"),
    "ayanami rei": ("neon genesis evangelion", "blue"),
    "kafuu chino": ("gochuumon wa usagi desu ka?", "blue"),
    "tohsaka rin": ("fate (series)", "black"),
    "yor briar": ("spy x family", "black"),
    "komi shouko": ("komi-san wa komyushou desu", "black"),
    "misaka mikoto": ("toaru kagaku no railgun", "brown"),
    "hirasawa yui": ("k-on!", "brown"),
    "hoshino ai (oshi no ko)": ("oshi no ko", "purple"),
    "c.c.": ("code geass", "green"),
}
SPACES = ("te", "emb", 4, 8, 14, 20, 24, 30, 36, 41)
INJECT = {
    "vocaloid": ("headset", "detached sleeves", "necktie", "sleeveless shirt"),
    "touhou": ("bow", "frills", "hair bow", "ribbon"),
    "hololive": ("virtual youtuber", "animal ears", "hair ornament", "streaked hair"),
    "bocchi the rock!": ("school uniform", "guitar", "jacket", "hair ornament"),
}
RHOS = (1.0, 2.0)


def grouped():
    counts = {}
    for s, _ in CHARS.values():
        counts[s] = counts.get(s, 0) + 1
    return [n for n, (s, _) in CHARS.items() if counts[s] > 1]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def spearman(x, y):
    def rank(v):
        r = torch.empty(len(v))
        r[torch.tensor(v).argsort()] = torch.arange(len(v), dtype=torch.float)
        return r

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def geometry(vecs, fake, labels, targets, perms, seed=0):
    """``vecs``: name -> (D,), ``fake``: (D,). LOO accuracy over ``targets``, the
    permutation null, per-target ``align`` and margin."""
    names = list(vecs)
    stack = torch.stack([vecs[n] for n in names] + [fake])
    mu, sd = stack.mean(0), stack.std(0).clamp_min(1e-6)
    d = {n: (vecs[n] - mu) / sd - (fake - mu) / sd for n in names}
    m = torch.stack(list(d.values())).mean(0)
    d = {n: v - m for n, v in d.items()}

    mat = torch.stack([d[n] for n in names])
    unit = mat / mat.norm(dim=-1, keepdim=True)
    tgt = torch.tensor([names.index(n) for n in targets])

    def loo(lab):
        classes = sorted(set(lab.values()))
        k = torch.tensor([classes.index(lab[n]) for n in names])
        onehot = torch.nn.functional.one_hot(k, len(classes)).float()
        sums, counts = onehot.T @ mat, onehot.sum(0)
        hits, per = 0, {}
        for n, i in zip(targets, tgt.tolist()):
            own = k[i]
            cents = sums.clone()
            cents[own] -= mat[i]
            cnt = counts.clone()
            cnt[own] -= 1
            alive = cnt > 0
            scores = (cents / cnt.clamp_min(1)[:, None]) @ unit[i]
            scores = scores / (cents / cnt.clamp_min(1)[:, None]).norm(
                dim=-1
            ).clamp_min(1e-9)
            scores[~alive] = -2.0
            if not alive[own]:
                continue
            rival = torch.cat((scores[:own], scores[own + 1 :])).max()
            hits += int(scores.argmax() == own)
            per[n] = {"align": float(scores[own]), "margin": float(scores[own] - rival)}
        return hits / len(targets), per

    acc, per = loo(labels)
    gen = torch.Generator().manual_seed(seed)
    vals = list(labels.values())
    null = []
    for _ in range(perms):
        idx = torch.randperm(len(names), generator=gen)
        null.append(loo({n: vals[i] for n, i in zip(names, idx.tolist())})[0])
    null = torch.tensor(null)
    return {
        "loo_acc": acc,
        "null_mean": float(null.mean()),
        "null_p95": float(null.quantile(0.95)),
        "p": float((null >= acc).float().mean()),
        "per": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument("--probe_sigma", type=float, default=0.9)
    ap.add_argument("--name_seed", type=int, default=11)
    ap.add_argument("--inject_seeds", default="11,22")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--tagger", default="models/captioners/anima-tagger-dbv4")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/media/sorryhyun/new/qwen35_sae/entity_series"),
    )
    args = ap.parse_args()
    out_dir = args.out / time.strftime("%Y%m%d-%H%M")
    (out_dir / "latents").mkdir(parents=True, exist_ok=True)
    inject_seeds = [int(s) for s in args.inject_seeds.split(",")]
    model_dir = default_moss_dir()
    labels = {n: s for n, (s, _) in CHARS.items()}
    targets = grouped()

    tok = loader.load_tokenizer(model_dir)
    probe = [
        dict(
            who=who,
            text=caption(who, ctx),
            span=span(tok, caption(who, ctx), who, '"characters"'),
        )
        for who in (*CHARS, *FAKES)
        for ctx in CONTEXTS
    ]
    fake_text = caption(FAKES[0], CONTEXTS[0])
    fake_span = span(tok, fake_text, FAKES[0], '"characters"')
    series_text = {s: caption(FAKES[0], CONTEXTS[0], s) for s in INJECT}
    name_text = {n: caption(n, CONTEXTS[0]) for n in targets}
    texts = [p["text"] for p in probe] + [fake_text, *series_text.values(), ""]
    texts += list(name_text.values())
    log(f"{len(probe)} probe captions, {len(targets)} grouped characters")
    enc = loader.encode_prompts(model_dir, texts, log=log, precision=args.text_encoder)

    device = torch.device("cuda")
    transformer = loader.load_transformer(model_dir, "base", device=device)
    transformer.requires_grad_(False)
    state = {"grab": None, "emb_out": None, "keep": False, "add": None}
    grab = {}

    def emb_hook(_m, _i, out):
        if state["grab"] is not None:
            a, b = state["grab"]
            grab["emb"] = out[0, a:b].float().mean(0).cpu()
        if state["keep"]:
            state["emb_out"] = out[0].float().cpu()
        if state["add"] is not None and out.shape[1] == state["add"][0]:
            _, rows, vec = state["add"]
            out = out.clone()
            out[:, rows] += vec.to(out.device, out.dtype)
        return out

    def layer_hook(index, x, n_img):
        if state["grab"] is not None and index in SPACES:
            a, b = state["grab"]
            grab[index] = x[0, n_img + a : n_img + b].float().mean(0).cpu()
        return x

    transformer.text_embedder.register_forward_hook(emb_hook)

    @torch.no_grad()
    def x_hat(noisy, sigma, emb):
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        v = transformer(
            noisy, t, emb[None].to(device, torch.bfloat16), return_dict=False
        )[0]
        return noisy - sigma * v.float()

    h, w = args.height // 16, args.width // 16
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn((1, 128, h, w), generator=gen).to(device)

    # ---- geometry
    transformer.layer_hook = layer_hook
    raw = {}
    for i, p in enumerate(probe):
        e = enc[p["text"]]
        a, b = p["span"]
        grab.clear()
        grab["te"] = e[a:b].float().mean(0)
        state["grab"] = p["span"]
        x_hat(args.probe_sigma * noise, args.probe_sigma, e)
        state["grab"] = None
        for s in SPACES:
            raw.setdefault(s, {}).setdefault(p["who"], []).append(grab[s])
        if i % 30 == 0:
            log(f"probe {i}/{len(probe)}")
    transformer.layer_hook = None
    vecs = {s: {n: torch.stack(v).mean(0) for n, v in raw[s].items()} for s in SPACES}
    save_file(
        {f"{s}/{n}": v.contiguous() for s in SPACES for n, v in vecs[s].items()},
        str(out_dir / "vectors.safetensors"),
    )
    report = {"geometry": {}, "render": {}, "inject": {}, "chars": CHARS}
    for s in SPACES:
        names = {n: vecs[s][n] for n in CHARS}
        fake = torch.stack([vecs[s][f] for f in FAKES]).mean(0)
        g = geometry(names, fake, labels, targets, args.perms)
        report["geometry"][str(s)] = g
        log(
            f"[geom] {str(s):>4s}: loo acc {g['loo_acc']:.2f} (null {g['null_mean']:.2f}, "
            f"p95 {g['null_p95']:.2f}, p {g['p']:.3f})"
        )

    # ---- series directions in the text_embedder output
    state["keep"] = True
    x_hat(0.9 * noise, 0.9, enc[fake_text])
    state["keep"] = False
    fake_rows = state["emb_out"][fake_span[0] : fake_span[1]]
    scale = float(fake_rows.norm(dim=-1).mean())
    fake_emb = torch.stack([vecs["emb"][f] for f in FAKES]).mean(0)
    rows = list(range(*fake_span))
    adds = {}
    for s in INJECT:
        c = torch.stack([vecs["emb"][n] for n in targets if labels[n] == s]).mean(0)
        c = c - fake_emb
        report["inject"].setdefault(s, {})["centroid_over_name_row"] = float(
            c.norm() / scale
        )
        for rho in RHOS:
            adds[f"{s}@ρ{rho}"] = rho * scale * c / c.norm()
    rnd = torch.randn(fake_emb.shape[0], generator=gen)
    adds["rand@ρ2.0"] = 2.0 * scale * rnd / rnd.norm()

    # ---- renders
    loader.compile_blocks(transformer.layers)
    uncond = loader.pad_embeds([enc[""]], device=device)
    sigmas = snapped_sigmas(50, 5.0)
    jobs = [(f"name_{n}", name_text[n], None, args.name_seed) for n in targets]
    for seed in inject_seeds:
        jobs.append((f"fake_s{seed}", fake_text, None, seed))
        for s, t in series_text.items():
            jobs.append((f"written_{s}_s{seed}", t, None, seed))
        for k, vec in adds.items():
            jobs.append((f"inject_{k}_s{seed}", fake_text, vec, seed))
    latents = {}
    for i, (key, text, vec, seed) in enumerate(jobs):
        cond = enc[text]
        state["add"] = None if vec is None else (cond.shape[0], rows, vec)
        c, cm = loader.pad_embeds([cond], device=device)
        t0 = time.time()
        lat = denoise(
            transformer,
            initial_noise(seed, args.height, args.width).to(device),
            c,
            cm,
            *uncond,
            sigmas,
            cfg=3.0,
            uncond_skip=(22,),
            spectrum=MossSpectrum(),
            sampler="er_sde",
            seed=seed,
        ).cpu()
        state["add"] = None
        latents[key] = lat
        save_file(
            {"latent": lat.contiguous()},
            str(out_dir / "latents" / f"{key}.safetensors"),
        )
        log(f"[gen] {i + 1}/{len(jobs)} {key}: {time.time() - t0:.1f}s")

    # ---- one-step reads on the invented-name render
    x0 = latents[f"fake_s{inject_seeds[0]}"].to(device)
    for sigma in (0.9, 0.75):
        noisy = (1 - sigma) * x0 + sigma * noise
        base = x_hat(noisy, sigma, enc[fake_text])
        for s in INJECT:
            teacher = x_hat(noisy, sigma, enc[series_text[s]]) - base
            row = {}
            for k, vec in adds.items():
                if not (k.startswith(s + "@") or k.startswith("rand")):
                    continue
                state["add"] = (enc[fake_text].shape[0], rows, vec)
                eff = x_hat(noisy, sigma, enc[fake_text]) - base
                state["add"] = None
                row[k] = {
                    "r": float((eff * teacher).sum() / teacher.pow(2).sum()),
                    "cos": cos(eff.flatten(), teacher.flatten()),
                }
            report["inject"][s][f"onestep_σ{sigma}"] = row
            log(
                f"[onestep] {s} σ={sigma}: "
                + "  ".join(f"{k}={v['r']:.2f}/{v['cos']:.2f}" for k, v in row.items())
            )

    transformer.to("cpu")
    loader.empty_cache()

    # ---- decode + tag
    from diffusers.image_processor import VaeImageProcessor

    vae = loader.load_vae(model_dir, device=device)
    processor = VaeImageProcessor(vae_scale_factor=16)
    images = {}
    for key, lat in latents.items():
        img = processor.postprocess(decode(lat.to(device), vae), output_type="pil")[0]
        img.save(out_dir / f"{key}.png")
        images[key] = img
    del vae
    loader.empty_cache()

    from anime_tools.tagger.tagger import AnimaTagger

    tagger = AnimaTagger(resolve_under_home(args.tagger), device="cuda")
    tags = {
        k: tagger.predict(img.convert("RGB"))["scores"] for k, img in images.items()
    }
    (out_dir / "tags.json").write_text(
        json.dumps(
            {
                k: {t: round(p, 4) for t, p in sc.items() if p >= 0.05}
                for k, sc in tags.items()
            },
            ensure_ascii=False,
        )
    )

    for n in targets:
        sc = tags[f"name_{n}"]
        s, hair = CHARS[n]
        report["render"][n] = {
            "series_p": sc.get(s),
            "hair_p": sc.get(f"{hair} hair"),
            "char_p": sc.get(n),
        }
    for metric in ("series_p", "hair_p", "char_p"):
        have = [n for n in targets if report["render"][n][metric] is not None]
        if len(have) < 5:
            log(
                f"[render] {metric}: only {len(have)} characters in the tagger vocab, skipped"
            )
            continue
        line = []
        for s in SPACES:
            per = report["geometry"][str(s)]["per"]
            for kind in ("align", "margin"):
                rho = spearman(
                    [per[n][kind] for n in have],
                    [report["render"][n][metric] for n in have],
                )
                report["geometry"][str(s)].setdefault("spearman", {})[
                    f"{kind}~{metric}"
                ] = rho
            line.append(
                f"{s}={report['geometry'][str(s)]['spearman'][f'align~{metric}']:+.2f}"
            )
        log(f"[render] spearman(align, {metric}) n={len(have)}: " + "  ".join(line))

    for s, shared in INJECT.items():
        rec = report["inject"][s]
        keys = [
            "fake",
            f"written_{s}",
            *(f"inject_{s}@ρ{r}" for r in RHOS),
            "inject_rand@ρ2.0",
        ]
        for k in keys:
            ps = [tags[f"{k}_s{seed}"] for seed in inject_seeds]
            rec[k] = {
                "series_p": sum(p.get(s, 0.0) for p in ps) / len(ps),
                "shared_p": sum(p.get(t, 0.0) for p in ps for t in shared)
                / (len(ps) * len(shared)),
            }
        log(
            f"[inject] {s}: "
            + "  ".join(
                f"{k}={rec[k]['series_p']:.2f}/{rec[k]['shared_p']:.2f}" for k in keys
            )
            + "  (series p / shared-tag p)"
        )

    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    log(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
