"""Where does the MOSS DiT expand a character name into her appearance?

Two reads on the base DiT, int8 TE (the shipped stack):

- ``probe``: hair colour at the name tokens, per space (TE output, ``text_embedder``
  output, each of the 42 blocks). Colour directions come from ``{colour} hair`` tag
  spans in captions that name an invented character (``tag`` probe: does the name
  token look like the tag?), and from the other characters' name tokens
  (``loo``: is colour linearly laid out over names at all?). Name vectors are the
  mean over contexts minus the invented names' mean. One fixed ``x_t`` (σ 0.9,
  zero latent).
- ``knockout``: the name's effect on the one-step ``x̂₀``, ``E = x̂₀(name) −
  x̂₀(invented name)``, with every query barred from the name's keys from block
  ``L`` on (``pre`` = also in the two text blocks). ``r = ⟨E_ko, E⟩/‖E‖²``. ``rows``
  splices the name's TE rows into the invented-name caption; ``ctx`` is the
  name's caption with the invented name's rows (what the later TE tokens carry).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.train import NUM_TRAIN_TIMESTEPS, place, step_reserve_gb

COLOURS = (
    "aqua",
    "blonde",
    "red",
    "pink",
    "blue",
    "black",
    "brown",
    "white",
    "purple",
    "green",
)
CHARS = {
    "hatsune miku": "aqua",
    "akita neru": "blonde",
    "kagamine rin": "blonde",
    "kirisame marisa": "blonde",
    "flandre scarlet": "blonde",
    "ijichi nijika": "blonde",
    "kasane teto": "red",
    "kita ikuyo": "red",
    "houshou marine": "red",
    "hong meiling": "red",
    "erza scarlet": "red",
    "megurine luka": "pink",
    "gotou hitori": "pink",
    "ram (re:zero)": "pink",
    "komeiji satori": "pink",
    "kaname madoka": "pink",
    "ayanami rei": "blue",
    "rem (re:zero)": "blue",
    "cirno": "blue",
    "yamada ryo": "blue",
    "kafuu chino": "blue",
    "akemi homura": "black",
    "tohsaka rin": "black",
    "yor briar": "black",
    "komi shouko": "black",
    "hakurei reimu": "brown",
    "inugami korone": "brown",
    "misaka mikoto": "brown",
    "hirasawa yui": "brown",
    "frieren": "white",
    "shirakami fubuki": "white",
    "inubashiri momiji": "white",
    "patchouli knowledge": "purple",
    "fern (sousou no frieren)": "purple",
    "nekomata okayu": "purple",
    "hoshino ai (oshi no ko)": "purple",
    "komeiji koishi": "green",
    "kochiya sanae": "green",
    "uruha rushia": "green",
    "c.c.": "green",
}
# Not counted in accuracy: the DiT does not draw them from the name (phase 0).
UNKNOWN = {"toujou koneko": "white", "malfoid": None}
FAKES = ("okuda reni", "tamura kaede", "mizuno shiori")
CONTEXTS = (
    "1girl, solo, full body, standing, looking at viewer, smile, simple background, white background",
    "1girl, solo, upper body, looking at viewer, open mouth, outdoors, blue sky",
    "1girl, solo, portrait, closed mouth, indoors, bedroom",
)
KO_STARTS = ("pre", 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
KO_CHARS = ("hatsune miku", "akita neru", "kasane teto")
X0 = {
    "miku_img": "output/tests/neru_phase0/latents/012_spaceM_miku_s11_base_s11.safetensors",
    "neru_img": "output/tests/neru_phase0/latents/000_spaceA_name_s11_base_s11.safetensors",
}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def caption(name, tags, series=None):
    s = {"meta": ["sfw"], "characters": [name]}
    if series:
        s["series"] = [series]
    s["caption"] = tags
    return dumps({"summary": s})


def span(tok, text, needle, after=None):
    """Token range ``[a, b)`` of ``needle`` in the chat-formatted ``text`` (the first
    occurrence after the substring ``after``)."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    offs = tok(formatted, return_offsets_mapping=True).offset_mapping
    start = formatted.find(needle, formatted.find(after) if after else 0)
    assert start >= 0, (needle, text)
    end = start + len(needle)
    idx = [i for i, (a, b) in enumerate(offs) if a < end and b > start]
    return idx[0], idx[-1] + 1


def build(tok):
    """Probe captions ``[{text, kind, label, span}]`` and knockout pairs."""
    probe = []
    for f in FAKES:
        for ci, ctx in enumerate(CONTEXTS):
            text = caption(f, ctx)
            probe.append(
                dict(
                    text=text,
                    kind="fake",
                    who=f,
                    span=span(tok, text, f, '"characters"'),
                )
            )
            for c in COLOURS:
                tags = ctx.replace("solo, ", f"solo, {c} hair, ", 1)
                text = caption(f, tags)
                probe.append(
                    dict(
                        text=text,
                        kind="tag",
                        who=c,
                        span=span(tok, text, f"{c} hair", '"caption"'),
                    )
                )
    for name in (*CHARS, *UNKNOWN):
        for ctx in CONTEXTS:
            text = caption(name, ctx)
            probe.append(
                dict(
                    text=text,
                    kind="name",
                    who=name,
                    span=span(tok, text, name, '"characters"'),
                )
            )
    ko = []
    for name in KO_CHARS:
        real = caption(name, CONTEXTS[0], "vocaloid")
        fake = caption(FAKES[0], CONTEXTS[0], "vocaloid")
        ko.append(
            dict(
                name=name,
                real=real,
                fake=fake,
                real_span=span(tok, real, name, '"characters"'),
                fake_span=span(tok, fake, FAKES[0], '"characters"'),
            )
        )
    return probe, ko


class Knockout:
    """Wraps the text blocks and the 42 blocks: from ``start`` on, every query is
    barred from the keys at ``pos`` (text-relative). ``start=None`` passes an
    all-true mask, so every condition takes the same masked attention path."""

    def __init__(self, transformer):
        self.start, self.pos, self.n_img = None, None, 0
        for i, block in enumerate(transformer.text_embedder.blocks):
            self._wrap(block, i - 2, text_only=True)
        for i, layer in enumerate(transformer.layers):
            self._wrap(layer, i, text_only=False)

    def _mask(self, x, index, text_only):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if self.pos is not None and self.start is not None:
            first = -2 if self.start == "pre" else self.start
            if index >= first:
                mask[:, self.pos + (0 if text_only else self.n_img)] = False
        return mask

    def _wrap(self, block, index, text_only):
        inner = block.forward

        if text_only:

            def forward(x, rope, key_mask=None):
                return inner(x, rope, self._mask(x, index, True))
        else:

            def forward(x, rope, modulation, key_mask=None, *a, **kw):
                return inner(x, rope, modulation, self._mask(x, index, False), *a, **kw)

        block.forward = forward


def standardise(vectors):
    """Per-dimension z-scoring over every vector of one space (massive channels)."""
    stack = torch.stack(list(vectors))
    return stack.mean(0), stack.std(0).clamp_min(1e-6)


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def analyse(space, probe, labelled):
    """``space``: per probe row ``{"mean": (D,), "last": (D,)}``. Returns accuracies
    and Miku's / the unknowns' colour scores."""
    out = {}
    for read in ("mean", "last"):
        mu, sd = standardise(space[i][read] for i in range(len(probe)))

        def z(i):
            return (space[i][read] - mu) / sd

        tag = {
            c: torch.stack(
                [z(i) for i, p in enumerate(probe) if p["kind"] == "tag" and p["who"] == c]
            ).mean(0)
            for c in COLOURS
        }
        tag_mu = torch.stack(list(tag.values())).mean(0)
        u = {c: v - tag_mu for c, v in tag.items()}
        fake = torch.stack(
            [z(i) for i, p in enumerate(probe) if p["kind"] == "fake"]
        ).mean(0)
        names = {}
        for i, p in enumerate(probe):
            if p["kind"] == "name":
                names.setdefault(p["who"], []).append(z(i))
        d = {n: torch.stack(v).mean(0) - fake for n, v in names.items()}

        def scores(vec, dirs):
            return {c: cos(vec, w) for c, w in dirs.items()}

        hits_tag, hits_loo, per = 0, 0, {}
        for n, colour in labelled.items():
            s_tag = scores(d[n], u)
            others = {
                c: torch.stack([d[m] for m, cc in labelled.items() if cc == c and m != n]).mean(0)
                for c in COLOURS
                if any(cc == c and m != n for m, cc in labelled.items())
            }
            o_mu = torch.stack(list(others.values())).mean(0)
            s_loo = scores(d[n], {c: v - o_mu for c, v in others.items()})
            pred_tag = max(s_tag, key=s_tag.get)
            pred_loo = max(s_loo, key=s_loo.get) if colour in others else None
            hits_tag += pred_tag == colour
            hits_loo += pred_loo == colour
            per[n] = {
                "tag_pred": pred_tag,
                "tag_rank": sorted(s_tag, key=s_tag.get, reverse=True).index(colour) + 1,
                "loo_pred": pred_loo,
            }
        n_loo = sum(
            sum(cc == colour for cc in labelled.values()) > 1 for colour in labelled.values()
        )
        unknown = {n: scores(d[n], u) for n in UNKNOWN}
        out[read] = {
            "tag_acc": hits_tag / len(labelled),
            "loo_acc": hits_loo / n_loo,
            "miku_aqua_cos": scores(d["hatsune miku"], u)["aqua"],
            "miku_tag_rank": per["hatsune miku"]["tag_rank"],
            "name_norm_vs_fake": float(
                torch.stack([v.norm() for v in d.values()]).mean()
            ),
            "per": per,
            "unknown_top": {
                n: sorted(s, key=s.get, reverse=True)[:3] for n, s in unknown.items()
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument("--probe_sigma", type=float, default=0.9)
    ap.add_argument("--ko_sigmas", default="0.9,0.6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=Path("/media/sorryhyun/new/qwen35_sae/entity_dit")
    )
    args = ap.parse_args()
    out_dir = args.out / time.strftime("%Y%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    ko_sigmas = [float(s) for s in args.ko_sigmas.split(",")]

    tok = loader.load_tokenizer(default_moss_dir())
    probe, ko = build(tok)
    texts = [p["text"] for p in probe] + [t for k in ko for t in (k["real"], k["fake"])]
    log(f"{len(probe)} probe captions, {len(ko)} knockout pairs")
    enc = loader.encode_prompts(
        default_moss_dir(), texts, log=log, precision=args.text_encoder
    )

    device = torch.device("cuda")
    transformer = loader.load_transformer(default_moss_dir(), "base")
    transformer.requires_grad_(False)
    longest = max(e.shape[0] for e in enc.values()) + 8
    place(transformer, device, -1, step_reserve_gb(76 * 52 + longest, grad_checkpointing=True))
    knock = Knockout(transformer)

    grab = {}

    def hook(index, x, n_img):
        if grab.get("spans") is not None:
            a, b = grab["spans"]
            rows = x[0, n_img + a : n_img + b].float()
            grab[index] = {"mean": rows.mean(0).cpu(), "last": rows[-1].cpu()}
        return x

    def emb_hook(_m, _i, out):
        if grab.get("spans") is not None:
            a, b = grab["spans"]
            rows = out[0, a:b].float()
            grab["emb"] = {"mean": rows.mean(0).cpu(), "last": rows[-1].cpu()}

    transformer.text_embedder.register_forward_hook(emb_hook)

    @torch.no_grad()
    def velocity(noisy, sigma, emb):
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        return transformer(
            noisy, t, emb[None].to(device, torch.bfloat16), return_dict=False
        )[0].float()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((1, 128, 76, 52), generator=gen).to(device)

    # ---- probe
    transformer.layer_hook = hook
    spaces = {}
    for i, p in enumerate(probe):
        e = enc[p["text"]]
        a, b = p["span"]
        rows = e[a:b].float()
        spaces.setdefault("te", {})[i] = {"mean": rows.mean(0), "last": rows[-1]}
        grab.clear()
        grab["spans"] = p["span"]
        velocity(args.probe_sigma * noise, args.probe_sigma, e)
        for k, v in grab.items():
            if k != "spans":
                spaces.setdefault(k, {})[i] = v
        if i % 40 == 0:
            log(f"probe {i}/{len(probe)}")
    transformer.layer_hook = None
    grab.clear()

    order = ["te", "emb", *range(len(transformer.layers))]
    report = {"probe": {}, "knockout": {}, "chars": CHARS, "colours": COLOURS}
    for s in order:
        res = analyse(spaces[s], probe, CHARS)
        report["probe"][str(s)] = res
        m, la = res["mean"], res["last"]
        log(
            f"[probe] {str(s):>4s}: tag acc {m['tag_acc']:.2f}/{la['tag_acc']:.2f}  "
            f"loo acc {m['loo_acc']:.2f}/{la['loo_acc']:.2f}  miku aqua cos "
            f"{m['miku_aqua_cos']:+.3f} (rank {m['miku_tag_rank']})  "
            f"koneko {m['unknown_top']['toujou koneko']}  (mean/last)"
        )
    del spaces

    # ---- knockout
    def splice(dst, dst_span, src, src_span):
        return torch.cat(
            (dst[: dst_span[0]], src[src_span[0] : src_span[1]], dst[dst_span[1] :])
        )

    for k in ko:
        real, fake = enc[k["real"]], enc[k["fake"]]
        rs, fs = k["real_span"], k["fake_span"]
        conds = {
            "full": (real, None),
            "rows": (splice(fake, fs, real, rs), None),
            "ctx": (splice(real, rs, fake, fs), None),
            **{f"ko{L}": (real, L) for L in KO_STARTS},
        }
        rec = {}
        for x0_name, x0_path in X0.items():
            x0 = load_file(resolve_under_home(x0_path))["latent"].to(device)
            for sigma in ko_sigmas:
                noisy = (1 - sigma) * x0 + sigma * noise
                knock.start = knock.pos = None
                knock.n_img = 76 * 52

                def x_hat(emb, start=None, pos=None):
                    knock.start, knock.pos = start, pos
                    v = velocity(noisy, sigma, emb)
                    knock.start = knock.pos = None
                    return noisy - sigma * v

                base = x_hat(fake)
                full = x_hat(real) - base
                row = {"effect_over_xhat": float(full.norm() / base.norm())}
                for c, (emb, start) in conds.items():
                    if c == "full":
                        continue
                    pos = (
                        torch.arange(rs[0], rs[1], device=device)
                        if start is not None
                        else None
                    )
                    eff = x_hat(emb, start, pos) - base
                    row[c] = {
                        "r": float((eff * full).sum() / full.pow(2).sum()),
                        "cos": cos(eff.flatten(), full.flatten()),
                    }
                rec[f"{x0_name}/σ{sigma}"] = row
                log(
                    f"[ko] {k['name']} x0={x0_name} σ={sigma} ‖E‖/‖x̂₀‖={row['effect_over_xhat']:.3f}  "
                    + "  ".join(
                        f"{c}={row[c]['r']:.2f}/{row[c]['cos']:.2f}"
                        for c in conds
                        if c != "full"
                    )
                )
        report["knockout"][k["name"]] = rec

    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    log(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
