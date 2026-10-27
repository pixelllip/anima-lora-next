"""Qwen-Scope Base SAEs on the MOSS TE (Qwen3.5-9B post-trained, bf16).

Encodes MOSS prompts exactly as ``library.moss.loader.encode_prompt`` does (chat
template, one prompt per forward, no padding), hooks ``resid_post`` of the given
layers, and reports per layer × source × segment (template prefix / prompt body /
template suffix):

- reconstruction: FVU, NMSE, cosine, with the card's encoder (no ``b_dec``
  centring) and the centred variant;
- ``norm(h31)`` against ``last_hidden_state`` and against the cached INT8 embeds;
- last layer only, per feature: DiT readability ``‖P(g⊙d)‖ / ‖g⊙d‖`` with ``P`` =
  ``text_embedder.proj`` and ``g`` the final-norm gain, and the energy it carries
  into the DiT on body tokens, ``‖P(g⊙d)‖² Σ_t (a_t / rms_t)²`` (linearised
  through the norm, SAE error term ignored), with top-activating contexts.

Raw-text controls (no chat template) separate template effects from the
Base→post-trained shift.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from safetensors import safe_open

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader

LAYOUT_SETS = (
    "ama_mitsuki_layout",
    "group_wsw_layout",
    "hews_layout",
    "sincos_layout",
    "channel_(caststation)_layout",
)
NL_PROMPTS = (
    "project/pe_lora/prompts_handwritten.txt",
    "project/pe_lora/prompts_heldout.txt",
)
SEGMENTS = ("prefix", "body", "suffix")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def collect_prompts(cache_root: Path, n_per_set: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for name in LAYOUT_SETS:
        files = sorted((cache_root / name).glob("*.moss_te.safetensors"))
        for f in rng.sample(files, min(n_per_set, len(files))):
            with safe_open(f, "pt") as g:
                meta = g.metadata()
            if meta.get("caption_format") != "layout_json":
                continue
            rows.append(
                {
                    "source": "layout",
                    "set": name,
                    "text": meta["caption"],
                    "cache": str(f),
                }
            )
    for p in NL_PROMPTS:
        for line in resolve_under_home(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(
                    {
                        "source": "nl",
                        "set": Path(p).stem,
                        "text": line.strip(),
                        "cache": None,
                    }
                )
    return rows


def tokenize(tok, text, templated: bool):
    """``(ids (1, S), segment ids (S,))``; ids match ``loader.prompt_ids``."""
    if not templated:
        ids = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=loader.MAX_SEQUENCE_LENGTH,
        ).input_ids
        return ids, torch.full((ids.shape[1],), 1, dtype=torch.int8)
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tok(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=loader.MAX_SEQUENCE_LENGTH,
        return_offsets_mapping=True,
    )
    assert torch.equal(enc.input_ids, loader.prompt_ids(tok, text).input_ids)
    start = formatted.find(text)
    end = start + len(text)
    seg = torch.empty(enc.input_ids.shape[1], dtype=torch.int8)
    for i, (a, b) in enumerate(enc.offset_mapping[0].tolist()):
        seg[i] = 0 if b <= start else (2 if a >= end else 1)
    return enc.input_ids, seg


@torch.no_grad()
def capture(args, rows, layers):
    from transformers import AutoModel

    tok = loader.load_tokenizer(default_moss_dir())
    te = AutoModel.from_pretrained(args.inst_dir, dtype=torch.bfloat16).eval()
    norm_w = te.language_model.norm.weight.detach().float().clone()
    norm_eps = te.language_model.norm.eps
    loader.offload_by_layer(te, torch.device("cuda"), loader.te_reserve_gb(2048))
    grabbed = {}
    hooks = [
        te.language_model.layers[li].register_forward_hook(
            lambda m, i, o, li=li: grabbed.__setitem__(
                li, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
            )
        )
        for li in layers
    ]
    seqs = []
    for n, row in enumerate(rows):
        for templated in (True, False):
            ids, seg = tokenize(tok, row["text"], templated)
            out = te(
                input_ids=ids.cuda(),
                attention_mask=torch.ones_like(ids).cuda(),
                use_cache=False,
            )
            seqs.append(
                {
                    "row": n,
                    "templated": templated,
                    "ids": ids[0],
                    "seg": seg,
                    "resid": {li: grabbed[li] for li in layers},
                    "last": out.last_hidden_state[0].float().cpu(),
                }
            )
        if n % 25 == 0:
            log(f"[te] {n + 1}/{len(rows)}  S={ids.shape[1]}")
    for h in hooks:
        h.remove()
    del te
    torch.cuda.empty_cache()
    return tok, seqs, norm_w, norm_eps


def final_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def te_checks(rows, seqs, norm_w, norm_eps, last_layer):
    out = {"norm_h_last_vs_last_hidden": [], "bf16_vs_cached_int8": []}
    for s in seqs:
        if not s["templated"]:
            continue
        x = s["resid"][last_layer]
        e = final_norm(x, norm_w, norm_eps)
        out["norm_h_last_vs_last_hidden"].append(
            float((e - s["last"]).norm() / s["last"].norm())
        )
        cache = rows[s["row"]]["cache"]
        if cache:
            with safe_open(cache, "pt") as g:
                c = g.get_tensor("embeds").float()
            if c.shape == s["last"].shape:
                d = (c - s["last"]).norm(dim=-1) / s["last"].norm(dim=-1)
                body = s["seg"] == 1
                out["bf16_vs_cached_int8"].append(
                    (float(d.mean()), float(d[body].mean()))
                )
    a = out["norm_h_last_vs_last_hidden"]
    b = out["bf16_vs_cached_int8"]
    return {
        "norm_h_last_vs_last_hidden_rel_max": max(a),
        "int8_vs_bf16_token_rel_err_mean_all": sum(x for x, _ in b) / max(len(b), 1),
        "int8_vs_bf16_token_rel_err_mean_body": sum(y for _, y in b) / max(len(b), 1),
        "int8_vs_bf16_n": len(b),
    }


def load_sae(sae_dir, layer):
    sd = torch.load(
        Path(sae_dir) / f"layer{layer}.sae.pt", map_location="cpu", weights_only=True
    )
    return {k: v.float().cuda() for k, v in sd.items()}


@torch.no_grad()
def encode(sae, x, k, centred=False):
    pre = (x - sae["b_dec"] if centred else x) @ sae["W_enc"].T + sae["b_enc"]
    vals, idx = pre.topk(k, dim=-1)
    xhat = torch.zeros_like(pre).scatter_(-1, idx, vals) @ sae["W_dec"].T + sae["b_dec"]
    return vals, idx, xhat


def recon_table(sae, seqs, layer, k, rows, chunk=1024):
    """Per (source, templated, segment): FVU / NMSE / cosine for both encoders,
    plus the card-encoder codes per sequence."""
    groups = {}
    codes = []
    for s in seqs:
        x = s["resid"][layer].cuda()
        vals, idx, card_err, cent_err, cos = [], [], [], [], []
        for c in range(0, x.shape[0], chunk):
            xc = x[c : c + chunk]
            v, i, xh = encode(sae, xc, k)
            _, _, xh2 = encode(sae, xc, k, centred=True)
            vals.append(v.half().cpu())
            idx.append(i.int().cpu())
            card_err.append((xc - xh).pow(2).sum(-1))
            cent_err.append((xc - xh2).pow(2).sum(-1))
            cos.append(torch.nn.functional.cosine_similarity(xc, xh, dim=-1))
        card_err, cent_err, cos = (
            torch.cat(t).cpu() for t in (card_err, cent_err, cos)
        )
        codes.append((torch.cat(vals), torch.cat(idx)))
        src = rows[s["row"]]["source"] + ("" if s["templated"] else "_raw")
        xs = x.cpu()
        for g in range(3):
            m = s["seg"] == g
            if not m.any():
                continue
            key = f"{src}/{SEGMENTS[g]}"
            acc = groups.setdefault(
                key, {"x": [], "card": 0.0, "cent": 0.0, "cos": 0.0, "n": 0, "neg": 0}
            )
            acc["x"].append(xs[m])
            acc["card"] += float(card_err[m].sum())
            acc["cent"] += float(cent_err[m].sum())
            acc["cos"] += float(cos[m].sum())
            acc["n"] += int(m.sum())
            acc["neg"] += int((codes[-1][0][m] < 0).sum())
    table = {}
    for key, acc in groups.items():
        xs = torch.cat(acc["x"])
        var = float((xs - xs.mean(0)).pow(2).sum())
        sq = float(xs.pow(2).sum())
        norms = xs.norm(dim=-1)
        table[key] = {
            "tokens": acc["n"],
            "fvu_card": acc["card"] / var,
            "fvu_centred": acc["cent"] / var,
            "nmse_card": acc["card"] / sq,
            "cos_card": acc["cos"] / acc["n"],
            "neg_code_frac": acc["neg"] / (acc["n"] * k),
            "norm_median": float(norms.median()),
            "norm_max": float(norms.max()),
        }
    return table, codes


def dit_projection(moss_dir: Path):
    tdir = moss_dir / "MOSS-Image-Base" / "transformer"
    wm = json.loads(
        (tdir / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    key = "text_embedder.proj.weight"
    with safe_open(tdir / wm[key], "pt") as g:
        return g.get_tensor(key).float().cuda()


@torch.no_grad()
def readability(sae, P, norm_w, norm_eps, seqs, codes, tok, layer_k, top_n=40, n_ctx=6):
    g = (1.0 + norm_w).cuda()
    gd = g[:, None] * sae["W_dec"]  # (4096, F)
    read = (P @ gd).norm(dim=0) / gd.norm(dim=0)
    te_gain2 = gd.pow(2).sum(0)
    rnd = torch.randn(4096, 4096, device="cuda") * g[:, None]
    read_rnd = (P @ rnd).norm(dim=0) / rnd.norm(dim=0)

    F = read.shape[0]
    a2 = torch.zeros(F, dtype=torch.float64)
    freq = torch.zeros(F, dtype=torch.int64)
    data_read = []
    split = {"x": 0.0, "rec": 0.0, "err": 0.0}
    split_te = dict(split)
    for s, (vals, idx) in zip(seqs, codes):
        if not s["templated"]:
            continue
        x = s["resid"][layer_k]
        body = s["seg"] == 1
        rms = x.pow(2).mean(-1).sqrt()
        e = final_norm(x[body], norm_w, norm_eps).cuda()
        data_read.append((P @ e.T).norm(dim=0) / e.norm(dim=-1))
        v = vals[body].float() / rms[body, None]
        a2.index_add_(0, idx[body].reshape(-1).long(), v.pow(2).reshape(-1).double())
        freq.index_add_(
            0,
            idx[body].reshape(-1).long(),
            torch.ones(idx[body].numel(), dtype=torch.int64),
        )
        xb = x[body].cuda()
        rec = (
            sae["W_dec"].T[idx[body].long().cuda()]
            * vals[body].float().cuda()[..., None]
        ).sum(1) + sae["b_dec"]
        scale = g / rms[body, None].cuda()
        for name, part in (("x", xb), ("rec", rec), ("err", xb - rec)):
            split[name] += float((P @ (part * scale).T).pow(2).sum())
            split_te[name] += float((part * scale).pow(2).sum())
    dit_energy = read.double().cpu().pow(2) * a2
    te_energy = te_gain2.double().cpu() * a2
    data_read = torch.cat(data_read)

    n_body = sum(int((s["seg"] == 1).sum()) for s in seqs if s["templated"])

    def contexts(f):
        hits = []
        for s, (vals, idx) in zip(seqs, codes):
            if not s["templated"]:
                continue
            m = (idx == f) & (s["seg"] == 1)[:, None]
            if m.any():
                t, j = m.nonzero(as_tuple=True)
                for tt, jj in zip(t.tolist(), j.tolist()):
                    hits.append((float(vals[tt, jj]), s, tt))
        hits.sort(key=lambda h: -h[0])
        out = []
        for a, s, t in hits[:n_ctx]:
            ids = s["ids"].tolist()
            left = tok.decode(ids[max(0, t - 10) : t])
            out.append(
                f"{a:.1f}  …{left}⟦{tok.decode([ids[t]])}⟧{tok.decode(ids[t + 1 : t + 3])}"
            )
        return out

    def describe(order, n):
        rows = []
        for f in order[:n].tolist():
            rows.append(
                {
                    "feature": f,
                    "dit_energy_share": float(dit_energy[f] / dit_energy.sum()),
                    "te_energy_share": float(te_energy[f] / te_energy.sum()),
                    "readability": float(read[f]),
                    "body_freq": float(freq[f]) / n_body,
                    "contexts": contexts(f),
                }
            )
        return rows

    q = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device="cuda")
    active = freq > 0
    ratio = torch.where(
        te_energy > 0, dit_energy / te_energy, torch.zeros_like(te_energy)
    )
    heavy = active & (te_energy > te_energy[active].quantile(0.9))
    suppressed = torch.where(heavy, ratio, torch.full_like(ratio, float("inf")))
    amplified = torch.where(heavy, ratio, torch.full_like(ratio, float("-inf")))
    return {
        "readability_quantiles_all_features": read.quantile(q).tolist(),
        "readability_quantiles_random_dirs": read_rnd.quantile(q).tolist(),
        "readability_quantiles_body_tokens": data_read.quantile(q).tolist(),
        "body_tokens": n_body,
        "features_active_on_body": int(active.sum()),
        "dit_energy_top100_share": float(
            dit_energy.sort(descending=True).values[:100].sum() / dit_energy.sum()
        ),
        "top_by_dit_energy": describe(dit_energy.argsort(descending=True), top_n),
        "energy_split_dit": {k: v / split["x"] for k, v in split.items()},
        "energy_split_te": {k: v / split_te["x"] for k, v in split_te.items()},
        "suppressed_by_dit": describe(suppressed.argsort(), 20),
        "amplified_by_dit": describe(amplified.argsort(descending=True), 20),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inst_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_pe/inst")
    )
    ap.add_argument(
        "--sae_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_sae")
    )
    ap.add_argument("--layers", default="15,31")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--n_per_set", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    last = max(layers)
    out = args.out or args.sae_dir / "probe" / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)

    rows = collect_prompts(
        resolve_under_home("post_image_dataset/moss"), args.n_per_set, args.seed
    )
    log(
        f"{len(rows)} prompts ({sum(r['source'] == 'layout' for r in rows)} layout, "
        f"{sum(r['source'] == 'nl' for r in rows)} nl)"
    )
    tok, seqs, norm_w, norm_eps = capture(args, rows, layers)
    summary = {
        "layers": layers,
        "prompts": len(rows),
        "te_checks": te_checks(rows, seqs, norm_w, norm_eps, last),
    }
    log(f"te checks {summary['te_checks']}")

    P = dit_projection(default_moss_dir())
    for layer in layers:
        sae = load_sae(args.sae_dir, layer)
        table, codes = recon_table(sae, seqs, layer, args.k, rows)
        summary[f"layer{layer}"] = {"recon": table}
        for key in sorted(table):
            r = table[key]
            log(
                f"L{layer} {key:22s} n={r['tokens']:6d} fvu={r['fvu_card']:.3f} (centred {r['fvu_centred']:.3f}) "
                f"cos={r['cos_card']:.3f} |x|med={r['norm_median']:.1f} max={r['norm_max']:.0f}"
            )
        if layer == last:
            summary[f"layer{layer}"]["dit"] = readability(
                sae, P, norm_w, norm_eps, seqs, codes, tok, layer
            )
            torch.save(
                {
                    "rows": rows,
                    "seqs": [
                        {k: v for k, v in s.items() if k not in ("resid", "last")}
                        for s in seqs
                    ],
                    "codes": codes,
                },
                out / f"codes_layer{layer}.pt",
            )
        del sae
        torch.cuda.empty_cache()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
