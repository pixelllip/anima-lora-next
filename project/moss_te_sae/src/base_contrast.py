"""Qwen3.5-9B Base vs instruct × {raw, shipped template} on the probe's caption set.

Separates "instruct weights" from "user-turn framing" on the TE side (hypothesis.md,
test 3). Both models read the same token ids (MOSS tokenizer; Base ships the same
vocab / merges). Caption rows are compared position by position: the templated body
ids must equal the raw ids, and the first ``--skip_head`` body tokens are dropped (in
raw text the first token is the attention sink).

Per layer (15, 31 ``resid_post``; ``out`` = each model's ``last_hidden_state``;
``proj`` = ``text_embedder.proj`` on ``out``) and per edge of the square:

- relative change ``‖b − a‖ / ‖a‖`` per token, cosine, and cosine after removing each
  cell's mean row (the shared offset dominates raw cosine);
- by body position bin, since the template's influence may decay along the caption;
- Base-SAE FVU per cell and top-k code overlap per edge (layers 15 / 31).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from probe import collect_prompts, dit_projection, encode, load_sae, tokenize

CELLS = ("base/raw", "base/tmpl", "inst/raw", "inst/tmpl")
EDGES = {
    "framing|base": ("base/raw", "base/tmpl"),
    "framing|inst": ("inst/raw", "inst/tmpl"),
    "weights|raw": ("base/raw", "inst/raw"),
    "weights|tmpl": ("base/tmpl", "inst/tmpl"),
    "total": ("base/raw", "inst/tmpl"),
}
POS_BINS = ((0, 16), (16, 64), (64, 10**9))


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


@torch.no_grad()
def capture(model_dir, tok, rows, layers):
    """``{framing: [ {layer|'out': (T_body, D) bf16, 'ids': (T_body,)} per row ]}``."""
    from transformers import AutoModel

    te = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
    loader.offload_by_layer(te, torch.device("cuda"), loader.te_reserve_gb(2048))
    grabbed = {}
    hooks = [
        te.language_model.layers[li].register_forward_hook(
            lambda m, i, o, li=li: grabbed.__setitem__(
                li, (o[0] if isinstance(o, tuple) else o)[0]
            )
        )
        for li in layers
    ]
    out = {"raw": [], "tmpl": []}
    for n, row in enumerate(rows):
        for framing in ("raw", "tmpl"):
            ids, seg = tokenize(tok, row["text"], framing == "tmpl")
            o = te(
                input_ids=ids.cuda(),
                attention_mask=torch.ones_like(ids).cuda(),
                use_cache=False,
            )
            body = (seg == 1).nonzero()[:, 0].cuda()
            rec = {li: grabbed[li][body].bfloat16().cpu() for li in layers}
            rec["out"] = o.last_hidden_state[0][body].bfloat16().cpu()
            rec["ids"] = ids[0][body.cpu()]
            out[framing].append(rec)
        if n % 50 == 0:
            log(f"[{Path(model_dir).name}] {n + 1}/{len(rows)}  S={ids.shape[1]}")
    for h in hooks:
        h.remove()
    del te
    torch.cuda.empty_cache()
    return out


def stack(cap, keep, key, skip):
    """Concatenate one cell's rows for ``key``; returns (X, row index, position)."""
    xs, rid, pos = [], [], []
    for n in keep:
        x = cap[n][key][skip:]
        xs.append(x)
        rid.append(torch.full((x.shape[0],), n))
        pos.append(torch.arange(x.shape[0]))
    return torch.cat(xs), torch.cat(rid), torch.cat(pos)


def edge_stats(a, b, masks):
    rel = ((b - a).norm(dim=-1) / a.norm(dim=-1)).cpu()
    cos = F.cosine_similarity(a, b, dim=-1).cpu()
    res = {}
    for name, m in masks.items():
        ac = a[m] - a[m].mean(0)
        bc = b[m] - b[m].mean(0)
        ccos = F.cosine_similarity(ac, bc, dim=-1).cpu()
        res[name] = {
            "tokens": int(m.sum()),
            "rel_mean": float(rel[m].mean()),
            "rel_median": float(rel[m].median()),
            "cos_mean": float(cos[m].mean()),
            "centred_cos_mean": float(ccos.mean()),
        }
    return res


def pos_bins(a, b, pos, src_masks):
    rel = ((b - a).norm(dim=-1) / a.norm(dim=-1)).cpu()
    out = {}
    for name, m in src_masks.items():
        for lo, hi in POS_BINS:
            pm = m & (pos >= lo) & (pos < hi)
            if pm.any():
                out[f"{name}/{lo}-{hi if hi < 10**9 else 'end'}"] = {
                    "tokens": int(pm.sum()),
                    "rel_mean": float(rel[pm].mean()),
                }
    return out


@torch.no_grad()
def sae_stats(sae, cells, masks, k, chunk=2048):
    fvu, idx = {}, {}
    for c, x in cells.items():
        err, ii = [], []
        for s in range(0, x.shape[0], chunk):
            xc = x[s : s + chunk].cuda()
            _, i, xh = encode(sae, xc, k)
            err.append((xc - xh).pow(2).sum(-1).cpu())
            ii.append(i.int().cpu())
        err, idx[c] = torch.cat(err), torch.cat(ii)
        fvu[c] = {}
        for name, m in masks.items():
            xm = x[m]
            fvu[c][name] = float(err[m].sum() / (xm - xm.mean(0)).pow(2).sum())
    n_feat = sae["W_enc"].shape[0]
    overlap = {}
    for e, (ca, cb) in EDGES.items():
        ov = []
        for s in range(0, idx[ca].shape[0], chunk):
            ia, ib = idx[ca][s : s + chunk].cuda(), idx[cb][s : s + chunk].cuda()
            ma = torch.zeros(ia.shape[0], n_feat, dtype=torch.bool, device="cuda")
            ma.scatter_(1, ia.long(), True)
            ov.append(ma.gather(1, ib.long()).float().mean(-1).cpu())
        ov = torch.cat(ov)
        overlap[e] = {name: float(ov[m].mean()) for name, m in masks.items()}
    return fvu, overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        type=Path,
        default=Path("/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base"),
    )
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
    ap.add_argument("--skip_head", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = args.out or resolve_under_home(
        f"output/tests/base_contrast/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)

    rows = collect_prompts(
        resolve_under_home("post_image_dataset/moss"), args.n_per_set, args.seed
    )
    tok = loader.load_tokenizer(default_moss_dir())
    caps = {}
    for model, d in (("base", args.base_dir), ("inst", args.inst_dir)):
        c = capture(d, tok, rows, layers)
        caps[f"{model}/raw"], caps[f"{model}/tmpl"] = c["raw"], c["tmpl"]

    keep = [
        n
        for n in range(len(rows))
        if torch.equal(caps["base/raw"][n]["ids"], caps["base/tmpl"][n]["ids"])
        and caps["base/raw"][n]["ids"].shape[0] > args.skip_head
    ]
    log(f"{len(keep)}/{len(rows)} rows with matching body ids")

    P = dit_projection(default_moss_dir())
    summary = {
        "prompts": len(rows),
        "rows_kept": len(keep),
        "skip_head": args.skip_head,
        "edges": {e: list(v) for e, v in EDGES.items()},
    }
    for key in [*layers, "out", "proj"]:
        src_key = "out" if key == "proj" else key
        cells, rid, pos = {}, None, None
        for c in CELLS:
            x, rid, pos = stack(caps[c], keep, src_key, args.skip_head)
            x = x.float()
            if key == "proj":
                x = torch.cat(
                    [
                        (x[s : s + 4096].cuda() @ P.T).cpu()
                        for s in range(0, len(x), 4096)
                    ]
                )
            cells[c] = x
        src = [rows[n]["source"] for n in rid.tolist()]
        is_layout = torch.tensor([s == "layout" for s in src])
        masks = {"layout": is_layout, "nl": ~is_layout}
        res = {
            "edges": {
                e: edge_stats(cells[a], cells[b], masks) for e, (a, b) in EDGES.items()
            },
            "pos_bins": {
                e: pos_bins(cells[a], cells[b], pos, masks)
                for e, (a, b) in EDGES.items()
            },
        }
        if isinstance(key, int):
            sae = load_sae(args.sae_dir, key)
            res["fvu"], res["code_overlap"] = sae_stats(sae, cells, masks, args.k)
            del sae
            torch.cuda.empty_cache()
        summary[f"layer{key}" if isinstance(key, int) else key] = res
        for e, r in res["edges"].items():
            for name, v in r.items():
                log(
                    f"{key!s:>4} {e:13s} {name:6s} rel={v['rel_mean']:.3f} "
                    f"(med {v['rel_median']:.3f}) cos={v['cos_mean']:.3f} "
                    f"ccos={v['centred_cos_mean']:.3f}"
                    + (
                        f" overlap={res['code_overlap'][e][name]:.3f}"
                        if "code_overlap" in res
                        else ""
                    )
                )
        if "fvu" in res:
            for c, v in res["fvu"].items():
                log(
                    f"{key!s:>4} fvu {c:9s} "
                    + "  ".join(f"{n}={f:.3f}" for n, f in v.items())
                )
        del cells
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
