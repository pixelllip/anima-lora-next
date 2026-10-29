"""Compare ``template_attn.py`` arms (Base vs LoRAs) from their ``raw.pt``. CPU only.

- attention mass / value contribution per key segment, image and caption queries;
- per-block caption mass (block mean and max head);
- per template token mass;
- value rows: cross-prompt spread of each template row against the caption-mean row,
  and at step 0 (same input to every arm) the LoRA's change to each row and how alike
  that change is across prompts.

    .venv/bin/python project/moss_te_sae/src/template_attn_cmp.py <dir> base arm1 arm2
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch

SEGS = ("image", "prefix", "body", "suffix")


def load(path):
    return torch.load(path / "raw.pt", weights_only=False)


def seg_table(arm, key, steps=None):
    """(4,) mean over prompts, records and heads."""
    rows = [
        r[key].mean(0)
        for p in arm["prompts"]
        for r in p["records"]
        if steps is None or r["step"] in steps
    ]
    return torch.stack(rows).mean(0)


def by_block(arm, key):
    """(42, H, 4) mean over prompts and steps."""
    acc = torch.zeros(42, *arm["prompts"][0]["records"][0][key].shape)
    n = torch.zeros(42)
    for p in arm["prompts"]:
        for r in p["records"]:
            acc[r["block"]] += r[key]
            n[r["block"]] += 1
    return acc / n[:, None, None]


def token_mass(arm, key):
    t = torch.stack(
        [
            torch.stack([r[key] for r in p["records"]]).mean((0, 1))
            for p in arm["prompts"]
        ]
    )
    labels = [lab.split(":", 1)[1] for lab in arm["prompts"][0]["labels"]]
    tags = [f"pre{j}" if j < 3 else f"suf{j - 3}" for j in range(len(labels))]
    return {f"{tg} {lab}": float(x) for tg, lab, x in zip(tags, labels, t.mean(0))}


def value_spread(arm):
    """Cross-prompt ``mean_p ‖v_p − v̄‖ / ‖v̄‖`` per template row and for the caption
    mean, averaged over the saved (step, block)."""
    keys = arm["prompts"][0]["vals"].keys()
    tm, cm = [], []
    for k in keys:
        t = torch.stack([p["vals"][k][0].float() for p in arm["prompts"]])  # (P, T, HD)
        c = torch.stack([p["vals"][k][1].float() for p in arm["prompts"]])  # (P, HD)
        tm.append((t - t.mean(0)).norm(dim=-1).mean(0) / t.mean(0).norm(dim=-1))
        cm.append((c - c.mean(0)).norm(dim=-1).mean(0) / c.mean(0).norm(dim=-1))
    return torch.stack(tm).mean(0), float(torch.stack(cm).mean())


def lora_delta(base, arm, step=0):
    """At ``step``: per row ``‖v_lora − v_base‖ / ‖v_base‖`` and the mean pairwise
    cosine of the per-prompt deltas (1 = the same shift for every prompt)."""
    keys = [k for k in base["prompts"][0]["vals"] if k[0] == step]
    rel_t, rel_c, cos_t, cos_c = [], [], [], []
    for k in keys:
        bt = torch.stack([p["vals"][k][0].float() for p in base["prompts"]])
        at = torch.stack([p["vals"][k][0].float() for p in arm["prompts"]])
        bc = torch.stack([p["vals"][k][1].float() for p in base["prompts"]])
        ac = torch.stack([p["vals"][k][1].float() for p in arm["prompts"]])
        dt, dc = at - bt, ac - bc
        rel_t.append((dt.norm(dim=-1) / bt.norm(dim=-1)).mean(0))
        rel_c.append(float((dc.norm(dim=-1) / bc.norm(dim=-1)).mean()))
        pairs = list(combinations(range(dt.shape[0]), 2))
        cos_t.append(
            torch.stack(
                [
                    torch.nn.functional.cosine_similarity(dt[i], dt[j], -1)
                    for i, j in pairs
                ]
            ).mean(0)
        )
        cos_c.append(
            float(
                torch.stack(
                    [
                        torch.nn.functional.cosine_similarity(dc[i], dc[j], -1)
                        for i, j in pairs
                    ]
                ).mean()
            )
        )
    return {
        "rel_tmpl_rows": torch.stack(rel_t).mean(0).tolist(),
        "rel_caption_mean": sum(rel_c) / len(rel_c),
        "delta_cos_tmpl_rows": torch.stack(cos_t).mean(0).tolist(),
        "delta_cos_caption_mean": sum(cos_c) / len(cos_c),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("arms", nargs="+", help="first is the reference")
    args = ap.parse_args()
    arms = {a: load(args.root / a) for a in args.arms}
    ref = args.arms[0]
    n_steps = 1 + max(r["step"] for r in arms[ref]["prompts"][0]["records"])
    thirds = {
        "early": set(range(0, n_steps // 3)),
        "mid": set(range(n_steps // 3, 2 * n_steps // 3)),
        "late": set(range(2 * n_steps // 3, n_steps)),
    }
    out = {}
    for q in ("img", "body"):
        for m in ("mass", "contrib"):
            key = f"{q}_{m}"
            print(f"\n{key}  {'/'.join(SEGS)}")
            for a, arm in arms.items():
                cells = {t: seg_table(arm, key, s) for t, s in thirds.items()}
                out.setdefault(a, {})[key] = {t: c.tolist() for t, c in cells.items()}
                print(
                    f"  {a:28s} "
                    + " | ".join(
                        f"{t} " + "/".join(f"{x:.3f}" for x in c)
                        for t, c in cells.items()
                    )
                )
    print("\nimage→caption mass by block: mean (max head)")
    bb = {a: by_block(arm, "img_mass") for a, arm in arms.items()}
    for b in range(42):
        print(
            f"  {b:2d} "
            + "  ".join(
                f"{a[-6:]}: {bb[a][b, :, 2].mean():.3f} ({bb[a][b, :, 2].max():.3f})"
                for a in arms
            )
        )
    print("\ntemplate token mass (image queries)")
    tok = {a: token_mass(arm, "img_tmpl_tok") for a, arm in arms.items()}
    for lab in tok[ref]:
        print(f"  {lab:20s} " + "  ".join(f"{tok[a][lab]:.4f}" for a in arms))
    print("\nvalue rows: cross-prompt spread (template rows | caption mean)")
    labels = list(tok[ref])
    for a, arm in arms.items():
        t, c = value_spread(arm)
        out[a]["value_spread"] = {"tmpl": dict(zip(labels, t.tolist())), "caption": c}
        print(f"  {a:28s} " + " ".join(f"{x:.2f}" for x in t) + f" | {c:.2f}")
    for a in args.arms[1:]:
        d = lora_delta(arms[ref], arms[a])
        out[a]["lora_delta_step0"] = d
        print(f"\n{a} vs {ref}, step 0 value change (template rows | caption mean)")
        print(
            "  rel   "
            + " ".join(f"{x:.2f}" for x in d["rel_tmpl_rows"])
            + f" | {d['rel_caption_mean']:.2f}"
        )
        print(
            "  Δcos  "
            + " ".join(f"{x:.2f}" for x in d["delta_cos_tmpl_rows"])
            + f" | {d['delta_cos_caption_mean']:.2f}"
        )
    (args.root / "compare.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.root}/compare.json")


if __name__ == "__main__":
    main()
