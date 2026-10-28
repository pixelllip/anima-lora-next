"""Can tag directions added onto a name's rows stand in for the spelled-out tags?

For each character, ``N`` is the phase-0 name-only caption and ``T`` the same with the
appearance tags (``output/tests/<c>_phase0/manifest.json``). The teacher effect on the
one-step ``x̂₀`` is ``E = x̂₀(T) − x̂₀(N)``; an injection's is ``x̂₀(N + inject) −
x̂₀(N)``, read as ``r = ⟨E_s, E⟩/‖E‖²`` and cos. No training.

Injections add to the name's rows of ``N``, in the TE output (``te``) or the
``text_embedder`` output (``emb``), a direction built from ``T`` in the same space:

- ``mean``: every name row gets ``mean(T's added tag rows) − mean(N's base tag rows)``;
- ``split``: the added tags are dealt round-robin over the name rows, each row gets the
  mean of its tags' rows minus the same base mean;
- ``rand``: a random direction (control).

Each is scaled to ``ρ ×`` the mean name-row norm in that space.

``textonly``: the text rows through the 42 blocks without the image tokens, against the
joint forward (relative error per block, name rows and all text rows). If it is small,
the blocks' text path can be distilled without the image.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from entity_dit import log, span
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.train import NUM_TRAIN_TIMESTEPS, place, step_reserve_gb

CHARS = {
    "akita neru": "neru_phase0",
    "malfoid": "malfoid_phase0",
    "toujou koneko": "koneko_phase0",
}
RHOS = (0.25, 0.5, 1.0, 2.0)
SIGMAS = (0.9, 0.75)


def phase0(folder):
    """``(N, T, {"name": latent path, "tags": latent path})`` from a phase-0 manifest."""
    root = resolve_under_home(f"output/tests/{folder}")
    images = json.loads((root / "manifest.json").read_text())["images"]
    def by(cond):
        im = next(i for i in images if f"space{cond}_" in i["key"])
        return im["prompt"], root / "latents" / f"{Path(im['image']).stem}.safetensors"

    (n_text, n_lat), (t_text, t_lat) = by("A"), by("B")
    return n_text, t_text, {"name": n_lat, "tags": t_lat}


def added_tags(tok, n_text, t_text):
    """Token ranges in ``T`` of each tag ``T`` adds, and of ``N``'s base tags in ``N``."""
    base = json.loads(n_text)["summary"]["caption"]
    full = json.loads(t_text)["summary"]["caption"]
    at = full.find(base)
    assert at >= 0, (base, full)

    def formatted(text):
        f = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return f, tok(f, return_offsets_mapping=True).offset_mapping

    def rows(offs, a, b):
        return [i for i, (s, e) in enumerate(offs) if s < b and e > a]

    f_t, offs_t = formatted(t_text)
    cap_t = f_t.find(full, f_t.find('"caption"'))
    tags = []
    for part_start, part in ((0, full[:at]), (at + len(base), full[at + len(base) :])):
        pos = 0
        for tag in part.split(","):
            tag_start = part.find(tag, pos)
            pos = tag_start + len(tag)
            stripped = tag.strip()
            if not stripped:
                continue
            a = cap_t + part_start + tag_start + tag.find(stripped)
            tags.append((stripped, rows(offs_t, a, a + len(stripped))))
    f_n, offs_n = formatted(n_text)
    cap_n = f_n.find(base, f_n.find('"caption"'))
    base_rows = rows(offs_n, cap_n, cap_n + len(base))
    return tags, base_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_encoder", default="int8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", type=Path, default=Path("/media/sorryhyun/new/qwen35_sae/entity_add")
    )
    args = ap.parse_args()
    out_dir = args.out / time.strftime("%Y%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = loader.load_tokenizer(default_moss_dir())
    jobs = []
    for name, folder in CHARS.items():
        n_text, t_text, lat = phase0(folder)
        tags, base_rows = added_tags(tok, n_text, t_text)
        jobs.append(
            dict(
                name=name,
                N=n_text,
                T=t_text,
                lat=lat,
                tags=tags,
                base_rows=base_rows,
                name_span=span(tok, n_text, name, '"characters"'),
            )
        )
        log(
            f"{name}: {len(tags)} added tags, name rows {jobs[-1]['name_span']}, "
            f"{len(base_rows)} base rows"
        )
    enc = loader.encode_prompts(
        default_moss_dir(),
        [t for j in jobs for t in (j["N"], j["T"])],
        log=log,
        precision=args.text_encoder,
    )

    device = torch.device("cuda")
    transformer = loader.load_transformer(default_moss_dir(), "base")
    transformer.requires_grad_(False)
    longest = max(e.shape[0] for e in enc.values()) + 8
    place(
        transformer,
        device,
        -1,
        step_reserve_gb(76 * 52 + longest, grad_checkpointing=True),
    )

    state = {"emb_add": None, "emb_out": None, "text_rows": None}

    def emb_hook(_m, _i, out):
        state["emb_out"] = out[0].float().cpu()
        if state["emb_add"] is not None:
            rows, vec = state["emb_add"]
            out = out.clone()
            out[0, rows] += vec.to(out.device, out.dtype)
        return out

    transformer.text_embedder.register_forward_hook(emb_hook)

    def layer_hook(index, x, n_img):
        if state["text_rows"] is not None:
            state["text_rows"][index] = x[0, n_img:].float().cpu()
        return x

    @torch.no_grad()
    def x_hat(noisy, sigma, emb, emb_add=None):
        state["emb_add"] = emb_add
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        v = transformer(
            noisy, t, emb[None].to(device, torch.bfloat16), return_dict=False
        )[0].float()
        state["emb_add"] = None
        return noisy - sigma * v

    @torch.no_grad()
    def text_states(noisy, sigma, emb, text_only):
        """Per block, the text rows ``(S, D)`` of the joint or the text-only forward."""
        state["text_rows"] = {}
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        transformer.layer_hook = layer_hook
        if not text_only:
            transformer(noisy, t, emb[None].to(device, torch.bfloat16))
        else:
            mod, _ = transformer.time_conditioning(t, 1)
            x, rope = transformer.embed_sample(
                noisy, emb[None].to(device, torch.bfloat16)
            )
            n_img = noisy.shape[2] * noisy.shape[3]
            transformer.run_layers(x[:, n_img:], rope[:, n_img:], mod, n_img=0)
        transformer.layer_hook = None
        rows, state["text_rows"] = state["text_rows"], None
        return rows

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn((1, 128, 76, 52), generator=gen).to(device)
    rand = torch.randn(4096, generator=gen), torch.randn(3072, generator=gen)
    report = {}
    for j in jobs:
        a, b = j["name_span"]
        name_rows = list(range(a, b))
        rec = {"tags": [t for t, _ in j["tags"]], "conds": {}, "textonly": {}}
        n_emb, t_emb = enc[j["N"]], enc[j["T"]]
        # emb-space rows of N and T: one forward each at a throwaway x_t
        spaces = {"te": (n_emb.float(), t_emb.float())}
        x_hat(0.9 * noise, 0.9, n_emb)
        n_e = state["emb_out"]
        x_hat(0.9 * noise, 0.9, t_emb)
        spaces["emb"] = (n_e, state["emb_out"])

        injections = {}
        for sp, (n_rows, t_rows) in spaces.items():
            base = n_rows[j["base_rows"]].mean(0)
            scale = n_rows[name_rows].norm(dim=-1).mean()
            all_tag = [r for _, rs in j["tags"] for r in rs]
            mean_dir = t_rows[all_tag].mean(0) - base
            split = torch.zeros(len(name_rows), n_rows.shape[1])
            for k in range(len(name_rows)):
                mine = [
                    r for i, (_, rs) in enumerate(j["tags"]) if i % len(name_rows) == k for r in rs
                ]
                split[k] = t_rows[mine].mean(0) - base
            rnd = rand[0] if sp == "te" else rand[1]
            dirs = {
                "mean": mean_dir.expand(len(name_rows), -1),
                "split": split,
                "rand": rnd.expand(len(name_rows), -1),
            }
            for dname, d in dirs.items():
                unit = d / d.norm(dim=-1, keepdim=True)
                for rho in RHOS if dname != "rand" else (1.0,):
                    injections[f"{sp}/{dname}/ρ{rho}"] = (sp, rho * scale * unit)
            rec.setdefault("dir_norm_over_name_row", {})[sp] = float(
                mean_dir.norm() / scale
            )

        for x0_name, path in j["lat"].items():
            x0 = load_file(path)["latent"].to(device)
            for sigma in SIGMAS:
                noisy = (1 - sigma) * x0 + sigma * noise
                xn = x_hat(noisy, sigma, n_emb)
                teacher = x_hat(noisy, sigma, t_emb) - xn
                row = {"teacher_over_xhat": float(teacher.norm() / xn.norm())}
                for cname, (sp, add) in injections.items():
                    if sp == "te":
                        e = n_emb.clone()
                        e[name_rows] = (e[name_rows].float() + add).to(e.dtype)
                        eff = x_hat(noisy, sigma, e) - xn
                    else:
                        eff = x_hat(noisy, sigma, n_emb, (name_rows, add)) - xn
                    row[cname] = {
                        "r": float((eff * teacher).sum() / teacher.pow(2).sum()),
                        "cos": float(
                            torch.nn.functional.cosine_similarity(
                                eff.flatten(), teacher.flatten(), dim=0
                            )
                        ),
                        "norm_ratio": float(eff.norm() / teacher.norm()),
                    }
                rec["conds"][f"{x0_name}/σ{sigma}"] = row
                best = max(
                    (k for k in row if k != "teacher_over_xhat"),
                    key=lambda k: row[k]["cos"],
                )
                log(
                    f"[add] {j['name']} x0={x0_name} σ={sigma} ‖E‖/‖x̂₀‖={row['teacher_over_xhat']:.3f}  "
                    + "  ".join(
                        f"{k}={row[k]['r']:.2f}/{row[k]['cos']:.2f}"
                        for k in row
                        if k != "teacher_over_xhat"
                    )
                    + f"  best={best}"
                )

        x0 = load_file(j["lat"]["name"])["latent"].to(device)
        for sigma in (0.9, 0.6):
            noisy = (1 - sigma) * x0 + sigma * noise
            for cap in ("N", "T"):
                joint = text_states(noisy, sigma, enc[j[cap]], False)
                alone = text_states(noisy, sigma, enc[j[cap]], True)
                errs = {}
                for blk in sorted(joint):
                    jt, al = joint[blk], alone[blk]
                    errs[blk] = {
                        "name": float(
                            (al[name_rows] - jt[name_rows]).norm() / jt[name_rows].norm()
                        )
                        if cap == "N"
                        else None,
                        "all": float((al - jt).norm() / jt.norm()),
                    }
                rec["textonly"][f"{cap}/σ{sigma}"] = errs
                pick = [0, 8, 16, 24, 31, 41]
                log(
                    f"[textonly] {j['name']} {cap} σ={sigma} rel err (all rows"
                    + (", name rows" if cap == "N" else "")
                    + "): "
                    + "  ".join(
                        f"b{k}={errs[k]['all']:.3f}"
                        + (f"/{errs[k]['name']:.3f}" if cap == "N" else "")
                        for k in pick
                    )
                )
        report[j["name"]] = rec

    (out_dir / "summary.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    log(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
