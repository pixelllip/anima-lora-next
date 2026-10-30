"""ΔW = PE − Qwen3.5-9B spectra under three truncation rules, scored on activations.

For every decoder linear (and lm_head) with input Gram ``C`` from ``calib.py``:

  plain    SVD of ΔW (Frobenius-optimal)
  asvd     SVD of ΔW·diag(√diag C) — the turbo extractor's ``--act_scales`` rule
  white    SVD of ΔW·C^½ — optimal for E‖ΔW x‖² on the calibration activations

Each rule's rank-r truncation is scored by the same functional metric, the share of
``E‖ΔW x‖² = tr(ΔW C ΔWᵀ)`` it keeps. Every rule is evaluated through the in×in
Gram ``ΔWᵀΔW`` (eigh), so lm_head (248320 × 4096) costs the same as an MLP.
Per-module arrays go to ``spectra.npz``; ``summary.json`` holds the uniform-rank
and global-budget tables.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

GROUP_OF = {
    "self_attn.q_proj": "attn_in",
    "self_attn.k_proj": "attn_in",
    "self_attn.v_proj": "attn_in",
    "linear_attn.in_proj_qkv": "attn_in",
    "linear_attn.in_proj_z": "attn_in",
    "linear_attn.in_proj_a": "attn_in",
    "linear_attn.in_proj_b": "attn_in",
    "self_attn.o_proj": "attn_out",
    "linear_attn.out_proj": "attn_out",
    "mlp.gate_proj": "mlp_in",
    "mlp.up_proj": "mlp_in",
    "mlp.down_proj": "mlp_down",
}
PREFIX = "model.language_model."
RANKS = (16, 32, 64, 128, 256, 512, 1024)


class Ckpt:
    def __init__(self, d: Path):
        self.d = d
        self.wm = json.loads((d / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.handles = {}

    def _h(self, key):
        shard = self.wm[key]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.d / shard), "pt", device="cpu")
        return self.handles[shard]

    def get(self, key):
        return self._h(key).get_tensor(key)

    def get_slice(self, key):
        return self._h(key).get_slice(key)


def sym_sqrt(c):
    lam, q = torch.linalg.eigh(c)
    return (q * lam.clamp(min=0).sqrt()) @ q.T


def grams(pe, base, key, c, chunk=16384):
    """Row-chunked ``ΔWᵀΔW``, ``tr(W C Wᵀ)``, ``‖W‖²`` (lm_head does not fit whole)."""
    n_rows = pe.get_slice(key).get_shape()[0]
    g = e_base = f_base = 0
    for a in range(0, n_rows, chunk):
        w0 = base.get_slice(key)[a : a + chunk].cuda().float()
        dw = pe.get_slice(key)[a : a + chunk].cuda().float() - w0
        g = g + dw.T @ dw
        e_base += float(((w0 @ c) * w0).sum())
        f_base += float(w0.square().sum())
    return g, e_base, f_base


def spectra(g, c, c_half):
    """Per-component functional energies, each rule's components sorted by its own
    singular value (descending)."""
    out = {}
    # plain: eig of ΔWᵀΔW → s², v; functional weight s²·vᵀCv
    s2, v = torch.linalg.eigh(g)
    s2, v = s2.flip(0).clamp(min=0), v.flip(1)
    out["plain"] = s2 * ((v.T @ c) * v.T).sum(1)
    out["plain_frob"] = s2
    # asvd: D = √diag C (clamped like extract_delta_lora.svd_truncate)
    d = c.diagonal().clamp(min=0).sqrt()
    d = d.clamp(min=1e-3 * d.mean())
    s2a, va = torch.linalg.eigh(d[:, None] * g * d[None, :])
    s2a, va = s2a.flip(0).clamp(min=0), va.flip(1)
    vd = va / d[:, None]
    out["asvd"] = s2a * ((vd.T @ c) * vd.T).sum(1)
    # whitened: eig of C½ ΔWᵀΔW C½ = σ²
    out["white"] = torch.linalg.eigvalsh(c_half @ g @ c_half).flip(0).clamp(min=0)
    return out


def capture(w, total, r):
    return float(w[:r].sum() / total) if total > 0 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--base_dir", type=Path, required=True)
    ap.add_argument("--calib_dir", type=Path, required=True)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    pe, base = Ckpt(args.pe_dir), Ckpt(args.base_dir)
    cov_dir = args.calib_dir / "cov"
    names = sorted(
        {
            k[len(PREFIX) : -len(".weight")]
            for k in pe.wm
            if k.startswith(PREFIX + "layers.")
            and k.endswith(".weight")
            and k[len(PREFIX) :].split(".", 2)[2][: -len(".weight")] in GROUP_OF
        },
        key=lambda n: (int(n.split(".")[1]), n),
    )
    log(f"[analyze] {len(names)} linears")
    rec = {}
    t0 = time.time()
    cur_layer, covs = None, {}
    with safe_open(str(cov_dir / "final.safetensors"), "pt") as f:
        n_tok = int(f.metadata()["tokens"])
    jobs = [(n, int(n.split(".")[1]), GROUP_OF[n.split(".", 2)[2]]) for n in names]
    jobs.append(("lm_head", None, "final"))
    for name, layer, group in jobs:
        if layer != cur_layer:
            covs.clear()
            cur_layer = layer
            path = cov_dir / (
                "final.safetensors"
                if layer is None
                else f"layer{layer:02d}.safetensors"
            )
            with safe_open(str(path), "pt", device="cuda") as f:
                for k in f.keys():
                    c = f.get_tensor(k) / n_tok
                    covs[k] = (c, sym_sqrt(c))
        c, c_half = covs[group]
        key = "lm_head.weight" if layer is None else f"{PREFIX}{name}.weight"
        g, e_base, f_base = grams(pe, base, key, c)
        sp = spectra(g, c, c_half)
        e_delta = float((c * g).sum())
        shape = pe.get_slice(key).get_shape()
        rec[name] = {
            "shape": list(shape),
            "e_base": e_base,
            "e_delta": e_delta,
            "frob_rel": float(np.sqrt(float(g.trace()) / f_base)),
            **{k: v.cpu().numpy() for k, v in sp.items()},
        }
        r = rec[name]
        log(
            f"[{time.time() - t0:5.0f}s] {name:38s} {tuple(shape)} "
            f"func_rel={np.sqrt(e_delta / e_base):.3f} "
            + " ".join(
                f"r128 {m}={capture(r[m], e_delta, 128):.2f}"
                for m in ("plain", "asvd", "white")
            )
        )
        del g
    # non-linear params: relative deltas only
    small = {}
    for k in pe.wm:
        if k.startswith("model.visual") or k in ("lm_head.weight",):
            continue
        n = k[len(PREFIX) :][: -len(".weight")] if k.endswith(".weight") else None
        if n in rec:
            continue
        a, b = pe.get(k).float(), base.get(k).float()
        small[k] = {
            "numel": a.numel(),
            "rel": float((a - b).norm() / b.norm().clamp(min=1e-12)),
        }
    out = args.calib_dir
    np.savez(
        out / "spectra.npz",
        **{
            f"{n}|{k}": v
            for n, r in rec.items()
            for k, v in r.items()
            if isinstance(v, np.ndarray)
        },
    )
    summary = summarize(rec)
    summary["tokens"] = n_tok
    summary["small_params"] = small
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: v for k, v in summary.items() if k != "small_params"}, indent=1))


def summarize(rec):
    names = [n for n in rec if n != "lm_head"]
    uniform = {}
    for m in ("plain", "asvd", "white"):
        uniform[m] = {}
        for r in RANKS:
            caps = [capture(rec[n][m], rec[n]["e_delta"], r) for n in names]
            uniform[m][r] = {
                "mean": float(np.mean(caps)),
                "min": float(np.min(caps)),
                "energy_weighted": float(
                    sum(rec[n][m][:r].sum() for n in names)
                    / sum(rec[n]["e_delta"] for n in names)
                ),
            }
    by_kind = {}
    for n in names:
        kind = re.sub(r"^layers\.\d+\.", "", n)
        by_kind.setdefault(kind, []).append(n)
    kinds = {
        kind: {
            "n": len(ns),
            "func_rel": float(
                np.mean([np.sqrt(rec[n]["e_delta"] / rec[n]["e_base"]) for n in ns])
            ),
            **{
                f"white_r{r}": float(
                    np.mean(
                        [capture(rec[n]["white"], rec[n]["e_delta"], r) for n in ns]
                    )
                )
                for r in (64, 128, 256)
            },
        }
        for kind, ns in by_kind.items()
    }
    # global budget: components ranked by σ²/E_base (relative output perturbation),
    # each costing (in + out) params; budgets matched to uniform rank r
    comps = []
    for i, n in enumerate(names):
        o, k = rec[n]["shape"]
        w = rec[n]["white"]
        comps.append((w / rec[n]["e_base"], np.full(len(w), o + k), np.full(len(w), i)))
    gain = np.concatenate([c[0] for c in comps])
    cost = np.concatenate([c[1] for c in comps])
    owner = np.concatenate([c[2] for c in comps])
    order = np.argsort(-gain)
    csum = np.cumsum(cost[order])
    budget = {}
    for r in (32, 64, 128, 256):
        params = sum(r * sum(rec[n]["shape"]) for n in names)
        take = order[: int(np.searchsorted(csum, params, side="right"))]
        kept = np.zeros(len(names))
        np.add.at(kept, owner[take], gain[take])
        ranks = np.bincount(owner[take], minlength=len(names))
        caps = [
            kept[i] * rec[n]["e_base"] / rec[n]["e_delta"] for i, n in enumerate(names)
        ]
        budget[r] = {
            "params_M": params / 1e6,
            "mean": float(np.mean(caps)),
            "min": float(np.min(caps)),
            "rank_min": int(ranks.min()),
            "rank_median": int(np.median(ranks)),
            "rank_max": int(ranks.max()),
        }
    lm = rec["lm_head"]
    return {
        "uniform": uniform,
        "global_budget_white": budget,
        "by_kind": kinds,
        "lm_head": {
            "func_rel": float(np.sqrt(lm["e_delta"] / lm["e_base"])),
            "frob_rel": lm["frob_rel"],
            **{
                f"white_r{r}": capture(lm["white"], lm["e_delta"], r)
                for r in (64, 128, 256)
            },
        },
    }


if __name__ == "__main__":
    main()
