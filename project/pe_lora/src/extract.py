"""Whitened LoRA extraction of ΔW = PE − Qwen3.5-9B, top ``--rank`` components.

With ``C`` the calibration input Gram (``calib.py``) and ``A = ΔW·C^½ = UΣVᵀ``, the
rank-r truncation optimal for ``E‖ΔW x‖²`` is ``U_r U_rᵀ ΔW``. It is formed from the
in×in eig of ``AᵀA = C^½ (ΔWᵀΔW) C^½`` alone (no ``C^-½``, so no damping):

  lora_up   = ΔW C^½ V_r Σ_r^-½          (out, r)
  lora_down = Σ_r^-3/2 V_rᵀ C^½ ΔWᵀΔW    (r, in)

Components are stored in descending σ, so any prefix ``[:, :k]`` / ``[:k]`` is the
optimal rank-k extraction — the eval truncates one file to every rank it tests.
Keys are the checkpoint's: ``<module>.lora_up`` / ``<module>.lora_down`` for every
decoder linear and ``lm_head``; ``small.<key>`` holds PE's full value for every
other language-model tensor except ``embed_tokens`` (norms, GDN conv1d / A_log /
dt_bias). Basis: bf16 Qwen3.5-9B, unsmoothed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from analyze import GROUP_OF, PREFIX, Ckpt, grams, sym_sqrt


def factors(pe, base, key, c, c_half, rank, chunk=16384):
    g, _, _ = grams(pe, base, key, c, chunk)
    e_delta = float((c * g).sum())
    s2, v = torch.linalg.eigh(c_half @ g @ c_half)
    s2, v = s2.flip(0), v.flip(1)
    r = int(min(rank, (s2 > s2[0] * 1e-10).sum()))
    s2, v = s2[:r], v[:, :r]
    cv = c_half @ v  # (in, r)
    down = (s2.pow(-0.75)[:, None] * (cv.T @ g)).bfloat16().cpu()
    n_rows = pe.get_slice(key).get_shape()[0]
    up = []
    for a in range(0, n_rows, chunk):
        dw = (
            pe.get_slice(key)[a : a + chunk].cuda().float()
            - base.get_slice(key)[a : a + chunk].cuda().float()
        )
        up.append((dw @ cv * s2.pow(-0.25)[None, :]).bfloat16().cpu())
    capture = float(s2.sum() / e_delta)
    return torch.cat(up), down, capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--base_dir", type=Path, required=True)
    ap.add_argument("--calib_dir", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)  # noqa: E731
    pe, base = Ckpt(args.pe_dir), Ckpt(args.base_dir)
    cov_dir = args.calib_dir / "cov"
    with safe_open(str(cov_dir / "final.safetensors"), "pt") as f:
        n_tok = int(f.metadata()["tokens"])
    linears = sorted(
        (
            k[: -len(".weight")]
            for k in pe.wm
            if k.startswith(PREFIX + "layers.")
            and k.endswith(".weight")
            and k[len(PREFIX) :].split(".", 2)[2][: -len(".weight")] in GROUP_OF
        ),
        key=lambda n: (int(n[len(PREFIX) :].split(".")[1]), n),
    )
    jobs = [
        (
            n,
            int(n[len(PREFIX) :].split(".")[1]),
            GROUP_OF[n[len(PREFIX) :].split(".", 2)[2]],
        )
        for n in linears
    ] + [("lm_head", None, "final")]
    out, caps = {}, {}
    cur, covs = "unset", {}
    t0 = time.time()
    for name, layer, group in jobs:
        if layer != cur:
            covs.clear()
            cur = layer
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
        up, down, cap = factors(pe, base, f"{name}.weight", c, c_half, args.rank)
        out[f"{name}.lora_up"], out[f"{name}.lora_down"] = up, down
        caps[name] = cap
        log(f"[{time.time() - t0:5.0f}s] {name} r={up.shape[1]} capture={cap:.3f}")
    n_small = 0
    for k in pe.wm:
        if not k.startswith(PREFIX) or "embed_tokens" in k:
            continue
        if k[: -len(".weight")] in caps:
            continue
        out[f"small.{k}"] = pe.get(k)
        n_small += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        out,
        str(args.out),
        metadata={
            "basis": "Qwen/Qwen3.5-9B bf16 (unsmoothed)",
            "target": "Qwen/Qwen-Image-2.1-PE-T2I",
            "rule": "whitened (calibration Gram), descending sigma",
            "rank": str(args.rank),
            "capture": json.dumps(caps),
        },
    )
    log(f"[extract] {len(caps)} modules, {n_small} small tensors -> {args.out}")


if __name__ == "__main__":
    main()
