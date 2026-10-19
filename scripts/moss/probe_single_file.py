"""Probe: can the ComfyUI single-file int8 MOSS weights be mapped onto the
diffusers-style modules the trainer builds? Read-only; prints key diffs and one
dequantized layer's stats.

    python scripts/moss/probe_single_file.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

DIT = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models\moss_image_sft_int8_convrot.safetensors")
TE = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\text_encoders\qwen3.5_9b_w4a8_gptq.safetensors")
VAE = Path(r"D:\Comfy-Desktop\ComfyUI-Shared\models\vae\flux2-vae_small.safetensors")
BASE = Path(r"D:\myProject\anima-lora-next\models\moss_image\MOSS-Image-Base")


def load_cfg(p: Path) -> dict:
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def probe_dit() -> None:
    from library.moss.model import MOSSImageTransformer

    with torch.device("meta"):
        model = MOSSImageTransformer.from_config(load_cfg(BASE / "transformer" / "config.json"))
    expected = set(model.state_dict())
    with safe_open(str(DIT), framework="pt") as f:
        keys = set(f.keys())
        aux = {
            k
            for k in keys
            if k.endswith("comfy_quant")
            or k.endswith("weight_scale")
            or k.endswith("comfy_attention.config")
            or k.endswith(".qkv.weight")
        }
        base = keys - aux
        qkv = sorted(k for k in keys if k.endswith(".qkv.weight"))
        print(f"[dit] file={len(keys)} aux={len(aux)} qkv={len(qkv)}")
        missing = expected - base
        extra = base - expected
        print(f"[dit] missing={len(missing)} extra={len(extra)}")
        if missing:
            print("  missing sample:", sorted(missing)[:8])
            print("  missing tail:", sorted(missing)[-6:])
        print("  qkv keys:", qkv[:3], "…", qkv[-3:])
        if extra:
            print("  extra sample:", sorted(extra)[:8])
        q = f.get_tensor("layers.0.proj.qkv.weight")
        s = f.get_tensor("layers.0.proj.qkv.weight_scale")
        print(f"[dit] qkv {tuple(q.shape)} {q.dtype} scale {tuple(s.shape)} {s.dtype}")
        qs = torch.chunk(q, 3, dim=0)
        ss = torch.chunk(s, 3, dim=0)
        w = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(qs[0], ss[0], 256, 2)
        print(f"[dit] dequant q {tuple(w.shape)} {w.dtype} mean={w.float().mean():.5f} std={w.float().std():.5f}")


def probe_te() -> None:
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(BASE / "text_encoder")
    with torch.device("meta"):
        te = AutoModel.from_config(cfg)
    expected = set(te.state_dict())
    with safe_open(str(TE), framework="pt") as f:
        keys = set(f.keys())
        print(f"[te] {type(te).__name__} model_keys={len(expected)} file_keys={len(keys)}")
        print("  model sample:", sorted(expected)[:6])
        print("  file  sample:", sorted(keys)[:6])
        quant = {k for k in keys if k.endswith(".weight_s_rel")}
        print(f"  quantized linears={len(quant)}")
        print(f"  has embed scale: {'model.language_model.embed_tokens.weight_scale' in keys}")
        p = "model.language_model.layers.0.mlp.down_proj"
        w = f.get_tensor(p + ".weight")
        s_rel = f.get_tensor(p + ".weight_s_rel")
        s_ch = f.get_tensor(p + ".weight_s_channel")
        cb = f.get_tensor(p + ".weight_codebook")
        print(f"[te] packed {tuple(w.shape)} {w.dtype} s_rel {tuple(s_rel.shape)} {s_rel.dtype} s_ch {tuple(s_ch.shape)} cb {tuple(cb.shape)}")
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

        dw = dequantize_w4a8_int8_weight(
            w, s_rel, s_ch, cb, None, 16, 256, torch.bfloat16
        )
        print(f"[te] dequant {tuple(dw.shape)} {dw.dtype} mean={dw.float().mean():.5f} std={dw.float().std():.5f}")


def probe_vae() -> None:
    from diffusers import AutoencoderKLFlux2

    cfg = load_cfg(BASE / "vae" / "config.json")
    with torch.device("meta"):
        vae = AutoencoderKLFlux2.from_config(cfg)
    expected = set(vae.state_dict())
    with safe_open(str(VAE), framework="pt") as f:
        keys = set(f.keys())
    print(f"[vae] model_keys={len(expected)} file_keys={len(keys)} missing={len(keys - expected)} extra={len(expected - keys)}")
    if expected - keys:
        print("  extra sample:", sorted(expected - keys)[:5])
    if keys - expected:
        print("  missing sample:", sorted(keys - expected)[:5])


def main() -> None:
    import comfy_kitchen  # noqa: F401

    probe_dit()
    probe_te()
    probe_vae()


if __name__ == "__main__":
    main()
