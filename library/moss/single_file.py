"""Load MOSS-Image from ComfyUI's single-file quantized checkpoints.

The trainer normally reads a diffusers folder (``transformer/`` + ``text_encoder/``
+ ``vae/``). Those folders are the 27 GB bf16 release; the machine here only has
ComfyUI's quantized single files. This module maps them onto the same modules so
the int8 / w4a8 weights stay quantized in RAM and dequantize per forward:

* DiT  ``moss_image_*_int8_convrot.safetensors`` — int8 + per-row scale + a
  group-wise Hadamard rotation (``comfy_quant`` says ``convrot``). The weight is
  kept as an int8 ``nn.Parameter`` so the existing block swapper moves 113 MB per
  block instead of 226 MB; ``forward`` calls ``comfy_kitchen``'s dequant op.
* TE   ``qwen3.5_9b_w4a8_gptq.safetensors`` — packed w4a8 GPTQ linears, same idea
  (``--no-grad`` cache only).
* VAE  ``flux2-vae*.safetensors`` — plain bf16, just a state_dict load.

Everything is derived from the checkpoints + the small ``config.json`` files from
the HF release (no 27 GB download).
"""

from __future__ import annotations

import json
from pathlib import Path

import comfy_kitchen  # noqa: F401  (registers torch.ops.comfy_kitchen.*)
import torch
from torch import nn
from torch.nn import functional as F

_DTYPE_CODES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _dtype_code(dtype: torch.dtype) -> int:
    return _DTYPE_CODES.get(dtype, 2)


def _read_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- #
# quantized Linear shells
# --------------------------------------------------------------------------- #
class Int8ConvRotLinear(nn.Linear):
    """Frozen int8 weight (ComfyUI ``int8_tensorwise`` + ``convrot``) as a Linear.

    ``weight`` is an int8 Parameter — not a buffer — because
    ``library/runtime/device.py::weighs_to_device`` only moves ``.weight`` and the
    block swapper matches on ``module.weight.shape``. A buffer would keep all
    5 GiB on the card and defeat ``blocks_to_swap``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        convrot_groupsize: int = 256,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias is not None)
        self.weight = nn.Parameter(qdata.to(torch.int8).contiguous(), requires_grad=False)
        self.register_buffer("weight_scale", scale.to(torch.float32).contiguous())
        self.convrot_groupsize = int(convrot_groupsize)
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)

    def dequantized_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        weight = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            self.weight, self.weight_scale, self.convrot_groupsize, _dtype_code(dtype)
        )
        return weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.is_floating_point() else torch.bfloat16
        return F.linear(x, self.dequantized_weight(dtype), self.bias)


class W4A8Linear(nn.Linear):
    """Frozen packed w4a8 GPTQ weight (ComfyUI's Qwen3.5 text encoder)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qdata: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        codebook: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        group_size: int = 16,
        convrot_groupsize: int = 256,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias is not None)
        self.weight = nn.Parameter(qdata.contiguous(), requires_grad=False)
        self.register_buffer("weight_s_rel", s_rel.contiguous())
        self.register_buffer("weight_s_channel", s_channel.contiguous())
        self.register_buffer(
            "weight_codebook",
            codebook.contiguous() if codebook is not None else torch.empty(0),
        )
        self.group_size = int(group_size)
        self.convrot_groupsize = int(convrot_groupsize)
        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)

    def dequantized_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        from comfy_kitchen.tensor.w4a8_int8 import dequantize_w4a8_int8_weight

        codebook = self.weight_codebook if self.weight_codebook.numel() else None
        return dequantize_w4a8_int8_weight(
            self.weight,
            self.weight_s_rel,
            self.weight_s_channel,
            codebook,
            None,
            self.group_size,
            self.convrot_groupsize,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype if x.is_floating_point() else torch.bfloat16
        return F.linear(x, self.dequantized_weight(dtype), self.bias)


# --------------------------------------------------------------------------- #
# DiT
# --------------------------------------------------------------------------- #
def _quant_json(raw: torch.Tensor) -> dict:
    return json.loads(bytes(raw.flatten().tolist()).decode("utf-8"))


def load_dit_single_file(path: str | Path, config_path: str | Path) -> nn.Module:
    """``MOSSImageTransformer`` with its 168 int8 linears kept quantized."""
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from library.moss.model import MOSSImageTransformer

    cfg = _read_config(config_path)
    num_layers = int(cfg["num_layers"])

    with init_empty_weights():
        model = MOSSImageTransformer.from_config(cfg)
    model.to(torch.bfloat16)

    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        for layer in range(num_layers):
            prefix = f"layers.{layer}.proj."
            block = model.layers[layer].proj

            qkv = f.get_tensor(prefix + "qkv.weight")
            qkv_scale = f.get_tensor(prefix + "qkv.weight_scale")
            q_parts = torch.chunk(qkv, 3, dim=0)
            s_parts = torch.chunk(qkv_scale, 3, dim=0)
            for name, qdata, scale in zip(("q", "k", "v"), q_parts, s_parts):
                block[name] = Int8ConvRotLinear(
                    int(qdata.shape[1]), int(qdata.shape[0]), qdata, scale
                )
                state[f"layers.{layer}.proj.{name}.weight"] = qdata
                state[f"layers.{layer}.proj.{name}.weight_scale"] = scale

            for name in ("o", "ffn_up_gate", "ffn_down"):
                qdata = f.get_tensor(prefix + name + ".weight")
                scale = f.get_tensor(prefix + name + ".weight_scale")
                block[name] = Int8ConvRotLinear(
                    int(qdata.shape[1]), int(qdata.shape[0]), qdata, scale
                )
                state[prefix + name + ".weight"] = qdata
                state[prefix + name + ".weight_scale"] = scale

        for key in keys:
            if key.endswith((".comfy_quant", ".comfy_attention.config")):
                continue
            if key.startswith("layers.") and ".proj.qkv." in key:
                continue
            value = f.get_tensor(key)
            if value.dtype == torch.uint8 and key.endswith(".weight"):
                # only the fused qkv block above is uint8; anything else here is a bug
                raise ValueError(f"unexpected packed weight without a split: {key}")
            state[key] = value

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    bad_missing = [k for k in missing if "rope.bands" not in k]
    if bad_missing:
        raise RuntimeError(f"DiT checkpoint is missing {len(bad_missing)} keys: {bad_missing[:8]}")
    if unexpected:
        raise RuntimeError(f"DiT checkpoint has unexpected keys: {sorted(unexpected)[:8]}")
    return model.eval()


# --------------------------------------------------------------------------- #
# text encoder
# --------------------------------------------------------------------------- #
def load_te_single_file(
    path: str | Path,
    config_path: str | Path,
) -> nn.Module:
    """``Qwen3_5Model`` with 200 w4a8 linears kept packed; embeddings dequantized."""
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(str(config_path))
    with init_empty_weights():
        model = AutoModel.from_config(cfg)
    model.to(torch.bfloat16)

    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        quantized = sorted(k[: -len(".weight_s_rel")] for k in keys if k.endswith(".weight_s_rel"))
        for module_path in quantized:
            bare = module_path.removeprefix("model.")
            qdata = f.get_tensor(module_path + ".weight")
            s_rel = f.get_tensor(module_path + ".weight_s_rel")
            s_channel = f.get_tensor(module_path + ".weight_s_channel")
            codebook = (
                f.get_tensor(module_path + ".weight_codebook")
                if module_path + ".weight_codebook" in keys
                else None
            )
            quant = (
                _quant_json(f.get_tensor(module_path + ".comfy_quant"))
                if module_path + ".comfy_quant" in keys
                else {}
            )
            bias_key = module_path + ".bias"
            bias = f.get_tensor(bias_key) if bias_key in keys else None
            in_features = int(qdata.shape[1]) * 2
            out_features = int(qdata.shape[0])
            module = W4A8Linear(
                in_features,
                out_features,
                qdata,
                s_rel,
                s_channel,
                codebook,
                bias,
                group_size=int(quant.get("group_size", 16)),
                convrot_groupsize=int(quant.get("convrot_groupsize", 256)),
            )
            parent_path, _, attr = bare.rpartition(".")
            parent = model.get_submodule(parent_path)
            setattr(parent, attr, module)
            state[bare + ".weight"] = qdata
            state[bare + ".weight_s_rel"] = s_rel
            state[bare + ".weight_s_channel"] = s_channel
            state[bare + ".weight_codebook"] = (
                codebook if codebook is not None else torch.empty(0)
            )
            if bias is not None:
                state[bare + ".bias"] = bias

        # int8 convrot embedding: dequantize the whole table once (bf16 ~2 GB)
        emb_key = "model.language_model.embed_tokens"
        if emb_key + ".weight_scale" in keys:
            qdata = f.get_tensor(emb_key + ".weight")
            scale = f.get_tensor(emb_key + ".weight_scale")
            quant = _quant_json(f.get_tensor(emb_key + ".comfy_quant"))
            emb = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                qdata, scale, int(quant.get("convrot_groupsize", 256)), 2
            )
            state["language_model.embed_tokens.weight"] = emb.to(torch.bfloat16)

        embed_prefix = "model.language_model.embed_tokens."
        for key in keys:
            if key.startswith(("lm_head.", "mtp.")):
                continue
            if key.startswith(embed_prefix) and key != embed_prefix + "weight":
                continue
            if key.endswith((".comfy_quant", ".weight_s_rel", ".weight_s_channel", ".weight_codebook")):
                continue
            if key.endswith(".weight") and key[: -len(".weight")] in quantized:
                continue
            if key.startswith("model."):
                bare = key[len("model.") :]
            else:
                continue
            if bare == "language_model.embed_tokens.weight":
                continue
            state[bare] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"TE checkpoint is missing {len(missing)} keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"TE checkpoint has unexpected keys: {sorted(unexpected)[:8]}")
    return model.eval()


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #
def load_vae_single_file(path: str | Path, config_path: str | Path) -> nn.Module:
    """``AutoencoderKLFlux2`` from a ComfyUI single file (keys are identical)."""
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file

    cfg = _read_config(config_path)
    vae = AutoencoderKLFlux2.from_config(cfg)
    sd = load_file(str(path))
    vae.load_state_dict(sd, strict=True)
    return vae.eval()
