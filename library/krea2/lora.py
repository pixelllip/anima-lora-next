"""Krea-2 line —— LoRA 注入器（**functional / 参数放在 block 外**）。

⚠ 为什么不是"把 nn.Linear 换成 LoRALinear 包一层"（第一版就是这么写的，撞了墙）：
块交换（`ModelOffloader`）会把 block 下**每一个 `.weight`** 在 CPU/GPU 之间搬来搬去
（它是按名字**成对交换两个 block 的 data**）。嵌在 block 里的可训权重于是会在优化器
`step()` 时出现"参数在卡上、`.grad` 在 CPU"——现象是
`Expected all tensors to be on the same device, cuda:0 and cpu`，而且**只在开块交换时出现**。
上游 `library/qwen21/CLAUDE.md` 与 `library/moss/CLAUDE.md` 都专门写了这条：
**adapter 要放在 block 外**。

所以这里：`lora_down` / `lora_up` 挂在一个**独立 holder**（挂在 DiT 根下，根不参与交换），
目标 Linear 的 `forward` 用闭包补上 `up(down(x)) * (alpha/rank)` —— 计算照旧，参数常驻显存。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

from library.models.krea2_raw.lora_targets import (
    KREA2_ATTENTION_ONLY_SPEC,
    KREA2_LORA_TARGET_SPEC,
    KREA2_OFFICIAL_GATE_SPEC,
)

# 档位表（与 krea2-webui 的 `lora_targets.py::_KREA2_SPEC_PRESETS` 同源）：
#   official_gate = **264** = 全部 Linear（官方 232 + 32 个 `mlp.gate`）—— 2026-10-13 起的口径，
#                   r16 下 58.65M 参数；gate 与 mlp.up 同形状**不引入新张量宽度**，
#                   所以 8GB 卡上 512 档 + swap26 实测 reserved 4.41–4.62GB 跑得动。
#   official      = 232（官方 DreamBooth 集），r16 下 47.95M —— 8GB 上 1024 档会越线换页。
#   attention     = 160（只训注意力投影），r16 下 25.03M —— 窄档，留给"就想试注意力学没学到"的场景。
#   attn_mlp      = 196 —— **已弃用**（2026-10-07 的中间档，被 official_gate 取代），不再接受。
SPEC_BY_PRESET = {
    "official_gate": KREA2_OFFICIAL_GATE_SPEC,
    "official": KREA2_LORA_TARGET_SPEC,
    "attention": KREA2_ATTENTION_ONLY_SPEC,
}
DEPRECATED_PRESETS = ("attn_mlp",)
SUPPORTED_PRESETS = tuple(SPEC_BY_PRESET)


def resolve_spec(preset: str):
    if preset in DEPRECATED_PRESETS:
        raise SystemExit(
            f"target_preset={preset!r} 已弃用（196 档的中间档）——"
            f" 改用 {SUPPORTED_PRESETS[0]!r}（264，含 mlp.gate）或 'official'（232）"
        )
    if preset not in SPEC_BY_PRESET:
        raise SystemExit(
            f"target_preset={preset!r} 暂不支持（已接：{list(SUPPORTED_PRESETS)}）——"
            " 不静默换成别的档，请显式选一个"
        )
    return SPEC_BY_PRESET[preset]


class LoRALinear(nn.Module):
    """独立的 rank-r 补丁：`y = up(down(x)) * (alpha / rank)`（零初始化 ⇒ 起步 ΔW=0）。"""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def trainable_param_count(self) -> int:
        return self.lora_down.weight.numel() + self.lora_up.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return (
            self.lora_up(self.lora_down(self.dropout(x)))
            * self.scale
            * self.multiplier
        )


@dataclass
class LoRAHandles:
    holder: nn.Module
    modules: dict[str, LoRALinear]
    patched: dict[str, object] = field(default_factory=dict)  # path -> 原 forward

    @property
    def param_count(self) -> int:
        return sum(m.trainable_param_count for m in self.modules.values())

    def parameters(self):
        for mod in self.modules.values():
            yield from mod.parameters()

    def set_multiplier(self, value: float) -> None:
        for mod in self.modules.values():
            mod.multiplier = float(value)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load normalized ``lora_unet_*`` tensors into this handle's modules.

        The file's module token is the dotted DiT path with dots replaced by
        underscores (``blocks_0_attn_wq``, ``txtfusion_refiner_blocks_0_attn_wo``),
        so match on that form instead of trying to reverse the underscores.
        """
        by_token = {path.replace(".", "_"): mod for path, mod in self.modules.items()}
        consumed = 0
        for key, value in state_dict.items():
            if not key.startswith("lora_unet_"):
                continue
            body = key[len("lora_unet_") :]
            for suffix, field_name in (
                (".lora_down.weight", "lora_down"),
                (".lora_up.weight", "lora_up"),
            ):
                if not body.endswith(suffix):
                    continue
                mod = by_token.get(body[: -len(suffix)])
                if mod is None:
                    continue
                target = getattr(mod, field_name).weight
                target.data.copy_(
                    value.to(device=target.device, dtype=target.dtype)
                )
                consumed += 1
                break
        if consumed == 0:
            raise SystemExit("no LoRA tensors matched this DiT's target modules")

    def state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for path, mod in self.modules.items():
            prefix = "lora_unet_" + path.replace(".", "_")
            out[f"{prefix}.lora_down.weight"] = mod.lora_down.weight.detach().cpu()
            out[f"{prefix}.lora_up.weight"] = mod.lora_up.weight.detach().cpu()
            out[f"{prefix}.alpha"] = torch.tensor(float(mod.alpha))
        return out


def _iter_linear(root: nn.Module, prefix: str = "") -> Iterable[tuple[str, nn.Linear]]:
    for name, child in root.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):  # bnb Linear4bit 也是 nn.Linear 子类
            yield path, child
        else:
            yield from _iter_linear(child, path)


def select_targets(dit: nn.Module, spec) -> list[tuple[str, nn.Linear]]:
    """按 spec 的容器白名单 + exclude 正则挑目标 Linear（语义与 targeting.py 一致）。"""
    excl = [re.compile(p) for p in spec.exclude_patterns]
    inc = [re.compile(p) for p in spec.include_patterns]
    picked: list[tuple[str, nn.Linear]] = []
    for path, lin in _iter_linear(dit):
        if path.startswith("_krea2_lora"):  # 别把自己的 holder 也扫进去
            continue
        if any(p.search(path) for p in excl) and not any(p.search(path) for p in inc):
            continue
        picked.append((path, lin))
    return picked


def attach_lora(dit: nn.Module, *, rank: int, alpha: float, preset: str,
                dropout: float = 0.0, holder_name: str = "_krea2_lora",
                dtype: torch.dtype | None = None) -> LoRAHandles:
    """把 LoRA 挂到 DiT 上（参数常驻在 block 外的 holder 里；返回句柄）。

    ⚠ `dtype` 必须给成**模型的 compute dtype（bf16）**：LoRA 权重留 fp32 时，
    `F.linear(bf16_x, fp32_w)` 会提升成 fp32，`_orig(x) + _pair(x)` 于是把**整个下游图**
    拖成 fp32 —— 激活翻倍、步时变慢（本机实测：1024 档前向峰值 4.56GB，而 krea2-webui
    同参数只有 3.47GB）。参数给 bf16、优化器状态让它自己按参数 dtype 走。
    """
    spec = resolve_spec(preset)
    holder = nn.Module()
    modules: dict[str, LoRALinear] = {}
    patched: dict[str, object] = {}

    for path, lin in select_targets(dit, spec):
        key = path.replace(".", "__")  # holder 里的子模块名（导出时另按 path 拼）
        pair = LoRALinear(lin.in_features, lin.out_features, rank, alpha, dropout)
        holder.add_module(key, pair)
        modules[path] = pair

        orig_forward = lin.forward

        def forward(x, _orig=orig_forward, _pair=pair, _path=path):
            return _orig(x) + _pair(x)

        lin.forward = forward  # type: ignore[method-assign]
        patched[path] = orig_forward

    if not modules:
        raise SystemExit("没扫到任何目标 Linear —— spec 与 DiT 的模块名对不上？")
    if dtype is not None:
        holder.to(dtype=dtype)
    dit.add_module(holder_name, holder)
    return LoRAHandles(holder=holder, modules=modules, patched=patched)


def normalize_lora_weights(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize ai-toolkit / PEFT keys to the repo's ``lora_unet_*`` layout."""
    out: dict[str, torch.Tensor] = {}
    suffixes = (
        (".lora_A.weight", ".lora_down.weight"),
        (".lora_B.weight", ".lora_up.weight"),
        (".lora_down.weight", ".lora_down.weight"),
        (".lora_up.weight", ".lora_up.weight"),
    )
    for key, value in raw.items():
        if key.endswith(".alpha"):
            module = key[: -len(".alpha")]
            if module.startswith("diffusion_model."):
                module = module[len("diffusion_model.") :]
            if module.startswith("lora_unet_"):
                module = module[len("lora_unet_") :]
            out[f"lora_unet_{module}.alpha"] = value
            continue
        for suffix, replacement in suffixes:
            if not key.endswith(suffix):
                continue
            module = key[: -len(suffix)]
            if module.startswith("diffusion_model."):
                module = module[len("diffusion_model.") :]
            if module.startswith("lora_unet_"):
                module = module[len("lora_unet_") :]
            # ``diffusion_model.blocks.0.attn.wq`` -> ``blocks_0_attn_wq``;
            # existing underscores (``layerwise_blocks``) stay intact.
            module = module.replace(".", "_")
            out[f"lora_unet_{module}{replacement}"] = value
            break
    down = ".lora_down.weight"
    for key in list(out):
        if not key.endswith(down):
            continue
        alpha_key = key[: -len(down)] + ".alpha"
        if alpha_key not in out:
            rank = out[key].shape[0]
            out[alpha_key] = torch.tensor(float(rank), dtype=torch.float32)
    return out


def attach_lora_from_file(
    dit: nn.Module,
    path: str,
    *,
    holder_name: str,
    dtype: torch.dtype | None = None,
) -> LoRAHandles:
    """Attach the LoRA stored at ``path`` without a preset (keys decide targets).

    Used for frozen style LoRAs during character training: third-party exports
    (ai-toolkit) do not carry the repo's target metadata, so matching the file's
    own module names to the DiT's Linear paths is more robust than a preset.
    """
    from safetensors.torch import load_file

    weights = normalize_lora_weights(load_file(path))
    by_token = {
        path_.replace(".", "_"): (path_, lin)
        for path_, lin in _iter_linear(dit)
        if not path_.startswith(holder_name)
    }

    holder = nn.Module()
    modules: dict[str, LoRALinear] = {}
    patched: dict[str, object] = {}
    for key, down_weight in list(weights.items()):
        if not key.endswith(".lora_down.weight"):
            continue
        module = key[len("lora_unet_") : -len(".lora_down.weight")]
        target = by_token.get(module)
        if target is None:
            continue
        path_, lin = target
        up_key = f"lora_unet_{module}.lora_up.weight"
        if up_key not in weights:
            continue
        rank = int(down_weight.shape[0])
        alpha = float(weights.get(f"lora_unet_{module}.alpha", rank))
        pair = LoRALinear(lin.in_features, lin.out_features, rank, alpha)
        pair.lora_down.weight.data.copy_(down_weight.to(pair.lora_down.weight.dtype))
        pair.lora_up.weight.data.copy_(weights[up_key].to(pair.lora_up.weight.dtype))
        holder.add_module(path_.replace(".", "__"), pair)
        modules[path_] = pair

        orig_forward = lin.forward

        def forward(x, _orig=orig_forward, _pair=pair):
            return _orig(x) + _pair(x)

        lin.forward = forward  # type: ignore[method-assign]
        patched[path_] = orig_forward

    if not modules:
        raise SystemExit(f"no LoRA tensors matched this DiT's modules: {path}")
    if dtype is not None:
        holder.to(dtype=dtype)
    dit.add_module(holder_name, holder)
    return LoRAHandles(holder=holder, modules=modules, patched=patched)


def detach_lora(dit: nn.Module, handles: LoRAHandles, holder_name: str = "_krea2_lora") -> None:
    """把 forward 还原、holder 摘掉（推理 / 换 target 时用）。"""
    for path, orig_forward in handles.patched.items():
        mod = dit
        *parents, leaf = path.split(".")
        for part in parents:
            mod = getattr(mod, part)
        setattr(mod, leaf, orig_forward)  # 原样写回绑定方法即可
    delattr(dit, holder_name)


def lora_parameters(handles: LoRAHandles):
    yield from handles.parameters()
