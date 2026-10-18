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
)

SPEC_BY_PRESET = {
    "attention": KREA2_ATTENTION_ONLY_SPEC,
    "official": KREA2_LORA_TARGET_SPEC,
}
# `attn_mlp` / `official_gate` 在 krea2-webui 里是另外两档；这里先只放最常用的两档，
# 需要时从 lora_targets.py 追加（不许静默降级成别的档：见 resolve_spec）。
SUPPORTED_PRESETS = tuple(SPEC_BY_PRESET)


def resolve_spec(preset: str):
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
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def trainable_param_count(self) -> int:
        return self.lora_down.weight.numel() + self.lora_up.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.lora_up(self.lora_down(self.dropout(x))) * self.scale


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
                dropout: float = 0.0, holder_name: str = "_krea2_lora") -> LoRAHandles:
    """把 LoRA 挂到 DiT 上（参数常驻在 block 外的 holder 里；返回句柄）。"""
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
