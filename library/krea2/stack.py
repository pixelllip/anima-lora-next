"""Krea-2 line —— style-in-the-loop 叠加调度（训练时挂冻结画风 LoRA）。

人物 LoRA 单独训练时只见过 base 特征；叠画风后画风改写 attention/MLP 输入，
人物残差落不到原特征上（鳍耳、瞳色、尾巴等离散细节丢失）。这里把冻结画风
LoRA 加进来，让人物残差直接在画风特征上优化。

为什么 ``pinned`` 才是正解
--------------------------
数据集只有一种画风、caption 又不描述画风，于是「身份」和「画风」对 loss 是同一团
没被解释的残差。冻结画风 LoRA 决定这团残差里的画风成分要不要由人物 LoRA 承担：

* ``prob < 1``：没挂画风的那些步，人物 delta 必须自己补画风；挂着的步再补就是
  重复计算。最优解是折中 ``Δ_char ≈ α·画风 + 身份``（α>0）。推理时 α 随强度一起涨
  ⟹ 「开 1 压画风、开低丢细节」。
* ``mode="pinned"``：画风每步都在、强度钉死在实际出图那一档，人物 delta 里的画风
  成分纯粹是重复计算，梯度把它压掉 —— 只留画风 LoRA 解释不了的部分。
  多条画风各按自己的强度常挂，delta 还要对「挂的是哪条画风」不敏感。

与 krea2-webui 的 `library/training/stack_lora.py` 同一语义；本仓是独立
Krea-2 line，直接用 `library/krea2/lora.py` 的 handle/foward 链。
"""

from __future__ import annotations

import random
import re

import torch

from library.krea2.lora import LoRAHandles, attach_lora_from_file

STACK_MODES = ("cycle", "random", "independent", "pinned")

_STRENGTH_SPLIT_RE = re.compile(r"[;,\s]+")


def split_stack_paths(value: str | None) -> list[str]:
    if not value:
        return []
    parts: list[str] = []
    for chunk in str(value).replace(";", ",").replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def parse_stack_strengths(value) -> tuple[float, ...] | None:
    """``"1.25;1.0"`` / ``["1.25", "1.0"]`` → ``(1.25, 1.0)``."""
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [p for p in _STRENGTH_SPLIT_RE.split(value) if p]
    else:
        parts = [str(p) for p in value if str(p).strip()]
    if not parts:
        return None
    strengths = tuple(float(p) for p in parts)
    if any(s <= 0.0 for s in strengths):
        raise ValueError(f"stack_lora_strengths 必须 > 0（收到 {strengths}）")
    return strengths


def stack_enabled(
    *, paths: list[str], prob: float, mode: str, strengths: tuple[float, ...] | None
) -> bool:
    """pinned 自带开关语义；其余模式仍要求 ``prob > 0``。"""
    if not paths:
        return False
    if mode == "pinned":
        return True
    return float(prob) > 0.0


def resolve_stack_strengths(
    paths: list[str], raw, *, mode: str
) -> tuple[float, ...] | None:
    """校验逐条强度：条数必须与路径一一对应。"""
    strengths = parse_stack_strengths(raw)
    if strengths is not None and len(strengths) != len(paths):
        raise SystemExit(
            "stack_lora_strengths 必须与 stack_lora_paths 一一对应"
            f"（{len(strengths)} 个强度 vs {len(paths)} 条路径）"
        )
    if mode == "pinned" and strengths is None:
        raise SystemExit(
            'stack_lora_mode="pinned" 需要 stack_lora_strengths'
            '（例如 "1.25;1.0"，就是你的出图强度）'
        )
    return strengths


def build_stack_loras(
    dit: torch.nn.Module,
    paths: list[str],
    *,
    dtype: torch.dtype | None = torch.bfloat16,
    device: torch.device | str | None = None,
) -> list[LoRAHandles]:
    """Attach each style LoRA as a frozen additive patch (multiplier 0)."""
    handles: list[LoRAHandles] = []
    for index, path in enumerate(paths):
        handle = attach_lora_from_file(
            dit,
            path,
            holder_name=f"_krea2_stack_{index}",
            dtype=dtype,
        )
        handle.freeze()
        handle.set_multiplier(0.0)
        if device is not None:
            handle.holder.to(device=device)
        handles.append(handle)
    return handles


def _strength_for(
    index: int,
    rng: random.Random,
    *,
    strengths: tuple[float, ...] | None,
    strength_min: float,
    strength_max: float,
    jitter: float,
) -> float:
    if strengths is not None:
        base = float(strengths[index])
        if jitter <= 0.0:
            return base
        return base * (1.0 + rng.uniform(-jitter, jitter))
    return rng.uniform(float(strength_min), float(strength_max))


def apply_stack_schedule(
    handles: list[LoRAHandles],
    *,
    step: int,
    prob: float,
    strength_min: float,
    strength_max: float,
    warmup_steps: int = 0,
    seed: int = 0,
    mode: str = "cycle",
    strengths: tuple[float, ...] | None = None,
    jitter: float = 0.0,
) -> dict | None:
    """Set every frozen style handle's multiplier for ``step`` (deterministic)."""
    if not handles:
        return None
    if mode != "pinned" and prob <= 0.0:
        for handle in handles:
            handle.set_multiplier(0.0)
        return {"active": False, "reason": "prob", "step": step}
    if step < warmup_steps:
        for handle in handles:
            handle.set_multiplier(0.0)
        return {"active": False, "reason": "warmup", "step": step}

    values = dict(
        strengths=strengths,
        strength_min=strength_min,
        strength_max=strength_max,
        jitter=jitter,
    )
    rng = random.Random(int(seed) * 1_000_003 + int(step))
    if mode == "pinned":
        active = []
        for index, handle in enumerate(handles):
            strength = _strength_for(index, rng, **values)
            handle.set_multiplier(strength)
            active.append({"index": index, "strength": strength})
        return {
            "active": True,
            "active_styles": active,
            "mode": "pinned",
            "step": int(step),
        }

    if mode == "independent":
        active = []
        for index, handle in enumerate(handles):
            if rng.random() < float(prob):
                strength = _strength_for(index, rng, **values)
                handle.set_multiplier(strength)
                active.append({"index": index, "strength": strength})
            else:
                handle.set_multiplier(0.0)
        return {
            "active": bool(active),
            "active_styles": active,
            "step": int(step),
        }

    if rng.random() >= float(prob):
        for handle in handles:
            handle.set_multiplier(0.0)
        return {"active": False, "reason": "prob", "step": step}

    index = rng.randrange(len(handles)) if mode == "random" else int(step) % len(handles)
    strength = _strength_for(index, rng, **values)
    for position, handle in enumerate(handles):
        handle.set_multiplier(strength if position == index else 0.0)
    return {
        "active": True,
        "index": index,
        "strength": strength,
        "step": int(step),
    }
