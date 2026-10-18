"""Krea-2 line —— 请求定义（**torch-free**，可被 CLI / daemon / 将来的 WebUI 共用）。

键名即 `configs/krea2/*.toml` 的键名；优先级：字段默认值 → TOML → 命令行。
这条 line **不走** Anima 的 `base → preset → method → CLI` 链（那条链是 Anima 专属的）。
"""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainRequest:
    """一次 Krea-2 LoRA 训练需要的全部参数。"""

    # 数据 / 产物
    cache: str = "post_image_dataset/krea2"          # 缓存目录（*_anima.npz + *_krea2_te.safetensors）
    output: str = "output/ckpt_krea2/krea2_lora.safetensors"
    output_name: str = "krea2_lora"
    # 底模（自包含 NF4 v2 或 bf16；v2 会被自动识别）
    model: str = "models/diffusion_models/krea2_raw_nf4_self_contained.safetensors"
    # LoRA
    rank: int = 16
    alpha: float = 16.0
    target_preset: str = "official_gate"             # official_gate(264) | official(232) | attention(160)；attn_mlp 已弃用
    dropout: float = 0.0
    # 训练
    max_steps: int = 20
    batch_size: int = 1
    lr: float = 3e-4
    lr_scheduler: str = "constant"
    warmup_steps: int = 0
    weight_decay: float = 0.0
    # adamw | adamw8bit（8bit = bitsandbytes AdamW8bit，8GB 卡上实测再省 ~0.13GiB；
    # ⚠ 唯一"改数值"的一项：不能与旧的 fp32 AdamW state 混用续训）
    optimizer_type: str = "adamw"
    seed: int = 0
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    # 显存
    blocks_to_swap: int = 26
    grad_checkpointing: bool = True
    device: str = "cuda"
    # 宿主内存：常驻内存的样本窗口（TE 缓存每张 31.46MB；372 张全进内存是 11GB+，会顶爆宿主）
    mem_cache_size: int = 8
    # 保存
    save_every_n_steps: int = 0                      # 0 = 只存最后
    save_last_n: int = 2
    log_every_n_steps: int = 1

    EXTRA_KEYS: tuple[str, ...] = field(default=(), init=False, repr=False)

    @classmethod
    def load_config(cls, path: str | Path, base: "TrainRequest | None" = None) -> "TrainRequest":
        """`--config <toml>`：未知键直接报错（不静默忽略）。"""
        req = base or cls()
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SystemExit(f"{path}: 未知键 {unknown}（可用键：{sorted(known)}）")
        for key, value in data.items():
            setattr(req, key, value)
        return req

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init}

    def apply_cli(self, argv: list[str]) -> "TrainRequest":
        """命令行覆盖（只认本类字段；`--config` 由 scripts 层先处理）。"""
        ap = argparse.ArgumentParser(add_help=False)
        for f in dataclasses.fields(self):
            if not f.init:
                continue
            cur = getattr(self, f.name)
            if isinstance(cur, bool):
                ap.add_argument(f"--{f.name}", type=_bool, default=None)
            elif isinstance(cur, int):
                ap.add_argument(f"--{f.name}", type=int, default=None)
            elif isinstance(cur, float):
                ap.add_argument(f"--{f.name}", type=float, default=None)
            else:
                ap.add_argument(f"--{f.name}", type=str, default=None)
        ns, _ = ap.parse_known_args(argv)
        for f in dataclasses.fields(self):
            if f.init and getattr(ns, f.name) is not None:
                setattr(self, f.name, getattr(ns, f.name))
        return self


def _bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")
