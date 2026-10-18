"""Krea-2 line —— 缓存数据集（读 `*_anima.npz` + `*_krea2_te.safetensors`）。

命名约定就是 krea2-webui 的缓存布局（我们只读它，不改）：
    <cache>/10021549_0576x0448_anima.npz         VAE latent（键名带 WxH）
    <cache>/10021549_krea2_te.safetensors        TE 缓存（hiddens/mask/caption_dropout_rate）
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_LATENT_RE = re.compile(r"^(?P<stem>.+?)_(?P<w>\d+)x(?P<h>\d+)_anima\.npz$")


@dataclass(frozen=True)
class Sample:
    stem: str
    latent_path: Path
    te_path: Path
    size: tuple[int, int]


class Krea2Cache:
    """扫描缓存目录，配对 latent 与 TE 缓存（缺一半的样本直接跳过并计数）。

    `mem_cache_size`：常驻内存的样本数（默认 8）。**为什么必须有**：Krea-2 的 TE 缓存
    每张 **31.46MB**（`hiddens[512,12,2560]` bf16 + mask），每步都从磁盘读一遍会让步时
    从 7~8.5s 掉到 10~15s；但把 372 张全塞内存又是 11GB+（宿主内存顶不住）——
    所以按窗口缓存，默认 8 张 ≈ 250MB。0 = 关掉（纯流式）。
    """

    def __init__(self, cache_dir: str | Path, mem_cache_size: int = 8):
        self.dir = Path(cache_dir)
        if not self.dir.is_dir():
            raise SystemExit(f"缓存目录不存在：{self.dir}")
        self._mem_size = max(0, int(mem_cache_size))
        self._mem: "OrderedDict[int, tuple]" = OrderedDict()
        self.samples: list[Sample] = []
        self.skipped = 0
        for npz in sorted(self.dir.glob("*_anima.npz")):
            m = _LATENT_RE.match(npz.name)
            if not m:
                continue
            te = self.dir / f"{m['stem']}_krea2_te.safetensors"
            if not te.is_file():
                self.skipped += 1
                continue
            self.samples.append(
                Sample(m["stem"], npz, te, (int(m["w"]), int(m["h"])))
            )
        if not self.samples:
            raise SystemExit(f"{self.dir} 里没有成对的 latent + TE 缓存（跳过了 {self.skipped} 个）")

    def __len__(self) -> int:
        return len(self.samples)

    def load(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sample]:
        """返回 `(latents_4d (1,C,H,W) bf16, hiddens (1,L,12,2560) bf16, mask (1,L) bool, sample)`。"""
        i = index % len(self.samples)
        if self._mem_size and i in self._mem:
            self._mem.move_to_end(i)
            return self._mem[i]

        from library.models.krea2_raw.strategy import (
            Krea2TextEncoderOutputsCachingStrategy,
        )

        s = self.samples[i]
        with np.load(s.latent_path) as z:
            key = next((k for k in z.keys() if k.startswith("latents")), list(z.keys())[0])
            arr = np.ascontiguousarray(z[key])
        lat = torch.from_numpy(arr).float()
        while lat.dim() > 4:
            lat = lat.squeeze(0)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)

        conds = Krea2TextEncoderOutputsCachingStrategy().load_outputs_npz(str(s.te_path))
        hiddens, mask = conds[0], conds[1]
        if hiddens.dim() == 3:
            hiddens = hiddens.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        out = (
            lat.to(torch.bfloat16),
            hiddens.to(torch.bfloat16),
            mask.to(torch.bool),
            s,
        )
        if self._mem_size:
            self._mem[i] = out
            self._mem.move_to_end(i)
            while len(self._mem) > self._mem_size:
                self._mem.popitem(last=False)
        return out
