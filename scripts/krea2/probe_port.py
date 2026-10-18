"""Krea-2 移植通路探针（512 档桶用）。

回答一个具体问题：把 `scvxzf1/krea2-webui` 的 `library/models/krea2_raw/` **原样**搬进
本仓库（以上游为基线的 anima-lora-next）之后，**能不能在 512 档桶上 forward+backward 跑通**
—— 用真实的自包含 NF4 底模 + 真实的 512 档缓存（`*_anima.npz` + `*_krea2_te.safetensors`）。

它**不训练**（不建 LoRA、不写产物）：LoRA 注入/训练循环那层接线是要按上游 line 风格重写的部分，
本探针只证明"核心能在新仓库里加载并跑起来"。

用法：
    python scripts/krea2/probe_port.py \
        --dit  <krea2_raw_nf4_self_contained.safetensors> \
        --latent <post_image_dataset/.../xxxx_0576x0448_anima.npz> \
        --te   <post_image_dataset/.../xxxx_krea2_te.safetensors> \
        [--swap 0] [--min-free-gb 6] [--steps 1]

⚠ 内存纪律（用户还要用电脑）：默认要求可用物理内存 ≥ `--min-free-gb`（默认 6GB）才开跑；
每一步前后都打印可用内存与显存峰值。
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch


def free_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:  # psutil 不在时退回 Windows API
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1024**3


def load_latent(path: Path) -> tuple[torch.Tensor, str]:
    """从 npz 里拿 latent；返回 4D (1,C,H,W)。"""
    with np.load(path) as z:
        keys = list(z.keys())
        arr = None
        for k in ("latents", "latent", "arr_0"):
            if k in z:
                arr = z[k]
                break
        if arr is None:
            arr = z[keys[0]]
        desc = f"npz keys={keys[:4]} shape={arr.shape}"
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    while t.dim() > 4:
        t = t.squeeze(0)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t, desc


def load_text(path: Path) -> tuple[torch.Tensor, torch.Tensor, str]:
    """按**训练侧同一份实现**读 Krea-2 TE 缓存（`strategy.load_outputs_npz`）。

    自己按形状猜是错的来源：缓存里 `hiddens` 是 (L, layers, D) 的 12 层 MFA 堆栈，
    训练侧要的是 (B, L, D) —— 这个转换只有 strategy 那份实现说了算。
    """
    from library.models.krea2_raw.strategy import (
        Krea2TextEncoderOutputsCachingStrategy,
    )

    conds = Krea2TextEncoderOutputsCachingStrategy().load_outputs_npz(str(path))
    hiddens, mask = conds[0], conds[1]
    # 缓存约定（family.compute_noise_pred_and_target 的头注）：hiddens (B,L,12,2560)、mask (B,L)。
    # 单图缓存的 B 维可能被省掉，这里补回来 —— 这是探针自己的适配，不是移植问题。
    if hiddens.dim() == 3:
        hiddens = hiddens.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    desc = f"strategy.load_outputs_npz -> {len(conds)} 项；hiddens{tuple(hiddens.shape)} mask{tuple(mask.shape)}"
    return hiddens, mask.to(torch.bool), desc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", required=True, help="Krea-2 NF4 自包含底模 safetensors")
    ap.add_argument("--latent", required=True, help="512 档 VAE latent npz（*_anima.npz）")
    ap.add_argument("--te", required=True, help="512 档 Krea-2 TE 缓存（*_krea2_te.safetensors）")
    ap.add_argument("--swap", type=int, default=0, help="块交换块数（0 = 不交换）")
    ap.add_argument("--steps", type=int, default=1, help="前向+反向跑几次")
    ap.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="关掉梯度检查点（默认开 —— 512 档在 8GB 卡上靠它压激活）",
    )
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="可用内存低于这个值就拒绝开跑")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))

    before = free_gb()
    print(f"[probe] 宿主可用内存 {before:.1f} GB（下限 {args.min_free_gb:.1f} GB）")
    if before < args.min_free_gb:
        print("[probe] 可用内存不够，拒绝开跑（用户还要用电脑）")
        return 2
    if not torch.cuda.is_available():
        print("[probe] CUDA 不可用")
        return 2

    from library.models.krea2_raw.family import forward_for_loss
    from library.models.krea2_raw.weights import load_krea2_dit

    lat_4d, lat_desc = load_latent(Path(args.latent))
    hiddens, mask, te_desc = load_text(Path(args.te))
    b, c, h, w = lat_4d.shape
    latents_5d = lat_4d.unsqueeze(2).to(args.device, dtype=torch.bfloat16)
    hiddens = hiddens.unsqueeze(0) if hiddens.dim() == 2 else hiddens
    hiddens = hiddens.to(args.device, dtype=torch.bfloat16)
    mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
    mask = mask.to(args.device)
    print(f"[probe] latent: {lat_desc} -> 5D {tuple(latents_5d.shape)}")
    print(f"[probe] te: {te_desc}")

    t0 = time.time()
    print(f"[probe] 加载 NF4 DiT（{Path(args.dit).stat().st_size / 1024**3:.2f} GB 磁盘）…")
    # 先在 CPU 上 strict 加载，再决定哪些块留 GPU —— 否则 6.2GB 权重先占满 8GB 卡，
    # 后面的块交换已经没地方腾挪（fork 的做法也是 load→enable_block_swap→move_*）。
    dit = load_krea2_dit(args.dit, device="cpu", nf4_path=args.dit)
    dev = torch.device(args.device)
    if args.swap:
        dit.enable_block_swap(args.swap, dev)
        dit.move_to_device_except_swap_blocks(dev)
    else:
        dit.to(dev)
    dit.train()
    if not args.no_grad_ckpt:
        dit.enable_gradient_checkpointing()
    print(f"[probe] DiT 就绪 {time.time() - t0:.1f}s；参数 {sum(p.numel() for p in dit.parameters()):,}"
          f"；swap={args.swap}；grad_ckpt={not args.no_grad_ckpt}")
    print(f"[probe] 加载后可用内存 {free_gb():.1f} GB；显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    for step in range(1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats()
        target = torch.randn_like(latents_5d)
        t = torch.full((latents_5d.shape[0],), 0.6, device=args.device)
        if args.swap:
            # 训练步开始时把第一块搬上卡（fork 的训练循环在每步开头做同一件事）
            dit.prepare_block_swap_before_forward()
        ts = time.time()
        pred = forward_for_loss(dit, latents_5d, (hiddens, mask), t)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        loss.backward()
        dit.zero_grad(set_to_none=True)
        took = time.time() - ts
        peak_a = torch.cuda.max_memory_allocated() / 1024**3
        peak_r = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"[probe] step {step}/{args.steps}: loss={loss.item():.4f} {took:.1f}s "
            f"| 显存峰值 allocated {peak_a:.2f} GB / reserved {peak_r:.2f} GB "
            f"| 可用内存 {free_gb():.1f} GB"
        )

    del dit, latents_5d, hiddens, mask, target
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[probe] 收尾：可用内存 {free_gb():.1f} GB；显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print("[probe] OK —— 核心在本仓库里能加载并在 512 档桶上 forward+backward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
