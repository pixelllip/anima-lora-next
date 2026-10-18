"""Krea-2 line —— 训练循环（512 档在 8GB 卡上实测可跑：swap=20 + 梯度检查点）。

配方来源（krea2-webui 的实测结论，见 downstream/krea2-line-plan.md）：
* σ 采样：官方把 **未 shift 的线性网格 + u~U(0,1)** 当训练噪声 → `sigma` 档 = σ ~ U(0,1)；
* 目标：rectified-flow 的 velocity `noise - latents`；
* 底模：**NF4 自包含 v2**（8GB 卡加载 bf16 26GB 是不可能的）；
* `attn_mode=torch`、`torch_compile=false`（compile 在 8GB 上方向为负）；
* 显存靠 **块交换 + 全程梯度检查点**（探针实测 reserved 4.48GB @swap20/512）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from library.krea2.data import Krea2Cache
from library.krea2.lora import attach_lora, lora_parameters
from library.krea2.requests import TrainRequest


def _log(msg: str) -> None:
    print(f"[krea2] {msg}", flush=True)


def _save(handles, req: TrainRequest, step: int, *, final: bool) -> Path:
    out = Path(req.output)
    if not final:
        out = out.with_name(f"{out.stem}-step{step:08d}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tensors = handles.state_dict()
    metadata = {
        "modelspec.architecture": "krea2-raw/lora",
        "modelspec.title": req.output_name,
        "ss_network_dim": str(req.rank),
        "ss_network_alpha": str(req.alpha),
        "ss_target_preset": req.target_preset,
        "ss_base_model": Path(req.model).name,
        "ss_max_steps": str(req.max_steps),
        "ss_learning_rate": str(req.lr),
        "ss_timestep_sampling": "sigma",
        "ss_lr_scheduler": req.lr_scheduler,
        "ss_dataset_count": "unknown",
        "ss_network_module": "library.krea2.lora",
        "ss_down_init": "kaiming_zero_up",
    }
    save_file(tensors, str(out), metadata=metadata)
    snapshot = out.with_suffix(".snapshot.toml")
    lines = ["# Krea-2 line 训练快照（本次实际用的键值）"]
    for key, value in req.to_dict().items():
        if isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    snapshot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_train(req: TrainRequest) -> int:
    """跑 `req.max_steps` 步并落盘 LoRA；返回 0。内存/显存按 §"用户还要用电脑" 的纪律打印。"""
    from library.models.krea2_raw.family import forward_for_loss
    from library.models.krea2_raw.weights import load_krea2_dit

    torch.manual_seed(req.seed)
    device = torch.device(req.device)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    ds = Krea2Cache(req.cache)
    _log(f"缓存 {len(ds.samples)} 个样本（跳过 {ds.skipped} 个孤立的 latent）")

    t0 = time.time()
    dit = load_krea2_dit(req.model, device="cpu", nf4_path=req.model)
    dit.train()
    if req.grad_checkpointing:
        dit.enable_gradient_checkpointing()

    # ⚠ 顺序要紧：**先挂 LoRA，再开块交换**。反过来的话，offloader 是按"开交换那一刻
    # 块里有什么"来搬的，后加的 LoRA 参数就成了漏网之鱼 —— 现象就是
    # `Expected all tensors to be on the same device, cuda:0 and cpu`。
    handles = attach_lora(dit, rank=req.rank, alpha=req.alpha,
                          preset=req.target_preset, dropout=req.dropout)
    lora_params = list(lora_parameters(handles))
    for p in lora_params:
        p.data = p.data.to(dtype=torch.float32)

    if req.blocks_to_swap:
        dit.enable_block_swap(req.blocks_to_swap, device)
        dit.move_to_device_except_swap_blocks(device)
    else:
        dit.to(device)
    _log(
        f"DiT 就绪 {time.time() - t0:.1f}s；swap={req.blocks_to_swap}；"
        f"grad_ckpt={req.grad_checkpointing}；显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    _log(f"LoRA 挂在 {len(handles.modules)} 个 Linear 上，可训参数 {handles.param_count:,}")

    opt = torch.optim.AdamW(lora_params, lr=req.lr, weight_decay=req.weight_decay)
    sched = None
    if req.lr_scheduler == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    elif req.lr_scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, req.max_steps))
    else:
        raise SystemExit(f"lr_scheduler={req.lr_scheduler!r} 暂不支持（constant | cosine）")

    saved: list[Path] = []
    for step in range(1, req.max_steps + 1):
        lat_4d, hiddens, mask, sample = ds.load(step - 1)
        lat_4d = lat_4d.to(device)
        hiddens = hiddens.to(device)
        mask = mask.to(device)
        lat_5d = lat_4d.unsqueeze(2)
        noise = torch.randn_like(lat_5d)
        sigma = torch.empty(lat_5d.shape[0], device=device).uniform_(req.sigma_min, req.sigma_max)
        view = sigma.view(-1, 1, 1, 1, 1)
        noisy = (1.0 - view) * lat_5d + view * noise       # rectified flow: x_σ = (1-σ)x₀ + σε
        target = noise - lat_5d                            # velocity

        if req.blocks_to_swap:
            dit.prepare_block_swap_before_forward()
        ts = time.time()
        pred = forward_for_loss(dit, noisy, (hiddens, mask), sigma)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        if step % req.log_every_n_steps == 0 or step == 1:
            _log(
                f"step {step}/{req.max_steps} loss={loss.item():.4f} "
                f"σ={sigma.mean().item():.3f} {time.time() - ts:.1f}s "
                f"| 显存峰值 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
                f"| 样本 {sample.stem} {sample.size[0]}x{sample.size[1]}"
            )
        if req.save_every_n_steps and step % req.save_every_n_steps == 0:
            path = _save(handles, req, step, final=False)
            saved.append(path)
            _log(f"存档 {path.name}")
            if req.save_last_n > 0 and len(saved) > req.save_last_n:
                old = saved.pop(0)
                old.unlink(missing_ok=True)
                old.with_suffix(".snapshot.toml").unlink(missing_ok=True)

    final = _save(handles, req, req.max_steps, final=True)
    _log(f"最终 LoRA: {final}（{final.stat().st_size / 1024**2:.1f} MB，"
         f"{len(handles.modules)} 个模块）")
    return 0
