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
import os
import time

# 分配器设置必须在 import torch **之前**生效；守护进程/ComfyHub 拉起的子进程不会继承
# 我们 shell 里的环境变量，所以在代码里兜底（Windows 不支持 expandable_segments；
# krea2-webui 实测：带它 reserved 7.25 → 6.09GB）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
from pathlib import Path

import torch
from safetensors.torch import save_file

from library.krea2.data import Krea2Cache
from library.krea2.lora import attach_lora, lora_parameters
from library.krea2.requests import TrainRequest
from library.krea2.stack import (
    apply_stack_schedule,
    build_stack_loras,
    resolve_stack_strengths,
    split_stack_paths,
    stack_enabled,
)


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
        "ss_stack_lora_count": str(len(split_stack_paths(req.stack_lora_paths))),
        "ss_stack_lora_prob": str(req.stack_lora_prob),
        # 人物 delta 被训练成「叠在某档画风上的残差」；不记下这档强度，pinned
        # 成品在推理端不可复现。
        "ss_stack_lora_mode": req.stack_lora_mode,
        "ss_stack_lora_strengths": str(req.stack_lora_strengths or ""),
        "ss_stack_lora_strength_jitter": str(req.stack_lora_strength_jitter),
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


def _cast_model_bf16(dit) -> int:
    """把**非量化**的浮点参数统一降到 bf16（NF4 的 Params4bit 跳过）。

    为什么必须做：v2 自包含 NF4 文件里 `first` / `last` 这类没被量化的 Linear 与一堆
    norm/调制参数是 **fp32**，`load_krea2_dit(..., dtype=...)` 不会替你全转。fp32 权重
    一旦参与运算，就会把整条图拉成 fp32 —— 本机实测前向峰值 4.56GB（krea2-webui 同参数
    3.47GB），而且 LoRA 权重被迫跟着 fp32 才不会 dtype 报错。生产快照里写的就是
    `mixed_precision = "bf16"`。
    """
    from bitsandbytes.nn import Params4bit

    quantized = set()
    for mod in dit.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if isinstance(p, Params4bit):
                quantized.add(id(p))

    n = 0
    for p in list(dit.parameters()):
        if id(p) in quantized or not p.is_floating_point() or p.dtype == torch.bfloat16:
            continue
        p.data = p.data.to(torch.bfloat16)
        n += 1
    return n


def _configure_attention_backends() -> None:
    """把 SDPA 后端钉在 **mem-efficient** 上（8GB 卡的生死线）。

    为什么要显式钉（本机实测，L=4032 / heads=28 / D=128 / bf16）：

        math       peak 4.12 GB   0.432s     ← 一旦落到这里，1024 档直接越线换页
        efficient  peak 0.14 GB   0.073s
        cudnn      peak 0.14 GB   0.727s

    本项目 Krea-2 的 attention 走 `attention_backend._cudnn_attention`：它先用
    `sdpa_kernel(CUDNN_ATTENTION)`，**失败就退回裸 `scaled_dot_product_attention`**——
    而裸调用会自己挑后端，实测挑中了 **math**（于是在 1024 档 reserved 冲到 8.12GB、
    步时 76–155s、GPU 反而不干活）。禁用 math（再顺手禁掉更慢的 cuDNN）之后，
    退路只能落到 mem-efficient。上游 `networks/attention_dispatch.py` 用的也是
    `SDPBackend.EFFICIENT_ATTENTION`，口径一致。
    """
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)


def run_train(req: TrainRequest) -> int:
    """跑 `req.max_steps` 步并落盘 LoRA；返回 0。内存/显存按 §"用户还要用电脑" 的纪律打印。"""
    from library.models.krea2_raw.family import forward_for_loss
    from library.models.krea2_raw.weights import load_krea2_dit

    _configure_attention_backends()
    torch.manual_seed(req.seed)
    device = torch.device(req.device)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    ds = Krea2Cache(req.cache, mem_cache_size=req.mem_cache_size)
    _log(f"缓存 {len(ds.samples)} 个样本（跳过 {ds.skipped} 个孤立的 latent；"
         f"内存窗口 {req.mem_cache_size} 张 ≈ {req.mem_cache_size * 31.5:.0f} MB）")

    t0 = time.time()
    # dtype 必须按 `mixed_precision` 给死 bf16（生产快照里就是 `mixed_precision = "bf16"`）：
    # 留 None = "保持文件原 dtype"，那个自包含 NF4 里的 159 个非 Linear 张量是 fp32，
    # modulation 一乘就把整个 block 拉成 fp32 —— 激活翻倍，而且 LoRA 权重也必须跟着 fp32
    # 才不会 dtype 报错（本机实测：bf16 LoRA + fp32 图 = `expected mat1 and mat2 to have
    # the same dtype, but got: float != BFloat16`）。整张图统一 bf16 后前向峰值才对得上
    # krea2-webui 的 3.47GB。
    dit = load_krea2_dit(req.model, device="cpu", nf4_path=req.model,
                         dtype=torch.bfloat16)
    cast = _cast_model_bf16(dit)
    _log(f"fp32 → bf16 的参数 {cast} 个（NF4 的 Params4bit 跳过）")
    dit.train()
    if req.grad_checkpointing:
        dit.enable_gradient_checkpointing()

    # ⚠ 顺序要紧：**先挂 LoRA，再开块交换**。反过来的话，offloader 是按"开交换那一刻
    # 块里有什么"来搬的，后加的 LoRA 参数就成了漏网之鱼 —— 现象就是
    # `Expected all tensors to be on the same device, cuda:0 and cpu`。
    handles = attach_lora(dit, rank=req.rank, alpha=req.alpha,
                          preset=req.target_preset, dropout=req.dropout,
                          dtype=torch.bfloat16)  # ← 必须 bf16：留 fp32 会把整张图拖成 fp32
    lora_params = list(lora_parameters(handles))

    # style-in-the-loop：冻结画风 LoRA 也必须**先挂再开块交换**（同上）。
    stack_paths = split_stack_paths(req.stack_lora_paths)
    stack_strengths = None
    stack_handles = []
    if stack_paths:
        stack_strengths = resolve_stack_strengths(
            stack_paths, req.stack_lora_strengths, mode=req.stack_lora_mode
        )
    if stack_enabled(
        paths=stack_paths,
        prob=req.stack_lora_prob,
        mode=req.stack_lora_mode,
        strengths=stack_strengths,
    ):
        stack_handles = build_stack_loras(dit, stack_paths, dtype=torch.bfloat16)
        if stack_strengths is not None:
            detail = "/".join(f"{s:g}" for s in stack_strengths)
            detail = f"pinned={detail}"
        else:
            detail = f"强度 {req.stack_lora_strength_min:g}..{req.stack_lora_strength_max:g}"
        _log(
            f"style-in-the-loop：{len(stack_handles)} 个冻结画风 LoRA，"
            f"mode={req.stack_lora_mode}，{detail}，"
            f"jitter={req.stack_lora_strength_jitter:g}，"
            f"p={req.stack_lora_prob:g}，warmup={req.stack_lora_warmup_steps}"
        )
        if req.stack_lora_mode == "pinned" and req.stack_lora_warmup_steps:
            _log(
                "⚠️ pinned 模式下 warmup>0：预热步没有画风，人物 delta 会把画风"
                "吃回去（这正是本模式要避免的失败模式），建议 warmup=0"
            )

    if req.blocks_to_swap:
        dit.enable_block_swap(req.blocks_to_swap, device)
        dit.move_to_device_except_swap_blocks(device)
    else:
        dit.to(device)
    for handle in stack_handles:
        handle.holder.to(device)
    _log(
        f"DiT 就绪 {time.time() - t0:.1f}s；swap={req.blocks_to_swap}；"
        f"grad_ckpt={req.grad_checkpointing}；显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )
    _log(f"LoRA 挂在 {len(handles.modules)} 个 Linear 上，可训参数 {handles.param_count:,}")

    if req.optimizer_type == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(lora_params, lr=req.lr, weight_decay=req.weight_decay)
        _log("优化器 = bitsandbytes AdamW8bit（8bit m/v；不能与旧 fp32 state 混用续训）")
    elif req.optimizer_type == "adamw":
        opt = torch.optim.AdamW(lora_params, lr=req.lr, weight_decay=req.weight_decay)
    else:
        raise SystemExit(f"optimizer_type={req.optimizer_type!r} 不支持（adamw | adamw8bit）")
    sched = None
    if req.lr_scheduler == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    elif req.lr_scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, req.max_steps))
    else:
        raise SystemExit(f"lr_scheduler={req.lr_scheduler!r} 暂不支持（constant | cosine）")

    # ---- 预热：对齐 krea2-webui 的 `accelerator_prepared → loop_ready` --------------
    # 它进 loop 之前会把网络/优化器/分配器全部建好、**暖一次前向**，并把要用的缓存读进
    # 内存（宿主磁盘 100%、GPU 12% 就是那一段）；之后磁盘基本不动、reserved 从第 1 步
    # 就稳在 6.38GB。我原来加载完 1.8s 就开步、每步现读 30MB 的 `_krea2_te` →
    # 新分配不断产生，reserved 会从 4.81 爬到 6.75（`free` 掉到 0）。
    # ⚠ 预热后**不要** `torch.cuda.empty_cache()`：那会造成"先缩后涨"的碎片抖动。
    for i in range(min(req.mem_cache_size, len(ds))):
        ds.load(i)  # 只填 CPU 侧窗口，不动显存
    _log(f"缓存窗口已预读 {min(req.mem_cache_size, len(ds))} 张（不占显存）")
    # 预热要用**最大形状**那张（不是第 0 张）：allocator 的池子按"见过的最大块"长，
    # 用小图预热之后再遇到 1200x896 这类大桶就会一路往上长（实测第 5 步涨到 7.50GB、
    # 步时 86s = 换页）。先用最大形状把池子撑到位，后面各档就都落在里面了。
    worst = max(
        range(len(ds)),
        key=lambda i: ds.samples[i].size[0] * ds.samples[i].size[1],
    )
    _log(f"预热样本选最大形状：{ds.samples[worst].stem} "
         f"{ds.samples[worst].size[0]}x{ds.samples[worst].size[1]}")
    with torch.enable_grad():
        w_lat, w_hid, w_mask, _ = ds.load(worst)
        w_lat = w_lat.to(device)
        w_5d = w_lat.unsqueeze(2)
        w_noise = torch.randn_like(w_5d)
        w_sigma = torch.empty(w_5d.shape[0], device=device, dtype=torch.bfloat16).uniform_(
            req.sigma_min, req.sigma_max
        )
        w_view = w_sigma.view(-1, 1, 1, 1, 1)
        w_noisy = (1.0 - w_view) * w_5d + w_view * w_noise
        if req.blocks_to_swap:
            dit.prepare_block_swap_before_forward()
        w_pred = forward_for_loss(dit, w_noisy, (w_hid.to(device), w_mask.to(device)), w_sigma)
        torch.nn.functional.mse_loss(w_pred.float(), (w_noise - w_5d).float()).backward()
        opt.zero_grad(set_to_none=True)
        del w_lat, w_5d, w_noise, w_sigma, w_view, w_noisy, w_pred
        _log(f"预热完成：reserved {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
             f"（krea2-webui 同参数为 6.38）；此后开始计时统计")

    saved: list[Path] = []
    for step in range(1, req.max_steps + 1):
        lat_4d, hiddens, mask, sample = ds.load(step - 1)
        lat_4d = lat_4d.to(device)
        hiddens = hiddens.to(device)
        mask = mask.to(device)
        lat_5d = lat_4d.unsqueeze(2)
        noise = torch.randn_like(lat_5d)
        # ⚠ σ 必须与模型同为 bf16：fp32 的 σ 一乘进 `(1-σ)x₀ + σε`，整个 noisy 输入就被
        # 提升成 fp32，`self.first` 立刻 dtype 报错；就算不报错（权重也是 fp32 时）整张图
        # 会在 fp32 里跑 —— 本机实测就是这条让前向峰值从 3.5GB 涨到 4.56GB。
        sigma = torch.empty(lat_5d.shape[0], device=device,
                            dtype=torch.bfloat16).uniform_(req.sigma_min, req.sigma_max)
        view = sigma.view(-1, 1, 1, 1, 1)
        noisy = (1.0 - view) * lat_5d + view * noise       # rectified flow: x_σ = (1-σ)x₀ + σε
        target = noise - lat_5d                            # velocity

        def _phase(label: str, _step: int = step) -> None:
            if _step > 3:
                return
            _log(
                "  [%s] step %d alloc %.2f reserved %.2f max_alloc %.2f free %.2f"
                % (
                    label,
                    _step,
                    torch.cuda.memory_allocated() / 1024**3,
                    torch.cuda.memory_reserved() / 1024**3,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    torch.cuda.mem_get_info()[0] / 1024**3,
                )
            )

        _phase("step_enter")
        if stack_handles:
            apply_stack_schedule(
                stack_handles,
                step=step,
                prob=req.stack_lora_prob,
                strength_min=req.stack_lora_strength_min,
                strength_max=req.stack_lora_strength_max,
                warmup_steps=req.stack_lora_warmup_steps,
                seed=req.seed,
                mode=req.stack_lora_mode,
                strengths=stack_strengths,
                jitter=req.stack_lora_strength_jitter,
            )
        if req.blocks_to_swap:
            dit.prepare_block_swap_before_forward()
        ts = time.time()
        pred = forward_for_loss(dit, noisy, (hiddens, mask), sigma)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        _phase("after_forward")
        _phase("before_backward")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _phase("after_backward")
        _phase("before_optimizer")
        opt.step()
        _phase("after_optimizer")
        if sched is not None:
            sched.step()

        if step % req.log_every_n_steps == 0 or step == 1:
            alloc = torch.cuda.max_memory_allocated() / 1024**3
            reserved = torch.cuda.max_memory_reserved() / 1024**3
            _log(
                f"step {step}/{req.max_steps} loss={loss.item():.4f} "
                f"σ={sigma.mean().item():.3f} {time.time() - ts:.1f}s "
                # 判据看 reserved（活跃内存可能只有 1.7GB 而 reserved 涨到 7.7GB：
                # 多出来的是 allocator 池块）——krea2-webui 的实测口径
                f"| 显存 allocated {alloc:.2f} / reserved {reserved:.2f} GB "
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
