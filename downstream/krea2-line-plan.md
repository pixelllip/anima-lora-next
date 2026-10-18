# Krea-2 line —— 计划落点（按上游的 "line" 风格）

这个目录**目前只是占位**：Krea-2 训练/推理还没迁进来。放这里是为了先把落点定下来，
避免以后又变成"往 Anima 训练器里塞 model_family 开关"那种改法。

上游对"多一个模型"的既有做法（`library/moss/`、`library/qwen21/` + `scripts/<line>/` +
`configs/<line>/` + `gui/lines/`）是**一条独立 line**，而不是在 Anima 里加分支：

| 落点 | 装什么 | 参考上游哪条线 |
| --- | --- | --- |
| `library/krea2/requests.py` | torch-free 的 `CacheRequest` / `TrainRequest`（每个 flag 的唯一真源） | `library/moss/requests.py` |
| `library/krea2/loader.py` | 分阶段加载（TE → encode → free → DiT → VAE），NF4 自包含底模 | `library/moss/loader.py` |
| `library/krea2/lora.py` | LoRA/LoKr/LoHa 绑定，**放在 block 外**（块交换会搬走 block 内的 `.weight`） | `library/lines/adapters.py` |
| `library/krea2/dit.py` | SingleStream MMDiT（裸移植，非 diffusers） | `library/moss/model.py` |
| `library/krea2/strategy.py` | Qwen3-VL 文本链（ChatML tokenize + 12 层 MFA + R1 mask 契约） | `library/qwen21/` |
| `library/krea2/sampling.py` | 官方 Euler + log-sigmoid mu shift | `library/moss/generate.py` |
| `library/krea2/train.py` / `cache.py` / `generate.py` | `run_train` / `run_cache` / `run_generate(req)` | 同上 |
| `scripts/krea2/{cache,train,generate}.py` | 薄 CLI 壳（`--config <toml>`） | `scripts/moss/*.py` |
| `configs/krea2/lora.toml` | 键 = `TrainRequest` 字段名（不是 base/preset/method 链） | `configs/moss/lora.toml` |

**能直接搬的**（来自 `scvxzf1/krea2-webui`，MIT，见 `NOTICE`）：
`library/models/krea2_raw/` 那 4.2k LOC —— `dit.py` / `weights.py` / `quantize.py`(NF4) /
`lora_targets.py` / `sampling.py` / `strategy.py` / `family.py::forward_for_loss`。
它们 import 的上游模块在今天的上游里都还在（`library/anima/text_strategies.py`、
`library/runtime/offloading.py`、`library/runtime/accelerator.py::prepare_dtype`、
`library/io/cache.py`、`library/datasets/buckets.py`、`library/env.py`）。

**必须重写的**：接线。krea2-webui 把 Krea-2 挂在它自己那套训练器上
（`library/training/{bootstrap,model_loading,anima_strategies,batch_step,compat_matrix,adapter_resolver}.py`、
`networks/registry*.py`、`library/cache_pool/fingerprint.py` —— **这些文件上游一个都没有**），
而上游是 `train.py`(3480 行) + `library/training/loop.py` + `library/anima/training.py`。
所以不是"打补丁"，是照上游 line 风格重新接。

**一起要带的三块能力**（上游目前没有，而 8GB 配方离不开）：

1. **NF4 自包含底模**：`quantize.py` + `bitsandbytes` 依赖（上游主依赖里没有 bnb；
   上游是 cu132/torch2.14，krea2-webui 是 cu130/torch2.12，wheel 要重验）。
2. **`selective_checkpoint`（every_other）与 `compile_block_scope`**：上游没有这两个键。
3. **块交换的那批修复**：krea2-webui 的 `library/runtime/offloading.py` 是上游的超集
   （`_try_foreach_h2d_copy` / `_wait_for_async_cuda_copy_on_current_stream` /
   `_finalize_async_cuda_timings` —— 就是"被交换块 D2H 必须 non_blocking=False、
   swap≥12 撞宿主内存却伪装成 CUDA OOM"那一课）。

**许可**：先看 `DOWNSTREAM.md` §1 —— `krea-ai/krea-2` 仓库根目录没有 LICENSE，
移植其代码前必须确认代码许可。确认之前，这一条 line 只能待在私有仓库里。

**验收**：上游 `CONTRIBUTING.md` 对新增 line 的要求（bench + invariant test）照做；
另外至少要有一条"cache → train 20 步 → 存 LoRA → ComfyUI 能加载"的端到端用例。

---

## 探针实测（2026-10-18，本机 4060 Laptop 8GB / 23GB 宿主）

`scripts/krea2/probe_port.py`（512 档桶）已经跑过，结论分两半——**能跑的部分与卡住的部分都很具体**：

**已经验证 ✅**

| 事项 | 结果 |
| --- | --- |
| 核心按原路径搬进来 | `library/models/krea2_raw/` 12 个文件、4200 行，**逐文件 sha256 与 krea2-webui 一致** |
| 模块导入 | 11 个模块里 10 个直接 import 通过；只有 `quantize.py` 需要 `bitsandbytes`（`uv pip install --no-deps "bitsandbytes>=0.49.2"` 装上后通过，实测 0.50.2 + torch 2.14.1+cu132 不冲突） |
| NF4 自包含底模 | `load_krea2_dit(..., nf4_path=...)` 自动识别 **format=v2**：264 个 Linear4bit + 159 个模型状态，**1.4~1.9s** 加载完（6.16GB 磁盘），参数 6,410,783,302 |
| 512 档缓存直读 | latent `latents_56x72` → 5D (1,16,1,56,72)（=448×576，**1008 tokens**，正是 512 档的桶）；TE 走**训练侧同一份实现** `Krea2TextEncoderOutputsCachingStrategy.load_outputs_npz` → `hiddens (1,512,12,2560)` + `mask (1,512)`（缓存里是 12 层 MFA 堆栈，**不能自己按形状猜**） |
| 不交换时 | 权重占显存 6.21GB；8GB 卡上 forward 直接 OOM（桌面已占 1.4GB）——与预期一致 |

**当时卡住、现在已解决 ✅（2026-10-18 当天复验通过）**

先说结论：**上游那份 `ModelOffloader` 够用，不需要把 krea2-webui 那 1908 行的超集搬进来。**
之前判成"必须搬"是被两个真因掩盖了 —— 记录在这里免得以后再踩：

| 当时的现象 | 真因 | 修法 |
| --- | --- | --- |
| `ModelOffloader.__init__() got an unexpected keyword argument 'profile_jsonl'` | fork 的 offloader 参数是超集（多 `profile_jsonl` / `transfer_dtype` / `restore_mode`，都是 profiling / H2D 调优用） | 移植版 `dit.py` 里**按签名过滤**这些 kwargs（上游以后合入超集时自动生效）。profiling 能力暂时失去，**不影响训练** |
| `Expected all tensors to be on the same device` | 训练步开始时没人把第一块搬上卡（fork 的训练循环每步开头会调） | 探针/训练循环每步调一次 `prepare_block_swap_before_forward()` |
| `CUBLAS_STATUS_EXECUTION_FAILED`（swap=16）/ 直接 OOM（swap=24） | **激活内存**，不是 offloader：探针当时没开梯度检查点 | `enable_gradient_checkpointing()` + `swap=20` |

**512 档桶实测（RTX 4060 Laptop 8GB / 23GB 宿主，swap=20 + 全程梯度检查点）**

```
[probe] DiT 就绪 1.6s；参数 6,410,783,302；swap=20；grad_ckpt=True
[probe] step 1/2: loss=1.6977 8.9s | 显存峰值 allocated 3.77 GB / reserved 4.48 GB | 可用内存 7.5 GB
[probe] step 2/2: loss=1.6952 6.1s | 显存峰值 allocated 3.77 GB / reserved 4.48 GB | 可用内存 7.5 GB
[probe] OK —— 核心在本仓库里能加载并在 512 档桶上 forward+backward
```

要点：
* **显存只用 3.77GB（reserved 4.48GB）**，8GB 卡有充裕余量 —— 512 档不必像 1024 档那样精打细算；
* 宿主内存从 14.1GB 掉到 7.5GB 后**不再增长**（20 个块的 pinned CPU 副本 + 模型），
  探针自带"可用内存 < 6GB 拒绝开跑"的闸门；
* 稳定态 ~6.1s/step（单步 forward+backward，未含优化器/LoRA）；**没有开 torch.compile**（512 档也不需要）。

**剩下要做的**（属于"接线"而非"通路"）：按上游 line 风格把 `requests/loader/train/cache/generate`
写出来，把 LoRA 注入（`lora_targets.py` 的 target spec + `library/lines/adapters.py`）接上，
再跑一次真正的 20 步出 LoRA。8GB 上的验收参数照
`hiten_krea2_r264_r16_tagnotes_512.toml`（r16/α16、lr 3e-4 constant、
`timestep_sampling="sigma"`+shift 1.0+uniform、attn_mode=torch、full grad ckpt、NF4 自包含、TE 在 CPU 编）。

**待补的兼容件**：`library/training/anima_strategies.py::_is_krea2` 上游没有（fork 独有），
只在"未缓存、需要现场编码"分支用到；移植时要带上或改成上游等价物。


**下一步（按优先级）**

1. 把 `krea2-webui/library/runtime/offloading.py` 作为**补丁文件**评估搬入（先做差异审计：1908 行 vs 上游 386 行里哪些是 Krea 专用、哪些是对 Anima 也有效的修复）；
   或者先给 Krea-2 写最小适配层，只补 `profile_jsonl`/`supports_backward`/`flush_profile_events` 三个语义。
2. 8GB 上的正式验收参数照 `hiten_krea2_r264_r16_tagnotes_512.toml`：`resolution=512`、`min/max_bucket_reso=256/1024`、`network_dim=16`/`alpha=16`、`lr=3e-4` constant、`timestep_sampling="sigma"`+`discrete_flow_shift=1.0`+`weighting_scheme="uniform"`、`torch_compile=false`、`attn_mode="torch"`、full grad ckpt、NF4 自包含、TE 在 CPU 上编（8.9GB Qwen3-VL 放不进 8GB 卡）。
3. `library/training/anima_strategies.py::_is_krea2` 目前**上游没有**（fork 独有，`family.py` 里只在"未缓存、需要现场编码"的分支用到）——移植时要带上或改成上游的等价物。

