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
