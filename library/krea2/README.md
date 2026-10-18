# `library/krea2/` —— 占位目录

Krea-2 训练/推理的 line 内核将来落在这里（`requests.py` / `loader.py` / `lora.py` /
`dit.py` / `strategy.py` / `sampling.py` / `train.py` / `cache.py` / `generate.py`）。

现在还没有代码：落点、可搬的部分、必须重写的部分与许可前提都写在
[`downstream/krea2-line-plan.md`](../../downstream/krea2-line-plan.md)。

**核心已经按原路径搬进来了**（2026-10-18 探针跑过）：`library/models/krea2_raw/`
（12 个文件、4200 行、sha256 与 `scvxzf1/krea2-webui` 一致），
验证脚本 `scripts/krea2/probe_port.py`（512 档桶、NF4 自包含底模、真实缓存）。
实测结论与卡点（块交换需要那份 offloader 超集）都在上面那份计划里。

**这条 line 不是 Anima**：上游根 `CLAUDE.md` 的不变式（5D latent 单例在 dim 2、
文本必须 max-padding、free-fit 分桶…）在这里大部分不成立，别照搬。

**额外依赖**：NF4 路径需要 `bitsandbytes`（上游主依赖里没有；实测
`uv pip install --no-deps "bitsandbytes>=0.49.2"` 与本仓库的 torch 2.14.1+cu132 不冲突）。
等这条 line 正式落地时再把它加进 `pyproject.toml`（那会是一次"补丁上游文件"的改动）。
