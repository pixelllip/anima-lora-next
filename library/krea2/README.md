# `library/krea2/` —— 占位目录

Krea-2 训练/推理的 line 内核将来落在这里（`requests.py` / `loader.py` / `lora.py` /
`dit.py` / `strategy.py` / `sampling.py` / `train.py` / `cache.py` / `generate.py`）。

现在还没有代码：落点、可搬的部分、必须重写的部分与许可前提都写在
[`downstream/krea2-line-plan.md`](../../downstream/krea2-line-plan.md)。

**这条 line 不是 Anima**：上游根 `CLAUDE.md` 的不变式（5D latent 单例在 dim 2、
文本必须 max-padding、free-fit 分桶…）在这里大部分不成立，别照搬。
