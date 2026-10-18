# `scripts/krea2/` —— 占位目录

Krea-2 line 的薄 CLI 壳将来落在这里（`cache.py` / `train.py` / `generate.py`），
风格对齐 `scripts/moss/*.py`：**只做 argparse + 调 `library/krea2/*.run_*()`**，
参数定义唯一的真源是 `library/krea2/requests.py`（torch-free，可被 GUI / daemon / 脚本共用）。

计划与前提见 [`downstream/krea2-line-plan.md`](../../downstream/krea2-line-plan.md)。
