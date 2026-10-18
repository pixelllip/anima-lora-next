# `configs/krea2/` —— 占位目录

Krea-2 line 的配置将来落在这里，形如 `lora.toml`：**一张扁平的键值表，
键名 = `library/krea2/requests.py` 里 `TrainRequest` / `CacheRequest` 的字段名**
（`--config <file.toml>` 读取，优先级：字段默认值 → TOML → 命令行）。

⚠ 它**不走** Anima 的 `base.toml → presets.toml → methods/<method>.toml → CLI` 链 ——
那条链是 Anima 专属的（键、默认值、合并语义都是）。参考 `configs/moss/lora.toml`。

计划与前提见 [`downstream/krea2-line-plan.md`](../../downstream/krea2-line-plan.md)。
