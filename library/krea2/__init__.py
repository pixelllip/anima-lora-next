"""Krea-2 line（不是 Anima）。

内核在 `library/models/krea2_raw/`（自 krea2-webui 原样搬入，见 NOTICE）；
这里放 line 自己的请求定义、数据、LoRA 注入与训练循环。

落点与前提：`downstream/krea2-line-plan.md`。
"""

from library.krea2.requests import TrainRequest

__all__ = ["TrainRequest"]
