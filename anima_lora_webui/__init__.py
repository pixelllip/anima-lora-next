"""anima_lora 训练守护进程（anima_daemon）的自有 Web 界面。零运行时依赖。"""

from importlib import import_module

__all__ = ["__version__", "serve"]


def __getattr__(name: str):
    """按需加载 server 模块。

    不在包导入时拉 `.server`：`python -m anima_lora_webui.server` 会先导入本包，
    若这里已经导入过一次 server，runpy 再执行同一个模块就会报
    "found in sys.modules after import of package" 的 RuntimeWarning。
    """
    if name in __all__:
        return getattr(import_module("anima_lora_webui.server"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
