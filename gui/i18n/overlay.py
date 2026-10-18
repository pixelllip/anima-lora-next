"""中文显示补丁（Chinese display overlay）。

上游 ``gui/i18n/{en,ko,ja,cn}.py`` 覆盖的是「控件文案」；有几处界面是拿**数据**
直接渲染的，绕过了那四张表，于是任何语言下都是英文：

* 训练配置表单的字段名 —— 直接拿 TOML 键当标签（``learning_rate`` …）；
* ``gui/core/config_io.py::_GROUPS`` 的分组标题（Architecture / Paths / …）；
* ``presets.toml`` 里 ``[<name>.gui] label`` 的硬件预设名（Default (16GB+) …）；
* ``gui/tabs/config_tab.py`` / ``methods_tab.py`` / ``tabs/preprocess/tab.py``
  里硬编码的 ``"Method"``（该键已按上游的口径补进 ``gui/i18n/*.py``，故这三处直接用
  ``t("method")``，不再走本模块）。

这些字符串在源码里就是标识符，硬塞进 ``gui/i18n/*.py`` 会破坏
``tests/test_gui_i18n_parity.py``（键与 ``str.format`` 字段必须四个语言完全一致），
而且给 en/ko/ja 编一份「标识符的中文名」也没有意义。所以这里用一张**只对中文生效**
的补丁表 ``cn_overlay.json``：

* ``current_language() != "cn"`` 时**原样返回默认值** —— en/ko/ja 的渲染与上游逐字节相同；
* 补丁表里缺键时同样回落默认值，所以加键是纯增量，不改键就是上游行为。

数据与代码分离：**要加中文标签只改 ``cn_overlay.json``**，不用动任何 .py。
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from gui.i18n import current_language

_DATA_PATH = Path(__file__).with_name("cn_overlay.json")


@functools.lru_cache(maxsize=1)
def _data() -> dict[str, dict[str, str]]:
    """Load the overlay table once. A missing/corrupt file degrades to {}."""
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        section: {str(k): str(v) for k, v in value.items()}
        for section, value in raw.items()
        if isinstance(value, dict)
    }


def _lookup(section: str, key: str, default: str) -> str:
    if current_language() != "cn" or not key:
        return default
    return _data().get(section, {}).get(key, default)


def field_label(key: str) -> str:
    """Form label for a TOML key: Chinese when known, else the key itself."""
    return _lookup("fields", key, key)


def group_title(name: str) -> str:
    """``_GROUPS`` group-box title (Architecture → 架构)."""
    return _lookup("groups", name, name)


def preset_label(name: str, default: str) -> str:
    """Hardware preset display name, keyed by its ``presets.toml`` section."""
    return _lookup("presets", name, default)
