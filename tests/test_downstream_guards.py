"""下游整合版的守卫测试（**只用标准库**，CI 里不装 torch / PySide6 也能跑）。

守三件事：

1. **补丁面没有偷偷变大** —— 与基线相比，"改过的上游文件"必须正好等于
   ``.downstream/patched-upstream-files.txt`` 里的清单；多改一个上游文件就红。
   （我们改动越大，``git merge upstream/main`` 越容易冲突 —— 这就是那条线。）
2. **我们依赖的上游缝合点还在** —— 上游重构时最先断的就是这些名字，这里按字符串核对，
   比 import 更抗环境（CI 里没有 torch）。
3. **中文补丁表的形状** —— ``gui/i18n/cn_overlay.json`` 可解析、有值，且只有 ``cn`` 才生效。

跑法：``python -m pytest tests/test_downstream_guards.py -q``
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHED_LIST = ROOT / ".downstream" / "patched-upstream-files.txt"
OWNED_LIST = ROOT / "downstream" / "owned-paths.txt"
BASELINE_FILE = ROOT / ".downstream" / "upstream-base"
OVERLAY_JSON = ROOT / "gui" / "i18n" / "cn_overlay.json"
OVERLAY_PY = ROOT / "gui" / "i18n" / "overlay.py"


def _patched_files() -> list[str]:
    return [
        line.strip()
        for line in PATCHED_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _owned_files(kind: str) -> list[str]:
    out = []
    for line in OWNED_LIST.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == kind:
            out.append(parts[1])
    return out


# --- 1. 补丁面 ---------------------------------------------------------------


def test_patched_files_exist():
    missing = [p for p in _patched_files() if not (ROOT / p).is_file()]
    assert missing == [], f"清单里的补丁文件不存在：{missing}"


def test_owned_and_patched_lists_agree():
    assert sorted(_owned_files("patched")) == sorted(_patched_files()), (
        "downstream/owned-paths.txt 与 .downstream/patched-upstream-files.txt 不一致 —— 两份清单必须同步"
    )


def test_diff_against_baseline_touches_only_known_files():
    """与 ``.downstream/upstream-base`` 比，改过的文件只能是"补丁 + 自有新文件"。"""
    if not BASELINE_FILE.is_file():
        pytest.skip("没有基线记录")
    base = BASELINE_FILE.read_text(encoding="utf-8").strip()
    if not base:
        pytest.skip("基线记录为空")
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{base}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"基线提交 {base[:12]} 不在本地历史里（浅克隆？）")
    diff = subprocess.run(
        ["git", "diff", "--name-only", base, "--"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    allowed = set(_patched_files()) | set(_owned_files("additive"))
    unexpected = sorted(set(diff) - allowed)
    assert unexpected == [], (
        "这些文件既不在补丁清单、也不在自有新文件清单里 —— 要么把它们登记进 "
        f"downstream/owned-paths.txt，要么别改上游文件：{unexpected}"
    )


# --- 2. 上游缝合点 -----------------------------------------------------------


SEAMS: dict[str, tuple[str, ...]] = {
    # 我们（含未来的 Krea-2 line）直接依赖的上游面；名字变了就是要重新对接。
    "library/env.py": ("def anima_home", "def resolve_under_home"),
    "library/runtime/blockswap.py": ("def find_blocks", "def to_device_except_blocks"),
    "library/runtime/offloading.py": ("class ModelOffloader",),
    "library/runtime/accelerator.py": ("def prepare_dtype",),
    "library/anima/text_strategies.py": ("class ",),
    "library/lines/requests.py": ("class ",),
    "library/lines/adapters.py": ("def ",),
    "library/downloads.py": ("DEFAULT_SET", "def models_dir"),
    "library/config/io.py": ("def load_method_preset",),
    "library/preprocess/images.py": ("def resize_to_buckets",),
    "anima_daemon/client.py": ("class DaemonClient", "def ensure_daemon"),
    "anima_daemon/config.py": ("PIDFILE", "def discover_pidfile"),
    "gui/i18n/__init__.py": ("TRANSLATIONS", "def t", "def current_language"),
    "gui/explanations/__init__.py": ("def field_help",),
    "gui/core/config_io.py": ("_GROUPS", "def list_gui_variants"),
}


@pytest.mark.parametrize(("rel", "needles"), sorted(SEAMS.items()))
def test_upstream_seam_still_present(rel: str, needles: tuple[str, ...]):
    path = ROOT / rel
    assert path.is_file(), f"上游缝合点文件不见了：{rel}（上游重构了？）"
    text = path.read_text(encoding="utf-8")
    missing = [n for n in needles if n not in text]
    assert missing == [], f"{rel} 里找不到 {missing} —— 上游改了名字，下游要重新对接"


# --- 3. 中文补丁表 -----------------------------------------------------------


def test_overlay_json_shape():
    data = json.loads(OVERLAY_JSON.read_text(encoding="utf-8"))
    assert set(data) >= {"fields", "groups", "presets"}
    for section in ("fields", "groups", "presets"):
        assert data[section], f"cn_overlay.json 的 {section} 是空的"
        assert all(isinstance(v, str) and v.strip() for v in data[section].values())


def test_overlay_covers_every_group_title():
    """``_GROUPS`` 的每个分组标题都要有中文，否则界面上会漏出英文标题。"""
    from gui.core.config_io import _GROUPS  # 依赖-free，CI 里可导入

    data = json.loads(OVERLAY_JSON.read_text(encoding="utf-8"))
    missing = [g for g in (*_GROUPS, "Other") if g not in data["groups"]]
    assert missing == [], f"这些分组标题还没有中文：{missing}"


def test_overlay_guards_non_cn_languages():
    """en/ko/ja 必须与上游逐字节一致 —— overlay.py 里必须有 cn-only 的闸门。"""
    src = OVERLAY_PY.read_text(encoding="utf-8")
    assert 'current_language() != "cn"' in src, (
        "overlay.py 的 cn-only 闸门不见了：非中文界面会跟着变，等于改了上游行为"
    )


def test_overlay_module_importable_and_cn_only():
    """真导一次（无 Qt 依赖）：中文取中文，其它语言取默认值。"""
    probe = (
        "import sys; sys.path.insert(0, r'%s');"
        "import gui.i18n as i; i.set_language('en');"
        "from gui.i18n import overlay as o;"
        "assert o.field_label('learning_rate') == 'learning_rate', 'en 不该被改';"
        "i.set_language('cn');"
        "assert o.field_label('learning_rate') == '学习率', 'cn 没拿到中文';"
        "assert o.group_title('Paths') == '路径';"
        "assert o.preset_label('low_vram', 'Low VRAM (8GB)') == '低显存（8GB）';"
        "print('ok')"
    ) % ROOT.as_posix()
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT
    )
    if "ModuleNotFoundError" in done.stderr:
        pytest.skip(f"缺依赖，跳过真导入检查：{done.stderr.strip().splitlines()[-1]}")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# --- 4. 补丁点自身 -----------------------------------------------------------


PATCH_SITES: dict[str, tuple[str, ...]] = {
    "gui/tabs/config_tab.py": (
        "cn_overlay.field_label(k)",
        "cn_overlay.group_title(title)",
        "cn_overlay.preset_label(",
    ),
    "gui/tabs/easycontrol_tab.py": ("cn_overlay.field_label(k)",),
    "gui/tabs/methods_tab.py": ('t("method")',),
    "gui/tabs/preprocess/tab.py": ('t("method")',),
}


@pytest.mark.parametrize(("rel", "needles"), sorted(PATCH_SITES.items()))
def test_patch_sites_still_wired(rel: str, needles: tuple[str, ...]):
    """上游改了这些文件时，冲突解决有可能把我们的调用点吃掉 —— 这里点名守护。"""
    text = (ROOT / rel).read_text(encoding="utf-8")
    missing = [n for n in needles if n not in text]
    assert missing == [], f"{rel} 里的下游补丁点丢了：{missing}"


def test_i18n_tables_are_still_valid_python():
    for lang in ("en", "ko", "ja", "cn"):
        path = ROOT / "gui" / "i18n" / f"{lang}.py"
        ast.parse(path.read_text(encoding="utf-8"))
