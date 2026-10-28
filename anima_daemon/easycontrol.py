"""EasyControl variants — the Qt GUI's EasyControl tab, without Qt.

Two kinds of variant share the page:

* **method variants** — ``configs/gui-methods/*.toml`` whose
  ``[variant].family = "easycontrol"``. The variant stem *is* the train.py
  method; edits are sparse top-level writebacks into that file (the GUI's
  Config form does the same), so the ``[[datasets]]`` blueprint survives.
* **descriptor variants** — ``configs/easycontrol/<stem>.toml`` referenced by
  ``EASYADAPTER=<stem>``. The form edits the scalar knobs of the top level and
  the ``[staging] / [preprocess] / [training]`` tables in place (tomlkit keeps
  comments, arrays and the ``[[datasets]]`` blueprint untouched).

Preprocess and train both go through the daemon's serial queue, so the WebUI
and the AI tools see the same job ledger.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomlkit

from . import config, studio

DESC_DIR = config.ROOT / "configs" / "easycontrol"
_DESC_TABLE_ORDER = ("staging", "preprocess", "training")
_DESC_SKIP_TABLES = frozenset({"general", "datasets", "variant"})
_VARIANT_ENV = {"colorize": "colorize"}


class EasyControlError(ValueError):
    """Caller error → HTTP 400 (same contract as ``studio.StudioError``)."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _descriptors() -> list[dict]:
    out: list[dict] = []
    if not DESC_DIR.is_dir():
        return out
    for p in sorted(DESC_DIR.glob("*.toml")):
        try:
            doc = tomllib.loads(p.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        var = doc.get("variant")
        if isinstance(var, dict) and var.get("family") == "easycontrol":
            out.append(
                {
                    "value": p.stem,
                    "kind": "descriptor",
                    "label": str(var.get("label") or p.stem),
                    "order": int(var.get("order", 100)),
                    "name": str(doc.get("name") or p.stem),
                }
            )
    out.sort(key=lambda v: (v["order"], v["value"]))
    return out


def _method_variants() -> list[dict]:
    cio = studio._gui_config()
    out = []
    for name in cio.list_gui_variants("easycontrol"):
        if name.startswith("custom/"):
            continue
        meta = cio.variant_metadata(name) or {}
        out.append(
            {
                "value": name,
                "kind": "method",
                "label": str(meta.get("label") or name),
                "order": 0,
            }
        )
    return out


def _variants() -> list[dict]:
    return _method_variants() + _descriptors()


def _find(name: str) -> dict:
    for item in _variants():
        if item["value"] == name:
            return item
    raise EasyControlError(f"没有这个 EasyControl 变体：{name}")


def list_variants() -> dict:
    schema = studio.config_schema()
    return {
        "variants": _variants(),
        "presets": schema["presets"],
        "groups": schema["groups"],
        "descriptorDir": str(DESC_DIR),
    }


# --------------------------------------------------------------------------- #
# Form shapes
# --------------------------------------------------------------------------- #


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "string"


def _plain(value: Any) -> Any:
    """Strip tomlkit wrappers so the JSON form gets native Python values."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _editable(value: Any) -> bool:
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return not any(isinstance(x, dict) for x in value)
    return True


def _field(key: str, value: Any, group: str, help_text: dict[str, str]) -> dict:
    return {
        "key": key,
        "group": group,
        "type": _value_type(value),
        "value": _plain(value),
        "help": help_text.get(key, ""),
    }


def _descriptor_detail(item: dict) -> dict:
    path = DESC_DIR / f"{item['value']}.toml"
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    help_text = studio._field_help()
    groups: list[dict] = []

    top = {
        k: v
        for k, v in doc.items()
        if not isinstance(v, (dict, list)) and k not in _DESC_SKIP_TABLES
    }
    if top:
        groups.append(
            {
                "name": "",
                "label": "顶部标量",
                "fields": [_field(k, v, "", help_text) for k, v in top.items()],
            }
        )

    tables = [k for k in doc if isinstance(doc[k], dict)]
    ordered = [k for k in _DESC_TABLE_ORDER if k in tables] + [
        k for k in tables if k not in _DESC_TABLE_ORDER and k not in _DESC_SKIP_TABLES
    ]
    for table in ordered:
        fields = [
            _field(f"{table}.{k}" if table else k, v, table, help_text)
            for k, v in doc[table].items()
            if _editable(v)
        ]
        if fields:
            groups.append({"name": table, "label": f"[{table}]", "fields": fields})

    try:
        shown = str(path.relative_to(config.ROOT))
    except ValueError:
        shown = str(path)
    return {
        "value": item["value"],
        "kind": "descriptor",
        "label": item["label"],
        "path": shown,
        "preset": None,
        "groups": groups,
        "note": (
            "描述符变体：表单只改上面这些标量；注释、数组格式与 [[datasets]] "
            "蓝图原样保留。"
        ),
    }


def _method_detail(item: dict, preset: str) -> dict:
    fields = studio._config_fields(
        {"variant": item["value"], "preset": preset, "overrides": {}}
    )
    groups: list[dict] = []
    by_name: dict[str, dict] = {}
    for f in fields:
        group = str(f.get("group") or "Other")
        if group not in by_name:
            entry = {
                "name": group,
                "label": str(f.get("groupLabel") or group),
                "fields": [],
            }
            by_name[group] = entry
            groups.append(entry)
        by_name[group]["fields"].append(
            {
                "key": f["key"],
                "group": group,
                "type": f["type"],
                "value": f["value"],
                "help": f.get("help") or "",
                "basic": bool(f.get("basic")),
                "default": f.get("default"),
                "choices": f.get("choices") or [],
            }
        )
    cio = studio._gui_config()
    path = cio.variant_path(item["value"])
    return {
        "value": item["value"],
        "kind": "method",
        "label": item["label"],
        "path": str(path.relative_to(config.ROOT)),
        "preset": preset,
        "groups": groups,
        "note": (
            "方法变体：字段来自 base → preset → gui-methods 合并链；保存只写你改过的"
            "顶层字段，[[datasets]] 蓝图保持原样。"
        ),
    }


def detail(name: str, preset: str = "default") -> dict:
    item = _find(name)
    if item["kind"] == "descriptor":
        return _descriptor_detail(item)
    return _method_detail(item, preset or "default")


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


def _coerce(raw: Any, old: Any, key: str) -> Any:
    """Convert a JSON value to the type the TOML field already had."""
    try:
        if isinstance(old, bool):
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw)
        if isinstance(old, int) and not isinstance(old, bool):
            return int(raw)
        if isinstance(old, float):
            return float(raw)
        if isinstance(old, str):
            return "" if raw is None else str(raw)
        if isinstance(old, list):
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, str):
                # One item per line is the WebUI's list editor; JSON stays supported.
                text = raw.strip()
                if text.startswith("["):
                    import json

                    items = json.loads(text)
                    if not isinstance(items, list):
                        raise ValueError("要一个列表")
                else:
                    items = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                raise ValueError("要一个列表")
            # `target_res = [896, 1024]` must stay ints; the textarea only gives
            # strings, so type each item by the first existing element.
            if old and isinstance(old[0], (int, float)) and not isinstance(old[0], bool):
                item_type = int if isinstance(old[0], int) else float
                return [item_type(item) for item in items]
            return items
    except (TypeError, ValueError) as e:
        raise EasyControlError(f"{key} 的值不合法：{e}") from None
    return raw


def _apply_values(target: dict, values: dict[str, Any], prefix: str = "") -> int:
    changed = 0
    for key, raw in values.items():
        if not isinstance(target, dict) or key not in target:
            raise EasyControlError(f"未知字段 {prefix}{key}")
        old = target[key]
        if not _editable(old):
            raise EasyControlError(f"{prefix}{key} 不是可编辑的标量/列表")
        new = _coerce(raw, old, f"{prefix}{key}")
        if new != _plain(old):
            target[key] = new
            changed += 1
    return changed


def save(name: str, body: dict) -> dict:
    item = _find(name)
    values = body.get("values")
    if not isinstance(values, dict):
        raise EasyControlError("缺少 values")
    if item["kind"] == "descriptor":
        path = DESC_DIR / f"{name}.toml"
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
        changed = 0
        top: dict[str, Any] = {}
        per_table: dict[str, dict[str, Any]] = {}
        for raw_key, raw in values.items():
            key = str(raw_key)
            if "." in key:
                table, sub = key.split(".", 1)
                per_table.setdefault(table, {})[sub] = raw
            else:
                top[key] = raw
        changed += _apply_values(doc, top)
        for table, items in per_table.items():
            if table in _DESC_SKIP_TABLES or not isinstance(doc.get(table), dict):
                raise EasyControlError(f"未知分组 [{table}]")
            changed += _apply_values(doc[table], items, prefix=f"[{table}].")
        if changed:
            path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        return {**detail(name), "changed": changed}

    path = studio._gui_config().variant_path(name)
    if not path.is_file():
        raise EasyControlError(f"变体文件不存在：{path}")
    preset = str(body.get("preset") or "default")
    # Method variants merge base → preset → gui-methods/<variant>.toml. Compare
    # against that merged view, not against the file: a base-only key is absent
    # from the file and must not be copied in just because the form sent it.
    current = {
        f["key"]: f["value"]
        for g in _method_detail(item, preset)["groups"]
        for f in g["fields"]
    }
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    changed = 0
    for raw_key, raw in values.items():
        key = str(raw_key)
        if key not in current:
            raise EasyControlError(f"未知字段 {key}")
        new = _coerce(raw, current[key], key)
        if new != current[key]:
            doc[key] = new
            changed += 1
    if changed:
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return {**detail(name, preset), "changed": changed}


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def _cache_dir(name: str, item: dict) -> Path:
    if item["kind"] == "descriptor":
        slug = item.get("name") or name
        return config.ROOT / "post_image_dataset" / "easycontrol" / slug / "cond"
    rel = (
        "post_image_dataset/easycontrol/colorize/cond"
        if name == "colorize"
        else "post_image_dataset/easycontrol"
    )
    return config.ROOT / rel


def preprocess(name: str, manager: Any) -> dict:
    item = _find(name)
    argv = ["tasks.py", "easycontrol-preprocess"]
    if item["kind"] == "descriptor":
        argv.append("--no-skip_mangafy")
    adapter = name if item["kind"] == "descriptor" else _VARIANT_ENV.get(name)
    job = manager.submit_command(
        label=f"easycontrol:{name}:preprocess",
        argv=argv,
        extra_env={"EASYADAPTER": adapter} if adapter else None,
        start=True,
        stall_timeout=0,
    )
    return {
        "value": name,
        "job_id": job.id,
        "state": job.state,
        "stage": "preprocess",
        "cacheDir": str(_cache_dir(name, item)),
        "adapter": adapter,
    }


def train(name: str, preset: str, manager: Any) -> dict:
    item = _find(name)
    preset = preset or "default"
    if item["kind"] == "method":
        job = manager.submit(
            method=name,
            preset=preset,
            methods_subdir="gui-methods",
            start=True,
        )
        return {
            "value": name,
            "job_id": job.id,
            "state": job.state,
            "stage": "train",
            "kind": "method",
        }

    try:
        from scripts.tasks.training import _easy_train_extra
    except Exception as e:  # noqa: BLE001 — missing engine pieces must be reported
        raise EasyControlError(f"读不到引擎的训练辅助函数：{e}") from None
    try:
        extra = _easy_train_extra(name, [])
    except SystemExit as e:
        raise EasyControlError(str(e)) from None
    job = manager.submit(
        method="easycontrol",
        preset=preset,
        methods_subdir=None,
        extra=extra,
        start=True,
    )
    return {
        "value": name,
        "job_id": job.id,
        "state": job.state,
        "stage": "train",
        "kind": "descriptor",
        "extra": extra,
    }
