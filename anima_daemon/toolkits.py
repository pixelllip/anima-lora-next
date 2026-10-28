"""Toolkit pages ported from the Qt GUI: merge, TensorBoard, anime_tools.

The Qt GUI runs these as children of the GUI process. The WebUI can't do that
(it is a stdlib proxy), so they live in the daemon instead:

* **merge** rides the existing serial job queue (``manager.submit_command``),
  exactly like ``anime_studio_*`` preprocess/train jobs, so the merge log is
  the same SSE stream the jobs tab already renders;
* **TensorBoard** and the **anime_tools** panel are long-lived helpers owned
  by the daemon — a page reload (or the phone) re-attaches to the one already
  running instead of spawning a second copy.

Everything here is Qt-free and imports ``gui.core.*`` only lazily, matching
``anima_daemon.studio``. Paths are resolved against ``config.ROOT`` — never
``%APPDATA%``.
"""

from __future__ import annotations

import json
import shlex
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import tomlkit

from . import config
from .proc import create_time, is_alive, kill_tree, spawn_detached

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ToolkitError(ValueError):
    """Caller error → HTTP 400 (same contract as ``studio.StudioError``)."""


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

DEFAULT_DIT = config.ROOT / "models" / "diffusion_models" / "anima-base-v1.0.safetensors"
MERGE_MODES = ("dit", "loras", "extract")
MERGE_DTYPES = ("bf16", "fp16", "fp32")
MERGE_NORMALIZE = ("global", "per_module", "off")
_EXTRACT_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _adapter_dirs() -> dict[str, Path]:
    """Adapter roots that actually contain ``.safetensors`` files.

    Mirrors ``gui.core.discovery._adapter_dirs`` (the merge tab's dropdown) but
    reads from ``config.ROOT`` instead of importing the GUI package at module
    load.
    """
    dirs: dict[str, Path] = {}
    for name, path in (
        ("output/ckpt", config.ROOT / "output" / "ckpt"),
        ("output_temp", config.ROOT / "output_temp"),
        ("models/diffusion_models", config.ROOT / "models" / "diffusion_models"),
    ):
        if path.is_dir() and any(path.glob("*.safetensors")):
            dirs[name] = path
    for parent, label in (
        (config.ROOT / "output" / "ckpt", "output/ckpt"),
        (config.ROOT / "output_temp", "output_temp"),
    ):
        if not parent.is_dir():
            continue
        for p in sorted(parent.iterdir()):
            if (
                p.is_dir()
                and not p.name.endswith("-checkpoint-state")
                and any(p.glob("*.safetensors"))
            ):
                dirs[f"{label}/{p.name}"] = p
    return dirs


def _files_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix == ".safetensors"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _scan_adapter(path: Path) -> dict:
    """Read safetensors keys and classify bakeability.

    Same verdict rules as ``gui/tabs/merge_tab._scan_adapter``; kept here so
    importing the merge tab never drags PySide6 into the daemon.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return {
            "verdict": "safetensors 没装（引擎 venv 不完整）",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    counts = {
        "lora_down": 0,
        "lora_up_weight": 0,
        "lora_ups": 0,
        "postfix": 0,
        "other": 0,
    }
    metadata_mode: str | None = None
    try:
        with safe_open(str(path), framework="np") as f:
            keys = list(f.keys())
            try:
                meta = f.metadata() or {}
                metadata_mode = meta.get("ss_mode")
            except Exception:  # noqa: BLE001 — metadata is optional
                pass
    except Exception as exc:  # noqa: BLE001 — surface any read error in UI
        return {
            "verdict": f"读不出来：{exc!s}",
            "severity": "unknown",
            "details": [],
            "counts": {},
        }

    is_postfix = metadata_mode in {"postfix", "postfix_exp", "postfix_func", "cond"}
    for k in keys:
        if k.endswith(".lora_up_weight"):
            counts["lora_up_weight"] += 1
        elif ".lora_ups." in k:
            counts["lora_ups"] += 1
        elif k.endswith(".lora_down.weight"):
            counts["lora_down"] += 1
        elif is_postfix:
            counts["postfix"] += 1
        else:
            counts["other"] += 1

    details = []
    if counts["lora_down"]:
        details.append(f"{counts['lora_down']} 个 LoRA 键")
    if counts["lora_up_weight"] or counts["lora_ups"]:
        n = counts["lora_up_weight"] + counts["lora_ups"]
        details.append(f"{n} 个 HydraLoRA 键")
    if counts["postfix"]:
        details.append(f"{counts['postfix']} 个 {metadata_mode} 键")

    is_hydra = bool(counts["lora_up_weight"] or counts["lora_ups"])
    is_postfix_only = is_postfix and not counts["lora_down"]
    if is_hydra:
        verdict, severity = "HydraLoRA：不能直接烘焙进 DiT（先转成普通 LoRA）", "block"
    elif is_postfix_only:
        verdict, severity = "只有 postfix 键：不是可烘焙的 LoRA", "block"
    elif counts["lora_down"]:
        verdict, severity = "可烘焙（普通 LoRA）", "ok"
    else:
        verdict, severity = "认不出这是不是 LoRA", "unknown"

    return {"verdict": verdict, "severity": severity, "details": details, "counts": counts}


def _file_size(path: Path) -> dict:
    try:
        st = path.stat()
    except OSError:
        return {"size": None, "mtime": None}
    return {"size": st.st_size, "mtime": st.st_mtime}


def _safe_child(directory: Path, name: str) -> Path:
    """Resolve a user-supplied file name inside *directory* (no traversal)."""
    if not name or name != Path(name).name:
        raise ToolkitError("文件名不合法")
    target = (directory / name).resolve()
    try:
        target.relative_to(directory.resolve())
    except ValueError:
        raise ToolkitError("文件不在这个适配器目录里") from None
    if not target.is_file():
        raise ToolkitError(f"文件不存在：{name}")
    return target


def _abs_path(raw: Any) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else (config.ROOT / path)


def merge_state(dir_name: str = "", file_name: str = "") -> dict:
    """Adapter dirs + files + the selected file's scan (the merge tab's data)."""
    dirs = _adapter_dirs()
    names = list(dirs)
    active = dir_name or (names[0] if names else "")
    if active and active not in dirs:
        raise ToolkitError(f"没有这个适配器目录：{active}")
    files: list[dict] = []
    scan = None
    selected = ""
    if active:
        for p in _files_in(dirs[active]):
            files.append({"name": p.name, **_file_size(p)})
        wanted = file_name or (files[0]["name"] if files else "")
        if wanted:
            target = _safe_child(dirs[active], wanted)
            selected = target.name
            scan = {
                "name": target.name,
                **_file_size(target),
                **_scan_adapter(target),
            }
    return {
        "dirs": {name: str(path) for name, path in dirs.items()},
        "dir": active,
        "files": files,
        "selected": selected,
        "scan": scan,
        "defaultDit": str(DEFAULT_DIT),
        "modes": list(MERGE_MODES),
        "dtypes": list(MERGE_DTYPES),
        "normalize": list(MERGE_NORMALIZE),
    }


def _merge_argv(body: dict, dirs: dict[str, Path]) -> tuple[list[str], str]:
    mode = str(body.get("mode") or "dit").lower()
    if mode not in MERGE_MODES:
        raise ToolkitError("mode 只能是 dit / loras / extract")

    if mode == "dit":
        dir_name = str(body.get("dir") or "")
        if dir_name not in dirs:
            raise ToolkitError("先选一个适配器目录")
        adapter = _safe_child(dirs[dir_name], str(body.get("adapter") or ""))
        dit = _abs_path(body.get("dit")) or DEFAULT_DIT
        argv = [
            "scripts/toolkits/merge_to_dit.py",
            "--adapter",
            str(adapter),
            "--dit",
            str(dit),
            "--multiplier",
            f"{float(body.get('multiplier') or 1.0):g}",
            "--dtype",
            _choice(body.get("dtype"), MERGE_DTYPES, "bf16"),
        ]
        out = _abs_path(body.get("out"))
        if out:
            argv += ["--out", str(out)]
        if body.get("allow_partial"):
            argv.append("--allow-partial")
        return argv, f"merge:dit:{adapter.name}"

    if mode == "loras":
        dir_name = str(body.get("dir") or "")
        if dir_name not in dirs:
            raise ToolkitError("先选一个适配器目录")
        raw_names = body.get("adapters") or []
        if not isinstance(raw_names, list) or len(raw_names) < 2:
            raise ToolkitError("融合 LoRA 至少要选两个文件")
        adapters = [_safe_child(dirs[dir_name], str(n)) for n in raw_names]
        argv = [
            "scripts/toolkits/merge_loras.py",
            *[str(p) for p in adapters],
            "--normalize",
            _choice(body.get("normalize"), MERGE_NORMALIZE, "global"),
            "--dtype",
            _choice(body.get("dtype"), MERGE_DTYPES, "bf16"),
        ]
        out = _abs_path(body.get("out"))
        if out:
            argv += ["--out", str(out)]
        weights = str(body.get("weights") or "").strip()
        if weights:
            if len(weights.split(",")) != len(adapters):
                raise ToolkitError(
                    f"权重个数（{len(weights.split(','))}）要和选中的 LoRA 个数（{len(adapters)}）一致"
                )
            argv += ["--weights", weights]
        label = "merge:loras:" + "+".join(p.stem for p in adapters)
        return argv, label[:120]

    tuned = _abs_path(body.get("tuned"))
    out = _abs_path(body.get("out"))
    if tuned is None:
        raise ToolkitError("提取模式要先选 tuned 检查点")
    if out is None:
        raise ToolkitError("提取模式要先填输出路径")
    base = _abs_path(body.get("base")) or DEFAULT_DIT
    argv = [
        "scripts/toolkits/extract_delta_lora.py",
        "--base",
        str(base),
        "--tuned",
        str(tuned),
        "--rank",
        str(int(body.get("rank") or 96)),
        "--out",
        str(out),
        "--save_dtype",
        _EXTRACT_DTYPE[_choice(body.get("dtype"), MERGE_DTYPES, "bf16")],
    ]
    if body.get("include_adaln"):
        lay = str(body.get("adaln_layout") or "runtime").lower()
        if lay not in ("runtime", "comfy"):
            raise ToolkitError("adaln_layout 只能是 runtime / comfy")
        argv += ["--include_adaln", "--adaln_layout", lay]
    return argv, f"merge:extract:{tuned.stem}"


def _choice(raw: Any, allowed: tuple[str, ...], default: str) -> str:
    value = str(raw or default).lower()
    if value not in allowed:
        raise ToolkitError(f"只能是 {' / '.join(allowed)}")
    return value


def merge_run(body: dict, manager: Any) -> dict:
    """Submit one merge/extract/analyze command on the serial job queue."""
    dirs = _adapter_dirs()
    mode = str(body.get("mode") or "dit").lower()
    if mode == "analyze":
        dir_name = str(body.get("dir") or "")
        if dir_name not in dirs:
            raise ToolkitError("先选一个适配器目录")
        raw_names = body.get("adapters") or []
        if not isinstance(raw_names, list) or len(raw_names) < 2:
            raise ToolkitError("干扰分析至少要选两个 LoRA")
        adapters = [_safe_child(dirs[dir_name], str(n)) for n in raw_names]
        argv = [
            "scripts/toolkits/merge_loras.py",
            *[str(p) for p in adapters],
            "--analyze",
        ]
        weights = str(body.get("weights") or "").strip()
        if weights:
            if len(weights.split(",")) != len(adapters):
                raise ToolkitError("权重个数要和选中的 LoRA 个数一致")
            argv += ["--weights", weights]
        label = "merge:analyze:" + "+".join(p.stem for p in adapters)
    else:
        argv, label = _merge_argv(body, dirs)

    job = manager.submit_command(
        label=label[:120],
        argv=argv,
        start=True,
        stall_timeout=0,
    )
    return {
        "job_id": job.id,
        "state": job.state,
        "label": label,
        "argv": argv,
        "mode": mode,
    }


# --------------------------------------------------------------------------- #
# TensorBoard
# --------------------------------------------------------------------------- #

TB_PORT = 6006
TB_LOG = config.ROOT / "output" / "tensorboard.log"
TB_STATE = config.ROOT / "output" / "tensorboard.json"
TB_BASE = config.ROOT / "output" / "logs"


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _pick_port(preferred: int = TB_PORT) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise ToolkitError("6006–6025 都被占用了，腾一个再开 TensorBoard")


def _read_tb_state() -> dict:
    try:
        data = json.loads(TB_STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_tb_state(data: dict) -> None:
    TB_STATE.parent.mkdir(parents=True, exist_ok=True)
    TB_STATE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _run_dirs(base: Path | None = None, max_depth: int = 3) -> list[Path]:
    """Directories under *base* that hold TensorBoard event files."""
    found: list[Path] = []
    base = base or TB_BASE
    if not base.is_dir():
        return found

    def walk(d: Path, depth: int) -> None:
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        if any(
            e.name.startswith("events.out.tfevents") and e.is_file()
            for e in entries
        ):
            found.append(d)
            return
        if depth >= max_depth:
            return
        for e in entries:
            if e.is_dir():
                walk(e, depth + 1)

    walk(base, 0)
    return found


def tb_runs() -> list[dict]:
    rows = []
    for d in _run_dirs():
        try:
            st = d.stat()
            rel = d.relative_to(TB_BASE).as_posix()
        except OSError:
            continue
        rows.append({"path": rel, "mtime": st.st_mtime})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def _tb_alive(state: dict) -> bool:
    pid = state.get("pid")
    ct = state.get("createTime")
    if not pid or not is_alive(int(pid), float(ct or 0)):
        return False
    if _port_open(int(state.get("port") or 0)):
        return True
    # TensorBoard imports torch-adjacent packages before it binds; treat the
    # first few seconds as "starting", not "dead", or the UI would clear the
    # state we just wrote and the stop button would lose the pid.
    age = time.time() - float(state.get("startedAt") or 0)
    return age < 45.0


def tensorboard_state() -> dict:
    state = _read_tb_state()
    running = _tb_alive(state)
    if not running and state:
        # The process is gone (machine reboot / manual kill): clear the stale
        # marker so the UI doesn't offer "stop" for nothing.
        state = {}
        try:
            TB_STATE.unlink()
        except OSError:
            pass
    return {
        "running": running,
        "ready": bool(running and _port_open(int(state.get("port") or 0))),
        "port": state.get("port") if running else None,
        "url": f"http://127.0.0.1:{state['port']}" if running else None,
        "logDir": state.get("logDir") if running else None,
        "base": str(TB_BASE),
        "runs": tb_runs(),
    }


def tensorboard_start(body: dict) -> dict:
    run = str(body.get("run") or "").strip()
    log_dir = TB_BASE
    if run and run != "all":
        candidate = (TB_BASE / run).resolve()
        try:
            candidate.relative_to(TB_BASE.resolve())
        except ValueError:
            raise ToolkitError("run 不在 output/logs 里") from None
        if not candidate.is_dir():
            raise ToolkitError(f"没有这个 run：{run}")
        log_dir = candidate
    if not log_dir.is_dir():
        raise ToolkitError("还没有训练日志目录（先跑一次训练）")

    state = _read_tb_state()
    if _tb_alive(state):
        if str(state.get("logDir")) == str(log_dir):
            return tensorboard_state()
        tensorboard_stop()

    port = _pick_port()
    TB_LOG.parent.mkdir(parents=True, exist_ok=True)
    proc = spawn_detached(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            f"--logdir={log_dir}",
            f"--port={port}",
            "--bind_all",
        ],
        cwd=config.ROOT,
        stdout_path=TB_LOG,
    )
    _write_tb_state(
        {
            "pid": proc.pid,
            "createTime": create_time(proc.pid),
            "port": port,
            "logDir": str(log_dir),
            "startedAt": time.time(),
        }
    )
    return tensorboard_state()


def tensorboard_stop() -> dict:
    state = _read_tb_state()
    if state.get("pid"):
        try:
            kill_tree(int(state["pid"]), grace_seconds=3.0)
        except Exception:  # noqa: BLE001 — already gone is fine
            pass
    try:
        TB_STATE.unlink()
    except OSError:
        pass
    return tensorboard_state()


def tensorboard_remove(body: dict) -> dict:
    run = str(body.get("run") or "").strip()
    if not run or run == "all":
        raise ToolkitError("要指定一个 run")
    target = (TB_BASE / run).resolve()
    try:
        target.relative_to(TB_BASE.resolve())
    except ValueError:
        raise ToolkitError("run 不在 output/logs 里") from None
    if target == TB_BASE.resolve():
        raise ToolkitError("不能删整个 logs 目录")
    if not target.is_dir():
        raise ToolkitError(f"没有这个 run：{run}")
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    return tensorboard_state()


# --------------------------------------------------------------------------- #
# anime_tools panel
# --------------------------------------------------------------------------- #

_ANIME_CACHE: dict[str, Any] = {"at": 0.0, "url": None}
_ANIME_LOCK = threading.Lock()


def _anime_panel():
    """Import the Qt-free launcher module (lazy: pulls psutil / anime_tools)."""
    from gui.core import anime_tools_panel

    return anime_tools_panel


def anime_state() -> dict:
    panel = _anime_panel()
    home = config.ROOT.resolve()
    now = time.time()
    with _ANIME_LOCK:
        if now - float(_ANIME_CACHE.get("at") or 0) < 2.0:
            url = _ANIME_CACHE.get("url")
        else:
            url = panel.find_running(home, timeout=0.5)
            _ANIME_CACHE.update(at=now, url=url)
    return {
        "running": bool(url),
        "url": url,
        "home": str(home),
        "log": str(panel.log_path(home)),
        "port": _port_of(url),
    }


def anime_start(body: dict | None = None) -> dict:
    body = body or {}
    panel = _anime_panel()
    home = config.ROOT.resolve()
    current = anime_state()
    if current["running"]:
        return current
    source = str(body.get("source") or config.ROOT / "image_dataset")
    warnings: list[str] = []
    try:
        seeded = panel.seed_settings(source, home=home)
        warnings = list(seeded.warnings)
    except Exception as exc:  # noqa: BLE001 — seeding is best effort
        warnings = [f"设置预置失败：{exc}"]
    log = panel.launch(home)
    with _ANIME_LOCK:
        _ANIME_CACHE.update(at=0.0, url=None)
    return {
        **anime_state(),
        "starting": True,
        "log": str(log),
        "warnings": warnings,
    }


def _port_of(url: str | None) -> Optional[int]:
    if not url:
        return None
    try:
        return int(url.rsplit(":", 1)[1].split("/")[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------- #
# Experimental distill pages: Turbo + Soup
# --------------------------------------------------------------------------- #
#
# Both are *sectioned* TOMLs (flat top-level scalars + one level of tables),
# not the Anima method/preset chain, so they share one generic form. Editing is
# sparse + tomlkit-based: comments and ordering survive a save, which matters
# because those comments are the only in-file documentation of the knobs.

DISTILL = {
    "turbo": {
        "path": "configs/methods/turbo.toml",
        "label": "Turbo 蒸馏",
        "task": "turbo",
        "note": (
            "Turbo 走自己的蒸馏循环（scripts/distill_turbo/），不是 train.py；"
            "数据复用 LoRA 的 post_image_dataset/lora 缓存。"
        ),
    },
    "soup": {
        "path": "configs/soup/soup.toml",
        "label": "ΔW-soup（uncond 初始化 + 权重汤）",
        "task": "soup",
        "note": (
            "Soup 直接提交 python -m scripts.soup.pipeline（不经 tasks.py，"
            "否则会在守护进程里嵌套提交、卡死串行队列）。"
        ),
    },
}


def _distill_spec(kind: str) -> dict:
    spec = DISTILL.get(kind)
    if spec is None:
        raise ToolkitError(f"没有这个实验页：{kind}")
    return spec


def _distill_path(kind: str) -> Path:
    return config.ROOT / _distill_spec(kind)["path"]


def _plain_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, list):
        return [_plain_value(v) for v in value]
    return value


def _editable(value: Any) -> bool:
    if isinstance(value, dict):
        return False
    if isinstance(value, list):
        return not any(isinstance(x, dict) for x in value)
    return True


def _toml_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "string"


def _toml_groups(doc, skip: set[str] | None = None) -> list[dict]:
    """Top-level scalars + one level of tables → form groups."""
    skip = skip or set()
    groups: list[dict] = []
    top = {
        k: v
        for k, v in doc.items()
        if not isinstance(v, dict) and _editable(v) and k not in skip
    }
    if top:
        groups.append(
            {
                "name": "",
                "label": "基础",
                "fields": [
                    {
                        "key": k,
                        "type": _toml_type(v),
                        "value": _plain_value(v),
                    }
                    for k, v in top.items()
                ],
            }
        )
    for table, body in doc.items():
        if not isinstance(body, dict) or table in skip:
            continue
        fields = [
            {
                "key": f"{table}.{k}",
                "type": _toml_type(v),
                "value": _plain_value(v),
            }
            for k, v in body.items()
            if _editable(v)
        ]
        if fields:
            groups.append({"name": table, "label": f"[{table}]", "fields": fields})
    return groups


def distill_state(kind: str) -> dict:
    spec = _distill_spec(kind)
    path = _distill_path(kind)
    if not path.is_file():
        raise ToolkitError(f"配置文件不存在：{spec['path']}")
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    return {
        "kind": kind,
        "label": spec["label"],
        "path": spec["path"],
        "note": spec["note"],
        "task": spec["task"],
        "groups": _toml_groups(doc),
        "soupArgs": kind == "soup",
    }


def _coerce_toml(raw: Any, old: Any, key: str) -> Any:
    try:
        if isinstance(old, bool):
            if isinstance(raw, str):
                return raw.strip().lower() in {"1", "true", "yes", "on"}
            return bool(raw)
        if isinstance(old, int):
            return int(raw)
        if isinstance(old, float):
            return float(raw)
        if isinstance(old, str):
            return "" if raw is None else str(raw)
        if isinstance(old, list):
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, str):
                text = raw.strip()
                if text.startswith("["):
                    items = json.loads(text)
                    if not isinstance(items, list):
                        raise ValueError("要一个列表")
                else:
                    items = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                raise ValueError("要一个列表")
            if old and isinstance(old[0], (int, float)) and not isinstance(old[0], bool):
                item_type = int if isinstance(old[0], int) else float
                return [item_type(item) for item in items]
            return items
    except (TypeError, ValueError) as e:
        raise ToolkitError(f"{key} 的值不合法：{e}") from None
    return raw


def distill_save(kind: str, values: dict) -> dict:
    path = _distill_path(kind)
    doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    changed = 0
    for raw_key, raw in values.items():
        key = str(raw_key)
        if "." in key:
            table, sub = key.split(".", 1)
            target = doc.get(table)
            if not isinstance(target, dict):
                raise ToolkitError(f"未知分组 [{table}]")
            if sub not in target or not _editable(target[sub]):
                raise ToolkitError(f"未知字段 {key}")
            new = _coerce_toml(raw, target[sub], key)
            if new != _plain_value(target[sub]):
                target[sub] = new
                changed += 1
        else:
            if key not in doc or not _editable(doc[key]):
                raise ToolkitError(f"未知字段 {key}")
            new = _coerce_toml(raw, doc[key], key)
            if new != _plain_value(doc[key]):
                doc[key] = new
                changed += 1
    if changed:
        path.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return {**distill_state(kind), "changed": changed}


def distill_run(kind: str, body: dict, manager: Any) -> dict:
    _distill_spec(kind)  # validates the kind
    if kind == "turbo":
        argv = ["tasks.py", "turbo"]
        label = "turbo"
    else:
        pattern = str(body.get("pathPattern") or "").strip()
        if not pattern:
            raise ToolkitError("Soup 要先填 path_pattern（例如 sincos/*）")
        argv = ["-m", "scripts.soup.pipeline", "--path_pattern", pattern]
        name = str(body.get("name") or "").strip()
        if name:
            argv += ["--name", name]
        try:
            argv += shlex.split(str(body.get("ftArgs") or ""))
        except ValueError as e:
            raise ToolkitError(f"附加参数解析失败：{e}") from None
        label = "soup"
    job = manager.submit_command(
        label=label,
        argv=argv,
        start=True,
        stall_timeout=0,
    )
    return {
        "kind": kind,
        "job_id": job.id,
        "state": job.state,
        "label": label,
        "argv": argv,
    }
