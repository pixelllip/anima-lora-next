"""Training studio — project/dataset state for the full LoRA pipeline.

The daemon already owns the **serial GPU queue**. This module owns the
*project* around it: which images were fetched, which ones a human kept, the
caption sidecars, the generated training config, and which job belongs to
which stage.

Layout (everything lives under the engine root, never in ``%APPDATA%``)::

    training_projects/<slug>/
        project.json          project settings (variant/preset/trigger/...)
        jobs.json             [{id, stage, label, at}] — the stage ledger
        history.jsonl         append-only audit (import/review/tag/train)
        inbox/                fetched images waiting for a human decision
        rejected/             rejected images (kept so they can be restored)
        train_config.toml     generated merged config (training + preprocess)
    image_dataset/<slug>/     accepted images + .txt captions (the master)
    post_image_dataset/resized/<slug>/   derived
    post_image_dataset/lora/<slug>/      VAE/TE/PE caches
    output/ckpt/<slug>/                  trained LoRA

Why the accepted images use ``image_dataset/<slug>`` rather than a private
folder: the engine's GUI already supports a per-run ``path_scope`` that maps
``image_dataset/<scope>`` → ``post_image_dataset/resized/<scope>`` →
``post_image_dataset/lora/<scope>`` → ``output/ckpt/<scope>``. Reusing that
one mechanism means the trainer, preprocess, masks and the ComfyUI trainer
node all agree on the same paths without a second implementation.

Endpoints are wired in :mod:`anima_daemon.server`; the stdlib WebUI
(``anima_lora_webui``) proxies them. Only stdlib + ``toml`` (already an engine
dependency) are used here; no torch / PySide6 import happens at module load.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECTS_DIR = config.ROOT / "training_projects"
IMAGE_DATASET_DIR = config.ROOT / "image_dataset"
RESIZED_DIR = config.ROOT / "post_image_dataset" / "resized"
CACHE_DIR = config.ROOT / "post_image_dataset" / "lora"
MASK_DIR = config.ROOT / "post_image_dataset" / "masks"
OUTPUT_DIR = config.ROOT / "output" / "ckpt"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".avif"}
MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_CAPTION_CHARS = 20_000
USER_AGENT = "ComfyHub-Training-Studio/1.0 (+local)"

#: Unicode slug safety net. Project slugs end up as directory names.
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_JOB_STAGES = ("tag", "preprocess", "train", "krea2")

_LOCK = threading.RLock()

# --------------------------------------------------------------------------- #
# Parameter schema (the Qt GUI's Config tab, exposed to the WebUI)
# --------------------------------------------------------------------------- #
#
# The Qt GUI builds its form from ``gui.core.config_io`` (``_GROUPS`` /
# ``_BASIC``) plus the Chinese help text in ``explanations/guides/cn/_fields.json``.
# The WebUI must show the **same** fields with the same grouping; keeping the
# metadata here (not in JS) means adding a field to the GUI chain surfaces it
# automatically after a refresh.
_GROUP_ORDER = ("Architecture", "Training", "Performance", "Samples", "Paths", "Other")
_GROUP_LABELS = {
    "Architecture": "网络结构",
    "Training": "训练预算",
    "Performance": "显存与性能",
    "Samples": "采样预览",
    "Paths": "路径与产物",
    "Other": "其它",
}

_FIELD_LABELS = {
    "network_dim": "LoRA 秩 network_dim",
    "network_alpha": "LoRA alpha",
    "network_module": "网络模块",
    "network_args": "网络额外参数 network_args",
    "learning_rate": "学习率 learning_rate",
    "max_train_epochs": "训练轮数 max_train_epochs",
    "save_every_n_epochs": "每 N 轮保存",
    "checkpointing_epochs": "可续训状态间隔",
    "gradient_accumulation_steps": "梯度累积",
    "caption_dropout_rate": "caption 丢弃率",
    "use_shuffled_caption_variants": "使用 caption 洗牌变体",
    "optimizer_type": "优化器",
    "lr_scheduler": "学习率调度",
    "timestep_sampling": "时间步采样",
    "discrete_flow_shift": "flow shift",
    "masked_loss": "只算未遮罩区域",
    "sample_prompts": "采样提示词",
    "sample_every_n_epochs": "每 N 轮采样",
    "sample_at_first": "开训先采样",
    "sample_decode_inline": "采样就地解码",
    "attn_mode": "注意力后端",
    "gradient_checkpointing": "梯度检查点",
    "unsloth_offload_checkpointing": "检查点卸载到 CPU",
    "activation_memory_budget": "激活内存预算",
    "blocks_to_swap": "块交换 blocks_to_swap",
    "torch_compile": "torch.compile",
    "mixed_precision": "混合精度",
    "vae_chunk_size": "VAE 分块",
    "vae_disable_cache": "关闭 VAE 缓存",
    "use_vae_cache": "VAE latent 缓存",
    "use_text_cache": "文本编码缓存",
    "skip_cache_check": "跳过缓存校验",
    "use_cmmd": "CMMD 验证",
    "pretrained_model_name_or_path": "底模路径",
    "qwen3": "文本编码器路径",
    "vae": "VAE 路径",
    "vocab_pack": "CJK 词表包",
    "output_dir": "产物目录",
    "output_name": "产物名",
    "save_model_as": "保存格式",
    "source_image_dir": "原图目录",
    "resized_image_dir": "resize 目录",
    "lora_cache_dir": "缓存目录",
    "path_pattern": "图片过滤 path_pattern",
    "training_comment": "训练备注",
    "use_repa": "REPA v2",
    "repa_mode": "REPA 模式",
    "repa_weight": "REPA 权重",
    "repa_layer": "REPA 层",
    "repa_encoder": "REPA 编码器",
    "repa_target_dog": "REPA DoG 带通",
    "sigma_lowres": "σ 降级训练",
    "sigma_lowres_route": "σ 降级路由",
    "sigma_lowres_threshold": "σ 降级阈值",
    "sigma_lowres_threshold_max": "σ 降级阈值上限",
    "sigma_lowres_yarnsig": "σ 降级 YaRN",
    "sigma_lowres_span": "σ 降级步区间",
    "sigma_lowres_route2": "次降级路由",
    "sigma_lowres_threshold2": "次降级阈值",
    "sigma_lowres_threshold2_max": "次降级阈值上限",
    "sigma_lowres_span2": "次降级步区间",
    "train_adaln": "训练 adaln 投影",
    "adaln_rank": "adaln 秩",
    "adaln_alpha": "adaln alpha",
    "network_train_unet_only": "只训 DiT",
    "network_weights": "热启动权重",
    "dim_from_weights": "秩取自热启动",
    "num_experts": "专家数",
    "use_moe_style": "MoE 布局",
    "use_timestep_mask": "T-LoRA 时间步掩码",
    "min_rank": "最小秩",
    "alpha_rank_scale": "alpha 随秩缩放",
    "sample_ratio": "采样比例",
    "artists_shard": "画师分片",
    "use_valid": "留出验证集",
    "validation_split_num": "验证集图片数",
    "repeat_by_folder_name": "按文件夹名重复",
}

_FIELD_CHOICES = {
    "optimizer_type": ["AdamW8bit", "AdamW", "Lion", "Prodigy", "adafactor"],
    "lr_scheduler": ["constant", "cosine", "cosine_with_restarts", "polynomial", "linear"],
    "mixed_precision": ["bf16", "fp16", "no"],
    "attn_mode": ["flex", "flash", "torch", "sdpa"],
    "timestep_sampling": ["sigmoid", "uniform", "sigma", "shift"],
    "save_model_as": ["safetensors"],
    "router_source": ["sigma", "fei", "pooled_text"],
    "use_moe_style": ["shared_A", "false"],
    "sample_decode_inline": ["auto", "true", "false"],
    "sample_ratio": ["1", "0.5", "0.25", "0.1"],
}

_FIELD_ORDER = {
    key: i
    for i, key in enumerate(
        (
            "network_dim", "network_alpha", "network_module", "network_args",
            "learning_rate", "max_train_epochs", "save_every_n_epochs",
            "checkpointing_epochs", "gradient_accumulation_steps",
            "optimizer_type", "lr_scheduler", "timestep_sampling",
            "discrete_flow_shift", "caption_dropout_rate",
            "use_shuffled_caption_variants", "masked_loss",
            "gradient_checkpointing", "blocks_to_swap", "mixed_precision",
            "attn_mode", "torch_compile", "output_name", "output_dir",
            "pretrained_model_name_or_path", "qwen3", "vae",
        )
    )
}

_FIELD_HELP: dict[str, str] | None = None


def _field_help() -> dict[str, str]:
    """Chinese field help, shared with the Qt GUI (read once, lazily)."""
    global _FIELD_HELP
    if _FIELD_HELP is None:
        path = config.ROOT / "gui" / "explanations" / "guides" / "cn" / "_fields.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        _FIELD_HELP = {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    return _FIELD_HELP


def _gui_config():
    """Import ``gui.core.config_io`` lazily (never at module load)."""
    try:
        from gui.core import config_io

        return config_io
    except Exception as e:  # noqa: BLE001 — report the real reason to the user
        raise StudioError(f"引擎的配置模块不可用（{e}）；请确认在引擎 venv 里跑 WebUI/守护进程")


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "string"


def _config_merged(project: dict) -> dict:
    """The effective Anima config: base → preset → method, then project overrides."""
    cio = _gui_config()
    variant = str(project.get("variant") or "lora")
    preset = str(project.get("preset") or "low_vram")
    try:
        merged, _origin = cio.merged_gui_variant_preset(variant, preset)
    except Exception as e:  # noqa: BLE001
        raise StudioError(f"合并 {variant}/{preset} 配置失败：{e}")
    merged["path_scope"] = str(project.get("slug") or "")
    merged["output_name"] = str(project.get("output_name") or project.get("slug") or "")
    merged.update(project.get("overrides") or {})
    return merged


def _config_baseline(project: dict) -> dict:
    """The preset defaults with **no** project overrides (used to reset a field)."""
    return _config_merged({**project, "overrides": {}})


def _config_fields(project: dict) -> list[dict]:
    cio = _gui_config()
    merged = _config_merged(project)
    baseline = _config_baseline(project)
    overrides = project.get("overrides") or {}
    help_text = _field_help()
    fields: list[dict] = []
    for key, value in merged.items():
        key = str(key)
        if key in cio._SKIP or key == "path_scope" or key.startswith("_"):
            continue
        group = cio._K2G.get(key, "Other")
        fields.append(
            {
                "key": key,
                "label": _FIELD_LABELS.get(key, key),
                "group": group,
                "groupLabel": _GROUP_LABELS.get(group, group),
                "basic": bool(cio.is_basic_field(key)),
                "type": _value_type(value),
                "value": value,
                "default": baseline.get(key),
                "overridden": key in overrides,
                "help": help_text.get(key, ""),
                "choices": list(_FIELD_CHOICES.get(key, ())),
            }
        )
    fields.sort(
        key=lambda f: (
            _GROUP_ORDER.index(f["group"]) if f["group"] in _GROUP_ORDER else len(_GROUP_ORDER),
            0 if f["basic"] else 1,
            _FIELD_ORDER.get(f["key"], 100),
            f["key"],
        )
    )
    return fields


def config_schema() -> dict:
    """Variant/preset dropdowns + group labels (the Qt GUI's Config tab shape)."""
    cio = _gui_config()
    variants = []
    for method in cio.list_methods():
        meta = cio.variant_metadata(method) or {}
        variants.append(
            {
                "value": method,
                "label": str(meta.get("label") or method),
                "family": method,
                "description": str(meta.get("description") or ""),
            }
        )
    presets = [
        {"value": name, "label": str((meta or {}).get("label") or name)}
        for name, meta in cio.list_hardware_presets()
    ]
    return {
        "variants": variants,
        "presets": presets,
        "groups": [
            {"name": name, "label": _GROUP_LABELS.get(name, name)} for name in _GROUP_ORDER
        ],
    }


class _Unset:
    """Sentinel: the caller asked to drop this override (reset to preset)."""


_UNSET = _Unset()


def _coerce_config_value(key: str, raw: Any, baseline: Any) -> Any:
    """Coerce a form value to the type of the merged config (never silently)."""
    if raw is None:
        return _UNSET
    if isinstance(baseline, bool):
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        low = str(raw).strip().lower()
        if low in {"true", "1", "yes", "on"}:
            return True
        if low in {"false", "0", "no", "off"}:
            return False
        if not low:
            return _UNSET
        raise StudioError(f"{key} 需要 true / false")
    if isinstance(baseline, int):
        if isinstance(raw, str) and not raw.strip():
            return _UNSET
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise StudioError(f"{key} 需要整数（收到 {raw!r}）")
    if isinstance(baseline, float):
        if isinstance(raw, str) and not raw.strip():
            return _UNSET
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise StudioError(f"{key} 需要数字（收到 {raw!r}）")
    if isinstance(baseline, (list, dict)):
        value = raw
        if isinstance(raw, str):
            if not raw.strip():
                return _UNSET
            try:
                value = json.loads(raw)
            except ValueError as e:
                raise StudioError(f"{key} 需要 JSON：{e}")
        if not isinstance(value, type(baseline)):
            raise StudioError(f"{key} 需要 {'数组' if isinstance(baseline, list) else '对象'}")
        return value
    # string / unknown: keep text as-is (paths and regexes must not be trimmed)
    if isinstance(raw, (list, dict)):
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def project_config(slug: str, variant: str | None = None, preset: str | None = None) -> dict:
    """Everything the parameter tab needs for one project.

    ``variant`` / ``preset`` preview another merge chain **without saving**;
    the WebUI uses that when the user switches the method dropdown.
    """
    project = _project(slug)
    previewing = bool(variant or preset)
    if variant:
        project = {**project, "variant": str(variant)}
    if preset:
        project = {**project, "preset": str(preset)}
    if previewing:
        baseline = _config_baseline(project)
        project = {
            **project,
            "overrides": {
                key: value
                for key, value in (project.get("overrides") or {}).items()
                if key in baseline
            },
        }
    cfg = _config_file(slug)
    text = ""
    if cfg.is_file() and not previewing:
        try:
            text = cfg.read_text(encoding="utf-8")[:80_000]
        except OSError:
            pass
    return {
        "slug": slug,
        "variant": str(project.get("variant") or "lora"),
        "preset": str(project.get("preset") or "low_vram"),
        "outputName": str(project.get("output_name") or slug),
        "overrides": project.get("overrides") or {},
        "fields": _config_fields(project),
        "configPath": str(cfg) if cfg.is_file() else None,
        "configText": text,
        "preview": previewing,
        "schema": config_schema(),
    }


def save_project_config(slug: str, body: dict) -> dict:
    """Persist only the fields the user changed; ``null`` removes an override."""
    with _LOCK:
        project = _project(slug)
        cio = _gui_config()
        if body.get("variant"):
            variant = str(body["variant"]).strip()
            if variant not in cio.list_methods():
                raise StudioError(
                    f"未知方法 {variant}（可选：{', '.join(cio.list_methods())}）"
                )
            project["variant"] = variant
        if body.get("preset"):
            preset = str(body["preset"]).strip()
            allowed = {name for name, _ in cio.list_hardware_presets()} | set(cio.list_presets())
            if preset not in allowed:
                raise StudioError(f"未知硬件档 {preset}（可选：{', '.join(sorted(allowed))}）")
            project["preset"] = preset
        if "output_name" in body:
            project["output_name"] = str(body["output_name"]).strip() or slug
        if "target_res" in body:
            project["target_res"] = body["target_res"] or None

        baseline = _config_baseline(project)
        overrides = dict(project.get("overrides") or {})
        incoming = body.get("overrides")
        if isinstance(incoming, dict):
            for raw_key, raw_value in incoming.items():
                key = str(raw_key)
                if key not in baseline and key not in overrides:
                    raise StudioError(f"未知参数 {key}（不在 {project.get('variant')} 的配置里）")
                value = _coerce_config_value(key, raw_value, baseline.get(key))
                if value is _UNSET:
                    overrides.pop(key, None)
                else:
                    overrides[key] = value
        # Switching method/preset can make an old override meaningless; drop
        # keys the new merge chain doesn't know instead of leaking them into
        # the generated TOML.
        overrides = {key: value for key, value in overrides.items() if key in baseline}
        project["overrides"] = overrides
        _save_project(slug, project)
        _append_history(
            slug,
            {"ev": "config", "variant": project.get("variant"), "preset": project.get("preset")},
        )

    warning = None
    try:
        _config_file(slug).unlink(missing_ok=True)
        build_config(slug, force=True)
    except (StudioError, OSError) as e:
        warning = str(e)
    result = project_config(slug)
    if warning:
        result["warning"] = warning
    return result


# --------------------------------------------------------------------------- #
# Krea-2 line: its own TrainRequest table (not the Anima merge chain)
# --------------------------------------------------------------------------- #
_KREA2_GROUPS = (
    ("identity", "身份与 LoRA"),
    ("data", "数据与叠加"),
    ("budget", "训练预算"),
    ("runtime", "显存与运行"),
    ("save", "保存与日志"),
)
_KREA2_META = {
    "output_name": {
        "label": "产物名 output_name",
        "group": "identity",
        "help": "写进 LoRA 元数据的基础名；产物文件另见 output。",
    },
    "model": {
        "label": "底模路径 model",
        "group": "identity",
        "help": "Krea-2 自包含 NF4 v2 或 bf16 权重（.safetensors）。",
    },
    "rank": {"label": "秩 rank", "group": "identity", "min": 1, "step": 1},
    "alpha": {"label": "alpha", "group": "identity", "min": 0, "step": 1},
    "target_preset": {
        "label": "注入目标 target_preset",
        "group": "identity",
        "choices": ["official_gate", "official", "attention"],
        "help": "official_gate=264（官方 232 + 32 个 mlp.gate，当前口径）；official=232；attention=160。",
    },
    "dropout": {"label": "dropout", "group": "identity", "min": 0, "max": 1, "step": 0.01},
    "cache": {
        "label": "缓存目录 cache",
        "group": "data",
        "help": "krea2-webui 预处理产物：*_anima.npz + *_krea2_te.safetensors。512 档必须重跑 resize + cache_latents，不能复用 1024 档。",
    },
    "output": {
        "label": "产物路径 output",
        "group": "data",
        "help": "训练写出的 .safetensors；Anima 元数据带 ss_adaln_layout=comfy，可直接复制到 ComfyUI 的 models\\loras。",
    },
    "stack_lora_paths": {
        "label": "叠加画风 LoRA（分号分隔）",
        "group": "data",
        "help": "style-in-the-loop：训练人物 LoRA 时挂冻结画风 LoRA。留空 = 不叠加。",
    },
    "stack_lora_mode": {
        "label": "叠加模式",
        "group": "data",
        "choices": ["cycle", "random", "independent", "pinned"],
        "help": "pinned = 每条画风每一步都在、强度钉死；warmup 必须 0，且 strengths 条数必须等于 paths 条数。",
    },
    "stack_lora_strengths": {
        "label": "叠加强度（分号分隔）",
        "group": "data",
        "help": "与 paths 一一对应，例如 1.25;1.0；pinned 模式必填。",
    },
    "stack_lora_strength_min": {"label": "随机强度下限", "group": "data", "step": 0.01},
    "stack_lora_strength_max": {"label": "随机强度上限", "group": "data", "step": 0.01},
    "stack_lora_strength_jitter": {
        "label": "强度抖动（±比例）",
        "group": "data",
        "step": 0.01,
        "help": "0.05 = ±5%；pinned 时也生效。",
    },
    "stack_lora_prob": {"label": "每步叠加概率", "group": "data", "min": 0, "max": 1, "step": 0.01},
    "stack_lora_warmup_steps": {
        "label": "叠加 warmup 步数",
        "group": "data",
        "min": 0,
        "step": 1,
        "help": "pinned 必须 0：任何一步没画风，人物 delta 就会把画风吃回去。",
    },
    "max_steps": {"label": "最大步数 max_steps", "group": "budget", "min": 1, "step": 1},
    "batch_size": {"label": "批大小", "group": "budget", "min": 1, "step": 1},
    "lr": {"label": "学习率 lr", "group": "budget", "min": 0, "step": 0.00001},
    "lr_scheduler": {
        "label": "学习率调度",
        "group": "budget",
        "choices": ["constant", "cosine", "linear"],
    },
    "warmup_steps": {"label": "warmup 步数", "group": "budget", "min": 0, "step": 1},
    "weight_decay": {"label": "weight decay", "group": "budget", "min": 0, "step": 0.01},
    "optimizer_type": {
        "label": "优化器",
        "group": "budget",
        "choices": ["adamw", "adamw8bit"],
        "help": "adamw8bit 在 8GB 卡上再省 ~0.13GiB；不能与旧的 fp32 AdamW state 混用续训。",
    },
    "seed": {"label": "随机种子", "group": "budget", "step": 1},
    "sigma_min": {"label": "σ 下限", "group": "budget", "min": 0, "max": 1, "step": 0.01},
    "sigma_max": {"label": "σ 上限", "group": "budget", "min": 0, "max": 1, "step": 0.01},
    "blocks_to_swap": {
        "label": "块交换 blocks_to_swap",
        "group": "runtime",
        "min": 0,
        "step": 1,
        "help": "0 = 全在 GPU；越高 CPU 卸载越多，适合低显存。",
    },
    "grad_checkpointing": {
        "label": "梯度检查点",
        "group": "runtime",
        "help": "用计算换显存；低显存必备。",
    },
    "device": {"label": "设备", "group": "runtime", "choices": ["cuda", "cpu"]},
    "mem_cache_size": {
        "label": "常驻样本窗口",
        "group": "runtime",
        "min": 0,
        "step": 1,
        "help": "TE 缓存每张约 31.46MB；372 张全进内存会顶爆宿主，默认 8。",
    },
    "save_every_n_steps": {
        "label": "每 N 步保存（0=只存最后）",
        "group": "save",
        "min": 0,
        "step": 1,
    },
    "save_last_n": {"label": "保留最近 N 个", "group": "save", "min": 0, "step": 1},
    "log_every_n_steps": {"label": "每 N 步打日志", "group": "save", "min": 1, "step": 1},
}
_KREA2_ORDER = {
    key: i for i, key in enumerate(_KREA2_META)
}


def _krea2_file(slug: str) -> Path:
    return project_dir(slug) / "krea2.toml"


def _krea2_request():
    try:
        from library.krea2.requests import TrainRequest

        return TrainRequest
    except Exception as e:  # noqa: BLE001
        raise StudioError(f"Krea-2 请求定义不可用（{e}）")


def _krea2_defaults(project: dict) -> dict:
    slug = str(project.get("slug") or "")
    values = _krea2_request()().to_dict()
    values["output_name"] = str(project.get("output_name") or slug)
    values["cache"] = f"post_image_dataset/krea2/{slug}"
    values["output"] = f"output/ckpt_krea2/{slug}.safetensors"
    return values


def _krea2_toml(values: dict) -> str:
    import toml

    return toml.dumps(values)


def _krea2_abs(path_text: str) -> Path:
    path = Path(str(path_text or ""))
    return path if path.is_absolute() else (config.ROOT / path)


def _krea2_outputs(values: dict) -> list[dict]:
    path = _krea2_abs(values.get("output"))
    root = path.parent
    if not root.is_dir():
        return []
    out: list[dict] = []
    # `_save` writes `<stem>.safetensors` and `<stem>-step00000020.safetensors`;
    # never list another project's files that happen to share the directory.
    for item in sorted(root.glob(f"{path.stem}*.safetensors")):
        try:
            out.append(
                {
                    "name": item.name,
                    "size": item.stat().st_size,
                    "mtime": item.stat().st_mtime,
                }
            )
        except OSError:
            continue
    return out[-50:]


def _krea2_checks(values: dict) -> dict:
    cache = _krea2_abs(values.get("cache"))
    model = _krea2_abs(values.get("model"))
    npz = 0
    if cache.is_dir():
        try:
            npz = sum(1 for _ in cache.rglob("*.npz"))
        except OSError:
            npz = 0
    return {
        "cache": str(cache),
        "cacheExists": cache.is_dir(),
        "cacheFiles": npz,
        "model": str(model),
        "modelExists": model.is_file(),
        "output": str(values.get("output") or ""),
    }


def _krea2_fields(values: dict, defaults: dict) -> list[dict]:
    labels = {name: label for name, label in _KREA2_GROUPS}
    fields: list[dict] = []
    for key, value in values.items():
        meta = _KREA2_META.get(key, {})
        group = str(meta.get("group") or "runtime")
        fields.append(
            {
                "key": key,
                "label": str(meta.get("label") or key),
                "group": group,
                "groupLabel": labels.get(group, "其它"),
                "type": _value_type(value),
                "value": value,
                "default": defaults.get(key),
                "help": str(meta.get("help") or ""),
                "choices": list(meta.get("choices") or ()),
                "min": meta.get("min"),
                "max": meta.get("max"),
                "step": meta.get("step"),
            }
        )
    order = {name: i for i, (name, _label) in enumerate(_KREA2_GROUPS)}
    fields.sort(key=lambda f: (order.get(f["group"], 99), _KREA2_ORDER.get(f["key"], 100), f["key"]))
    return fields


def _validate_krea2(values: dict) -> None:
    target = str(values.get("target_preset") or "")
    if target not in {"official_gate", "official", "attention"}:
        raise StudioError("target_preset 只能是 official_gate / official / attention")
    mode = str(values.get("stack_lora_mode") or "cycle")
    if mode not in {"cycle", "random", "independent", "pinned"}:
        raise StudioError("stack_lora_mode 只能是 cycle / random / independent / pinned")
    for key in ("rank", "max_steps", "batch_size"):
        try:
            if int(values.get(key)) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            raise StudioError(f"{key} 必须是正整数")
    try:
        if float(values.get("alpha", 0)) < 0:
            raise ValueError
    except (TypeError, ValueError):
        raise StudioError("alpha 必须是非负数")
    if mode == "pinned":
        paths = [p for p in re.split(r"[;,]", str(values.get("stack_lora_paths") or "")) if p.strip()]
        strengths = [
            p for p in re.split(r"[;,]", str(values.get("stack_lora_strengths") or "")) if p.strip()
        ]
        if not paths:
            raise StudioError("pinned 模式必须给 stack_lora_paths")
        if len(paths) != len(strengths):
            raise StudioError(
                f"pinned 模式的 strengths 条数（{len(strengths)}）必须等于 paths 条数（{len(paths)}）"
            )
        if int(values.get("stack_lora_warmup_steps") or 0) != 0:
            raise StudioError("pinned 模式的 stack_lora_warmup_steps 必须为 0")


def project_krea2(slug: str) -> dict:
    project = _project(slug)
    path = _krea2_file(slug)
    defaults = _krea2_defaults(project)
    values = dict(defaults)
    warning = None
    exists = path.is_file()
    if exists:
        try:
            values.update(_krea2_request().load_config(path).to_dict())
        except SystemExit as e:
            warning = str(e)
        except Exception as e:  # noqa: BLE001
            warning = f"读 krea2.toml 失败：{e}"
    text = ""
    if exists:
        try:
            text = path.read_text(encoding="utf-8")[:80_000]
        except OSError:
            pass
    if not text:
        text = _krea2_toml(values)
    return {
        "slug": slug,
        "exists": exists,
        "configPath": str(path) if exists else None,
        "values": values,
        "fields": _krea2_fields(values, defaults),
        "groups": [{"name": name, "label": label} for name, label in _KREA2_GROUPS],
        "checks": _krea2_checks(values),
        "outputs": _krea2_outputs(values),
        "text": text,
        "warning": warning,
    }


def save_project_krea2(slug: str, body: dict) -> dict:
    project = _project(slug)
    defaults = _krea2_defaults(project)
    current = project_krea2(slug)["values"]
    incoming = body.get("values") if isinstance(body.get("values"), dict) else body
    values = dict(current)
    for raw_key, raw_value in incoming.items():
        key = str(raw_key)
        if key in {"slug", "values", "fields", "groups", "checks", "text", "warning", "action"}:
            continue
        if key not in current:
            raise StudioError(f"未知 Krea-2 参数 {key}")
        value = _coerce_config_value(key, raw_value, current.get(key))
        values[key] = defaults.get(key) if value is _UNSET else value
    _validate_krea2(values)
    path = _krea2_file(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_krea2_toml(values), encoding="utf-8")
    _append_history(slug, {"ev": "krea2_config"})
    return project_krea2(slug)


def enqueue_krea2_train(slug: str, manager: Any, body: dict) -> dict:
    """Save the Krea-2 table, validate its cache/model, then enqueue the CLI."""
    values_in = body.get("values") if isinstance(body.get("values"), dict) else body
    krea2_keys = set(_KREA2_META)
    if isinstance(values_in, dict) and any(k in krea2_keys for k in values_in):
        info = save_project_krea2(slug, body)
    else:
        info = project_krea2(slug)
    checks = info["checks"]
    if not checks["cacheExists"]:
        raise StudioError(
            f"Krea-2 缓存目录不存在：{checks['cache']}；先用 krea2-webui 做 resize + cache_latents"
        )
    if not checks["cacheFiles"]:
        raise StudioError(f"Krea-2 缓存目录里没有 .npz：{checks['cache']}；先跑 cache_latents")
    if not checks["modelExists"]:
        raise StudioError(f"Krea-2 底模不存在：{checks['model']}")
    cfg = _krea2_file(slug)
    if not cfg.is_file():
        raise StudioError("还没有 krea2.toml；先保存参数")
    try:
        rel = cfg.relative_to(config.ROOT).as_posix()
    except ValueError:
        # Tests point PROJECTS_DIR at a temp tree; production always lives under ROOT.
        rel = str(cfg)
    label = f"krea2:{slug}:train"
    job = manager.submit_command(
        label=label,
        argv=["scripts/krea2/train.py", "--config", rel],
        extra_env={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync"},
        start=True,
        stall_timeout=0,
    )
    _record_job(slug, job.id, "krea2", label)
    return {
        "slug": slug,
        "stage": "krea2",
        "job_id": job.id,
        "state": job.state,
        "config_path": str(cfg),
        "output": info["values"].get("output"),
    }


class StudioError(ValueError):
    """A caller error that should surface as HTTP 400 (not a 500)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def slugify(name: str) -> str:
    raw = (name or "").strip().lower()
    raw = raw.replace(" ", "_")
    slug = _SLUG_RE.sub("_", raw).strip("_")
    slug = re.sub(r"_{2,}", "_", slug)
    if not slug:
        slug = "project"
    if slug[0].isdigit():
        slug = f"p_{slug}"
    return slug[:48]


def _safe_slug(slug: str) -> str:
    value = (slug or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise StudioError("非法项目 slug")
    if value.startswith("_"):
        raise StudioError("项目 slug 不能以下划线开头（`_inbox` 等是保留名）")
    if value != slugify(value):
        raise StudioError("项目 slug 只能用小写字母、数字、下划线与连字符")
    return value


def project_dir(slug: str) -> Path:
    return PROJECTS_DIR / _safe_slug(slug)


def dataset_dir(slug: str) -> Path:
    return IMAGE_DATASET_DIR / _safe_slug(slug)


def inbox_dir(slug: str) -> Path:
    return project_dir(slug) / "inbox"


def rejected_dir(slug: str) -> Path:
    return project_dir(slug) / "rejected"


def _project_file(slug: str) -> Path:
    return project_dir(slug) / "project.json"


def _jobs_file(slug: str) -> Path:
    return project_dir(slug) / "jobs.json"


def _history_file(slug: str) -> Path:
    return project_dir(slug) / "history.jsonl"


def _config_file(slug: str) -> Path:
    return project_dir(slug) / "train_config.toml"


def _inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except (ValueError, OSError):
        return False


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_history(slug: str, event: dict) -> None:
    record = {"at": _now(), **event}
    path = _history_file(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name.lower(),
    )


def _sidecar_for(image: Path) -> Path:
    return image.with_suffix(".txt")


def _read_caption(image: Path) -> str:
    try:
        return _sidecar_for(image).read_text(encoding="utf-8")
    except OSError:
        return ""


def _write_caption(image: Path, caption: str) -> None:
    text = (caption or "").replace("\r\n", "\n").strip()
    if len(text) > MAX_CAPTION_CHARS:
        raise StudioError(f"caption 超长（>{MAX_CAPTION_CHARS} 字符）")
    sidecar = _sidecar_for(image)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _unique_dest(dest_dir: Path, name: str) -> Path:
    """``foo.jpg`` → ``foo-1.jpg`` when the name is taken (never overwrite)."""
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    for i in range(1, 10_000):
        candidate = dest_dir / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise StudioError(f"目标目录重名太多：{name}")


def _move_with_sidecar(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(dest_dir, src.name)
    shutil.move(str(src), str(dest))
    sidecar = _sidecar_for(src)
    if sidecar.is_file():
        target = _sidecar_for(dest)
        try:
            sidecar.replace(target)
        except OSError:
            pass
    return dest


def _purge_derived(slug: str, image_name: str) -> list[str]:
    """Remove resized/mask/cache files derived from one image.

    Training reads the *resized* tree and the cache dir, so removing a rejected
    image from ``image_dataset/`` alone would leave it in the next run. This is
    the one place that has to know all three trees.
    """
    stem = Path(image_name).stem
    scoped_slug = _safe_slug(slug)
    removed: list[str] = []
    for root in (
        RESIZED_DIR / scoped_slug,
        CACHE_DIR / scoped_slug,
        MASK_DIR / scoped_slug,
    ):
        if not root.is_dir():
            continue
        for path in list(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            # 别用 `stem*` 一把梭：`foo` 会把 `foobar` 的缓存也删掉。
            # 允许的三种形状：同名（resized 的 foo.png）、`foo_…`（latent / TE）、
            # `foo-…` / `foo.…`（分页/分卷命名）。
            if name != stem and not any(
                name.startswith(stem + sep) for sep in ("_", "-", ".")
            ):
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                pass
    return removed


def _project(slug: str) -> dict:
    data = _read_json(_project_file(slug), None)
    if not isinstance(data, dict):
        raise StudioError(f"项目不存在：{slug}")
    data.setdefault("slug", slug)
    return data


def _save_project(slug: str, data: dict) -> dict:
    data["slug"] = slug
    data["updated_at"] = _now()
    _write_json(_project_file(slug), data)
    return data


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
def list_projects() -> list[dict]:
    out: list[dict] = []
    if not PROJECTS_DIR.is_dir():
        return out
    for path in PROJECTS_DIR.iterdir():
        if not path.is_dir() or path.name.startswith("_"):
            continue
        data = _read_json(path / "project.json", None)
        if not isinstance(data, dict):
            continue
        out.append(_summary(data))
    out.sort(key=lambda p: p.get("updatedAt") or "", reverse=True)
    return out


def _summary(project: dict) -> dict:
    slug = str(project.get("slug") or "")
    counts = _counts(slug)
    return {
        "slug": slug,
        "name": project.get("name") or slug,
        "trigger": project.get("trigger") or "",
        "variant": project.get("variant") or "lora",
        "preset": project.get("preset") or "low_vram",
        "outputName": project.get("output_name") or slug,
        "createdAt": project.get("created_at"),
        "updatedAt": project.get("updated_at"),
        "counts": counts,
        "stages": _stages(counts),
    }


def _counts(slug: str) -> dict:
    dataset = _image_files(dataset_dir(slug))
    captions = sum(1 for p in dataset if _sidecar_for(p).is_file())
    resized = _image_files(RESIZED_DIR / slug) if (RESIZED_DIR / slug).is_dir() else []
    cache_files = []
    cache_root = CACHE_DIR / slug
    if cache_root.is_dir():
        cache_files = [p for p in cache_root.rglob("*") if p.is_file()]
    npz = [p for p in cache_files if p.suffix.lower() == ".npz"]
    # REPA 的视觉特征（`*_anima_pe_spatial.safetensors` / `*_anima_pe.safetensors`）
    # **先分出来**：它们也是 .safetensors，混进 TE 计数会让“只做了 PE、没做
    # caption”的项目被误判成预处理完成。
    pe = [
        p
        for p in cache_files
        if p.suffix.lower() == ".safetensors"
        and ("_anima_pe" in p.name or "_anima_pe_spatial" in p.name)
    ]
    te = [
        p
        for p in cache_files
        if p.suffix.lower() == ".safetensors" and p not in pe
    ]
    outputs = []
    out_root = OUTPUT_DIR / slug
    if out_root.is_dir():
        outputs = [p for p in out_root.rglob("*.safetensors") if p.is_file()]
    return {
        "inbox": len(_image_files(inbox_dir(slug))),
        "dataset": len(dataset),
        "rejected": len(_image_files(rejected_dir(slug))),
        "captions": captions,
        "resized": len(resized),
        "latents": len(npz),
        "teCache": len(te),
        "peCache": len(pe),
        "outputs": len(outputs),
    }


def _stages(counts: dict) -> dict:
    have = counts["dataset"] > 0
    tagged = have and counts["captions"] >= counts["dataset"]
    preprocessed = (
        have
        and counts["resized"] >= counts["dataset"]
        and counts["latents"] >= counts["dataset"]
        and counts["teCache"] >= counts["dataset"]
    )
    return {
        "fetch": counts["inbox"] + counts["dataset"] + counts["rejected"] > 0,
        "review": have and counts["inbox"] == 0,
        "tag": tagged,
        "preprocess": preprocessed,
        "train": counts["outputs"] > 0,
    }


def create_project(body: dict) -> dict:
    name = str(body.get("name") or "").strip()
    if not name:
        raise StudioError("缺少项目名 name")
    slug = str(body.get("slug") or "").strip() or slugify(name)
    slug = _safe_slug(slug)
    with _LOCK:
        if _project_file(slug).is_file():
            raise StudioError(f"项目已存在：{slug}")
        variant = str(body.get("variant") or "lora").strip() or "lora"
        preset = str(body.get("preset") or "low_vram").strip() or "low_vram"
        overrides = body.get("overrides") if isinstance(body.get("overrides"), dict) else {}
        project = {
            "slug": slug,
            "name": name,
            "trigger": str(body.get("trigger") or "").strip(),
            "variant": variant,
            "preset": preset,
            "notes": str(body.get("notes") or "").strip(),
            "target_res": body.get("target_res") or None,
            "overrides": {str(k): v for k, v in overrides.items() if v not in (None, "")},
            "output_name": str(body.get("output_name") or slug).strip() or slug,
            "created_at": _now(),
            "updated_at": _now(),
        }
        inbox_dir(slug).mkdir(parents=True, exist_ok=True)
        rejected_dir(slug).mkdir(parents=True, exist_ok=True)
        dataset_dir(slug).mkdir(parents=True, exist_ok=True)
        _save_project(slug, project)
        _append_history(slug, {"ev": "create", "name": name})
    return detail(slug)


def update_project(slug: str, body: dict) -> dict:
    with _LOCK:
        project = _project(slug)
        for key in ("name", "trigger", "variant", "preset", "notes", "output_name"):
            if key in body and body[key] is not None:
                project[key] = str(body[key]).strip()
        if "target_res" in body:
            project["target_res"] = body["target_res"] or None
        if isinstance(body.get("overrides"), dict):
            project["overrides"] = {
                str(k): v for k, v in body["overrides"].items() if v not in (None, "")
            }
        _save_project(slug, project)
        _append_history(slug, {"ev": "update"})
    return detail(slug)


def detail(slug: str) -> dict:
    project = _project(slug)
    counts = _counts(slug)
    jobs = _project_jobs(slug)
    config_path = _config_file(slug)
    config_text = ""
    if config_path.is_file():
        try:
            config_text = config_path.read_text(encoding="utf-8")[:40_000]
        except OSError:
            pass
    return {
        **_summary(project),
        "notes": project.get("notes") or "",
        "overrides": project.get("overrides") or {},
        "targetRes": project.get("target_res"),
        "configPath": str(config_path) if config_path.is_file() else None,
        "configText": config_text,
        "outputFiles": _output_files(slug),
        "jobs": jobs[-20:],
        "paths": {
            "dataset": str(dataset_dir(slug)),
            "inbox": str(inbox_dir(slug)),
            "rejected": str(rejected_dir(slug)),
            "resized": str(RESIZED_DIR / slug),
            "cache": str(CACHE_DIR / slug),
            "output": str(OUTPUT_DIR / slug),
        },
    }


def _output_files(slug: str) -> list[dict]:
    root = OUTPUT_DIR / _safe_slug(slug)
    if not root.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(root.rglob("*.safetensors")):
        try:
            out.append(
                {
                    "name": str(path.relative_to(root)).replace("\\", "/"),
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
            )
        except OSError:
            continue
    return out[-50:]


def _project_jobs(slug: str, manager: Any = None) -> list[dict]:
    raw = _read_json(_jobs_file(slug), [])
    if not isinstance(raw, list):
        raw = []
    out: list[dict] = []
    for entry in raw[-30:]:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        if manager is not None:
            try:
                job = manager.get(str(item.get("id") or ""))
            except Exception:  # noqa: BLE001 — status must never break the page
                job = None
            if job is not None:
                record = job.public() if hasattr(job, "public") else job
                if isinstance(record, dict):
                    item.update(
                        {
                            "state": record.get("state"),
                            "error": record.get("error"),
                            "returncode": record.get("returncode"),
                            "startedAt": record.get("started_at"),
                            "endedAt": record.get("ended_at"),
                        }
                    )
        out.append(item)
    return out


def _record_job(slug: str, job_id: str, stage: str, label: str) -> None:
    path = _jobs_file(slug)
    raw = _read_json(path, [])
    if not isinstance(raw, list):
        raw = []
    raw.append({"id": job_id, "stage": stage, "label": label, "at": _now()})
    _write_json(path, raw[-200:])
    _append_history(slug, {"ev": stage, "job": job_id, "label": label})


# --------------------------------------------------------------------------- #
# Images: import / review / captions
# --------------------------------------------------------------------------- #
def list_images(slug: str, state: str = "all", limit: int = 500) -> dict:
    _project(slug)
    state = (state or "all").lower()
    if state not in {"all", "inbox", "dataset", "rejected"}:
        raise StudioError("state 只能是 all / inbox / dataset / rejected")
    requested = ["inbox", "dataset", "rejected"] if state == "all" else [state]
    out: list[dict] = []
    for kind in requested:
        directory = {
            "inbox": inbox_dir(slug),
            "dataset": dataset_dir(slug),
            "rejected": rejected_dir(slug),
        }[kind]
        for image in _image_files(directory):
            if len(out) >= max(1, min(limit, 5000)):
                break
            rel = f"{kind}/{image.name}"
            out.append(
                {
                    "state": kind,
                    "name": image.name,
                    "path": rel,
                    "caption": _read_caption(image),
                    "size": image.stat().st_size,
                    "mtime": image.stat().st_mtime,
                    "url": (
                        f"/studio/projects/{urllib.parse.quote(slug)}/file?path="
                        f"{urllib.parse.quote(rel, safe='')}"
                    ),
                }
            )
    out.sort(key=lambda row: (row["state"], row["name"].lower()))
    return {"slug": slug, "state": state, "count": len(out), "images": out}


def import_images(slug: str, body: dict) -> dict:
    _project(slug)
    target = inbox_dir(slug)
    target.mkdir(parents=True, exist_ok=True)
    paths = body.get("paths") or []
    urls = body.get("urls") or []
    if isinstance(paths, str):
        paths = [p for p in paths.splitlines() if p.strip()]
    if isinstance(urls, str):
        urls = [u for u in urls.splitlines() if u.strip()]
    recursive = bool(body.get("recursive"))
    caption = str(body.get("caption") or "").strip()
    added: list[str] = []
    skipped: list[str] = []
    errors: list[dict] = []

    def copy_one(src: Path) -> None:
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTS:
            skipped.append(str(src))
            return
        try:
            if src.stat().st_size > MAX_IMPORT_BYTES:
                errors.append({"path": str(src), "error": "文件超过 64MB"})
                return
            dest = _unique_dest(target, src.name)
            shutil.copy2(src, dest)
            if caption:
                _write_caption(dest, caption)
            elif _sidecar_for(src).is_file():
                shutil.copy2(_sidecar_for(src), _sidecar_for(dest))
            added.append(dest.name)
        except OSError as e:
            errors.append({"path": str(src), "error": str(e)})

    for raw in paths:
        try:
            path = Path(str(raw)).expanduser()
        except OSError:
            continue
        if path.is_dir():
            walker: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
            for src in walker:
                if src.is_file():
                    copy_one(src)
        elif path.is_file():
            copy_one(path)
        else:
            errors.append({"path": str(path), "error": "不存在"})

    for url in urls:
        try:
            added.append(_download(str(url), target, caption))
        except Exception as e:  # noqa: BLE001 — per-URL failure must not abort the batch
            errors.append({"url": str(url), "error": str(e)})

    if added:
        _append_history(
            slug,
            {"ev": "import", "added": len(added), "skipped": len(skipped), "errors": len(errors)},
        )
    return {
        "slug": slug,
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "counts": _counts(slug),
    }


def _download(url: str, target: Path, caption: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise StudioError("只支持 http/https URL")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        length = int(resp.headers.get("Content-Length") or 0)
        if length > MAX_IMPORT_BYTES:
            raise StudioError("文件超过 64MB")
        data = resp.read(MAX_IMPORT_BYTES + 1)
        if len(data) > MAX_IMPORT_BYTES:
            raise StudioError("文件超过 64MB")
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype and not ctype.startswith("image/"):
        raise StudioError(f"不是图片（Content-Type={ctype}）")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in IMAGE_EXTS:
        suffix = {
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/avif": ".avif",
        }.get(ctype, ".jpg")
    digest = hashlib.sha256(data).hexdigest()[:10]
    stem = Path(parsed.path).stem or "image"
    dest = _unique_dest(target, f"{slugify(stem) or 'image'}-{digest}{suffix}")
    dest.write_bytes(data)
    if caption:
        _write_caption(dest, caption)
    return dest.name


def _resolve_image(slug: str, rel: str) -> tuple[str, Path]:
    raw = (rel or "").replace("\\", "/").strip("/")
    if not raw:
        raise StudioError("缺少 path")
    parts = raw.split("/", 1)
    kind = parts[0]
    name = parts[1] if len(parts) > 1 else ""
    if kind not in {"inbox", "dataset", "rejected"} or not name or "/" in name:
        raise StudioError("path 必须是 inbox/…、dataset/… 或 rejected/…")
    directory = {
        "inbox": inbox_dir(slug),
        "dataset": dataset_dir(slug),
        "rejected": rejected_dir(slug),
    }[kind]
    path = directory / name
    if not _inside(directory, path) or not path.is_file():
        raise StudioError(f"文件不存在：{raw}")
    return kind, path


def review_images(slug: str, body: dict) -> dict:
    _project(slug)
    action = str(body.get("action") or "").strip().lower()
    if action not in {"accept", "reject", "restore", "delete"}:
        raise StudioError("action 只能是 accept / reject / restore / delete")
    files = body.get("files") or body.get("paths") or []
    if isinstance(files, str):
        files = [f for f in files.splitlines() if f.strip()]
    if not files:
        raise StudioError("缺少 files")
    moved: list[dict] = []
    errors: list[dict] = []
    with _LOCK:
        for rel in files:
            try:
                kind, path = _resolve_image(slug, str(rel))
                if action == "accept":
                    if kind == "dataset":
                        continue
                    dest = _move_with_sidecar(path, dataset_dir(slug))
                    moved.append({"from": rel, "to": f"dataset/{dest.name}"})
                elif action == "reject":
                    if kind == "rejected":
                        continue
                    dest = _move_with_sidecar(path, rejected_dir(slug))
                    if kind == "dataset":
                        _purge_derived(slug, path.name)
                    moved.append({"from": rel, "to": f"rejected/{dest.name}"})
                elif action == "restore":
                    if kind == "inbox":
                        continue
                    dest = _move_with_sidecar(path, inbox_dir(slug))
                    moved.append({"from": rel, "to": f"inbox/{dest.name}"})
                else:  # delete
                    if kind == "dataset":
                        _purge_derived(slug, path.name)
                    path.unlink(missing_ok=True)
                    _sidecar_for(path).unlink(missing_ok=True)
                    moved.append({"from": rel, "to": None})
            except (StudioError, OSError) as e:
                errors.append({"path": str(rel), "error": str(e)})
        if moved:
            _append_history(slug, {"ev": "review", "action": action, "count": len(moved)})
    return {"slug": slug, "action": action, "moved": moved, "errors": errors, "counts": _counts(slug)}


def get_caption(slug: str, rel: str) -> dict:
    _project(slug)
    _, path = _resolve_image(slug, rel)
    return {"slug": slug, "path": rel, "caption": _read_caption(path)}


def set_caption(slug: str, body: dict) -> dict:
    _project(slug)
    rel = str(body.get("path") or body.get("file") or "")
    _, path = _resolve_image(slug, rel)
    caption = body.get("caption")
    if caption is None:
        raise StudioError("缺少 caption")
    _write_caption(path, str(caption))
    _append_history(slug, {"ev": "caption", "path": rel})
    return {"slug": slug, "path": rel, "caption": _read_caption(path)}


def apply_trigger(slug: str, body: dict) -> dict:
    project = _project(slug)
    trigger = str(body.get("trigger") or project.get("trigger") or "").strip()
    if not trigger:
        raise StudioError("项目没有触发词（trigger），也没有在请求里给")
    changed = 0
    for image in _image_files(dataset_dir(slug)):
        caption = _read_caption(image).strip()
        low = f"{caption}\n".lower()
        if trigger.lower() in low:
            continue
        new = f"{trigger}, {caption}" if caption else trigger
        _write_caption(image, new)
        changed += 1
    if trigger != (project.get("trigger") or ""):
        project["trigger"] = trigger
        _save_project(slug, project)
    _append_history(slug, {"ev": "trigger", "trigger": trigger, "changed": changed})
    return {"slug": slug, "trigger": trigger, "changed": changed, "counts": _counts(slug)}


# --------------------------------------------------------------------------- #
# Config generation (merged chain + path_scope)
# --------------------------------------------------------------------------- #
def build_config(slug: str, force: bool = False) -> dict:
    """Regenerate ``train_config.toml`` and return ``{path, snapshot}``.

    Uses the engine's own GUI submit plan (``gui.core`` is Qt-free) so the
    config is exactly what the Qt GUI would submit: base → preset → method,
    then the GUI-only ``path_scope`` expands into concrete run paths.
    """
    project = _project(slug)
    path = _config_file(slug)
    if path.is_file() and not force:
        return {
            "path": str(path),
            "snapshot": _read_toml(path),
        }
    try:
        from gui.core.config_io import merged_gui_variant_preset
        from gui.core.submit import training_snapshot
    except Exception as e:  # noqa: BLE001 — report the real reason to the user
        raise StudioError(f"引擎的配置模块不可用（{e}）；请确认在引擎 venv 里跑 WebUI/守护进程")

    variant = str(project.get("variant") or "lora")
    preset = str(project.get("preset") or "low_vram")
    merged, _origin = merged_gui_variant_preset(variant, preset)
    merged["path_scope"] = slug
    merged["output_name"] = str(project.get("output_name") or slug)
    for key, value in (project.get("overrides") or {}).items():
        merged[str(key)] = value
    try:
        snapshot = training_snapshot(variant, merged, {})
    except Exception as e:  # noqa: BLE001
        raise StudioError(f"生成训练配置失败：{e}")
    # The GUI submit plan drops source_image_dir from the training snapshot
    # (training only reads the resized tree), but the caption stages that run
    # as command jobs resolve it through `CONFIG_FILE`. Put it back explicitly
    # so `studio_tag` writes the accepted masters, not the repo root.
    snapshot["source_image_dir"] = f"image_dataset/{slug}"
    if project.get("target_res"):
        try:
            snapshot["target_res"] = [int(x) for x in project["target_res"]][:2]
        except (TypeError, ValueError):
            raise StudioError("target_res 必须是 [宽, 高]")
    # The Qt GUI treats these three as virtual keys: they are encoded into the
    # first `[[datasets]]` block, never written as flat TOML keys. Reuse the
    # GUI's own helpers so the WebUI and the Qt form agree on the semantics.
    overrides = project.get("overrides") or {}
    if "use_valid" in overrides or "validation_split_num" in overrides:
        try:
            from gui.core.validation import apply_validation_choice

            apply_validation_choice(
                snapshot,
                bool(merged.get("use_valid")),
                merged.get("validation_split_num"),
            )
        except Exception as e:  # noqa: BLE001
            raise StudioError(f"写入验证集设置失败：{e}")
    if "repeat_by_folder_name" in overrides:
        try:
            from gui.core.validation import apply_folder_repeats_choice

            apply_folder_repeats_choice(
                snapshot, bool(merged.get("repeat_by_folder_name")), False
            )
        except Exception as e:  # noqa: BLE001
            raise StudioError(f"写入文件夹重复设置失败：{e}")
    if not snapshot.get("datasets"):
        raise StudioError("生成的配置里没有 [[datasets]]，无法训练")

    import toml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(snapshot), encoding="utf-8")
    return {"path": str(path), "snapshot": snapshot}


def _read_toml(path: Path) -> dict:
    try:
        import toml

        return toml.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — the preview is best-effort
        return {}


# --------------------------------------------------------------------------- #
# Pipeline stages → daemon jobs
# --------------------------------------------------------------------------- #
def enqueue_tag(slug: str, manager: Any, body: dict) -> dict:
    project = _project(slug)
    counts = _counts(slug)
    if counts["dataset"] == 0:
        raise StudioError("先筛选出至少一张训练图（dataset 为空）")
    mode = str(body.get("mode") or "missing").lower()
    if mode not in {"missing", "merge", "overwrite"}:
        raise StudioError("mode 只能是 missing / merge / overwrite")
    cfg = build_config(slug)["path"]
    argv = ["tasks.py", "caption-autotag", "--apply", "--mode", mode]
    label = f"studio:{slug}:tag"
    job = manager.submit_command(
        label=label,
        argv=argv,
        config_file=cfg,
        start=True,
        stall_timeout=0 if body.get("wait_forever") else None,
    )
    _record_job(slug, job.id, "tag", label)
    return {"slug": slug, "stage": "tag", "job_id": job.id, "state": job.state, "mode": mode}


def enqueue_preprocess(slug: str, manager: Any, body: dict) -> dict:
    project = _project(slug)
    counts = _counts(slug)
    if counts["dataset"] == 0:
        raise StudioError("先筛选出至少一张训练图（dataset 为空）")
    cfg = build_config(slug, force=True)["path"]
    chain = bool(body.get("chain_train"))
    chain_spec = None
    if chain:
        chain_spec = {
            "method": str(project.get("variant") or "lora"),
            "preset": str(project.get("preset") or "low_vram"),
            "methods_subdir": "gui-methods",
            "config_file": cfg,
        }
    argv = [
        "tasks.py",
        "preprocess-config",
        "--dataset_config",
        cfg,
        "--src",
        f"image_dataset/{slug}",
    ]
    label = f"studio:{slug}:preprocess"
    job = manager.submit_command(
        label=label,
        argv=argv,
        config_file=cfg,
        chain_train=chain_spec,
        start=True,
        stall_timeout=0,
    )
    _record_job(slug, job.id, "preprocess", label)
    return {
        "slug": slug,
        "stage": "preprocess",
        "job_id": job.id,
        "state": job.state,
        "chain_train": chain,
    }


def enqueue_train(slug: str, manager: Any, body: dict) -> dict:
    project = _project(slug)
    counts = _counts(slug)
    if counts["dataset"] == 0:
        raise StudioError("先筛选出至少一张训练图（dataset 为空）")
    if counts["latents"] < counts["dataset"] or counts["teCache"] < counts["dataset"]:
        raise StudioError(
            "预处理缓存还不齐"
            f"（latent {counts['latents']}、TE {counts['teCache']} / 图 {counts['dataset']}）；先跑预处理"
        )
    if body.get("overrides") and isinstance(body["overrides"], dict):
        project["overrides"] = {
            **(project.get("overrides") or {}),
            **{str(k): v for k, v in body["overrides"].items() if v not in (None, "")},
        }
        _save_project(slug, project)
    built = build_config(slug, force=True)
    cfg = built["path"]
    snapshot = built.get("snapshot") or {}
    use_repa = snapshot.get("use_repa") is True or str(snapshot.get("use_repa")).lower() in {
        "1",
        "true",
        "yes",
    }
    if use_repa and counts["peCache"] < counts["dataset"]:
        raise StudioError(
            "REPA 的 PE 特征缓存不齐"
            f"（{counts['peCache']} / 图 {counts['dataset']}）；重跑预处理（它会自动补 PE）"
        )
    variant = str(project.get("variant") or "lora")
    preset = str(project.get("preset") or "low_vram")
    job = manager.submit(
        method=variant,
        preset=preset,
        methods_subdir="gui-methods",
        config_file=cfg,
        overrides={},
        extra=[],
        start=True,
    )
    _record_job(slug, job.id, "train", f"studio:{slug}:train")
    return {
        "slug": slug,
        "stage": "train",
        "job_id": job.id,
        "state": job.state,
        "output_name": project.get("output_name") or slug,
        "output_dir": str(OUTPUT_DIR / slug),
    }


def status(slug: str, manager: Any = None) -> dict:
    data = detail(slug)
    data["jobs"] = _project_jobs(slug, manager)
    latest = {}
    for job in reversed(data["jobs"]):
        stage = job.get("stage")
        if stage and stage not in latest:
            latest[stage] = job
    data["latestJobs"] = latest
    return data


# --------------------------------------------------------------------------- #
# File serving
# --------------------------------------------------------------------------- #
def resolve_file(slug: str, rel: str) -> Path:
    _project(slug)
    kind, path = _resolve_image(slug, rel)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise StudioError("只提供图片文件")
    return path
