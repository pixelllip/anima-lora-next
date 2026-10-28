"""工具台后端：合并 / TensorBoard / Anime Tools / EasyControl / 实验。

只测 Qt-free 的那一层（``anima_daemon.toolkits`` + ``anima_daemon.easycontrol``）；
WebUI 的代理契约另有 ``test_webui_server.py`` 盯着。所有落盘都在 ``tmp_path``，
不碰真实的 output/ckpt 与 configs。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anima_daemon import easycontrol, toolkits  # noqa: E402


class FakeJob:
    id = "job-1"
    state = "queued"


class FakeManager:
    def __init__(self):
        self.commands = []
        self.submits = []

    def submit_command(self, **kw):
        self.commands.append(kw)
        return FakeJob()

    def submit(self, **kw):
        self.submits.append(kw)
        return FakeJob()


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def _adapter(tmp_path: Path, name: str = "a.safetensors", hydra: bool = False) -> Path:
    from safetensors.numpy import save_file

    tensors = {"block.0.lora_down.weight": np.zeros((4, 4), dtype=np.float32)}
    if hydra:
        tensors = {"block.0.lora_up_weight": np.zeros((4, 4), dtype=np.float32)}
    path = tmp_path / name
    save_file(tensors, str(path))
    return path


def test_scan_adapter_classifies_lora_and_hydra(tmp_path):
    lora = toolkits._scan_adapter(_adapter(tmp_path))
    assert lora["severity"] == "ok"
    assert lora["counts"]["lora_down"] == 1

    hydra = toolkits._scan_adapter(_adapter(tmp_path, "h.safetensors", hydra=True))
    assert hydra["severity"] == "block"
    assert "HydraLoRA" in hydra["verdict"]


def test_merge_state_lists_files_and_scans_first(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr(toolkits, "_adapter_dirs", lambda: {"output/ckpt/demo": tmp_path})
    state = toolkits.merge_state()
    assert state["dir"] == "output/ckpt/demo"
    assert [f["name"] for f in state["files"]] == [adapter.name]
    assert state["scan"]["name"] == adapter.name
    assert state["scan"]["severity"] == "ok"


def test_merge_run_dit_builds_gui_argv(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path)
    monkeypatch.setattr(toolkits, "_adapter_dirs", lambda: {"output/ckpt/demo": tmp_path})
    mgr = FakeManager()
    out = tmp_path / "out.safetensors"
    res = toolkits.merge_run(
        {
            "mode": "dit",
            "dir": "output/ckpt/demo",
            "adapter": adapter.name,
            "dit": str(tmp_path / "base.safetensors"),
            "multiplier": 1.25,
            "dtype": "fp16",
            "out": str(out),
        },
        mgr,
    )
    assert res["job_id"] == "job-1"
    argv = mgr.commands[0]["argv"]
    assert argv[0] == "scripts/toolkits/merge_to_dit.py"
    assert "--multiplier" in argv and "1.25" in argv
    assert "--dtype" in argv and "fp16" in argv
    assert "--out" in argv and str(out) in argv


def test_merge_run_loras_requires_two_and_checks_weights(tmp_path, monkeypatch):
    a = _adapter(tmp_path, "a.safetensors")
    b = _adapter(tmp_path, "b.safetensors")
    monkeypatch.setattr(toolkits, "_adapter_dirs", lambda: {"output/ckpt/demo": tmp_path})
    with pytest.raises(toolkits.ToolkitError):
        toolkits.merge_run({"mode": "loras", "dir": "output/ckpt/demo", "adapters": [a.name]}, FakeManager())
    mgr = FakeManager()
    toolkits.merge_run(
        {
            "mode": "loras",
            "dir": "output/ckpt/demo",
            "adapters": [a.name, b.name],
            "weights": "1,0.5",
            "normalize": "off",
            "dtype": "bf16",
        },
        mgr,
    )
    argv = mgr.commands[0]["argv"]
    assert argv[0] == "scripts/toolkits/merge_loras.py"
    assert "--weights" in argv and "1,0.5" in argv
    assert "--normalize" in argv and "off" in argv


def test_merge_analyze_and_extract(tmp_path, monkeypatch):
    a = _adapter(tmp_path, "a.safetensors")
    b = _adapter(tmp_path, "b.safetensors")
    monkeypatch.setattr(toolkits, "_adapter_dirs", lambda: {"output/ckpt/demo": tmp_path})
    mgr = FakeManager()
    toolkits.merge_run({"mode": "analyze", "dir": "output/ckpt/demo", "adapters": [a.name, b.name]}, mgr)
    assert "--analyze" in mgr.commands[-1]["argv"]

    tuned = tmp_path / "tuned.safetensors"
    out = tmp_path / "delta.safetensors"
    toolkits.merge_run(
        {
            "mode": "extract",
            "tuned": str(tuned),
            "out": str(out),
            "rank": 32,
            "dtype": "bf16",
            "include_adaln": True,
            "adaln_layout": "comfy",
        },
        mgr,
    )
    argv = mgr.commands[-1]["argv"]
    assert argv[0] == "scripts/toolkits/extract_delta_lora.py"
    assert "--save_dtype" in argv and "bfloat16" in argv
    assert "--include_adaln" in argv and "comfy" in argv


def test_merge_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(toolkits, "_adapter_dirs", lambda: {"d": tmp_path})
    with pytest.raises(toolkits.ToolkitError):
        toolkits.merge_run({"mode": "dit", "dir": "d", "adapter": "../evil.safetensors"}, FakeManager())


# --------------------------------------------------------------------------- #
# TensorBoard
# --------------------------------------------------------------------------- #
def test_run_dirs_finds_event_files(tmp_path, monkeypatch):
    a = tmp_path / "run_a"
    a.mkdir()
    (a / "events.out.tfevents.123").write_bytes(b"x")
    nested = tmp_path / "turbo" / "run_b"
    nested.mkdir(parents=True)
    (nested / "events.out.tfevents.456").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(toolkits, "TB_BASE", tmp_path)
    assert {p.name for p in toolkits._run_dirs()} == {"run_a", "run_b"}


def test_tensorboard_remove_stays_inside_logs(tmp_path, monkeypatch):
    base = tmp_path / "logs"
    run = base / "run_a"
    run.mkdir(parents=True)
    (run / "events.out.tfevents.1").write_bytes(b"x")
    monkeypatch.setattr(toolkits, "TB_BASE", base)
    monkeypatch.setattr(toolkits, "TB_STATE", tmp_path / "tb.json")
    toolkits.tensorboard_remove({"run": "run_a"})
    assert not run.exists()
    with pytest.raises(toolkits.ToolkitError):
        toolkits.tensorboard_remove({"run": "../outside"})


def test_tensorboard_state_empty_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(toolkits, "TB_BASE", tmp_path / "logs")
    monkeypatch.setattr(toolkits, "TB_STATE", tmp_path / "tb.json")
    state = toolkits.tensorboard_state()
    assert state["running"] is False
    assert state["port"] is None
    assert state["runs"] == []


# --------------------------------------------------------------------------- #
# EasyControl
# --------------------------------------------------------------------------- #
def test_easycontrol_lists_method_and_descriptors():
    catalog = easycontrol.list_variants()
    values = [v["value"] for v in catalog["variants"]]
    assert values[0] == "easycontrol"
    assert {"colorize", "inpaint", "sanitize", "region"} <= set(values)
    assert all(v["kind"] in {"method", "descriptor"} for v in catalog["variants"])


def test_easycontrol_descriptor_detail_uses_compound_keys():
    detail = easycontrol.detail("colorize")
    assert detail["kind"] == "descriptor"
    labels = [g["label"] for g in detail["groups"]]
    assert "顶部标量" in labels
    assert "[training]" in labels
    keys = {f["key"] for g in detail["groups"] for f in g["fields"]}
    assert "training.network_dim" in keys
    assert "network_dim" not in keys


def test_easycontrol_descriptor_save_preserves_comments_and_blueprint(tmp_path, monkeypatch):
    (tmp_path / "colorize.toml").write_text(
        (ROOT / "configs" / "easycontrol" / "colorize.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(easycontrol, "DESC_DIR", tmp_path)
    before = (tmp_path / "colorize.toml").read_text(encoding="utf-8")
    res = easycontrol.save("colorize", {"values": {"training.network_dim": 32}})
    after = (tmp_path / "colorize.toml").read_text(encoding="utf-8")
    assert res["changed"] == 1
    assert "network_dim = 32" in after
    # Comments + blueprint survive the tomlkit round-trip.
    assert "# Korean-text slice" in after
    assert "[[datasets.subsets]]" in after
    assert before != after


def test_easycontrol_save_rejects_unknown_key():
    with pytest.raises(easycontrol.EasyControlError):
        easycontrol.save("colorize", {"values": {"training.no_such_knob": 1}})


def test_easycontrol_descriptor_save_list_keeps_int_type(tmp_path, monkeypatch):
    (tmp_path / "colorize.toml").write_text(
        (ROOT / "configs" / "easycontrol" / "colorize.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(easycontrol, "DESC_DIR", tmp_path)
    easycontrol.save("colorize", {"values": {"training.target_res": "896\n1024"}})
    doc = tomllib.loads((tmp_path / "colorize.toml").read_text(encoding="utf-8"))
    assert doc["training"]["target_res"] == [896, 1024]
    assert all(isinstance(v, int) for v in doc["training"]["target_res"])


def test_easycontrol_preprocess_routes_adapter_env():
    mgr = FakeManager()
    toolkits_res = easycontrol.preprocess("colorize", mgr)
    call = mgr.commands[-1]
    assert call["argv"] == ["tasks.py", "easycontrol-preprocess", "--no-skip_mangafy"]
    assert call["extra_env"] == {"EASYADAPTER": "colorize"}
    assert toolkits_res["stage"] == "preprocess"


def test_easycontrol_train_descriptor_folds_training_table(monkeypatch):
    import scripts.tasks.training as training

    monkeypatch.setattr(training, "_easy_train_extra", lambda name, extra: ["--network_dim", "48"])
    mgr = FakeManager()
    res = easycontrol.train("colorize", "low_vram", mgr)
    call = mgr.submits[-1]
    assert call["method"] == "easycontrol"
    assert call["preset"] == "low_vram"
    assert call["methods_subdir"] is None
    assert call["extra"] == ["--network_dim", "48"]
    assert res["kind"] == "descriptor"


def test_easycontrol_train_method_routes_gui_methods():
    mgr = FakeManager()
    easycontrol.train("easycontrol", "default", mgr)
    call = mgr.submits[-1]
    assert call["method"] == "easycontrol"
    assert call["methods_subdir"] == "gui-methods"


# --------------------------------------------------------------------------- #
# 实验：Turbo / Soup
# --------------------------------------------------------------------------- #
def _stage_distill(tmp_path: Path, monkeypatch, kind: str) -> Path:
    rel = toolkits.DISTILL[kind]["path"]
    src = ROOT / rel
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(toolkits.config, "ROOT", tmp_path)
    return dst


def test_distill_state_groups_sections(tmp_path, monkeypatch):
    _stage_distill(tmp_path, monkeypatch, "turbo")
    state = toolkits.distill_state("turbo")
    labels = [g["label"] for g in state["groups"]]
    assert "基础" in labels
    assert "[network]" in labels and "[sampling]" in labels
    keys = {f["key"] for g in state["groups"] for f in g["fields"]}
    assert "network.student_rank" in keys


def test_distill_save_preserves_comments(tmp_path, monkeypatch):
    path = _stage_distill(tmp_path, monkeypatch, "turbo")
    res = toolkits.distill_save("turbo", {"network.student_rank": 32})
    text = path.read_text(encoding="utf-8")
    assert res["changed"] == 1
    assert "student_rank = 32" in text
    assert "# v2: masks are opt-in" in text


def test_distill_run_turbo_and_soup(tmp_path, monkeypatch):
    _stage_distill(tmp_path, monkeypatch, "turbo")
    _stage_distill(tmp_path, monkeypatch, "soup")
    mgr = FakeManager()
    toolkits.distill_run("turbo", {}, mgr)
    assert mgr.commands[-1]["argv"] == ["tasks.py", "turbo"]

    with pytest.raises(toolkits.ToolkitError):
        toolkits.distill_run("soup", {}, mgr)

    res = toolkits.distill_run(
        "soup",
        {"pathPattern": "sincos/*", "name": "soup_a", "ftArgs": "--network_dim 32"},
        mgr,
    )
    assert mgr.commands[-1]["argv"] == [
        "-m",
        "scripts.soup.pipeline",
        "--path_pattern",
        "sincos/*",
        "--name",
        "soup_a",
        "--network_dim",
        "32",
    ]
    assert res["state"] == "queued"


def test_distill_run_rejects_bad_kind():
    with pytest.raises(toolkits.ToolkitError):
        toolkits.distill_run("nope", {}, FakeManager())
