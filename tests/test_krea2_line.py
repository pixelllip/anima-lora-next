"""Krea-2 line 的单元测试（不需要 GPU、不需要底模）。

覆盖三件事：
1. `TrainRequest` 的配置加载：未知键**报错**（不静默忽略）、命令行覆盖、往返一致；
2. target 选择：spec 的 exclude 正则真的排掉了 SwiGLU gate 与各种 Parameter；
3. `LoRALinear`：零初始化起步（ΔW=0 → 输出等于原层）、形状与参数计数、导出键名是
   ComfyUI 认的形状（`lora_unet_<路径>.lora_down.weight` / `.lora_up.weight` / `.alpha`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from library.krea2.lora import LoRALinear, SPEC_BY_PRESET, attach_lora, resolve_spec, select_targets  # noqa: E402
from library.krea2.requests import TrainRequest  # noqa: E402


# --- 1. 请求定义 --------------------------------------------------------------
def test_train_request_defaults_and_cli():
    req = TrainRequest().apply_cli(["--max_steps", "7", "--rank", "32", "--lr", "1e-4"])
    assert (req.max_steps, req.rank, req.lr) == (7, 32, 1e-4)
    # 当前口径是 official_gate（264 = 官方 232 + 32 个 mlp.gate），不是 attention(160)
    assert req.target_preset == "official_gate"
    assert req.blocks_to_swap == 26  # 512 档生产口径


def test_train_request_config_roundtrip(tmp_path):
    cfg = tmp_path / "lora.toml"
    cfg.write_text('rank = 8\nalpha = 8.0\ntarget_preset = "official"\n', encoding="utf-8")
    req = TrainRequest.load_config(cfg)
    assert (req.rank, req.alpha, req.target_preset) == (8, 8.0, "official")
    assert req.to_dict()["rank"] == 8


def test_train_request_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "bad.toml"
    cfg.write_text("rank = 8\ntypo_key = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        TrainRequest.load_config(cfg)


def test_resolve_spec_presets_and_deprecation():
    assert set(SPEC_BY_PRESET) == {"official_gate", "official", "attention"}
    with pytest.raises(SystemExit):
        resolve_spec("attn_mlp")  # 196 档已弃用，不许静默当别的档用
    with pytest.raises(SystemExit):
        resolve_spec("nope")


# --- 2. target 选择 -----------------------------------------------------------
class _Dummy(torch.nn.Module):
    """形状对齐 Krea-2 的命名：主 block 的 attn/mlp（含 gate）+ 一些不该命中的 Parameter/norm。"""

    class _Block(torch.nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.wq = torch.nn.Linear(dim, dim)
            self.attn.wk = torch.nn.Linear(dim, dim)
            self.attn.wv = torch.nn.Linear(dim, dim)
            self.attn.wo = torch.nn.Linear(dim, dim)
            self.attn.gate = torch.nn.Linear(dim, dim)
            self.mlp = torch.nn.Module()
            self.mlp.gate = torch.nn.Linear(dim, dim)
            self.mlp.up = torch.nn.Linear(dim, dim)
            self.mlp.down = torch.nn.Linear(dim, dim)
            self.prenorm = torch.nn.RMSNorm(dim) if hasattr(torch.nn, "RMSNorm") else torch.nn.LayerNorm(dim)
            self.mod = torch.nn.Module()
            self.mod.lin = torch.nn.Parameter(torch.zeros(dim))
            self.modulation = torch.nn.Module()
            self.modulation.lin = torch.nn.Parameter(torch.zeros(dim))

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([self._Block() for _ in range(2)])

    def forward(self, x):  # pragma: no cover - 只为形状合法
        return x


def test_select_targets_attention_preset_excludes_mlp_and_params():
    dit = _Dummy()
    picked = [p for p, _ in select_targets(dit, resolve_spec("attention"))]
    assert any(p.endswith("attn.wq") for p in picked)
    # attention 档按官方口径保留 attn.{wq,wk,wv,wo,gate} —— gate 是**要训**的，
    # 被排掉的是 mlp.gate（SwiGLU 闸门）与各种 Parameter/norm。
    assert any(p.endswith("attn.gate") for p in picked)
    assert not any("mlp" in p for p in picked), picked
    assert not any("mod" in p or "norm" in p for p in picked), picked


def test_select_targets_official_keeps_mlp_but_not_gate():
    dit = _Dummy()
    picked = [p for p, _ in select_targets(dit, resolve_spec("official"))]
    assert any(p.endswith("mlp.up") for p in picked)
    assert not any(p.endswith("mlp.gate") for p in picked), picked


# --- 3. LoRA 模块 -------------------------------------------------------------
def test_lora_starts_as_identity_and_exports_comfy_keys():
    pair = LoRALinear(8, 4, rank=2, alpha=2.0)
    assert torch.count_nonzero(pair(torch.randn(3, 8))) == 0, "up 零初始化 ⇒ 起步 ΔW=0"

    dit = _Dummy()
    x = torch.randn(2, 8)
    base_out = dit.blocks[0].attn.wq(x)
    handles = attach_lora(dit, rank=2, alpha=2.0, preset="attention")
    # 挂上之后（零初始化）输出必须与原来逐元素相同
    assert torch.allclose(dit.blocks[0].attn.wq(x), base_out, atol=1e-6)
    # 参数必须挂在 block **外** 的 holder 里（块交换会把 block 下每个 .weight 搬来搬去）
    assert handles.holder is dit._krea2_lora
    child_names = {name for name, _ in handles.holder.named_children()}
    assert len(child_names) == len(handles.modules)
    for path, pair in handles.modules.items():
        assert pair.lora_down.weight.device == torch.device("cpu")  # 还没搬到卡上
        assert not path.startswith("_krea2_lora")

    sd = handles.state_dict()
    keys = sorted(sd)
    assert keys, "必须扫到目标"
    assert all(k.startswith("lora_unet_") for k in keys)
    assert any(k.endswith(".lora_down.weight") for k in keys)
    assert any(k.endswith(".lora_up.weight") for k in keys)
    assert any(k.endswith(".alpha") for k in keys)
    assert handles.param_count == sum(m.trainable_param_count for m in handles.modules.values())


def test_attach_reports_when_nothing_matches():
    class OnlyExcluded(torch.nn.Module):
        """唯一那个 Linear 命中 exclude（`.*\\.mod\\..*`）→ 应当明确报错，不许静默挂 0 个。"""

        def __init__(self):
            super().__init__()
            self.wrap = torch.nn.Module()
            self.wrap.mod = torch.nn.Module()
            self.wrap.mod.lin = torch.nn.Linear(4, 4)  # 路径 wrap.mod.lin → 命中 `.*\.mod\..*`

    with pytest.raises(SystemExit):
        attach_lora(OnlyExcluded(), rank=2, alpha=2.0, preset="attention")
