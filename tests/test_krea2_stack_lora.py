from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from library.krea2.lora import attach_lora_from_file, normalize_lora_weights
from library.krea2.stack import (
    apply_stack_schedule,
    build_stack_loras,
    parse_stack_strengths,
    resolve_stack_strengths,
    split_stack_paths,
    stack_enabled,
)


class _TinyDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.wq = torch.nn.Linear(4, 4, bias=False)
        block.attn.wq.weight.data.copy_(torch.eye(4))
        self.blocks.append(block)


def _write_lora(path, scale: float = 1.0) -> None:
    down = torch.zeros(2, 4)
    down[0, 0] = scale
    down[1, 1] = scale
    up = torch.zeros(4, 2)
    up[0, 0] = 1.0
    up[1, 1] = 1.0
    save_file(
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": down,
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": up,
        },
        str(path),
    )


def test_normalize_peft_keys_adds_alpha():
    out = normalize_lora_weights(
        {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.zeros(2, 4),
            "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.zeros(4, 2),
        }
    )
    assert "lora_unet_blocks_0_attn_wq.lora_down.weight" in out
    assert "lora_unet_blocks_0_attn_wq.lora_up.weight" in out
    assert float(out["lora_unet_blocks_0_attn_wq.alpha"]) == 2.0


def test_attach_from_file_multiplier_and_composition(tmp_path):
    dit = _TinyDiT()
    x = torch.randn(1, 4)
    base = dit.blocks[0].attn.wq(x)

    first = tmp_path / "first.safetensors"
    _write_lora(first, scale=1.0)
    handles = attach_lora_from_file(
        dit, str(first), holder_name="_stack_0", dtype=torch.float32
    )
    assert len(handles.modules) == 1
    handles.set_multiplier(0.0)
    assert torch.allclose(dit.blocks[0].attn.wq(x), base, atol=1e-6)

    handles.set_multiplier(1.0)
    mixed = dit.blocks[0].attn.wq(x)
    assert not torch.allclose(mixed, base)
    expected = base + torch.tensor([[x[0, 0], x[0, 1], 0.0, 0.0]])
    assert torch.allclose(mixed, expected, atol=1e-5)

    second = tmp_path / "second.safetensors"
    _write_lora(second, scale=2.0)
    handles2 = attach_lora_from_file(
        dit, str(second), holder_name="_stack_1", dtype=torch.float32
    )
    handles2.set_multiplier(1.0)
    expected2 = base + torch.tensor([[3 * x[0, 0], 3 * x[0, 1], 0.0, 0.0]])
    assert torch.allclose(dit.blocks[0].attn.wq(x), expected2, atol=1e-5)


def test_build_stack_loras_freezes_and_starts_disabled(tmp_path):
    dit = _TinyDiT()
    x = torch.randn(1, 4)
    base = dit.blocks[0].attn.wq(x)
    path = tmp_path / "style.safetensors"
    _write_lora(path)
    handles = build_stack_loras(dit, [str(path)], dtype=torch.float32)
    assert len(handles) == 1
    assert all(not p.requires_grad for p in handles[0].parameters())
    assert torch.allclose(dit.blocks[0].attn.wq(x), base, atol=1e-6)


def test_stack_schedule_warmup_and_prob():
    calls: list[float] = []
    handles = [SimpleNamespace(set_multiplier=calls.append) for _ in range(2)]

    info = apply_stack_schedule(
        handles,
        step=1,
        prob=1.0,
        strength_min=0.5,
        strength_max=0.5,
        warmup_steps=2,
        seed=3,
    )
    assert info["reason"] == "warmup"
    assert calls[-2:] == [0.0, 0.0]

    info = apply_stack_schedule(
        handles,
        step=2,
        prob=1.0,
        strength_min=0.5,
        strength_max=0.5,
        warmup_steps=2,
        seed=3,
    )
    assert info["active"] is True
    assert calls[-2:] == [0.5, 0.0]

    info = apply_stack_schedule(
        handles,
        step=3,
        prob=1.0,
        strength_min=0.5,
        strength_max=0.5,
        warmup_steps=2,
        seed=3,
    )
    assert calls[-2:] == [0.0, 0.5]


def test_stack_schedule_independent_can_enable_both():
    calls: list[float] = []
    handles = [SimpleNamespace(set_multiplier=calls.append) for _ in range(2)]
    info = apply_stack_schedule(
        handles,
        step=5,
        prob=1.0,
        strength_min=0.5,
        strength_max=0.5,
        warmup_steps=0,
        seed=1,
        mode="independent",
    )
    assert info["active"] is True
    assert len(info["active_styles"]) == 2
    assert calls[-2:] == [0.5, 0.5]


def test_split_stack_paths():
    assert split_stack_paths("a.safetensors; b.safetensors\nc.safetensors") == [
        "a.safetensors",
        "b.safetensors",
        "c.safetensors",
    ]
    assert split_stack_paths("") == []


def test_parse_stack_strengths():
    assert parse_stack_strengths("1.25;1.0") == (1.25, 1.0)
    assert parse_stack_strengths(["1.25", "1.0"]) == (1.25, 1.0)
    assert parse_stack_strengths("") is None
    assert parse_stack_strengths(None) is None
    with pytest.raises(ValueError):
        parse_stack_strengths("0;1.0")


def test_resolve_requires_one_strength_per_path_and_pinned_needs_them():
    paths = ["a.safetensors", "b.safetensors"]
    assert resolve_stack_strengths(paths, "1.25;1.0", mode="pinned") == (1.25, 1.0)
    with pytest.raises(SystemExit):
        resolve_stack_strengths(paths, "1.25", mode="pinned")
    with pytest.raises(SystemExit):
        resolve_stack_strengths(paths, "", mode="pinned")
    assert resolve_stack_strengths(paths, "", mode="cycle") is None


def test_stack_enabled_pinned_ignores_prob():
    assert stack_enabled(paths=["a"], prob=0.0, mode="pinned", strengths=None) is True
    assert stack_enabled(paths=["a"], prob=0.0, mode="cycle", strengths=None) is False
    assert stack_enabled(paths=[], prob=1.0, mode="cycle", strengths=None) is False


def test_pinned_keeps_every_style_on_at_its_own_strength():
    calls: list[float] = []
    handles = [SimpleNamespace(set_multiplier=calls.append) for _ in range(2)]
    info = apply_stack_schedule(
        handles,
        step=7,
        prob=0.0,  # pinned 不看 prob
        strength_min=0.1,
        strength_max=0.1,
        warmup_steps=0,
        seed=2,
        mode="pinned",
        strengths=(1.25, 1.0),
    )
    assert info["mode"] == "pinned"
    assert calls[-2:] == [1.25, 1.0]

    for step in range(20):
        apply_stack_schedule(
            handles,
            step=step,
            prob=0.0,
            strength_min=0.1,
            strength_max=0.1,
            warmup_steps=0,
            seed=2,
            mode="pinned",
            strengths=(1.25, 1.0),
        )
        assert all(value > 0.0 for value in calls[-2:])


def test_pinned_jitter_bounds_and_determinism():
    handles = [SimpleNamespace(set_multiplier=lambda value: None)]
    seen = []

    for step in range(25):
        network = SimpleNamespace(set_multiplier=seen.append)
        info = apply_stack_schedule(
            [network],
            step=step,
            prob=1.0,
            strength_min=0.5,
            strength_max=0.5,
            warmup_steps=0,
            seed=9,
            mode="pinned",
            strengths=(1.0,),
            jitter=0.1,
        )
        assert 0.9 - 1e-9 <= info["active_styles"][0]["strength"] <= 1.1 + 1e-9
        assert seen[-1] == info["active_styles"][0]["strength"]


def test_pinned_strengths_override_sampled_range_in_cycle_mode():
    calls: list[float] = []
    handles = [SimpleNamespace(set_multiplier=calls.append) for _ in range(2)]
    info = apply_stack_schedule(
        handles,
        step=0,
        prob=1.0,
        strength_min=0.1,
        strength_max=0.2,
        warmup_steps=0,
        seed=1,
        mode="cycle",
        strengths=(1.25, 1.0),
    )
    assert info["strength"] == pytest.approx(1.25)
    assert calls[-2:] == [1.25, 0.0]
