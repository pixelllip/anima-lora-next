# Krea-2-Raw LoRA target spec.
#
# Krea-2 是 single-stream MMDiT，block 命名 `blocks.N.attn.{wq,wk,wv,wo,gate}` +
# `mlp.{up,down,gate}` + `mod.lin`(Parameter) + norms，另有文本融合
# `txtfusion.{layerwise_blocks,refiner_blocks}.N.*` 与投影层。
#
# ## 注入点以官方口径为准（2026-10-05 修正）
#
# 官方 diffusers DreamBooth trainer（`examples/dreambooth/train_dreambooth_lora_krea2.py`）
# 的默认层集合是：
#   img_in, final_layer.linear, to_q, to_k, to_v, to_out.0, to_gate,
#   ff.up, ff.down, text_fusion.projector, txt_in.linear_1, txt_in.linear_2,
#   time_embed.linear_1, time_embed.linear_2, time_mod_proj
# 用 PEFT 的**后缀**匹配，所以 `to_*` / `ff.*` 同时命中主 block 与文本融合 block。
# 脚本自带的 mmdit 命名映射（同文件 1358-1362 行）为：
#   first -> img_in, last.linear -> final_layer.linear,
#   wq/wk/wv/wo -> to_q/to_k/to_v/to_out.0, gate -> to_gate,
#   mlp.up/mlp.down -> ff.up/ff.down, txtfusion.projector -> text_fusion.projector,
#   txtmlp.1/txtmlp.3 -> txt_in.linear_1/linear_2,
#   tmlp.0/tmlp.2 -> time_embed.linear_1/linear_2, tproj.1 -> time_mod_proj
#
# 该集合与「DiT 里除 SwiGLU gate 之外的全部 Linear」**完全等价**：两种模型里
# Linear 的全集减去 `ff.gate` 正好等于官方白名单。所以这里不需要逐个白名单
# 正则，直接以 `SingleStreamDiT` 为容器扫全部 Linear，再用 exclude 精确排除
# SwiGLU gate 即可（`targeting.py` 的语义：excluded and not included 才跳过）。
#
# 覆盖（28 主 block + 4 文本融合 block）：
#   blocks.N.attn.{wq,wk,wv,wo,gate}     + txtfusion.*_blocks.N.attn.{...}
#   blocks.N.mlp.{up,down}               + txtfusion.*_blocks.N.mlp.{up,down}
#   顶层：first, last.linear, tmlp.0/2, tproj.1, txtmlp.1/3, txtfusion.projector
# 合计 232 个 Linear（旧保守集只有 196：漏掉全部 attn.gate 与整条文本融合链路，
# 且多挂了官方不训的 SwiGLU gate）。
#
# 官方对长跑的建议是「加 rank 并把层收窄到注意力层」
# （`to_q,to_k,to_v,to_out.0,to_gate`）——即 `KREA2_ATTENTION_ONLY_SPEC`。
# 该变体保留给显式选择，默认仍是官方完整集。
# 另有中间档 `KREA2_ATTN_MLP_SPEC`（主 block attn + mlp.up/down = 196 个）：
# 给「attention-only 学不出东西、但 232 集在 8GB 卡上越线」的场景用。
# 四档通过 `krea2_target_kwargs(preset)` / `--krea2_target_preset` 选择
# （`official` / `official_gate` / `attention` / `attn_mlp`）。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class Krea2LoRATargetSpec:
    """Krea-2-Raw 的 LoRA target 参数。

    通过 LoRANetworkCfg 的 unet_target_replace_modules / exclude_patterns 字段
    注入，覆盖 anima 默认。anima 路径不设这些字段 → 回退到
    ANIMA_TARGET_REPLACE_MODULE + _DEFAULT_EXCLUDE，行为不变。
    """

    # 容器白名单：以 DiT 根模块（SingleStreamDiT）为容器，扫描其全部 Linear。
    # 这里**只能列这一个类名**：`targeting.py` 对每个命中的容器实例独立扫描子模块，
    # 若同时再列 `SingleStreamBlock`，主 block 的 Linear 会被扫两遍而产生重复
    # LoRA 模块。用根容器 + exclude 才是无重复的等价表达。
    unet_target_replace_modules: Sequence[str] = ("SingleStreamDiT",)

    # include_patterns：Krea-2 不用 anima 的 adaln 豁免那套，留空。
    # 注入控制完全由 exclude_patterns 精确排除实现。
    include_patterns: List[str] = field(default_factory=list)

    # exclude_patterns：精确排除官方不训的 Linear。
    # targeting.py 语义：excluded and not included 才跳过。
    # 官方白名单 = 全部 Linear − SwiGLU gate，所以这里只需排掉 gate：
    #   blocks.N.mlp.gate                 (主 block SwiGLU gate)
    #   txtfusion.*_blocks.N.mlp.gate     (文本融合 block SwiGLU gate)
    # 其余条目都是兜底防御（norm / mod 是 Parameter，本不命中）。
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            r"blocks\.\d+\.mlp\.gate",  # 官方 ff.up/ff.down 不含 ff.gate
            r"txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+\.mlp\.gate",
            r".*\.mod\..*",  # DoubleSharedModulation.lin 是 Parameter
            r".*\.qknorm\..*",  # RMSNorm
            r".*\.prenorm\..*",  # RMSNorm
            r".*\.postnorm\..*",  # RMSNorm
            r".*\.norm\..*",  # LastLayer.norm / txtmlp.0 RMSNorm
            r".*\.modulation\..*",  # SimpleModulation.lin 是 Parameter
        ]
    )

    # Krea-2 无 q/k/v fuse（独立权重），fuse spec 为空。
    fuse_specs: tuple = ()

    # 文本编码器：Krea-2 用 Qwen3-VL-4B。官方 DreamBooth 也不训 TE。
    text_encoder_target_replace_modules: Sequence[str] = ()
    train_text_encoder: bool = False


# 官方默认（作者推荐，rank/alpha 32 起步，适配大多数画风）。
KREA2_LORA_TARGET_SPEC = Krea2LoRATargetSpec()

# 官方对长跑的建议变体：加 rank + 收窄到注意力层，避免 prompt 遵循退化。
# 只保留 attn.{wq,wk,wv,wo,gate}（主 block 28 + 文本融合 4 = 32 × 5 = 160 个），
# 排除全部 mlp 与顶层投影。
KREA2_ATTENTION_ONLY_SPEC = Krea2LoRATargetSpec(
    exclude_patterns=[
        r"blocks\.\d+\.mlp\..*",
        r"txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+\.mlp\..*",
        r"first",
        r"last\.linear",
        r"tmlp\..*",
        r"tproj\..*",
        r"txtmlp\..*",
        r"txtfusion\.projector",
        r".*\.mod\..*",
        r".*\.qknorm\..*",
        r".*\.prenorm\..*",
        r".*\.postnorm\..*",
        r".*\.norm\..*",
        r".*\.modulation\..*",
    ],
)

# 中间档：28 个主 block 的 attn（wq/wk/wv/wo/gate）+ mlp.up/mlp.down = **196** 个。
# 动机（2026-10-07 TwinBox 审计）：attention-only 把整条 MLP 链路排除了，而 anima 那份
# 的容量大头正好落在 MLP（`mlp_layer1` 参与度最高）——krea2 的 delta 因此塌成近似
# rank-2（参与度 1.92/16、top-1 独占 74.9% 能量）。本档只补 MLP，**不引入**官方 232 集里
# 那批宽张量（`tproj.1` 36864、`txtmlp`/`tmlp` 6144、`txtfusion` 2560、`first`/`last`），
# 因为 8GB 卡上 232 集的 reserved（7.69/7.97GB）越线换页，而差额主要来自这些新宽度。
# 覆盖：blocks.N.{attn ×5, mlp.up, mlp.down}；排除全部 txtfusion.*、first、last.linear、
# tmlp.*、tproj.*、txtmlp.* 与 SwiGLU gate。显存待量（见 findings 的 2026-10-07 补充）。
KREA2_ATTN_MLP_SPEC = Krea2LoRATargetSpec(
    exclude_patterns=[
        r"blocks\.\d+\.mlp\.gate",  # 官方 ff.up/ff.down 不含 ff.gate
        r"txtfusion\..*",  # 整条文本融合链路（含 projector 与 4 个融合 block）
        r"first",
        r"last\.linear",
        r"tmlp\..*",
        r"tproj\..*",
        r"txtmlp\..*",
        r".*\.mod\..*",
        r".*\.qknorm\..*",
        r".*\.prenorm\..*",
        r".*\.postnorm\..*",
        r".*\.norm\..*",
        r".*\.modulation\..*",
    ]
)

# SwiGLU gate 是官方 232 集**唯一整类排除**的 Linear（见 KREA2_LORA_TARGET_SPEC 上方注释）。
KREA2_MLP_GATE_PATTERNS = (
    r"blocks\.\d+\.mlp\.gate",
    r"txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+\.mlp\.gate",
)

# 档位：官方 232 集 + 全部 32 个 `mlp.gate`（SwiGLU 门）= **264**，即"全部 Linear"。
#
# 动机（2026-10-13，见 docs/findings/twinbox_style_incompleteness_audit.md §3.2）：
# 用户实际在用的第三方 Krea-2 LoRA 都把 **17.7–26.4%** 的 delta 能量放在
# `mlp.gate` 上（`nai_ba_style_krea2-st6000` 里它是第一大项，且因与 `mlp.up` 形状相同
# 可直接比，是 `mlp.up` 的 1.53 倍；`KNP_000003000` 是 17.7%，**但它不是画风 LoRA**，
# 只作旁证），而 `official` 集把它整类排除 —— 本仓自己的 232 产物在这一类是 **0**。
#
# 为什么它是"低风险"的一档：
#   * `mlp.gate` 是 `SwiGLU.forward = down(silu(gate(x)) * up(x))` 里的普通
#     `nn.Linear`（`dit.py:206`），所以它的 LoRA 键**完全走 vanilla ComfyUI 的原生
#     路径**，不会被静默丢弃（对比 `train_modulation` 的 `modulation_deltas.*`）。
#     它是 MLP 分支的**乘性逐通道门**，是渲染质感/增益类风格的自然旋钮。
#   * 它与 `mlp.up` / `mlp.down` **形状相同**，不引入新的张量宽度 —— 而 8GB 卡上
#     160→232 越线的主因被实测归因于**新增宽度**（`tproj.1` 6144→36864 等），
#     不是参数量（`max_alloc` 相同而 `reserved` +1.66GB）。
#
# 代价：r16 下 +10.70M 参数（47.95M → 58.65M，+22%）。**上线前必须跑显存探针**，
# 且仓库已被"3 步探针看不到稳态"坑过两次 —— 探针 PASS 不等于可以长跑，仍要看
# 正式跑的前 30 步。
KREA2_OFFICIAL_GATE_SPEC = Krea2LoRATargetSpec(
    # = official 减去 mlp.gate 这两条排除（单一真相源，避免两处漂移）
    exclude_patterns=[
        pattern
        for pattern in KREA2_LORA_TARGET_SPEC.exclude_patterns
        if pattern not in KREA2_MLP_GATE_PATTERNS
    ]
)

_KREA2_SPEC_PRESETS = {
    "official": KREA2_LORA_TARGET_SPEC,
    "official_gate": KREA2_OFFICIAL_GATE_SPEC,
    "attention": KREA2_ATTENTION_ONLY_SPEC,
    "attn_mlp": KREA2_ATTN_MLP_SPEC,
}


def krea2_target_kwargs(preset: str = "attention") -> dict:
    """返回注入 LoRANetworkCfg 的 kwargs（覆盖 anima 默认 target）。

    用法（在 family-aware network 构造处）：
        cfg = LoRANetworkCfg.from_kwargs({
            **krea2_target_kwargs(),
            "lora_dim": 16, "alpha": 8, ...
        }, network_dim=..., network_alpha=..., neuron_dropout=...,
           module_class=LoRAModule)

    `preset`：
      - ``"attention"``（**默认**）：只训注意力投影（160 个 target），配合更高
        rank —— 官方对长跑的建议。选它作默认是实测决定的，不是偏好：
        在 RTX 4060 Laptop 8GB 上，8.19GB 显存扣掉桌面 0.52GB 只剩 7.67GB
        torch pool 可用，而两种集合的实测峰值 `reserved` 是

            official (232, rank16)  eager   7.69GB  → 越线，free=0，WDDM 换页，
                                                    功耗跌到 36-46W，步时 79-99s
            official (232, rank16)  compile 7.97GB  → 更差（compile 方向为负）
            attention(160, rank16)  eager   6.97GB  → 留 0.7GB 余量，功耗 96-99W，
                                                    步时 21.5s

        LoRA 参数只差 47.95M vs 25.03M（fp32 179MB vs 94MB），所以差额不在权重，
        而在**新增模块带来的新张量宽度**（tproj.1 的 6144→36864、txtmlp/tmlp 的
        6144、txtfusion 的 2560）→ allocator pool 多留块。详见
        docs/findings/krea2_official_lora_recipe_audit.md。
      - ``"official"``：官方 DreamBooth 的完整层集合（232 个）。适合 >=12GB 的卡，
        或显存够、要最大容量时。
      - ``"official_gate"``：官方 232 集 + 全部 32 个 `mlp.gate`（**264**，即"全部
        Linear"）。动机、风险与代价见 `KREA2_OFFICIAL_GATE_SPEC` 上方注释：画风参照
        `nai_ba_style_krea2-st6000` 把 gate 当作第一大项，而 official 集排除了它。
        它与 `mlp.up` 同形状、不引入新宽度，且 LoRA 键完全被 vanilla ComfyUI 认账。
      - ``"attn_mlp"``：主 block 的 attn + `mlp.up/down`（**196** 个，中间档）。
        目的见 `KREA2_ATTN_MLP_SPEC` 上方注释：补回 MLP 广度而不引入 232 集的宽张量。
        显存尚未定档，上线前先用 `--memory_probe_jsonl` 复量。
    """
    if preset not in _KREA2_SPEC_PRESETS:
        raise ValueError(
            f"unknown Krea-2 LoRA target preset: {preset!r} "
            f"(expected one of {sorted(_KREA2_SPEC_PRESETS)})"
        )
    spec = _KREA2_SPEC_PRESETS[preset]
    return {
        "unet_target_replace_modules": list(spec.unet_target_replace_modules),
        "text_encoder_target_replace_modules": list(
            spec.text_encoder_target_replace_modules
        ),
        "include_patterns": list(spec.include_patterns) or None,
        # from_kwargs 读的是 "exclude_patterns" 键（会再 append _DEFAULT_EXCLUDE）。
        "exclude_patterns": list(spec.exclude_patterns),
        "train_text_encoder": spec.train_text_encoder,
    }


# 注入点的人类可读清单（调试/文档用）。
# 官方默认：每主 block 7 个（attn.wq/wk/wv/wo/gate + mlp.up/down），
# 每文本融合 block 同样 7 个（共 4 个：2 layerwise + 2 refiner），
# 加 8 个顶层 Linear → 28×7 + 4×7 + 8 = 232。
KREA2_LORA_INJECTION_POINTS = {
    "blocks.N.attn.wq": "query proj (GQA 48 头, 6144→6144)",
    "blocks.N.attn.wk": "key proj (GQA 12 头, 6144→1536)",
    "blocks.N.attn.wv": "value proj (GQA 12 头, 6144→1536)",
    "blocks.N.attn.gate": "sigmoid 输出门控 (官方 to_gate, 6144→6144)",
    "blocks.N.attn.wo": "output proj (6144→6144)",
    "blocks.N.mlp.up": "SwiGLU up (6144→16384)",
    "blocks.N.mlp.down": "SwiGLU down (16384→6144)",
    "txtfusion.{layerwise,refiner}_blocks.N.attn.{wq,wk,wv,wo,gate}":
        "文本融合注意力 (官方 to_*, 2560→2560)",
    "txtfusion.{layerwise,refiner}_blocks.N.mlp.{up,down}":
        "文本融合 SwiGLU (官方 ff.up/ff.down)",
    "txtfusion.projector": "12 层 MFA 投影 (12→1)",
    "txtmlp.1 / txtmlp.3": "text_in 投影 (官方 txt_in.linear_1/2)",
    "tmlp.0 / tmlp.2": "timestep embedding MLP (官方 time_embed.linear_1/2)",
    "tproj.1": "timestep modulation 投影 (官方 time_mod_proj, 6144→36864)",
    "first": "patch embed (官方 img_in)",
    "last.linear": "final layer 输出投影 (官方 final_layer.linear)",
}

# 明确不挂的点（及原因）。
KREA2_LORA_EXCLUDED_POINTS = {
    "blocks.N.mlp.gate": "SwiGLU gate —— 官方白名单只含 ff.up/ff.down，不含 ff.gate",
    "txtfusion.*_blocks.N.mlp.gate": "同上（文本融合 SwiGLU gate）",
    "blocks.N.mod.lin": "DoubleSharedModulation.lin 是 Parameter(6*dim)，非 Linear",
    "blocks.N.prenorm.scale / postnorm.scale": "RMSNorm Parameter，非 Linear",
    "blocks.N.attn.qknorm.{q,k}norm.scale": "RMSNorm Parameter，非 Linear",
    "last.norm.scale / txtmlp.0.scale": "RMSNorm Parameter，非 Linear",
}
