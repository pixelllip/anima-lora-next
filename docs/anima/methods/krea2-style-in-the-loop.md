# Krea-2 style-in-the-loop（人物 LoRA × 画风 LoRA 叠加鲁棒）

状态：实验（代码已接入 `library/krea2` 训练线，单测通过；**krea2-webui 侧三画风 `cycle`/600 步已出图复核：
角色卡 > 人物 LoRA，`@1.0` 仍压画风，见下「边界」**）
适用版本：当前 main 工作区（2026-10-20）
入口：`configs/krea2/whale_mature_768_stackaware.toml`
相关代码：`library/krea2/stack.py`、`library/krea2/lora.py`、`library/krea2/train.py`
相关文档（机制与结论，krea2-webui 仓）：
`docs/findings/krea2_character_style_entanglement.md`、`docs/methods/stack-aware-character-lora.md`

## 问题

人物 LoRA 单独训练时只见过 base 特征。推理时叠画风 LoRA，画风改写
attention/MLP 的输入特征，人物 LoRA 的残差落不到原特征上 —— 鳍耳变尖耳、
瞳色变、鲸尾消失、服装走样；画风之间互叠不会这样。

更深一层：数据集只有一种画风、caption 又不描述画风，所以「身份」和「画风」对 loss
来说是**同一团没被解释的残差**。画风 LoRA 挂在训练里，就是在给这团残差里的画风
成分定价：

| 训练方式 | 最优人物 delta | 推理后果 |
| --- | --- | --- |
| `prob<1`（有画风关闭的步） | `α·画风 + 身份`，α>0 | 强度 w 同时缩放两者：w=1 压画风、w↓ 丢细节 |
| **`pinned`**（画风每步都在、强度钉死） | 画风成分纯重复计算 → `α→0` | 叠同一档画风即复现，两者不抢预算 |

## 做法

训练人物 LoRA 时，**每一步**都挂**冻结**画风 LoRA（`attach_lora_from_file`，不更新其
权重），强度取你实际出图的逐条固定值（±抖动）。人物残差直接在画风改写的特征上优化，
且因为画风从不缺席，它不需要、也不允许把画风写进自己的权重。

组合是纯加法：`base + character + style`。人物先挂，画风后挂；后挂的 forward
包裹前者，顺序无关，不需要 merge（NF4 权重也不反量化合并）。

## 配置

```toml
target_preset = "official_gate"     # 264，人物不只靠 attention 承载身份
stack_lora_paths = "KNP.safetensors;twinbox.safetensors"
stack_lora_mode = "pinned"
stack_lora_strengths = "1.25;1.0"   # 与 paths 一一对应 = 出图强度
stack_lora_strength_jitter = 0.05   # ±5%
stack_lora_prob = 1.0               # pinned 不看它；写出来是为了快照可读
stack_lora_warmup_steps = 0         # ⚠️ 必须 0
```

要点：

- `pinned` **必须**给 `stack_lora_strengths`，且条数等于路径数；缺了直接 `SystemExit`，
  不会静默退化成随机强度。
- `warmup` 必须 0：任何一步没画风，delta 就会把画风吃回去。
- 想让 delta 对「挂哪条画风」也不敏感：`mode="cycle"` + `prob=1.0` + 逐条强度，
  每步恰有一条、永不落到 0。
- 元数据写入 `ss_stack_lora_mode` / `ss_stack_lora_strengths` /
  `ss_stack_lora_strength_jitter`；pinned 成品脱离这套强度不可复现。

推荐：264 注入集；pinned；jitter 0.05；warmup 0；700 步（30 张 × 23 遍）。
完整示例见 `configs/krea2/whale_mature_768_stackaware.toml`。

## 验证

同 seed、同精确触发词、同分辨率，出三张：人物单挂 / 画风单挂 / 人物+画风
（强度与训练同档）；检查鳍耳、瞳色、鲸尾、发色、标志服装。目标：人物+画风的身份特征
≥ 人物单挂的 80%，画风强度不下降。反向检查：pinned 成品不带画风时应偏「欠风格」
（它本来就是残差）——不能接受就说明用法必须永远带画风。

## 边界

- 仅 Krea-2 line；画风 LoRA 通过文件里的模块名匹配 DiT，兼容
  `diffusion_model.*`（ai-toolkit）与 `lora_unet_*`（kohya）两种键。
- 冻结画风网络常驻显存（约 48–228MB/个），不增加优化器状态。
- 每步多一次（冻结）LoRA 加法；不开 `torch.compile` 时无额外图问题。
- **2026-10-20 出图复核（krea2-webui，`test_00609–00619`）**：三画风 `cycle`/600 步成品
  `Δ` 从 17.81 涨到 **29.43**，`@1.0` 仍压画风；`@0.5` 在重画风栈下身份出不来；
  **853 字角色卡（不加人物 LoRA）身份最完整、画风随强度平滑变化** ⟹ 卡 > LoRA。
- **换一条没参与训练的画风**：pinned/cycle 只保证「人物 LoRA 不再自带画风压你」，
  不保证「身份细节在新画风里不掉」。人物 264 集的 delta 实测 **83% 在 MLP**
  （画风写的地方），碰撞面最大；新画风改写特征后身份仍可能退化。
  要外推就把要用的画风放进 `cycle`+`prob=1.0` 训练池，并**留一条留出画风**评估
  （细节见 krea2-webui 仓 `docs/findings/krea2_character_style_entanglement.md` §2.1/§8）。

## 测试

`python -m pytest tests/test_krea2_stack_lora.py -q`
