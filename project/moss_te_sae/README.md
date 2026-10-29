# moss_te_sae — Qwen-Scope SAEs on the MOSS text encoder

What the MOSS TE (post-trained Qwen3.5-9B) and the DiT do with character names and with
the caption rows. Read with Qwen-Scope's Base SAEs and with direct probes of the DiT. Two
questions run through it:

- how to give the DiT characters it cannot draw from the name
  ([`docs/moss/proposal/te_character_knowledge.md`](../../docs/moss/proposal/te_character_knowledge.md));
- why a LoRA trigger does not bind on MOSS ([hypothesis.md](hypothesis.md)).

## Setup

- SAEs: `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100` (TopK k=100, 64K features,
  `resid_post` per layer), layers 15 / 31 in `/media/sorryhyun/new/qwen35_sae/`.
- TEs: instruct bf16 `/media/sorryhyun/new/qwen35_pe/inst`, Base bf16
  `/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base`.
- Scripts in `src/` run as daemon jobs:
  `make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/<script>.py …"`.
  `template_attn_cmp.py` is CPU only.
- Results: newer runs in `output/tests/<name>/`, earlier ones under
  `/media/sorryhyun/new/qwen35_sae/<name>/<stamp>/`. Each report names its folder.

## Facts settled (2026-10-27)

- The MOSS TE is post-trained **Qwen3.5-9B** (instruct), not Base: every tensor the INT8
  file leaves unquantized (GDN `A_log` / `dt_bias` / `conv1d`, `q_norm` / `k_norm`, final
  norm, `embed_tokens`) matches `/media/sorryhyun/new/qwen35_pe/inst` bit for bit. Qwen-Scope
  ships only Base SAEs for 9B.
- `last_hidden_state` = final norm (gain `1 + w`) of layer 31 `resid_post` (rel err 1.7e-3).
- INT8 vs bf16 TE: 2.1 % mean per-token relative error on cached layout-JSON captions.

## Reports

| date | report | result |
|---|---|---|
| 10-27 | [sae_recon](reports/sae_recon_2026_10_27.md) | Base SAEs fit the caption rows at FVU 0.25–0.46; the DiT amplifies tag-content features, and ~1/3 of what it reads is in the SAE error term |
| 10-27 | [entity_neru](reports/entity_neru_2026_10_27.md) | the TE does not know akita neru; she reads like an invented name |
| 10-27 | [lora_use](reports/lora_use_2026_10_27.md) | the channel LoRA is a prompt-independent style shift; its tag is not a trigger (2.2 % of gradient × input) |
| 10-27 | [phase0_teacher_ceiling](reports/phase0_teacher_ceiling_2026_10_27.md) | the DiT draws Neru without the TE; `malfoid` / Koneko are DiT-unknown, tags lift recall to 0.83 |
| 10-27 | [entity_dit](reports/entity_dit_2026_10_27.md) | the identity sits in the name's own rows; the DiT writes it there and the image reads it in blocks 16–31 |
| 10-27 | [entity_add](reports/entity_add_2026_10_27.md) | tag directions on the name rows recover 1/4–1/3 of the teacher effect, no training |
| 10-27 | [entity_patch_malfoid](reports/entity_patch_malfoid_2026_10_27.md) | the distilled name-row patch fails the gate (24 % of the gain against 80 %); lr confounds the arms |
| 10-27 | [prompt_form_miku](reports/prompt_form_miku_2026_10_27.md) | the prompt form does not move a DiT-known character |
| 10-28 | [entity_series](reports/entity_series_2026_10_28.md) | series is linear over name vectors; the Vocaloid centroid transfers the outfit grammar |
| 10-29 | [base_contrast](reports/base_contrast_2026_10_29.md) | the template's effect on caption rows is an instruct × template interaction (0.10 / 0.24 alone, 0.60 together) |
| 10-29 | [template_attn](reports/template_attn_2026_10_29.md) | ~95 % of image→text attention lands on 10 template rows whose heavy members carry a prompt-constant value; the channel LoRA does not move it |
| 10-29 | [template_causal](reports/template_causal_2026_10_29.md) | swapping suffix rows changes nothing; cutting the caption keys loses the prompt; cutting the template keys collapses the image: the template gates, the caption carries |

## Open

- Causal check on the SAE side: ablate / boost single features in the TE and generate
  with a fixed seed. The error term carries ~1/3 of what the DiT reads, which bounds any
  SAE-space edit.
- Which blocks carry the caption read: a per-block `no_caption_keys` cut
  ([template_causal](reports/template_causal_2026_10_29.md) cut all blocks at once).
- [hypothesis.md](hypothesis.md): test 3 (Base contrast) is done; tests 1 (wrapper sweep),
  2 (name × context split) and 4 (training arm) are not.
