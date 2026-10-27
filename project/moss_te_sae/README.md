# moss_te_sae — Qwen-Scope SAEs on the MOSS text encoder

Probe: `src/probe.py` (daemon job, ~1 min). SAEs: `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100`
(TopK k=100, 64K features, `resid_post` per layer), layers 15 / 31 in
`/media/sorryhyun/new/qwen35_sae/`; results in `…/qwen35_sae/probe/<stamp>/summary.json`.

## Facts settled (2026-10-27)

- The MOSS TE is post-trained **Qwen3.5-9B** (instruct), not Base: every tensor the INT8
  file leaves unquantized (GDN `A_log` / `dt_bias` / `conv1d`, `q_norm` / `k_norm`, final
  norm, `embed_tokens`) matches `/media/sorryhyun/new/qwen35_pe/inst` bit for bit. Qwen-Scope
  ships only Base SAEs for 9B.
- `last_hidden_state` = final norm (gain `1 + w`) of layer 31 `resid_post` (rel err 1.7e-3).
- INT8 vs bf16 TE: 2.1 % mean per-token relative error on cached layout-JSON captions.

## Run 20261027-1609 (150 layout-JSON captions from five `*_layout` caches + 80 pe_lora NL prompts)

Reconstruction (card encoder, no `b_dec` centring; the centred variant is far worse):

| layer | layout body | NL body | NL raw (no template) | template prefix / suffix |
|---|---|---|---|---|
| 31 FVU | 0.34 | 0.25 | 0.21 | 0.31 / 0.31–0.36 |
| 15 FVU | 0.46 | 0.41 | 0.24 | 0.07 / 0.49–0.60 |

DiT side (layer 31 body tokens; `P` = `text_embedder.proj`, linearised through the final
norm; the two text blocks after `P` are not modelled):

- Readability `‖P(g⊙d)‖/‖g⊙d‖`: random directions 2.08 (median), SAE features 1.71–3.09
  (1–99 %), actual body tokens 1.93.
- Energy into the DiT: reconstruction 59 %, **SAE error term 31 %** (TE side: 73 / 26 %), so
  the DiT reads the unexplained part more than the explained part.
- Top 100 features carry 49 % of feature energy into the DiT, mostly structural ones (JSON
  `":"`, the `@` artist marker, bbox digits, a near-always-on feature f13752).
- `P` suppresses the high-energy generic and structural features (read 0.8–1.0: f26383 carries
  15 % of TE-side energy, the `@` artist marker, JSON delimiters, the first token of a quoted
  string). It amplifies low-energy tag-content features, mostly second subwords of
  multi-token tags (read 3.6–4.0: `pantyh|ose`, `thigh|high`, `tor|og|ao`, `rand|oser|u`,
  `speech| bubble`).

## Open

- Causal check: ablate / boost single features in the TE and generate with a fixed seed.
- The error term carries ~1/3 of what the DiT reads, which bounds any SAE-space edit.

## Entity probe: akita neru (`src/entity.py`, 20261027-1615)

The TE does not know her; at every read she looks like an invented name (`okuda reni`):

- QA (greedy, thinking off): "a Touhou shrine maiden, black hair, red ribbon"; `亞北ネル` →
  "Danganronpa 2, pink hair". It also gets Kasane Teto half wrong (pink wavy hair, balloon)
  and invents Triple Baka (DECO*27, Kaito). Miku is correct.
- Cloze "X is a derivative character of the": `' Vocal'` 0.68 (miku), 0.47 (teto), absent
  for neru (`' anime'` 0.08, `' Tou'` 0.07) and for the fake name. Hair colour: no blonde or
  yellow in neru's top 6. `' phone'` 0.05 for neru against 0.09 for the fake name.
- SAE top-100 overlap at the name's last token (same slots of `channel6.layout.txt`): neru ∩
  fake 60 / 54 at L31 and 47 / 66 at L15, above neru ∩ miku (47 / 45, 25 / 31). The fake
  name sits in the same slots, which inflates its overlap. The strongest features at name
  tokens are structural (JSON closers, a generic "Japanese" feature that fires on the fake
  name too), so the QA and cloze reads settle the question.

## What the channel LoRA reads (`src/lora_use.py`, `caststation_ema097_lr1e4`, 20261027-1626)

8 training captions (+ plain) and 4 `hews_layout` captions, σ 0.3 / 0.7. Δ = v_lora − v_base.
The bf16 re-encode differs from the cached embeds by 2–4 % (int8-sized), so the cache was
probably not encoded with the bf16 TE despite its stamp.

- ‖Δ‖/‖v‖ = 5.5 % on its own captions and 6.2 % on another artist's captions.
- Caption edits leave ‖Δ‖ unchanged (ratio 0.99–1.01). cos(Δ_edit, Δ_full) at σ 0.3 / 0.7:
  no artist 0.82 / 0.94, no characters 0.81 / 0.91, no tag bag 0.78 / 0.89, plain caption
  0.79 / 0.92, artist only 0.75 / 0.83, **empty prompt 0.74 / 0.83**. The LoRA is a
  prompt-independent style shift; the `@channel (caststation)` tag is not its trigger.
- Gradient × input on the full caption: tag bag 39 %, items 35 %, template 12.5 %, artist
  2.2 %, characters 2.2 %. Top tokens are commas and JSON delimiters, which pool context.
- The split over SAE features cancels heavily (Σ|c| = 4.6 × the token-level total). Its top
  features are the generic and structural ones, with no semantic feature standing out.

## Phase 0: teacher ceiling (`docs/moss/proposal/te_character_knowledge.md`)

Base MOSS, 832×1216, seeds 11 / 22 / 33. Recall = Anima Tagger hits on the character's
solo-post co-occurrence appearance tags (fetched through gelcrawl's Danbooru session)
that are in the tagger vocab. Generations: `output/tests/{neru,malfoid}_phase0/`. The
`src/tag_recall.py` job writes `tag_recall.json` there.

| condition | akita neru (17 ref tags, Spectrum off) | malfoid (8 ref tags, Spectrum on) |
|---|---|---|
| name only | 0.55 (visually Neru in all three) | **0.29** (generic Hogwarts students, Gryffindor ties) |
| name + co-occurrence tags | 0.94 | 0.83 |
| + wiki sentence | 0.82 | — |
| invented name + tags | 0.88 | 0.75 |
| control | Miku name only: Miku in all three | `draco malfoy` + genderswap: male Draco, 0.17 |

- **The DiT knows Neru from her name although the TE does not**: it memorised the name
  tokens from MOSS pretraining. The TE-knowledge premise does not hold for her, and the
  render problems seen with her come from elsewhere (multi-character binding, or the
  channel LoRA).
- `malfoid` (a recent meme, absent from the 2024 KB and the 2026 wiki dump) is unknown to
  the DiT, and spelled-out tags fix it. The 0.29 → 0.83 gap is the method's target.
  Targets are characters the **DiT** cannot draw from the name, not characters the TE
  cannot describe.
- The free-text wiki sentence lowers recall against tags alone (0.82 vs 0.94).
