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
that are in the tagger vocab. Generations: `output/tests/{neru,malfoid,koneko}_phase0/`. The
`src/tag_recall.py` job writes `tag_recall.json` there.

| condition | akita neru (17 ref tags, Spectrum off) | malfoid (8 ref tags, Spectrum on) | toujou koneko (6 of 8 ref tags in vocab, Spectrum on) |
|---|---|---|---|
| name only | 0.55 (visually Neru in all three) | **0.29** (generic Hogwarts students, Gryffindor ties) | **0.06** (High School DxD's anime style, three other girls) |
| name + co-occurrence tags | 0.94 | 0.83 | 0.83 |
| + wiki sentence | 0.82 | — | — |
| invented name + tags | 0.88 | 0.75 | 0.83 |
| control | Miku name only: Miku in all three | `draco malfoy` + genderswap: male Draco, 0.17 | — |

- **The DiT knows Neru from her name although the TE does not**: it memorised the name
  tokens from MOSS pretraining. The TE-knowledge premise does not hold for her, and the
  render problems seen with her come from elsewhere (multi-character binding, or the
  channel LoRA).
- `malfoid` (a recent meme, absent from the 2024 KB and the 2026 wiki dump) is unknown to
  the DiT, and spelled-out tags fix it. The 0.29 → 0.83 gap is the method's target.
  Targets are characters the **DiT** cannot draw from the name, not characters the TE
  cannot describe.
- The free-text wiki sentence lowers recall against tags alone (0.82 vs 0.94).
- **`toujou koneko` is case ④ although Danbooru has thousands of her posts** (2026-10-27).
  The DiT draws the series' style from `high school dxd` but not her (0.06; with tags
  0.83, the same as the invented name). `src/entity.py` on the bf16 instruct TE (greedy,
  thinking off): QA calls her a white-furred fox from *Kemono Friends* with red eyes, and
  the hair-colour cloze matches the invented name's (`' white'` 0.02). Rias Gremory, run
  alongside: series right, no red in the hair cloze. post_count does not predict what
  MOSS saw.
- A chat on Hugging Face suggested the TE knows her; the local probe says it does not.
  Candidates from a hosted chat (thinking, sampling, possibly another size) need
  `entity.py` before they count as TE-known. No case-② character is found yet.

## Where the DiT expands a name (`src/entity_dit.py`, 20261027-2108)

Base DiT, int8 TE, 832×1216. Results in `/media/sorryhyun/new/qwen35_sae/entity_dit/20261027-2108/`.

**Probe.** Hair colour at the name tokens, 40 characters the DiT should know, 10
colours (majority class 0.125). Directions come from `{colour} hair` tag spans next to an
invented name. Name vector = mean over 3 contexts minus the invented names' mean. One
`x_t` (σ 0.9, zero latent), so the image tokens carry no character. Tag-direction
accuracy, span mean / last token:

| space | TE out | `text_embedder` out | block 0 | 4 | 8 | 14 | 20 | 30 | 36 | 41 |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | 0.07 / 0.15 | 0.20 / 0.25 | 0.25 / 0.17 | 0.28 / 0.33 | 0.60 / 0.47 | 0.65 / 0.40 | 0.55 / 0.40 | **0.75** / 0.30 | 0.57 / 0.25 | 0.50 / 0.28 |

- The TE output does not put colour on the name in a tag-like direction, and the two
  text blocks add little. Colour appears **inside the DiT blocks at the name's own
  positions**: from block ~6, peaking at block 30. Miku's `aqua` ranks 1st from block 20
  on (4th in the TE output). This is subject enrichment by the DiT's own weights; the
  image tokens hold nothing to copy from.
- A name-trained leave-one-out centroid probe stays at 0.1–0.38. With ~4 names per colour
  that probe is weak, so this run does not show whether colour is laid out consistently
  over names.
- `toujou koneko` and `malfoid` both get `white` on top in the late blocks, and so do
  several misses. That looks like a bias of the white direction, not knowledge.

**Knockout.** σ 0.9 (at σ 0.6 the name's effect is 1–2 % of x̂₀ and the read is noisy). `r` =
the share of the one-step name effect `E` that is kept. One noise seed, two x0 images:

| | `rows` (name rows into the invented caption) | `ctx` (rest of the caption only) | ko pre / 0 | ko 16 | ko 24 | ko 28 | ko 32 |
|---|---|---|---|---|---|---|---|
| miku | 0.97–0.98 | 0.23–0.25 | 0.27–0.43 | 0.23–0.34 | 0.22–0.45 | 0.04–0.59 | 0.95–0.99 |
| neru | 0.87–0.98 | 0.04–0.21 | 0.08–0.27 | 0.07–0.31 | 0.17–0.32 | 0.33–0.46 | 0.91–0.98 |
| teto | 0.96 | 0.01–0.08 | −0.03–0.08 | 0.15–0.26 | 0.25–0.34 | 0.38–0.62 | 0.85–0.98 |

- **The identity sits in the name's own TE rows.** Splicing them into the invented-name
  caption keeps 0.96–0.98 of the effect. The later TE tokens carry 0.0–0.25, and `ctx`
  ≈ `ko pre` everywhere (a consistency check: both are what is left once the name rows
  are gone).
- **The image reads the enriched name in blocks ~16–31.** Knocking it out from block 32
  costs almost nothing, from block 28 or earlier most of the effect. Teto's
  ko 0–12 ≈ 0 means nothing useful is read before block 12. Miku's ko 28 < ko 24 on one x0
  is not monotone, so read single cells loosely.
- **For the proposal:** the target only has to change the name rows. That is B, or a
  `text_embedder` / text-position adapter. Nothing inside the TE has to change. The
  native mechanism is name rows → DiT blocks 0–30 enrich them at the name positions →
  image reads them in 16–31. The two text blocks are not where MOSS does it, but they are
  bidirectional, unmodulated and sit before every block, so an adapter there can still
  write a tag-like vector.
- Not checked: whether a full generation with ko ≥ 32 still draws Miku and ko ≤ 24 does
  not (tagger recall, 3 seeds × 2 arms ≈ 6 min).

## Tag directions added onto the name rows, no training (`src/entity_add.py`, 20261027-2120)

`N` / `T` = the phase-0 name / name + tags captions. `E = x̂₀(T) − x̂₀(N)`. An
injection adds to `N`'s name rows, in the TE output or the `text_embedder` output, the
`T` added-tag rows' mean minus `N`'s base-tag mean. `mean` gives every name row the same
vector; `split` deals the tags round-robin over the rows. Each is scaled to ρ × the
name-row norm, ρ 0.25–2; `rand` is a random direction at ρ 1. One noise seed; x0 = the
name and the tags generation (seed 11). `r / cos`:

| σ 0.9 | best injection | its r / cos | rand r / cos |
|---|---|---|---|
| malfoid, x0 tags | emb mean ρ2 (emb split ρ2 the same) | 0.35–0.37 / 0.60 | 0.06 / 0.21 |
| malfoid, x0 name | emb split ρ2 | 0.12 / 0.34 | 0.02 / 0.10 |
| koneko, x0 name | emb split ρ2 | 0.27 / 0.38 | 0.03 / 0.06 |
| koneko, x0 tags | te split ρ2 | 0.25 / 0.33 | 0.10 / 0.17 |
| neru, x0 name | te mean ρ0.5 | 0.16 / 0.23 (ρ ≥ 1 negative) | −0.19 / −0.09 |

- **Superposition partly works for the DiT-unknown characters.** Tag directions on the
  name rows recover a quarter to a third of the teacher effect, against ~0.05 for a
  random direction. The gain still rises at ρ 2, the largest scale tried. `split` ≥ `mean`
  and `emb` ≈ `te`. For Neru (DiT-known) a large injection hurts: it pushes the
  existing identity away.
- **σ 0.75 cannot tell them apart.** The random direction already reaches r 0.17–0.33,
  as high as the tag injections: any change to the name rows moves x̂₀ along a generic
  part of `E`. Read σ 0.9 only.
- **The image cannot be dropped.** Text rows run through the blocks without the image
  tokens differ from the joint forward by 13–36 % (all rows, blocks 0–31). The name rows
  differ by 20–56 %, most in blocks 24–31, which is the read window. The text path reads
  the image heavily, so a text-only distillation proxy is off. Distil with the full DiT
  (B / `text_embedder` LoRA with the velocity loss, a normal LoRA step's cost).
- The untrained `split` injection is a warm start for B's rows.

## Phase 2 B on `malfoid` (`src/entity_patch.py`, 20261027-2128)

A patch on the 3 name rows of the TE output (`m|alf|oid`), distilled from the phase-0
tags caption by the velocity loss: 4 teacher generations (seeds 101–104), σ from the
training distribution, 200 steps, Adam at lr 0.02 × the name-row element rms, 3.1 s/step.
Generation reads the patched `N` from the arm folder's `_te_cache`, with phase-0 flags
(Spectrum on) at seeds 11 / 22 / 33. Output in `output/tests/malfoid_patch/`.

| arm | val gap σ 0.9 / 0.6 / 0.3 (0 → 200 steps) | ‖Δ‖/‖row‖ | tagger recall | mean prob |
|---|---|---|---|---|
| name only (phase 0) | — | — | 0.29 | 0.26 |
| `split` init (ρ 2) | 0.96 / 0.99 / 0.99 → 0.46 / 0.38 / 0.87 | 2.0 (stuck at init) | 0.42 | 0.37 |
| `zero` init | 1.0 → 0.66 / 0.99 / 1.04 | 0.32 | 0.42 | 0.29 |
| tags (phase 0 / seeds 101–104) | — | — | 0.83 / 0.72 | 0.62 / 0.61 |

- **The gate fails on recall.** Both arms recover 24 % of the teacher's gain; the gate is 80 %.
- **By eye `split` is clearly ahead.** All three are Slytherin (green tie, crest), and s11 / s22 are blonde
  with green eyes. `zero` gives a Gryffindor (s22) and a Ravenclaw-like dark-haired
  girl (s33). The recall scores 8 in-vocab tags, `sweater vest` and `hoop earrings`
  are missed even by the teacher, and at 1/8 per tag it does not separate the arms. The
  mean probability does: `split` 31 % of the gain, `zero` 8 %.
- **`split` s33 has dark green hair.** The green of the tie, eyes and robe binds onto the hair: a
  superposition binding error.
- **The learning rate confounds the arms.** `zero` grew to only 0.32 of a row in 200 steps, and `split`
  only turned its direction at fixed norm. The next step is a larger lr and more steps,
  not a verdict on the init.

## Prompt form on a DiT-known character (Miku, 20261027-2200)

The phase-0 Miku caption, Spectrum on, seeds 11 / 22 / 33, scored against 19 canonical
outfit tags (Danbooru solo-post co-occurrence). Output in `output/tests/miku_promptform/`.

| form | recall | mean prob |
|---|---|---|
| plain | 0.67 (0.37 / 0.68 / 0.95) | 0.56 |
| `What does hatsune miku look like in this picture?\n` + caption | 0.46 | 0.48 |
| caption + `\nWhat does hatsune miku look like?` | 0.77 | 0.61 |
| caption + `\nLet's think step by step.` | 0.65 | 0.54 |

- **All twelve images are canonical Miku.** Within a seed the composition barely changes:
  the seed sets the image, the prompt form does not.
- **The spread across seeds is larger than across forms.** Plain alone runs 0.37–0.95, and +0.10 for
  the question suffix is about 2 of 19 tags per image. Not separable with 3 seeds.
- **The question prefix is lowest.** A prefix changes every caption token (causal TE), so it moves
  the DiT off its training distribution. A suffix changes only the rows after the caption.
- **No sign that a question or a CoT line enhances the prompt.** Expected: one forward,
  no generation.

## Series directions in the name vectors (`src/entity_series.py`, 20261028-1309)

Base DiT, int8 TE. 56 characters (46 in series with ≥ 2 members: Vocaloid family 10
incl. Teto, Touhou 13, Hololive 7, Bocchi 4, Madoka 4, Re:Zero 3, DxD 3, Frieren 2;
10 singletons as distractor classes). Captions carry no `series` field. Results in
`/media/sorryhyun/new/qwen35_sae/entity_series/20261028-1309/` (vectors, PNGs,
`tags.json`, `summary.json`).

**Geometry.** Leave-one-out nearest-series-centroid accuracy (46 targets, null from 1000
label permutations: mean 0.04, p95 0.11):

| TE out | `text_embedder` out | block 4 | 8 | 14 | 20 | 24 | 30 | 36 | 41 |
|---|---|---|---|---|---|---|---|---|---|
| 0.39 | 0.50 | 0.57 | 0.63 | **0.65** | 0.63 | 0.57 | 0.59 | 0.22 | 0.20 |

- Series is laid out linearly over names already in the TE output, unlike hair colour
  (≈ chance there), peaks in blocks 8–20 and fades after the read window (36, 41).
- Lexical overlap inflates part of it: `ram / rem / emilia (re:zero)`, `frieren` /
  `fern (sousou no frieren)`, `meiko / kaito (vocaloid)` share series tokens and have
  the highest `align` (0.55–0.89).

**Render.** Name only, seed 11; the tagger's series-tag probability (`series_p`) per
character against `align` (cos to her series' LOO centroid). Spearman, the 7 lexical
names left out (n = 39):

| | TE out | emb out | block 8 | 14 | 20 | 30 |
|---|---|---|---|---|---|---|
| pooled | +0.41 (p 0.01) | +0.48 (p 0.004) | +0.13 | +0.09 | +0.07 | −0.06 |
| within Vocaloid (8) | +0.55 | +0.69 | +0.64 | +0.69 | +0.69 | +0.69 |
| within Touhou (13) | +0.36 | +0.53 | +0.54 | +0.57 | +0.57 | +0.49 |
| within Hololive (7) | −0.07 | +0.18 | +0.54 | +0.29 | +0.46 | +0.32 |

- **`series_p` is not render quality.** By eye nearly every character is drawn on model
  from the name alone: all ten Vocaloids (Neru, GUMI, Haku, MEIKO included, `series_p`
  ≤ 0.16), Reimu (0.07) and Marisa (0.03). The tagger's copyright tag fires on some
  designs and not others. The table above is `align` against that tag, not against
  whether the character is drawn.
- **Misses by eye:** Koneko (a generic blonde girl), Toyosatomimi no Miko (generic brown
  hair), Yorigami Shion (blue hair, not her outfit); Homura and Akeno partly. Miko and
  Shion have the lowest Touhou `align` (−0.09, −0.03), Koneko 0.00. Too few misses to
  test the hypothesis; it needs a per-character identity read (reference tags, as in
  phase 0) on a set with more misses.
- Hololive and Bocchi: drawn on model with low `align` (0.1–0.3). Their members are
  known one by one, not through a shared direction.
- `hair_p` saturates (most ≥ 0.9) and does not separate characters. `char_p` exists for
  only 10 names; not read.

**Injection.** `okuda reni` + the series centroid (emb out, members' mean minus the
invented names' mean) on every name row at ρ × the name-row norm. Tagger series prob /
mean prob of 4 shared tags, seeds 11 / 22:

| series | invented name | series written in the caption | inject ρ 1 | inject ρ 2 | random ρ 2 |
|---|---|---|---|---|---|
| Vocaloid (headset, detached sleeves, necktie, sleeveless shirt) | 0.00 / 0.10 | 0.26 / 0.19 | **0.84 / 0.74** | 0.50 / 0.56 | 0.00 / 0.00 |
| Touhou (bow, frills, hair bow, ribbon) | 0.00 / 0.01 | 0.04 / 0.25 | 0.13 / 0.45 | 0.09 / 0.43 | 0.00 / 0.00 |
| Hololive | 0.00 / 0.00 | 0.04 / 0.12 | 0.00 / 0.11 | 0.01 / 0.25 | 0.00 / 0.01 |
| Bocchi | 0.00 / 0.08 | 0.02 / 0.21 | 0.00 / 0.01 | 0.00 / 0.00 | 0.00 / 0.11 |

One-step `r` / cos against `E = x̂₀(series written) − x̂₀(invented)` at σ 0.9, ρ 1:
Vocaloid 0.92 / 0.67, Touhou 1.03 / 0.76, Hololive 0.27 / 0.44, Bocchi 0.06 / 0.19
(random 0.08–0.19 / 0.13–0.27).

- **The Vocaloid centroid carries the series' outfit grammar, more strongly than
  writing `vocaloid` in the caption.** It also pulls toward the prototype: `hatsune miku`
  0.39 / 0.92 at ρ 1 (s11 is a blonde Vocaloid in Miku's outfit, s22 is Miku).
- **Touhou transfers more than the tagger says.** Both seeds at ρ 1 and 2 are
  Scarlet-sister-like Touhou girls (mob cap, frilled dress, red ribbon), while the
  invented name alone is a plain schoolgirl; the tagger gives `touhou` 0.07–0.17.
- Hololive adds animal ears and a tail at ρ 2 (the shared kemonomimi of Fubuki, Okayu,
  Korone), nothing else. Bocchi does nothing.
- **For B's init:** a series centroid is a usable warm start only for series with a
  shared visual grammar (Vocaloid, Touhou-like), and it brings the prototype with it. A
  centroid without the best-known member, or with her direction projected out, is the
  variant to try. It says nothing for `malfoid` or Koneko as such; their series are not
  this kind.
- One seed per character for render, two for injection; read single cells loosely.
