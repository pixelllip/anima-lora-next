# MOSS: character knowledge in the text encoder

Status: proposal (2026-10-27). Phase 0 passed on `malfoid` (name only 0.29 → with tags 0.83
tagger recall); `akita neru` turned out to be known to the DiT already. Evidence:
`project/moss_te_sae/README.md`.

Goal: a prompt that names a character (`akita neru`) should reach the DiT as if it also
spelled out her appearance (`blonde hair, long hair, side ponytail, yellow eyes, holding
phone, …`). Hatsune Miku already behaves this way: the TE knows her, and the name alone
renders her. The knowledge goes into the frozen Qwen3.5-9B TE (or its output). The DiT is
not touched.

## Why the TE and not a DiT LoRA

Measured on the shipped stack (`project/moss_te_sae/`):

1. **The TE does not know minor characters.** Asked about Akita Neru it invents a Touhou
   shrine maiden. Cloze `"X is a derivative character of the"` gives `' Vocal'` 0.68 for
   Miku, 0.47 for Teto, and nothing for Neru, the same as for an invented name. At the
   name's last token, Neru's SAE features overlap more with the invented name's than with
   Miku's.
2. **The DiT reads appearance tags well.** `text_embedder.proj` passes tag-content SAE
   features at 3.6–4.0× gain, against 2.1× for random directions and 0.8–1.0× for the
   high-energy structural ones. Spelled-out appearance already has a strong path into the
   DiT.
3. **A DiT LoRA does not bind to names.** The channel LoRA (`caststation_ema097_lr1e4`)
   moves the velocity by 5.5 % on its own captions and 6.2 % on another artist's. Its
   effect keeps cos 0.74 / 0.83 (σ 0.3 / 0.7) with an *empty* prompt. Artist and
   character tokens each carry 2.2 % of its gradient × input. It learned a
   prompt-independent style shift. A character learned that way needs images and competes
   with the style for the same rank. Neru appears in 1 of the 30 channel images.

That gives a clean split: **style stays in DiT LoRAs, character identity goes into the
TE**. The two compose without retraining each other.

## Teacher: what "detailed" means for one character

For a character tag `c`, build an appearance block `A(c)`:

- **Co-occurrence tags.** Danbooru `related_tag` for `c` (general category). Keep
  appearance categories by the taxonomy KB (`models/danbooru_tags_classified.csv`: hair,
  eyes, clothing, accessories, held items). Drop tags below a frequency share
  (alternate outfits). This is the visual consensus.
- **Wiki.** The 2026 dump (`kierarkia/danbooru-wiki-2026`, HF cache, 202k rows). The TE
  condenses the body into appearance clauses with the body in context; it does not need
  to know the character itself. Coverage varies: Miku's page describes her outfit item by
  item, while Neru's gives only "image colour orange-ish yellow" and "cell phone". So the
  wiki supplements the tags and cannot replace them.
- **PE mode, used in reverse.** `project/pe_lora` puts the Qwen-Image-2.1 prompt enhancer
  on the shipped INT8 TE as switchable LoRA branches. With the wiki and tags in its
  context, it writes the appearance clause in the register MOSS was trained on.

Teacher caption: the layout-JSON caption with `A(c)` merged into the summary tag bag and
into that character's `items` entry, so position clauses keep binding.

## Approaches, cheapest first

**A. Expansion at caption time (baseline, no training).** A `name → A(c)` table applied
by an anime_tools caption stage (and at generate time). Every learned approach has to
beat this. It wins on tag prompts. It fails on prompts that do not go through the stage
(ComfyUI, natural language), and it spends 20–40 tokens per character.

**B. Output patch (distilled textual inversion).** Learn a small delta on the TE output
rows at the name's tokens (optionally plus k appended soft rows), per character. Loss is
**DiT-output distillation**:

```
L = E_{x_t, σ} ‖ v(x_t, σ, E(short) + patch) − v(x_t, σ, E(teacher)) ‖²
```

`x_t` comes from teacher generations, and σ is drawn from the training distribution.
Only the DiT is backpropagated, to the rows: `lora_use.py` already runs exactly this
gradient path on 16 GB. The cost is minutes per character and a few KB per patch. Being
context-free is acceptable because the name occurs where the character is meant. The open
question is **compression**: can 2–3 rows carry what 20–40 teacher tokens did?

**C. Knowledge LoRA on the TE (the PE carrier, used the other way round).** One TE-mode
LoRA for N characters, carried by the same switchable-branch mechanism as PE mode (INT8
TE, branches off = the shipped TE bit for bit). Two losses:

- **Read-out matching (length-invariant).** The DiT consumes text only through joint
  attention. At a few blocks, take image-token queries from teacher runs and match the
  attention read-out over text keys and values, student against teacher. Sequence-length
  differences drop out, and no full DiT backward is needed per step. `L_dit` (as in B)
  stays as the final check.
- **Preservation.** On a calibration set without the trained names (and on names the TE
  already knows), TE output MSE and DiT velocity change stay at zero.

**D. Make the TE *know* the character (the ambitious one).** LM-loss fine-tuning of the
TE on wiki text plus appearance QA, so that Neru becomes a Miku-like entity inside Qwen.
The DiT representation follows only if the DiT reads appearance out of entity knowledge.
It might instead have memorised name-token identity from MOSS pretraining. Gate D on a
probe first: across ~500 characters, does a linear probe at the name token predict hair
and eye colour, and does that accuracy track the cloze knowledge score? If the DiT for
Miku runs on token identity, D cannot work and C is the line.

## Metrics

- **Tagger recall.** Run the Anima Tagger on generations and compute recall of `A(c)`'s
  tags, plus the tagger's own score for `c`. Fixed seeds; A / B / C against the teacher
  and the bare name.
- **DiT distance.** ‖v_student − v_teacher‖ at fixed `x_t`, no decode. Cheap enough per
  step.
- **Preservation.** Velocity change on unrelated prompts, and Miku / Teto unchanged.
- **Binding.** Multi-character captions (the Triple Baka page) with per-character
  attributes in the right position.

## Phases and gates

0. **Teacher ceiling.** Neru, Teto (half known), Miku (control) and three under-known
   characters from our sets. Base MOSS, same seeds: name / + tags / + tags + wiki.
   *Gate:* the teacher clearly raises tagger recall over the bare name. If it does not,
   the DiT cannot draw the character from a description either, and only a DiT LoRA is
   left.
1. **Baseline A** measured on the same seeds.
2. **B on Neru.** *Gate:* the patch recovers ≥ 80 % of the teacher's recall gain. Below
   that, try soft rows. If those fail too, compression is the blocker and C's read-out
   loss has to carry it.
3. **C on ~50 characters** chosen from the KB as post_count ≥ 500 and failing the cloze
   test, with preservation.
4. **D probe**, then D if the probe says entity knowledge drives the DiT.

## Costs and risks

- TE caches must stamp the knowledge LoRA's digest (as they stamp the CJK vocab pack).
  A switch is otherwise invisible to `_te_current`.
- `related_tag` is an external read-only API: cache its answers into the KB once.
- Teacher tags from alternate outfits or genderswaps blur `A(c)`. Use the frequency-share
  floor and drop tags that come with a `(cosplay)` or outfit-variant tag.
- B and C are trained on INT8 TE outputs. The bf16 TE differs by ~2 %, so stamp the
  precision.
