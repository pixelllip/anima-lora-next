# Phase 0: teacher ceiling

Proposal: `docs/moss/proposal/te_character_knowledge.md`. Scoring: `src/tag_recall.py`.

**Result.** The DiT draws Neru from her name although the TE does not know her. `malfoid` and `toujou koneko` are unknown to the DiT, and spelled-out tags lift recall 0.29 → 0.83 and 0.06 → 0.83. The targets are characters the DiT cannot draw, not characters the TE cannot describe.

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
