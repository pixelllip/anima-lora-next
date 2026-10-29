# Does the TE know akita neru

`src/entity.py`, run 20261027-1615, bf16 instruct TE.

**Result.** No. QA, cloze and SAE overlap all place her with an invented name (`okuda reni`).

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
