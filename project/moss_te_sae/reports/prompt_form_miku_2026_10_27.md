# Prompt form on a DiT-known character (Miku)

Run 20261027-2200; output `output/tests/miku_promptform/`.

**Result.** The prompt form does not move a DiT-known character. The spread across seeds is larger than across forms, and a question prefix scores lowest.

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
