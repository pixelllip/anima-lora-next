# Phase 2 B on `malfoid`

`src/entity_patch.py`, run 20261027-2128; output `output/tests/malfoid_patch/`.

**Result.** The gate fails: both arms recover 24 % of the teacher's recall gain against an 80 % gate. By eye `split` init is ahead. The learning rate confounds the arms.

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
