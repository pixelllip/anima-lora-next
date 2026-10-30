# Tag directions added onto the name rows, no training

`src/entity_add.py`, run 20261027-2120.

**Result.** For DiT-unknown characters, tag directions on the name rows recover a quarter to a third of the teacher effect (random ~0.05). Only σ 0.9 separates them. The text path reads the image heavily, so distil with the full DiT.

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
