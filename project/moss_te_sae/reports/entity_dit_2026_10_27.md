# Where the DiT expands a name

`src/entity_dit.py`, run 20261027-2108.

**Result.** The identity sits in the name's own TE rows (splicing them keeps 0.96–0.98 of the effect). The DiT writes hair colour onto the name positions inside its blocks (peak block 30), and the image reads it in blocks ~16–31.

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
