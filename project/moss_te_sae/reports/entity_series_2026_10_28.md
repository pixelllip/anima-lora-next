# Series directions in the name vectors

`src/entity_series.py`, run 20261028-1309.

**Result.** Series is laid out linearly over names (LOO accuracy 0.65 at block 14, null p95 0.11). The Vocaloid centroid injected on an invented name carries the outfit grammar (series prob 0.84) more strongly than writing `vocaloid`. Touhou transfers partly; Hololive and Bocchi do not.

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
