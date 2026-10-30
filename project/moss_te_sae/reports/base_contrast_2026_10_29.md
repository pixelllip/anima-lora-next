# Base vs instruct × raw vs template

`src/base_contrast.py`, run 20261029-2123; results `output/tests/base_contrast/20261029-2123/summary.json`. [hypothesis.md](../hypothesis.md) test 3.

**Result.** The template's effect on the caption rows is an interaction. The template alone moves Base's layout rows 0.10, the instruct weights alone 0.24, both together 0.60 (L31). Instruct holds the template's state over the whole caption; Base lets it fade.

The probe's 230 captions (150 layout-JSON, 80 NL) through
Qwen3.5-9B-Base and the instruct TE (bf16), each raw and in the shipped template. Caption
rows compared position by position (body ids match in 230 / 230); the first 4 body tokens
dropped (the raw sink). Mean per-token `‖b − a‖ / ‖a‖`, layout / NL:

| edge | L15 | L31 | `out` | `proj` |
|---|---|---|---|---|
| framing, Base (raw → tmpl) | 0.10 / 0.21 | 0.10 / 0.25 | 0.11 / 0.24 | 0.12 / 0.27 |
| framing, instruct | 0.27 / 0.24 | **0.57 / 0.40** | 0.56 / 0.37 | 0.58 / 0.40 |
| weights, raw (Base → inst) | 0.19 / 0.17 | 0.24 / 0.24 | 0.23 / 0.21 | 0.25 / 0.24 |
| weights, tmpl | 0.32 / 0.25 | 0.60 / 0.44 | 0.61 / 0.42 | 0.67 / 0.48 |
| total (Base raw → shipped) | 0.36 / 0.33 | 0.60 / 0.49 | 0.61 / 0.46 | 0.68 / 0.54 |

- **The shift is an interaction.** The template barely moves Base's layout rows (0.10),
  the weights alone move them 0.24; the two together move them 0.60. Shipped is as far
  from instruct-raw as from Base-raw. Centred cosine on layout at L31: 0.99 / 0.76 / 0.96 /
  0.72 (framing Base / framing inst / weights raw / weights tmpl).
- **It grows along the caption in instruct, fades in Base.** Layout `out`, body position
  0–16 / 16–64 / 64+: framing in Base 0.19 / 0.11 / 0.09, framing in instruct 0.37 / 0.54 /
  0.59. The template sets a state that the instruct model holds over the whole caption.
- **It is mostly a late-layer effect, larger on layout-JSON than NL** (L15 0.27 → L31 0.57
  on layout; Base shows the opposite order, NL > layout).
- H1's kill test (framing change within ~2× the 2.1 % INT8 gap) does not fire: the
  instruct framing change is ~26× the gap at `out`. This shows the framing matters to the
  rows, not that the rows encode a request; that is test 1.
- **The SAE barely sees it.** Base-SAE FVU, layout / NL at L31: Base raw 0.30 / 0.20, Base
  tmpl 0.30 / 0.21, inst raw 0.31 / 0.23, inst tmpl 0.34 / 0.27 (L15: 0.41–0.47). Top-100
  code overlap on layout at L31 drops to 0.57 under instruct framing (Base framing 0.92), so
  the features change but stay reconstructable. Run 20261027-1609's L15 gap (raw 0.24 vs
  body 0.41) does not reproduce with the head dropped (0.39 vs 0.43); it was most likely the
  raw sink token inflating the variance.
