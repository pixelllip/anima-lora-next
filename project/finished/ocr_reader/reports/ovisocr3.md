# OvisOCR3 — the successor on the crop gates (2026-10-28)

[`ATH-MaaS/OvisOCR3`](https://huggingface.co/ATH-MaaS/OvisOCR3) (commit
`4833d5d`, released 2026-10-21; Qwen3.5-0.8B base, Apache-2.0). It has the same
`qwen3_5` body as [OvisOCR2](ovisocr2.md) without the MTP layer. It adds
general-OCR task prompts (`prompts/plain_ocr.txt`, `word_spotting`, …) next to
the page parse; its card reports OCRBench 88.8 and OCRBench v2 ZH 61.3. Reader:
`eval_manga109.Ovis3Reader` (`--reader ovis3`), weights in `models/ovis_ocr3`.

The recipe is the card's: `plain_ocr` prompt (stripped), thinking off, greedy,
448² / 2880² pixels. The repo now ships a `generation_config.json`, and its
eos pair is the one OvisOCR2 needed patched in. The eval scripts need `pandas`,
which is now only in the `sr` group: `uv pip install pandas` into the venv.

## Crop mode

| gate | stock VL-1.6 | v4 (P3 ep2) | OvisOCR2 | OvisOCR3 | OvisOCR3 vs stock |
|---|---|---|---|---|---|
| K3 KO exact (n = 48) | 21 | **26** | 12 | 22 | 5 / 4, z +0.3 |
| K3 ZH exact (n = 48) | **31** | 30 | 32 | 27 | 5 / 9, z −1.1 |
| sincos SFX strict (n = 597) | — | **346** | 2 | 9 | — |
| sincos speech strict (n = 272) | — | **123** | 35 | 41 | — |

23.9 crops/s on sincos (OvisOCR2 10.5); SFX runaways 40 → 3.

- **KO is fixed, not won.** 12 → 22. That is stock level, with no kana leak and
  one runaway, but still 4 below v4. Clean balloon Hangul reads right. Stylised
  Hangul still goes to Han (`조로` → `圣君`, `샤코` → `灶台`) or to digits
  (`어머니—!!!` → `0101`).
- **ZH loses traditional script.** trad 3 / 12 (OvisOCR2 7 / 12). The prompt
  says "no correction", yet the model rewrites traditional to simplified:
  `臉頰` → `脸颊`, `新年快樂` → `新年快乐`, `我還是覺得…` → `我还是觉得…`.
  Multi-column vertical trad comes back in scrambled column order. Hearts are
  dropped (`早安，我的女主角♡` → no ♡).
- **Hand-lettered SFX is still unread: 9 / 597.** The runaways are gone, but
  kana bursts still come back as digits, Latin, or other kana
  (`ぱちゅ♡` → `1854`, `ズプ` → `70`, `ぴく♡` → `W<Y`). The 9 hits are
  printed-looking short SFX (`おっ`, `ガタ`, `ぱん`).

Verdict: not a candidate for the crop reader. Page mode was not run, because
OvisOCR2's page read already put SFX recall at 3 / 597.

## Page detection (`word_spotting`) — stopped

`ocr/eval_page_det.py` scores boxed page output on 200 Manga109-s COO test
pages, which are exhaustively annotated (speech `<text>` + COO SFX). The
AnimeText YOLO baseline: speech found 97.7 %, SFX found 82.3 %, precision
87.5 %, 0.1 min. Its training set may include Manga109.

On an 8-page smoke, OvisOCR3 took 2.4 min (~18 s / page, bs 8). Half the
pages ran away, repeating one box at the page bottom until the token cap.
It found 44 % of speech and 20 % of SFX (cover ≥ 0.5), at precision
74.5 %. The full run was cancelled as too slow to be a detector.

## Kernels — `fla` on, `causal_conv1d` A/B

`fla` 0.5.2 has been a dependency since 2026-10-21 (`b903df03`), so the GDN
layers ran on `fla.ops.gated_delta_rule` for every run above. OvisOCR2's "no
fla" note predates it. `causal-conv1d` was not one: PyPI ships only
the sdist. 1.7.0 built against torch 2.14 / CUDA 13.2 in 8 min 17 s, nine
gencode archs with no knob to narrow them (sm_120 alone is about 1/9 of
that). `ocr/bench_ovis_kernels.py`, one process per arm; `--no-cc1d` hides
the package, so transformers falls back to `F.conv1d`:

| load | torch conv | `causal_conv1d` | gain |
|---|---|---|---|
| 908 sincos crops, `plain_ocr`, bs 32 | 31.2 s (29.1 / s) | 27.0 s (33.6 / s) | +15 % |
| 16 pages, `word_spotting`, bs 8 (~67 out tok / page, prefill-heavy) | 8.3 s | 7.5 s | +11 % |
| train step, bs 4, fwd + bwd, full params | 197 ms, 10.5 GB | 190 ms, 10.2 GB | +4 % |

The bs 16 train step ran out of memory on the 16 GB card in both arms. The
outputs are not bit-identical: 828 / 908 crop reads and 3 / 16 page reads
match, which is bf16 drift in greedy decode. Decode gains the most, because
the torch path runs a `cat` + `conv1d` per GDN layer per token. Training
barely moves.

The build is now a dependency. The wheel (version re-tagged
`1.7.0+cu132torch2.14`) is the pre-release asset `deps-causal-conv1d-1.7.0`
on `sorryhyun/anima_lora`, pinned in the `torch-214` group (Linux x86_64).
