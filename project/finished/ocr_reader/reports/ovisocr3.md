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
a core dependency since 2026-10-29.

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

## LoRA micro arm (`ovis3_lora_micro`, 2026-10-28)

`ocr/finetune_ovis3_lora.py`: P3's data (81 002 rows: COO SFX + speech + the
gelnote KO / ZH pseudo rows), the card's `plain_ocr` turn as the prompt, and
loss on the target only. LoRA goes on every vision-tower + merger Linear at
r 64, and on the LM layers' GDN / attention / MLP projections at r 16 (20.3 M
trainable). lr 1e-4, bs 16, 1 000 steps = 16 k crops ≈ 0.2 epoch, 21.4 min,
12.5 crops/s, peak 10.4 GB. Final loss 0.386. One seed.

```
ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/finetune_ovis3_lora.py --run ovis3_lora_micro --steps 1000 --bs 16 --grad_accum 1"
```

| gate | OvisOCR3 stock | **LoRA, 0.2 ep** | v4 (VL P3, 2 ep) |
|---|---|---|---|
| sincos SFX strict / ♡-blind (n = 597) | 9 / — | **166 / 198** | 346 / 387 |
| sincos speech strict (n = 272) | 41 | 92 | 123 |
| COO SFX (n = 2 558) | 224 (8.8 %) | **1 850 (72.3 %)** | 2 223 (86.9 %) |
| COO speech (n = 2 559) | 1 171 (45.8 %) | 2 189 (85.5 %) | 2 273 (88.8 %) |
| COO runaway SFX / speech | 42 / 12 | 36 / 13 | 19 / 9 |
| K3 KO / ZH exact (n = 48) | 22 / 27 | 23 / 29 | 26 / 30 |
| K3 kana leak KO / ZH | 0 / 0 | 2 / 4 | 3 / 0 |

- **A tenth of v4's compute closes most of the COO gap.** COO SFX goes from
  8.8 % to 72.3 % (v4 86.9 %) and COO speech from 45.8 % to 85.5 % (v4 88.8 %).
  The sincos misses are now near-kana, not digits (`びく♡` → `びくん`,
  `イグッ` → `ズグッ`), and the `♥` target spelling is learnt.
- **Sincos is half of v4** (166 vs 346 strict). For scale: the finished line's
  arm B (VL, LM LoRA, tower frozen, 2 epochs) read 110 strict / 157 ♡-blind,
  though that was on the older 617 basis.
- **K3 does not move**, and the Japanese-heavy mix starts a kana leak on
  both languages.
- Every VL number above is 1–2 epochs; the arm below matches the budget.

## VL arm at the same budget (`vl16_p3_micro`, 2026-10-29)

The P3 recipe (tower r64 LoRA + LM full FT, lr 1e-4 / 1e-5, same mix) cut to
1 000 steps of 8 × 2 = 16 k crops by `finetune_vl16_lora.py --steps 1000`
(LR decays over the 1 000 steps, as in the Ovis arm). 25.6 min, 12.0 crops/s,
peak 12.1 GB, 290.6 M trainable. One seed.

```
ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="--stall-timeout 900 \
  project/finished/ocr_reader/ocr/finetune_vl16_lora.py --run vl16_p3_micro --steps 1000 --bs 8 --grad_accum 2 \
  --tower_rank 64 --lm_full --lm_lr 1e-5 --skip_stock_val \
  --extra_manifest pseudo_gelnote_ko --extra_manifest pseudo_gelnote_zh --speech_ratio 0.909525"
```

| gate | Ovis3 LoRA, 0.2 ep | **VL P3, 0.2 ep** | paired VL-only / Ovis-only (z) | v4 (2 ep) |
|---|---|---|---|---|
| sincos SFX strict / ♡-blind (n = 597) | 166 / 198 | **233 / 279** | 108 / 41 (+5.5); ♡-blind 123 / 42 (+6.3) | 346 / 387 |
| sincos speech strict (n = 272) | 92 | 101 | 30 / 21 (+1.3) | 123 |
| COO SFX (n = 2 558) | 1 850 (72.3 %) | **1 924 (75.2 %)** | 258 / 184 (+3.5) | 2 223 (86.9 %) |
| COO speech (n = 2 559) | 2 189 (85.5 %) | **2 237 (87.4 %)** | 143 / 95 (+3.1) | 2 273 (88.8 %) |
| COO runaway SFX / speech | 36 / 13 | 20 / 16 | | 19 / 9 |
| K3 KO / ZH exact (n = 48) | 23 / 29 | 25 / 33 | 5 / 3, 9 / 5 | 26 / 30 |
| K3 kana leak KO / ZH | 2 / 4 | 9 / 4 | | 3 / 0 |

- **OvisOCR3 does not learn faster.** At the same 16 k crops VL wins every JA
  gate. The gap is largest on hand-lettered SFX (sincos +67 strict, z +5.5).
  The Ovis arm's COO jump is not an Ovis-specific speed: the VL arm gets
  there too and goes a little further.
- **Not a clean architecture A/B.** Each arm uses its own line's recipe: VL
  trains the LM in full (290.6 M), Ovis gets LoRA (20.3 M). Full-parameter
  Ovis fits at bs 4 (10.5 GB, kernel bench above) and is the one confound
  left. Losses are not compared across the two tokenizers.
- **K3 is noise at n = 48.** ZH 33 is above v4 (30) and stock (31); trad
  6 / 12. KO kana leak 9 is the worst of any P3-mix arm.

Verdict: OvisOCR3 has no edge as a fine-tune base at this budget either. The
line stays closed.
