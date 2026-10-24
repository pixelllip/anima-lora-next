# MOSS i2i (image to image)

An i2i LoRA learns a reference → target edit on MOSS-Image. The reference image is
VAE-encoded and goes into every forward (both CFG branches) as clean latent tokens on RoPE's
reserved `r = 1` axis, after the image and text rows. The flags below control it:

| Flag | Meaning | Stamped in the LoRA |
|---|---|---|
| `ref_scale` | reference side = target side / `ref_scale` (1 = full size, 2 = ¼ area) | checked against the cache |
| `ref_timestep` | the ref rows are modulated at this timestep in every block (0 = clean) instead of the step's | yes |
| `ref_isolate` | ref queries attend only to ref keys; image and text still read the ref | yes |
| `ref_vision` | the page also enters the text encoder through Qwen3.5's vision tower, ahead of the caption | no (cache recipe) |

`generate.py` and the ComfyUI Reference node read `ref_timestep` / `ref_isolate` from the
LoRA. `ref_scale`, `ref_vision` and the prompt format must match the cache the LoRA trained on.
With `ref_isolate` and a `ref_timestep`, the ref rows depend only on the page, so ComfyUI runs
them through the blocks once per page and caches their K/V (`ref_cache`).

## Tasks

Recipes: `configs/moss/i2i/<task>/{cache,train,generate}.toml`. The task → cache-request map
is `library.moss.requests.I2I_CACHE`.

| Task | Reference | Text encoder input | Status |
|---|---|---|---|
| `colorize` | B&W page, half size, joint | the page through the vision tower, then the colour caption's layout JSON plus one text item per OCR line | v2 shipped (`moss_colorize_3k_ocr`). v3 (full size, `ref_isolate`, text-only prompt) planned in `project/moss_lora/plan.md` |
| `sanitize` | texted image, full size, isolated | one fixed caption for every pair: `no sfx, no dialogue, no watermark` | trained, private on the Hub |
| `swap_text` | JA manga page, full size, isolated | layout JSON, one text item per line to letter (translated text + bbox) | probe run on 100 pairs (`moss_i2i_swap_text_muon_lr1e4`): English near-clean, Korean not yet |

## Running

```bash
make i2i-stage TASK=<task>                      # pair tree under post_image_dataset/i2i/<task>/
make i2i-preprocess TASK=<task> MODEL=moss      # cache (colorize: scripts/moss/colorize_cache.py; sanitize, swap_text: cache.py)
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/<task>/train.toml"
make daemon-run ARGS="scripts/moss/generate.py --config configs/moss/i2i/<task>/generate.toml \
    --lora output/ckpt_moss/moss_i2i_<task>.safetensors --ref_image <page or folder>"
```

`--ref_image` takes one image or a folder (one job per image and prompt). The output keeps
the reference's aspect ratio, with the area capped at `width × height`. The GUI's **i2i Preprocess**
(stage, preprocess) and **i2i Train** (train, test) tabs (`make gui-moss`, or MOSS mode in
`make gui`) run the same four steps. Staging is shared with
EasyControl: `i2i_tasks/README.md`.

## sanitize

Removes text, SFX, speech-bubble text and watermarks and repaints what was underneath.

**Data.** `make i2i-stage TASK=sanitize` mines in-artist near-twins
(`i2i_tasks/tools/near_twins/`, knobs in the `[staging]` table of
`i2i_tasks/configs/sanitize.toml`): a texted image (`{id}_tags`, the condition) and its
clean twin (`{id}_no_tags`, the target), listed in `staging/pairs.jsonl`. `add_identity_pairs`
adds some clean → clean pairs.

**Cache.** `caption_format = "fixed"` encodes `fixed_caption` for every pair instead of the
pair's own `.txt`. The reference carries the edit, so the caption is just a constant. Changing
it re-encodes only the TE files. The reference is full size (`ref_scale = 1`). Copies of
`fixed_caption` live in `generate.toml` (as `prompt`, with `prompt_format = "text"`) and in
the ComfyUI MOSS Toolkit (`FIXED_CAPTIONS["sanitize"]`); change all three together.

**Train.** `ref_timestep = 0`, `ref_isolate = true`, other fields at the `TrainRequest`
defaults. The first run (`moss_i2i_sanitize`) used 49 pairs from 17 artist folders, rank 32,
12 epochs = 588 steps, 37 min on 16 GB (12.6 GB peak reserved, swap 28).

**Result.** On four test images from outside the staging tree (not checked against the
training pairs), the first run removes title text, handwritten
annotations and their arrows, unbubbled dialogue, a large stamped "reproduction prohibited"
watermark and corner signature watermarks. It repaints the background, clothing and legs
underneath, and leaves glyph-like art in place (a halo's red symbols). Line work, faces and
colour come through unchanged: outside the text regions, the output is ~31 dB PSNR against
the input. Small doodle hearts next to annotations go with the text. The prompt barely
matters: the same page with the layout JSON of `no sfx, no text` (not the training caption)
came out nearly identical. Still, use the fixed caption, since it is what the LoRA saw.

**ComfyUI.** [ComfyUI-MOSS-Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) 0.7.0
or later (0.6.x: Reference `task custom`, `ref_scale 1`), `example_workflows/moss-i2i-sanitize.json`:

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA, task auto) ─────────────── MODEL ─┐
                  ─ CLIP ─┬ CLIP Text Encode (Caption Helper, task sanitize) ── positive ┤
                          └ CLIP Text Encode ("") ──────────────────────────── negative ┤ MOSS Spectrum KSampler
                  ─ VAE ──── MOSS i2i Reference ────────────────────────────── LATENT ──┘ (30 steps, CFG 4, skip 22)
Load Image → Reference.image
```

The Caption Helper needs no image. It registers only when `anime-tools` is installed in
ComfyUI's environment. LoRA + workflow: `sorryhyun/MOSS-I2I-LORAs` on the Hub, `sanitize/`
(private for now; `colorize/` will go there too).

## swap_text

Re-letters a Japanese manga page in a translated edition's text, inside the balloons the
prompt names.

**Data.** `make i2i-stage TASK=swap_text` (`i2i_tasks/swap_text/prep.py`, knobs in
`i2i_tasks/configs/swap_text.toml`) registers hitocrawl edition twins (the JA page and
its `en` / `ko` edition). Each target is the JA page with clean text groups pasted from
the twin. `full` pairs paste every group; `partial` pairs (`partial_ratio`, 0.5) paste a
proper subset, and the rest stay Japanese. The caption is layout JSON with one text item
per pasted line, `abs_centralized` `yxyx` pixels of the staged page, read off the twin
(`--caption_ocr` reads the staged target instead). The current tree: 7,744 pairs from 254
works (en 5,990, ko 1,754; 3,887 partial), with 2,374 lines the OCR misread.

**Cache.** `caption_format = "layout_json"` passes a caption that already is layout JSON
through as written (`library/moss/cache.py::is_layout_json`), so no `embeds_plain` is
stored and `plain_caption_rate` has no effect. The reference is full size
(`ref_scale = 1`).

**Train.** `ref_timestep = 0`, `ref_isolate = true`. The first run
(`moss_i2i_swap_text_muon_lr1e4`) used `pairs_100.jsonl`: 100 pairs, one page per work
(en 60 / ko 40, full 54 / partial 46). It trained rank 32, Muon at lr 1e-4,
`dc_isolate` off, 6 epochs = 600 steps, `blocks_to_swap = 28`, at ~4 s/step (40 min) and
12.2 GB peak on 16 GB, with joint sequences up to 8.7k tokens. The global loss does not track the
lettering. A median 3 % of latent tokens differ between target and reference (p90 7 %), so
once the copy is learned the loss falls toward 0 (0.25 in epoch 1, 0.03–0.04 from epoch 3)
and what remains is the text regions.

**Generate.** `generate.toml`, with the page's own staged caption (or a hand-written one) as
`--prompt` and `--width` / `--height` set to the page size, so the bboxes land where they
were measured.

**Result.** Final checkpoint, seed 0, on three held-out pages from works outside the 100
(a fourth, `ko` partial, was generated but not reviewed):

- English (full and partial): every line lands in its bbox, legible, broken to the balloon
  shape, in the page's lettering style; the Japanese underneath is gone and the balloon
  art kept. Misses: one wrong glyph (`MOVE` → `MOWE`); a short exclamation drawn with its
  own outline lost the outline along with the Japanese; `♡` marks absent from the OCR
  caption are not drawn.
- Korean (full, 3 lines): one exact; one with a syllable dropped and others swapped
  (`처음 오시는 분인가요?` → `처음 시는 문이니요?`); one drew the caption's OCR misread
  (`벌거` for `씻겨`) faithfully, in a typeset font and colour unlike the page's hand
  lettering. 40 Korean pages leave most Hangul syllables seen once or never.
- Outside the text, 22–25 dB PSNR against the target. The difference is the screentone's
  dot phase (the VAE round trip), not drift.

Next: a larger cache weighted toward `ko`, with misread lines dropped from the captions.

**ComfyUI.** Toolkit 0.7.0 or later: the sanitize wiring, with the layout-JSON prompt as
the positive (typed into `CLIP Text Encode`, or through Caption Helper `task layout`, which
passes layout JSON through unchanged).

## More

- Probes and design notes: `docs/proposal/moss_i2i.md`. Colorize runs: `project/moss_lora/`
  (`plan.md`, `colorize_ocr.md`).
- Ref forward: `library/moss/model.py` (`ref=`, `_ref_only`). The ComfyUI copy is
  `reference.py` in the toolkit.
