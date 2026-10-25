#!/usr/bin/env python3
"""Synthetic faint-watermark pairs → an existing ``_tags`` / ``_no_tags`` staging tree.

The case the mined sanitize pairs miss: a URL watermark (``https://www.patreon.com/…``,
``….fanbox.cc/``) tiled over the whole page at a few percent opacity, barely visible
on flat colour (``output/moss_results/usual_case2.png``). Each pair here reuses a
staged pair: its clean ``_no_tags`` target stays the target, and the condition is
that target — or, ``--stack`` of the time, the pair's texted ``_tags`` side — under a
synthetic overlay. Both sides are decoded once and written as lossless WebP, so on
an unstacked pair ``cond − target`` is the overlay and nothing else.

The sources are the tree's hand-curated targets because corpus tags do not see
lettering: a tag-filtered corpus image still carries SFX or a title often enough,
and a pair that keeps text in its target teaches the sanitize adapter to keep text.

Per pair:
  * **source**: a ``pairs.jsonl`` row of ``--staging`` (not a ``wm_*`` one),
    downscaled to ``--max_side`` (INTER_AREA) when larger; the condition side is
    resized to the target's size.
  * **text**: 1–3 strings built from one handle (the artist name or a random one):
    patreon / fanbox / x / pixiv / fantia / skeb … URLs, ``@handle``, ``© Name``, and
    now and then a repost notice (CJK when a CJK font is found).
  * **layout**: ``tile`` (staggered rows over the page, alternating the strings —
    the reference case), ``scatter`` (a few stamps) or ``single`` (one stamp, centre
    or corner); horizontal or rotated.
  * **ink**: mostly white (else light grey / black / a tint) at opacity log-uniform in
    ``--opacity``, optional contrasting outline, optional sub-pixel blur. A colour
    that would vanish on the page (mean shift under ``--min_delta`` levels on the
    glyphs) is re-drawn.

Pairs land as ``<staging>/<artist>/wm_<stem>_{tags,no_tags}.webp`` + the source
pair's caption on both sides; a re-run replaces every ``wm_*`` pair (the draw is ``--seed``-fixed) and
rewrites ``pairs.jsonl`` from the tree. ``--preview DIR`` writes the conditions
there instead and leaves the tree alone.

    make i2i-stage TASK=watermark                     # into the sanitize tree
    python -m i2i_tasks.tools.watermark_pairs --n 8 --preview /tmp/wm
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from i2i_tasks.tools.near_twins.outputs import PAIR_MANIFEST, write_pair_manifest
from library.env import resolve_under_home

Image.MAX_IMAGE_PIXELS = None  # registered textless pages run past PIL's bomb limit

PREFIX = "wm_"
WEBP_LOSSLESS = [cv2.IMWRITE_WEBP_QUALITY, 101]

_SITES = (
    "https://www.patreon.com/{h}", "patreon.com/{h}", "https://{h}.fanbox.cc/",
    "{h}.fanbox.cc", "twitter.com/{h}", "https://x.com/{h}", "@{h}",
    "pixiv.net/users/{n}", "pixiv ID: {n}", "fantia.jp/fanclubs/{n}", "{h}.gumroad.com",
    "subscribestar.adult/{h}", "ci-en.dlsite.com/creator/{n}", "skeb.jp/@{h}",
    "instagram.com/{h}", "{h}.booth.pm", "© {H}", "{H}",
)  # fmt: skip
_NOTICES_LATIN = ("Do not repost", "DO NOT REPOST", "SAMPLE", "sample", "preview")
_NOTICES_CJK = (
    "無断転載禁止",
    "無断転載・AI学習禁止",
    "サンプル",
    "禁止转载",
    "무단 전재 금지",
)

# Font files by lowercase name: Linux (DejaVu / Liberation / Noto), Windows, macOS.
_LATIN_FONTS = {
    "dejavusans.ttf", "dejavusans-bold.ttf", "dejavusanscondensed.ttf",
    "liberationsans-regular.ttf", "liberationsans-bold.ttf", "notosans-regular.ttf",
    "notosans-light.ttf", "notosans-bold.ttf", "notosansmono-regular.ttf",
    "roboto-regular.ttf", "roboto-light.ttf", "arial.ttf", "arialbd.ttf", "segoeui.ttf",
    "segoeuil.ttf", "verdana.ttf", "tahoma.ttf", "calibri.ttf", "calibril.ttf",
    "helvetica.ttc", "arial unicode.ttf",
}  # fmt: skip
_CJK_FONTS = {
    "notosanscjk-regular.ttc", "notosanscjk-bold.ttc", "notosanscjk-light.ttc",
    "notoserifcjk-regular.ttc", "msgothic.ttc", "yugothr.ttc", "yugothm.ttc", "meiryo.ttc",
    "malgun.ttf", "msyh.ttc", "hiragino sans gb.ttc",
}  # fmt: skip
_FONT_DIRS = (
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.local/share/fonts", "~/.fonts",
    "C:/Windows/Fonts", "/System/Library/Fonts", "/Library/Fonts",
)  # fmt: skip


def find_fonts(dirs) -> tuple[list[str], list[str]]:
    """``(latin, cjk)`` font paths under ``dirs``, by the allow-lists above."""
    latin, cjk = [], []
    for d in dirs:
        root = Path(d).expanduser()
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            name = f.name.lower()
            if name in _LATIN_FONTS:
                latin.append(str(f))
            elif name in _CJK_FONTS:
                cjk.append(str(f))
    return sorted(set(latin)), sorted(set(cjk))


def _handle(artist: str, rng: random.Random) -> tuple[str, str]:
    """``(url handle, display name)`` — the artist's own name half the time."""
    name = re.sub(r"_\(.*?\)$", "", artist)
    h = re.sub(r"[^a-z0-9]", "", name.lower())
    if rng.random() < 0.5 and len(h) >= 3:
        return h, name.replace("_", " ").title()
    syl = [
        rng.choice("bdfghkmnprstwyz") + rng.choice("aeiou")
        for _ in range(rng.randint(2, 5))
    ]
    h = "".join(syl) + (str(rng.randint(1, 999)) if rng.random() < 0.3 else "")
    return h, h.capitalize()


def watermark_texts(artist: str, rng: random.Random, cjk: bool) -> list[str]:
    h, name = _handle(artist, rng)
    n = str(rng.randint(10_000, 99_999_999))
    k = rng.choices((1, 2, 3), weights=(0.4, 0.45, 0.15))[0]
    texts = [s.format(h=h, H=name, n=n) for s in rng.sample(_SITES, k)]
    if rng.random() < 0.15:
        texts.append(
            rng.choice(_NOTICES_CJK if cjk and rng.random() < 0.6 else _NOTICES_LATIN)
        )
    return texts


def _stamp(text: str, font, stroke: int, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """``(fill, fill ∪ outline)`` coverage in [0, 1] of one rotated text instance."""
    x0, y0, x1, y1 = font.getbbox(text, stroke_width=stroke)
    pad = stroke + 2
    size = (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad)
    out = []
    for sw in (0, stroke):
        im = Image.new("L", size, 0)
        ImageDraw.Draw(im).text(
            (pad - x0, pad - y0),
            text,
            font=font,
            fill=255,
            stroke_width=sw,
            stroke_fill=255,
        )
        out.append(
            np.asarray(im.rotate(angle, Image.BICUBIC, expand=True), np.float32) / 255
        )
    return out[0], out[1]


def _paste(canvas: np.ndarray, patch: np.ndarray, cx: float, cy: float) -> None:
    """``max``-composite ``patch`` centred at ``(cx, cy)``, clipped to the canvas."""
    ph, pw = patch.shape
    x, y = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    H, W = canvas.shape
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + pw), min(H, y + ph)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    region = canvas[dy0:dy1, dx0:dx1]
    np.maximum(region, patch[sy0 : sy0 + dy1 - dy0, sx0 : sx0 + dx1 - dx0], out=region)


def render_overlay(
    w: int, h: int, texts: list[str], fonts: dict[str, list[str]], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """``(fill, outline)`` coverage maps ``(h, w)`` in [0, 1] for one overlay."""
    layout = rng.choices(("tile", "scatter", "single"), weights=(0.6, 0.2, 0.2))[0]
    short = min(w, h)
    frac = {"tile": (0.018, 0.045), "scatter": (0.03, 0.07), "single": (0.04, 0.10)}[
        layout
    ]
    px = max(10, round(short * rng.uniform(*frac)))
    angle = (
        0.0
        if rng.random() < (0.5 if layout == "tile" else 0.6)
        else rng.uniform(-40, 40)
    )
    stroke = max(1, round(px * 0.06)) if rng.random() < 0.25 else 0
    latin = rng.choice(fonts["latin"])
    cjk = rng.choice(fonts["cjk"]) if fonts["cjk"] else latin

    stamps = []
    for t in texts:
        font = ImageFont.truetype(cjk if not t.isascii() else latin, px)
        stamps.append((_stamp(t, font, stroke, angle), font.getlength(t)))

    fill = np.zeros((h, w), np.float32)
    union = np.zeros((h, w), np.float32)
    a = math.radians(angle)
    along = np.array([math.cos(a), -math.sin(a)])  # text direction after a CCW rotate
    across = np.array([math.sin(a), math.cos(a)])
    centre = np.array([w / 2, h / 2])

    def put(i: int, p: np.ndarray) -> None:
        f, u = stamps[i][0]
        _paste(fill, f, *p)
        _paste(union, u, *p)

    if layout == "tile":
        reach = math.hypot(w, h) / 2 + px * 4
        pitch = px * rng.uniform(1.8, 4.0)
        gap = px * rng.uniform(0.6, 3.0)
        stagger = rng.random()
        for k, v in enumerate(np.arange(-reach, reach + pitch, pitch)):
            i = k % len(stamps)
            unit = stamps[i][1] + gap
            u = -reach - ((k * stagger * unit) % unit)
            while u < reach + unit:
                put(i, centre + (u + stamps[i][1] / 2) * along + v * across)
                u += unit
    elif layout == "scatter":
        for _ in range(rng.randint(2, 6)):
            put(
                rng.randrange(len(stamps)),
                np.array([rng.uniform(0, w), rng.uniform(0, h)]),
            )
    else:
        i = rng.randrange(len(stamps))
        (f, _), _len = stamps[i]
        m = px * 1.5
        spots = [
            (w / 2, h / 2),
            (m + f.shape[1] / 2, h - m - f.shape[0] / 2),
            (w - m - f.shape[1] / 2, h - m - f.shape[0] / 2),
            (w - m - f.shape[1] / 2, m + f.shape[0] / 2),
        ]
        put(i, np.array(rng.choice(spots)))

    sigma = rng.uniform(0.0, 0.8)
    if sigma > 0.3:
        fill = cv2.GaussianBlur(fill, (0, 0), sigma)
        union = cv2.GaussianBlur(union, (0, 0), sigma)
    return fill, np.clip(union - fill, 0, 1)


def _ink(rng: random.Random) -> np.ndarray:
    """An RGB ink: mostly white (the reference case), else light grey, black or a tint."""
    r = rng.random()
    if r < 0.8:
        return np.full(3, 255.0)
    if r < 0.9:
        return np.full(3, rng.uniform(200, 240))
    if r < 0.95:
        return np.zeros(3)
    hue = np.array(
        Image.new("HSV", (1, 1), (rng.randrange(256), 160, 230)).convert("RGB")
    ).reshape(3)
    return hue.astype(np.float64)


def composite(
    src: np.ndarray, fill: np.ndarray, outline: np.ndarray, rng: random.Random,
    opacity: tuple[float, float], min_delta: float,
) -> np.ndarray:  # fmt: skip
    """``src`` (RGB uint8) under the overlay; the ink is re-drawn (up to 6 times)
    until the glyphs shift the page by ≥ ``min_delta`` levels on average."""
    img = src.astype(np.float32)
    glyph = fill > 0.5
    lo, hi = np.log(opacity[0]), np.log(opacity[1])
    for _ in range(6):
        alpha = float(np.exp(rng.uniform(lo, hi)))
        ink = _ink(rng)
        out = img
        if outline.any():
            m = (outline * alpha)[..., None]
            out = out * (1 - m) + (255.0 - ink) * m
        m = (fill * alpha)[..., None]
        out = out * (1 - m) + ink * m
        if not glyph.any() or np.abs(out - img)[glyph].mean() >= min_delta:
            break
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _load_rgb(
    p: Path, max_side: int, size: tuple[int, int] | None = None
) -> np.ndarray:
    """RGB uint8, downscaled to ``max_side`` — or resized to ``size`` ``(w, h)``."""
    with Image.open(p) as im:
        im.seek(0)
        a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    if size is None:
        s = max_side / max(h, w)
        size = (round(w * s), round(h * s)) if s < 1 else (w, h)
    if size != (w, h):
        interp = cv2.INTER_AREA if size[0] < w else cv2.INTER_LANCZOS4
        a = cv2.resize(a, size, interpolation=interp)
    return a


def _write_webp(path: Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".webp", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), WEBP_LOSSLESS)
    if not ok:
        raise RuntimeError(f"webp encode failed: {path}")
    path.write_bytes(buf.tobytes())


def make_pair(job) -> str:
    cv2.setNumThreads(1)
    row, staging, out_dir, preview, fonts, knobs = job
    artist = Path(row["target"]).parent.name
    rng = random.Random(f"{knobs['seed']}:{artist}/{row['stem']}")
    target = _load_rgb(staging / row["target"], knobs["max_side"])
    h, w = target.shape[:2]
    stacked = rng.random() < knobs["stack"]
    base = _load_rgb(staging / row["cond"], 0, (w, h)) if stacked else target
    texts = watermark_texts(artist, rng, bool(fonts["cjk"]))
    fill, outline = render_overlay(w, h, texts, fonts, rng)
    cond = composite(base, fill, outline, rng, knobs["opacity"], knobs["min_delta"])
    pid = f"{PREFIX}{row['stem']}"
    tag = f"{artist}/{pid}{' (stacked)' if stacked else ''}: {' | '.join(texts)}"
    if preview:
        _write_webp(out_dir / f"{artist}__{pid}.webp", cond)
        return tag
    adir = out_dir / artist
    adir.mkdir(parents=True, exist_ok=True)
    _write_webp(adir / f"{pid}_no_tags.webp", target)
    _write_webp(adir / f"{pid}_tags.webp", cond)
    cap = (staging / row["caption"]).read_text(encoding="utf-8").strip()
    for side in ("_no_tags", "_tags"):
        (adir / f"{pid}{side}.txt").write_text(cap, encoding="utf-8")
    return tag


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--staging", default="post_image_dataset/i2i/sanitize/staging")
    p.add_argument(
        "--n", type=int, default=80, help="pairs to stage (capped at the staged pairs)"
    )
    p.add_argument(
        "--stack",
        type=float,
        default=0.5,
        help="share of conditions drawn over the texted side instead of the clean target",
    )
    p.add_argument(
        "--max_side",
        type=int,
        default=2048,
        help="longer sources are downscaled to this long side",
    )
    p.add_argument(
        "--opacity",
        nargs=2,
        type=float,
        default=[0.03, 0.15],
        metavar=("LO", "HI"),
        help="log-uniform ink opacity range",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=1.5,
        help="min mean glyph shift in 8-bit levels (else re-ink)",
    )
    p.add_argument("--font_dirs", nargs="+", default=list(_FONT_DIRS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="write the conditions here instead of staging",
    )
    a = p.parse_args(argv)

    latin, cjk = find_fonts(a.font_dirs)
    if not latin:
        raise SystemExit(f"no Latin font found under {a.font_dirs}; pass --font_dirs")
    if not cjk:
        print("  [warn] no CJK font found — notices stay Latin")
    fonts = {"latin": latin, "cjk": cjk}

    staging = resolve_under_home(a.staging)
    lines = (staging / PAIR_MANIFEST).read_text(encoding="utf-8").splitlines()
    rows = [
        r
        for r in map(json.loads, filter(str.strip, lines))
        if not r["stem"].startswith(PREFIX)
    ]
    rows = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
    print(
        f"{len(rows)} source pairs from {staging}; {len(latin)} Latin / {len(cjk)} CJK fonts"
    )

    out_dir = a.preview or staging
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.preview:
        stale = list(staging.glob(f"*/{PREFIX}*"))
        for f in stale:
            f.unlink()
        if stale:
            print(f"  removed {len(stale)} files of the previous {PREFIX}* draw")

    knobs = {
        "seed": a.seed,
        "max_side": a.max_side,
        "stack": a.stack,
        "opacity": tuple(a.opacity),
        "min_delta": a.min_delta,
    }
    jobs = [(r, staging, out_dir, bool(a.preview), fonts, knobs) for r in rows]
    with ProcessPoolExecutor(a.workers) as ex:
        for line in ex.map(make_pair, jobs, chunksize=2):
            print(f"  {line}")
    if a.preview:
        print(f"preview → {out_dir}")
        return
    n = write_pair_manifest(staging)
    print(f"staged {len(rows)} {PREFIX}* pairs; {staging / 'pairs.jsonl'} now {n} rows")


if __name__ == "__main__":
    main()
