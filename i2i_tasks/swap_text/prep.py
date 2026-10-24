#!/usr/bin/env python3
"""Stage the swap_text i2i task from hitocrawl edition twins.

Every hitocrawl work is the same pages lettered in JA and in a translated
edition (``en/``, ``ko/``). The condition is the JA page; the target is the
same page with the translated lettering, and the caption lists the target's
text. Three stages, each idempotent:

1. **build** (CPU, ``--workers``) — per work × language: phash-align pages,
   register the twin onto the JA page (ORB partial affine, refined by ECC),
   drop unregistered and off-scale pages, diff out the lettering regions, warp
   the twin into the JA frame, colour-match it there (``colour_match``) and
   fill the strip it does not cover from the JA page, then grow each region
   over the pixel diff connected to it (``grow_regions``). Regions whose grown
   labels touch form a *group*, the unit a partial swap pastes. Each page
   records ``resid`` (mean |src − tgt| outside the lettering) and
   ``uncovered``; each group its ``seam`` (the same just outside its paste
   mask) and ``leftover`` (strong diff connected to the paste mask but outside
   it — strokes a paste would leave half-swapped). Writes
   ``<root>/work/<lang>/<anchor>/{page}.src.webp`` / ``.tgt.webp`` /
   ``.lab.png`` (region ``k`` as ``k + 1``, 255 outside the warped twin) and
   ``pages.json``, the work's completion marker.
2. **ocr** (GPU) — reads every ``.tgt.webp`` with regions through the
   anime_tools reader, with the MOSS caption helper's floors
   (``library.moss.colorize.readable``), into ``<root>/ocr.json``.
3. **assemble** (CPU) — drops pages past ``--max_resid`` (another scan, not
   a relettering) or ``--max_uncovered``, and assigns each OCR line to the
   region its centre falls in. A line holding kana or CJK ideographs
   (``JA_SCRIPT``) inside a region is a misread of the translated lettering
   and never captions. Two modes:

   * **full** — the target is the whole translated page; the caption lists
     every line on it.
   * **partial** (``--partial_ratio`` of the pairs) — the target is the JA
     page with a random subset of the *clean* groups pasted from the twin
     (captioned, no misread line, ``seam`` ≤ ``--max_seam``, ``leftover`` ≤
     ``--max_leftover``), at least one group left JA; the caption lists the
     pasted groups' lines only, so whatever it does not name stays the source.

   The caption is layout JSON: ``{"summary": {"caption": "comic, english
   text"}, "items": [one text item per line]}``. Writes
   ``<root>/staging/<lang>/{stem}`` (``.tgt.webp``, ``.txt``, ``.src.webp``
   symlink; a full target is a symlink too), ``pairs.jsonl`` and
   ``stats.json``; reruns rewrite the whole tree.

Stems are ``{anchor}_{page}_{lang}``; ``pairs.jsonl`` rows carry ``work`` so a
holdout can split by work (the same work appears in both languages).

    make i2i-stage TASK=swap_text ARGS="--limit 4"
    make daemon-run ARGS="i2i_tasks/swap_text/prep.py --stages ocr assemble"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align import (  # noqa: E402
    align_pages,
    diff_regions,
    grow_regions,
    phash,
    refine_ecc,
    register,
)

from library.env import resolve_under_home  # noqa: E402

LANG_TAG = {"en": "english text", "ko": "korean text"}
# kana / CJK ideographs: in a translated edition's region these are the reader
# taking hand-lettered Hangul for katakana, or JA lettering left in place
JA_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)
INVALID = 255
WEBP = [cv2.IMWRITE_WEBP_QUALITY, 95]


def _stable_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _write_atomic(out: Path, write) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp" + out.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _imwrite(out: Path, img: np.ndarray, params=()) -> None:
    def write(tmp):
        if not cv2.imwrite(tmp, img, list(params)):
            raise OSError(f"cv2.imwrite failed: {out}")

    _write_atomic(out, write)


def _write_text(out: Path, text: str) -> None:
    _write_atomic(out, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))


# ------------------------------------------------------------------ build


def _pages(side: Path) -> list[Path]:
    """The side's downloaded pages, ad banners / credit cards (``promo``) dropped."""
    files = side / "files.json"
    if files.exists():
        recs = json.loads(files.read_text(encoding="utf-8"))
        names = [r["file"] for r in recs if r.get("file") and not r.get("promo")]
        return [side / n for n in names if (side / n).exists()]
    return sorted(side.glob("*.webp"))


def _linear_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``tgt`` with a per-channel gain / bias fit to ``src`` over ``keep``."""
    ys, xs = np.nonzero(keep)
    pick = np.random.default_rng(0).choice(len(xs), min(len(xs), 200_000), False)
    ys, xs = ys[pick], xs[pick]
    out = np.empty(tgt.shape, np.float32)
    for c in range(3):
        g, b = np.polyfit(tgt[ys, xs, c].astype(np.float32), src[ys, xs, c], 1)
        if not 0.7 <= g <= 1.4:
            g, b = 1.0, float(src[ys, xs, c].mean() - tgt[ys, xs, c].mean())
        out[..., c] = tgt[..., c] * g + b
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _lut_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``tgt`` through a per-channel quantile-matching LUT onto ``src`` over
    ``keep`` — a tone curve, for editions scanned at other levels / gamma."""
    q = np.linspace(0, 100, 257)
    out = np.empty_like(tgt)
    for c in range(3):
        qa = np.percentile(tgt[..., c][keep], q) + np.arange(257) * 1e-6
        qb = np.percentile(src[..., c][keep], q)
        lut = np.interp(np.arange(256), np.maximum.accumulate(qa), qb)
        out[..., c] = np.clip(lut + 0.5, 0, 255).astype(np.uint8)[tgt[..., c]]
    return out


def colour_match(tgt: np.ndarray, src: np.ndarray, keep: np.ndarray):
    """``(name, tgt')``: whichever of the twin as-is, a linear fit and a LUT
    leaves the lowest residual on ``keep``. Most pages need none — on 60 probe
    pages the linear fit alone raised the median residual 2.8 → 3.4 — while
    another scan's tone curve needs the LUT."""
    if keep.sum() < 1000:
        return "id", tgt
    cands = [
        ("id", tgt),
        ("linear", _linear_match(tgt, src, keep)),
        ("lut", _lut_match(tgt, src, keep)),
    ]
    return min(cands, key=lambda c: _absdiff(src, c[1])[keep].mean())


def _absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel mean |a - b| over the channels."""
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(-1)


def paste_mask(labels: np.ndarray, chosen, dilate_frac: float) -> np.ndarray:
    """The composite's paste mask for region indices ``chosen`` (uint8 0/1)."""
    mask = np.isin(labels, [k + 1 for k in chosen]).astype(np.uint8)
    d = max(2, round(min(labels.shape) * dilate_frac))
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2)
    )
    mask[labels == INVALID] = 0
    return mask


def region_groups(labels: np.ndarray, regions, near_frac=0.025) -> list[list[int]]:
    """Region indices grouped by grown labels within ``near_frac`` × the short
    edge of each other: one balloon's column split by a glyph gap stays one
    group, so a partial swap never pastes half of it."""
    n = len(regions)
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    d = max(2, round(near_frac * min(labels.shape)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1,) * 2)
    for k, r in enumerate(regions):
        x0, y0, x1, y1 = r["box"]
        p = 2 * d + round(0.05 * min(labels.shape))
        sl = (slice(max(0, y0 - p), y1 + p), slice(max(0, x0 - p), x1 + p))
        lab = labels[sl]
        near = lab[cv2.dilate((lab == k + 1).astype(np.uint8), ker) > 0]
        for other in np.unique(near):
            if 0 < other < INVALID and other != k + 1:
                parent[find(int(other) - 1)] = find(k)
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _group_metrics(groups, regions, labels, diff, dilate_frac, band=6, near_frac=0.03):
    """Per group: ``seam`` (mean ``diff`` over the band just outside its paste
    mask, other lettering excluded) and ``leftover`` (strong diff outside every
    region within ``near_frac`` × the short edge of the paste mask, over its
    area) — lettering a paste of the group would leave half-swapped."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1,) * 2)
    r_near = max(4, round(near_frac * min(labels.shape)))
    near_ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_near + 1,) * 2)
    strong = cv2.morphologyEx(
        (diff > 48).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    out = []
    for g in groups:
        x0 = min(regions[k]["box"][0] for k in g)
        y0 = min(regions[k]["box"][1] for k in g)
        x1 = max(regions[k]["box"][2] for k in g)
        y1 = max(regions[k]["box"][3] for k in g)
        p = 2 * r_near
        sl = (slice(max(0, y0 - p), y1 + p), slice(max(0, x0 - p), x1 + p))
        lab = labels[sl]
        m = paste_mask(lab, g, dilate_frac)
        free = (lab == 0) & (m == 0)
        ring = (cv2.dilate(m, ker) > 0) & free
        near = (cv2.dilate(m, near_ker) > 0) & free & (strong[sl] > 0)
        out.append(
            {
                "regions": g,
                "seam": round(float(diff[sl][ring].mean()), 2) if ring.any() else 0.0,
                "leftover": round(float(near.sum()) / max(int(m.sum()), 1), 4),
            }
        )
    return out


def _chroma(img: np.ndarray, keep: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1][keep].mean())


def build_work(job) -> str:
    try:
        return _build_work(*job)
    except Exception as e:  # one bad work never takes the pool down
        return f"{job[1]}/{job[0].name}: FAILED {type(e).__name__}: {e}"


def _read_gray(paths: list[Path]) -> tuple[list[Path], list[np.ndarray]]:
    """The readable pages of ``paths`` and their greyscale arrays."""
    kept = [(p, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    kept = [(p, g) for p, g in kept if g is not None and g.size]
    return [p for p, _ in kept], [g for _, g in kept]


def _build_work(work: Path, lang: str, a) -> str:
    out = a.root / "work" / lang / work.name
    if (out / "pages.json").exists() and not a.force:
        return f"{lang}/{work.name}: done, skipped"
    src_files, tw_files = _pages(work / "ja"), _pages(work / lang)
    if not src_files or not tw_files:
        return f"{lang}/{work.name}: no ja/{lang} pages"
    src_files, src_g = _read_gray(src_files)
    tw_files, tw_g = _read_gray(tw_files)
    if not src_files or not tw_files:
        return f"{lang}/{work.name}: no readable ja/{lang} pages"
    pairs = align_pages([phash(g) for g in src_g], [phash(g) for g in tw_g])
    regs = [register(src_g[i], tw_g[j]) for i, j, _ in pairs]
    scales = [
        float(np.hypot(M[0, 0], M[1, 0]))
        for M, inl in regs
        if M is not None and inl >= a.min_inliers
    ]
    work_scale = float(np.median(scales)) if scales else 0.0

    pages = []
    for (i, j, hd), (M, inl) in zip(pairs, regs):
        page = {
            "page": src_files[i].stem,
            "twin": tw_files[j].name,
            "phash": hd,
            "inliers": inl,
        }
        if M is None or inl < a.min_inliers:
            pages.append({**page, "status": "noreg"})
            continue
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        page["scale"] = round(scale, 4)
        # a page off the work's scan scale is a redrawn cover / different art
        if abs(scale / work_scale - 1) > a.scale_tol:
            pages.append({**page, "status": "scale_outlier"})
            continue
        M = refine_ecc(src_g[i], tw_g[j], M)
        H, W = src_g[i].shape
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        warped_g = cv2.warpAffine(tw_g[j], M, (W, H), flags=interp)
        valid = cv2.warpAffine(np.full(tw_g[j].shape, 255, np.uint8), M, (W, H))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        regions, labels, changed = diff_regions(src_g[i], warped_g, valid)
        page["changed_frac"] = round(changed, 4)
        if not regions:
            pages.append({**page, "status": "no_regions"})
            continue

        src = cv2.imread(str(src_files[i]), cv2.IMREAD_COLOR)
        tgt = cv2.warpAffine(
            cv2.imread(str(tw_files[j]), cv2.IMREAD_COLOR), M, (W, H), flags=interp
        )
        labels[valid == 0] = INVALID
        s = min(1.0, a.max_edge / max(H, W))
        if s < 1:
            size = (round(W * s), round(H * s))
            src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
            tgt = cv2.resize(tgt, size, interpolation=cv2.INTER_AREA)
            labels = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
            for r in regions:
                r["box"] = [round(v * s) for v in r["box"]]
                r["area"] = round(r["area"] * s * s)
        keep = labels == 0
        if abs(_chroma(src, keep) - _chroma(tgt, keep)) > a.max_chroma_gap:
            pages.append({**page, "status": "chroma_mismatch"})
            continue
        page["colour"], tgt = colour_match(tgt, src, keep)
        uncovered = labels == INVALID
        tgt[uncovered] = src[uncovered]
        diff = _absdiff(src, tgt)
        labels = grow_regions(labels, diff)
        keep = labels == 0
        groups = _group_metrics(
            region_groups(labels, regions), regions, labels, diff, a.dilate_frac
        )
        page["uncovered"] = round(float(uncovered.mean()), 4)
        tw_h, tw_w = tw_g[j].shape
        page["aspect"] = round((tw_w / tw_h) / (W / H) - 1, 4)
        page["resid"] = round(float(diff[keep].mean()), 2)
        stem = src_files[i].stem
        _imwrite(out / f"{stem}.src.webp", src, WEBP)
        _imwrite(out / f"{stem}.tgt.webp", tgt, WEBP)
        _imwrite(out / f"{stem}.lab.png", labels)
        pages.append(
            {
                **page,
                "size": [src.shape[1], src.shape[0]],
                "regions": regions,
                "groups": groups,
                "status": "ok",
            }
        )

    _write_text(out / "pages.json", json.dumps(pages, indent=1, ensure_ascii=False))
    ok = [p for p in pages if p["status"] == "ok"]
    return (
        f"{lang}/{work.name}: ja {len(src_files)} {lang} {len(tw_files)} "
        f"matched {len(pairs)} ok {len(ok)} "
        f"regions {sum(len(p['regions']) for p in ok)}"
    )


def stage_build(a) -> None:
    jobs = []
    for lang in a.langs:
        works = sorted(p.parent for p in a.hito.glob(f"*/*/{lang}") if p.is_dir())
        works = [w for w in works if (w / "ja").is_dir()]
        if a.works:
            works = [w for w in works if w.name in a.works]
        if a.limit:
            works = works[: a.limit]
        jobs += [(w, lang, a) for w in works]
    print(f"build: {len(jobs)} work × language jobs, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(build_work, jobs), 1):
            print(f"[{k}/{len(jobs)}] {msg}", flush=True)


# ------------------------------------------------------------------ pages


def ok_pages(a) -> list[dict]:
    """Every built page with regions: ``{key, lang, anchor, page, dir, size,
    regions}``, ``key = <lang>/<anchor>/<page>``."""
    out = []
    for lang in a.langs:
        for pj in sorted((a.root / "work" / lang).glob("*/pages.json")):
            anchor = pj.parent.name
            if a.works and anchor not in a.works:
                continue
            for p in json.loads(pj.read_text(encoding="utf-8")):
                if p["status"] == "ok":
                    out.append(
                        {
                            "key": f"{lang}/{anchor}/{p['page']}",
                            "lang": lang,
                            "anchor": anchor,
                            "page": p["page"],
                            "dir": pj.parent,
                            "size": p["size"],
                            "resid": p["resid"],
                            "uncovered": p["uncovered"],
                            "regions": p["regions"],
                            "groups": p["groups"],
                        }
                    )
    return out


# ------------------------------------------------------------------ ocr


def stage_ocr(a) -> None:
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    from library.moss.colorize import readable

    path = a.root / "ocr.json"
    reads = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [p for p in ok_pages(a) if p["key"] not in reads]
    print(f"ocr: {len(todo)} pages to read ({len(reads)} already read)", flush=True)
    if not todo:
        return
    reader = SfxReader.load(device=a.device)
    engine = RereadEngine(
        engine=load_ocr(device=a.device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    paths = [p["dir"] / f"{p['page']}.tgt.webp" for p in todo]
    t0 = time.time()
    for k, (p, lines) in enumerate(zip(todo, engine.read_iter(paths)), 1):
        reads[p["key"]] = [[ln.text, list(ln.box)] for ln in lines if readable(ln)]
        if k % 200 == 0 or k == len(todo):
            _write_text(path, json.dumps(reads, ensure_ascii=False))
            rate = k / (time.time() - t0)
            print(
                f"  ocr {k}/{len(todo)}, {rate:.2f} page/s, "
                f"eta {(len(todo) - k) / rate / 60:.0f} min",
                flush=True,
            )


# ------------------------------------------------------------------ assemble


def assign_lines(lines: list, regions: list[dict], pad_frac: float) -> list[int | None]:
    """Per OCR line, the index of the smallest padded region box holding its
    centre, else ``None``."""
    out = []
    for _, (x0, y0, x1, y1) in lines:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        best, best_area = None, None
        for k, r in enumerate(regions):
            bx0, by0, bx1, by1 = r["box"]
            p = pad_frac * max(bx1 - bx0, by1 - by0)
            if bx0 - p <= cx <= bx1 + p and by0 - p <= cy <= by1 + p:
                area = (bx1 - bx0) * (by1 - by0)
                if best_area is None or area < best_area:
                    best, best_area = k, area
        out.append(best)
    return out


def caption(lang: str, lines: list, size) -> str:
    from anime_tools.captions.layout_json import centered_box

    layout = {"summary": {"caption": f"comic, {LANG_TAG[lang]}"}}
    if lines:
        layout["items"] = [
            {
                "label": "text",
                "caption": "text",
                "text": text,
                "bbox": centered_box(box, size),
            }
            for text, box in lines
        ]
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def compose_job(job) -> None:
    """Write one pair: the target (a symlink for a full pair, else the
    composite), its caption, the source symlink."""
    p, chosen, lines, root, dilate, feather = job
    stem = f"{p['anchor']}_{p['page']}_{p['lang']}"
    base = root / "staging" / p["lang"]
    base.mkdir(parents=True, exist_ok=True)

    def link(name, target):
        (base / name).unlink(missing_ok=True)
        (base / name).symlink_to(os.path.relpath(target, base))

    if chosen is None:
        link(f"{stem}.tgt.webp", p["dir"] / f"{p['page']}.tgt.webp")
    else:
        src = cv2.imread(str(p["dir"] / f"{p['page']}.src.webp"), cv2.IMREAD_COLOR)
        tgt = cv2.imread(str(p["dir"] / f"{p['page']}.tgt.webp"), cv2.IMREAD_COLOR)
        labels = cv2.imread(
            str(p["dir"] / f"{p['page']}.lab.png"), cv2.IMREAD_UNCHANGED
        )
        mask = paste_mask(labels, chosen, dilate)
        alpha = mask.astype(np.float32)
        if feather:
            alpha = cv2.GaussianBlur(alpha, (0, 0), feather) * mask
        alpha = alpha[..., None]
        out = np.clip(src * (1 - alpha) + tgt * alpha + 0.5, 0, 255).astype(np.uint8)
        _imwrite(base / f"{stem}.tgt.webp", out, WEBP)
    _write_text(base / f"{stem}.txt", caption(p["lang"], lines, p["size"]))
    link(f"{stem}.src.webp", p["dir"] / f"{p['page']}.src.webp")


def stage_assemble(a) -> None:
    reads = json.loads((a.root / "ocr.json").read_text(encoding="utf-8"))
    pages, plans = [], []
    n = dict.fromkeys(
        ["unread", "resid", "uncovered", "uncaptioned", "misread_lines"], 0
    )
    for p in ok_pages(a):
        if p["key"] not in reads:
            n["unread"] += 1
            continue
        if p["resid"] > a.max_resid:
            n["resid"] += 1
            continue
        if p["uncovered"] > a.max_uncovered:
            n["uncovered"] += 1
            continue
        lines = [[t.replace('"', "”").strip(), b] for t, b in reads[p["key"]]]
        lines = [ln for ln in lines if ln[0]]
        owner = assign_lines(lines, p["regions"], a.pad_frac)
        misread = [
            k is not None and bool(JA_SCRIPT.search(t))
            for (t, _), k in zip(lines, owner)
        ]
        n["misread_lines"] += sum(misread)
        bad = {k for k, m in zip(owner, misread) if m}
        lines, owner = (
            [ln for ln, m in zip(lines, misread) if not m],
            [k for k, m in zip(owner, misread) if not m],
        )
        if all(k is None for k in owner):
            n["uncaptioned"] += 1
            continue
        group_of = {k: gi for gi, g in enumerate(p["groups"]) for k in g["regions"]}
        has_line = {group_of[k] for k in owner if k is not None}
        clean = [
            gi
            for gi, g in enumerate(p["groups"])
            if gi in has_line
            and not bad & set(g["regions"])
            and g["seam"] <= a.max_seam
            and g["leftover"] <= a.max_leftover
        ]
        pages.append((p, lines, owner, group_of, clean))

    def eligible(p, clean):
        return bool(clean) and len(p["groups"]) >= 2

    n_eligible = sum(eligible(p, c) for p, *_, c in pages)
    p_partial = min(1.0, a.partial_ratio * len(pages) / max(n_eligible, 1))
    for p, lines, owner, group_of, clean in pages:
        stem = f"{p['anchor']}_{p['page']}_{p['lang']}"
        rng = random.Random(_stable_seed(f"{a.seed}:{stem}"))
        if eligible(p, clean) and rng.random() < p_partial:
            k = rng.randint(1, min(len(clean), len(p["groups"]) - 1))
            groups = sorted(rng.sample(clean, k))
            chosen = sorted(r for gi in groups for r in p["groups"][gi]["regions"])
            keep = [
                ln
                for ln, r in zip(lines, owner)
                if r is not None and group_of[r] in groups
            ]
            plans.append((p, stem, chosen, keep, len(groups)))
        else:
            plans.append((p, stem, None, lines, len(p["groups"])))

    staging = a.root / "staging"
    for f in staging.glob("*/*"):
        f.unlink()
    jobs = [
        (p, chosen, keep, a.root, a.dilate_frac, a.feather)
        for p, _, chosen, keep, _ in plans
    ]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(compose_job, jobs, chunksize=4), 1):
            if k % 500 == 0 or k == len(jobs):
                print(
                    f"  assemble {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True
                )

    rows = [
        {
            "stem": stem,
            "target": f"{p['lang']}/{stem}.tgt.webp",
            "cond": f"{p['lang']}/{stem}.src.webp",
            "caption": f"{p['lang']}/{stem}.txt",
            "work": p["anchor"],
            "lang": p["lang"],
            "mode": "full" if chosen is None else "partial",
            "groups": len(p["groups"]),
            "swapped_groups": n_swapped,
            "lines": len(keep),
        }
        for p, stem, chosen, keep, n_swapped in plans
    ]
    _write_text(
        staging / "pairs.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stats = {
        "pairs": len(rows),
        "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
        "partial": sum(r["mode"] == "partial" for r in rows),
        "partial_eligible": n_eligible,
        "works": len({r["work"] for r in rows}),
        **{f"dropped_{k}": v for k, v in n.items() if k != "misread_lines"},
        "misread_lines": n["misread_lines"],
        "clean_group_frac": round(
            sum(len(c) for *_, c in pages)
            / max(sum(len(p["groups"]) for p, *_ in pages), 1),
            3,
        ),
    }
    _write_text(staging / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["build", "ocr", "assemble"],
        choices=["build", "ocr", "assemble"],
    )
    ap.add_argument("--hito", type=Path, default=Path.home() / "hitocrawl/retrieved")
    ap.add_argument("--root", default="post_image_dataset/i2i/swap_text")
    ap.add_argument("--langs", nargs="+", default=["en", "ko"], choices=list(LANG_TAG))
    ap.add_argument("--works", nargs="*", help="anchor ids; default every work")
    ap.add_argument("--limit", type=int, default=0, help="works per language (build)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild built works")
    ap.add_argument("--max_edge", type=int, default=2048, help="staged long edge cap")
    ap.add_argument("--min_inliers", type=int, default=50)
    ap.add_argument(
        "--scale_tol", type=float, default=0.02, help="max |scale / work median - 1|"
    )
    ap.add_argument(
        "--max_chroma_gap",
        type=float,
        default=20.0,
        help="mean HSV saturation gap above which a colour / B&W edition pair drops",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--partial_ratio",
        type=float,
        default=0.5,
        help="share of pairs that swap a proper subset of their captioned regions",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max_resid",
        type=float,
        default=8.0,
        help="page mean |src − tgt| outside the lettering above which it drops",
    )
    ap.add_argument(
        "--max_uncovered",
        type=float,
        default=0.005,
        help="page share the warped twin leaves uncovered above which it drops",
    )
    ap.add_argument(
        "--max_seam",
        type=float,
        default=8.0,
        help="group seam above which a partial pair never pastes it",
    )
    ap.add_argument(
        "--max_leftover",
        type=float,
        default=0.01,
        help="group leftover above which a partial pair never pastes it",
    )
    ap.add_argument(
        "--pad_frac",
        type=float,
        default=0.15,
        help="region box pad (× its long side) an OCR line centre may fall in",
    )
    ap.add_argument(
        "--dilate_frac",
        type=float,
        default=0.003,
        help="paste mask dilation / short edge",
    )
    ap.add_argument("--feather", type=float, default=1.5, help="paste edge blur σ (px)")
    a = ap.parse_args()
    a.root = resolve_under_home(a.root)
    a.works = set(a.works or [])
    for stage in a.stages:
        {"build": stage_build, "ocr": stage_ocr, "assemble": stage_assemble}[stage](a)


if __name__ == "__main__":
    main()
