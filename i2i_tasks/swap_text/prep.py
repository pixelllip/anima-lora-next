#!/usr/bin/env python3
"""Stage the swap_text i2i task from hitocrawl edition twins.

Every hitocrawl work is the same pages lettered in JA and in a translated
edition (``en/``, ``ko/``). The condition is the JA page; the target is the
same page with its translated lettering pasted where the OCR can read it, and
the caption lists the text read off that target. Four stages, each idempotent:

1. **build** (CPU, ``--workers``) — per work × language: phash-align pages,
   register the twin onto the JA page (ORB partial affine), drop unregistered
   and off-scale pages; then, at the MOSS cache's ``target_size`` for
   ``--resolution`` (so the cache encodes the staged pages as-is), refine the
   registration by ECC, diff out the lettering regions, warp
   the twin into the JA frame, colour-match it there (``colour_match``) and
   fill the strip it does not cover from the JA page, then grow each region
   over the pixel diff connected to it (``grow_regions``). Regions whose grown
   labels touch form a *group*, the unit a swap pastes. Each page
   records ``resid`` (mean |src − tgt| outside the lettering) and
   ``uncovered``; each group its ``seam`` (the same just outside its paste
   mask) and ``leftover`` (strong diff connected to the paste mask but outside
   it — strokes a paste would leave half-swapped). Writes
   ``<root>/work/<lang>/<anchor>/{page}.src.webp`` / ``.tgt.webp`` /
   ``.lab.png`` (region ``k`` as ``k + 1``, 255 outside the warped twin) and
   ``pages.json``, the work's completion marker.
2. **ocr** (GPU) — reads every twin ``.tgt.webp`` within ``--max_resid`` /
   ``--max_uncovered`` through the anime_tools
   reader, with the MOSS caption helper's floors
   (``library.moss.colorize.readable``), into ``<root>/ocr.json``.
3. **assemble** (CPU) — drops pages past ``--max_resid`` (another scan, not
   a relettering) or ``--max_uncovered``, and assigns each twin OCR line to the
   region its centre falls in. A line holding kana or CJK ideographs
   (``JA_SCRIPT``) inside a region is a misread of the translated lettering.
   A group is *clean* when a non-misread line landed in it, none misread, its
   ``seam`` ≤ ``--max_seam`` and ``leftover`` ≤ ``--max_leftover``; a page
   with none drops. Every target is the JA page with groups pasted from the
   twin — lettering the reader cannot read, or that a paste would tear, stays
   JA:

   * **full** — every clean group pasted.
   * **partial** (``--partial_ratio`` of the pairs) — a random proper subset
     of the clean groups.

   Writes ``<root>/staging/<lang>/{stem}`` (``.tgt.webp``, ``.src.webp``
   symlink), ``plan.jsonl`` and ``stats.json``; reruns rewrite the tree.
4. **caption** — captions each target with the twin read's lines in its
   pasted groups (CPU), or with ``--caption_ocr`` (GPU) with a second read of
   the staged target itself (``<root>/ocr_staged.json``, keyed by page and
   pasted regions); either way only lines holding no kana or CJK ideograph, so
   JA lettering — left by the translator or by a partial swap — is never named
   and stays the source; a target with no such line drops. The caption is layout JSON:
   ``{"summary": {"caption": "comic, english text"}, "items": [one text item
   (text + bbox) per line]}``. Writes ``{stem}.txt``, ``pairs.jsonl`` and the
   final ``stats.json``.

Stems are ``{anchor}_{page}_{lang}``; ``pairs.jsonl`` rows carry ``work`` so a
holdout can split by work (the same work appears in both languages).

    make i2i-stage TASK=swap_text ARGS="--limit 4"
    make daemon-run ARGS="i2i_tasks/swap_text/prep.py --stages ocr assemble caption"
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
from library.moss.cache import target_size  # noqa: E402

LANG_TAG = {"en": "english text", "ko": "korean text"}
# kana / CJK ideographs: in a translated edition's region these are the reader
# taking hand-lettered Hangul for katakana, or JA lettering left in place
JA_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)
INVALID = 255
WEBP = [cv2.IMWRITE_WEBP_QUALITY, 95]


def _single_thread() -> None:
    """Pool initializer: one OpenCV thread per worker. cv2 otherwise spawns a
    thread per core in every worker, and 12 workers ran at load ~100 on 12
    cores."""
    cv2.setNumThreads(1)


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


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) == size:
        return img
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


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
        # ECC and everything after run at the staged size; the twin is warped
        # at native size so both pages go through the same resize
        H, W = src_g[i].shape
        size = target_size(W, H, a.resolution)
        M = refine_ecc(src_g[i], tw_g[j], M, work_px=max(size))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

        def warp(img):
            return _resize(cv2.warpAffine(img, M, (W, H), flags=interp), size)

        valid = cv2.warpAffine(np.full(tw_g[j].shape, 255, np.uint8), M, (W, H))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
        regions, labels, changed = diff_regions(
            _resize(src_g[i], size), warp(tw_g[j]), valid
        )
        page["changed_frac"] = round(changed, 4)
        if not regions:
            pages.append({**page, "status": "no_regions"})
            continue

        src = _resize(cv2.imread(str(src_files[i]), cv2.IMREAD_COLOR), size)
        tgt = warp(cv2.imread(str(tw_files[j]), cv2.IMREAD_COLOR))
        labels[valid == 0] = INVALID
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
    with Pool(a.workers, initializer=_single_thread) as pool:
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


def _ocr_reader(a):
    """``(engine, readable)``: the anime_tools reader with the MOSS caption
    helper's floors (``library.moss.colorize.readable``)."""
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    from library.moss.colorize import readable

    reader = SfxReader.load(device=a.device)
    engine = RereadEngine(
        engine=load_ocr(device=a.device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    return engine, readable


def _read_into(a, path: Path, todo: list, paths: list[Path], label: str) -> dict:
    """Read ``paths`` into the ``{key: [[text, box], ...]}`` cache at ``path``;
    ``todo`` holds each path's key. Returns the whole cache.

    ``--ocr_procs`` > 1 splits the unread pages over that many processes, each
    with its own reader and its own ``{stem}.{i}of{n}.json``: the VL re-read
    decodes one page's boxes at a time, so one process leaves the GPU idle
    between launches. Shard files left by an interrupted run fold in first."""
    reads = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    shards = sorted(path.parent.glob(f"{path.stem}.*of*.json"))
    for sp in shards:
        reads.update(json.loads(sp.read_text(encoding="utf-8")))
    if shards:
        _write_text(path, json.dumps(reads, ensure_ascii=False))
        for sp in shards:
            sp.unlink()
    keep = [(k, p) for k, p in zip(todo, paths) if k not in reads]
    print(f"{label}: {len(keep)} pages to read ({len(reads)} cached)", flush=True)
    if not keep:
        return reads
    n = max(1, min(a.ocr_procs, len(keep)))
    if n == 1:
        reads.update(_read_shard((a, path, keep, label)))
        _write_text(path, json.dumps(reads, ensure_ascii=False))
        return reads
    import multiprocessing as mp

    jobs = [
        (a, path.with_name(f"{path.stem}.{i}of{n}.json"), keep[i::n], f"{label}[{i}]")
        for i in range(n)
    ]
    with mp.get_context("spawn").Pool(n) as pool:
        for part in pool.imap_unordered(_read_shard, jobs):
            reads.update(part)
    _write_text(path, json.dumps(reads, ensure_ascii=False))
    for _, sp, _, _ in jobs:
        sp.unlink(missing_ok=True)
    return reads


def _read_shard(job) -> dict:
    """Read ``keep`` (``[(key, path)]``), checkpointing to ``out`` every 200
    pages and reporting every 50."""
    a, out, keep, label = job
    engine, readable = _ocr_reader(a)
    reads, t0 = {}, time.time()
    lines_iter = engine.read_iter([p for _, p in keep])
    for k, ((key, _), lines) in enumerate(zip(keep, lines_iter), 1):
        reads[key] = [[ln.text, list(ln.box)] for ln in lines if readable(ln)]
        if k % 200 == 0 or k == len(keep):
            _write_text(out, json.dumps(reads, ensure_ascii=False))
        # the daemon kills a job silent for 120 s; a shard does ~1 page/s
        if k % 50 == 0 or k == len(keep):
            rate = k / (time.time() - t0)
            print(
                f"  {label} {k}/{len(keep)}, {rate:.2f} page/s, "
                f"eta {(len(keep) - k) / rate / 60:.0f} min",
                flush=True,
            )
    return reads


def stage_ocr(a) -> None:
    pages = [
        p
        for p in ok_pages(a)
        if p["resid"] <= a.max_resid and p["uncovered"] <= a.max_uncovered
    ]
    _read_into(
        a,
        a.root / "ocr.json",
        [p["key"] for p in pages],
        [p["dir"] / f"{p['page']}.tgt.webp" for p in pages],
        "ocr",
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
    """Write one pair's composite target and source symlink."""
    p, chosen, root, dilate, feather = job
    stem = f"{p['anchor']}_{p['page']}_{p['lang']}"
    base = root / "staging" / p["lang"]
    base.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(p["dir"] / f"{p['page']}.src.webp"), cv2.IMREAD_COLOR)
    tgt = cv2.imread(str(p["dir"] / f"{p['page']}.tgt.webp"), cv2.IMREAD_COLOR)
    labels = cv2.imread(str(p["dir"] / f"{p['page']}.lab.png"), cv2.IMREAD_UNCHANGED)
    mask = paste_mask(labels, chosen, dilate)
    alpha = mask.astype(np.float32)
    if feather:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather) * mask
    alpha = alpha[..., None]
    out = np.clip(src * (1 - alpha) + tgt * alpha + 0.5, 0, 255).astype(np.uint8)
    _imwrite(base / f"{stem}.tgt.webp", out, WEBP)
    (base / f"{stem}.src.webp").unlink(missing_ok=True)
    (base / f"{stem}.src.webp").symlink_to(
        os.path.relpath(p["dir"] / f"{p['page']}.src.webp", base)
    )


def stage_assemble(a) -> None:
    reads = json.loads((a.root / "ocr.json").read_text(encoding="utf-8"))
    pages, plans = [], []
    n = dict.fromkeys(
        ["unread", "resid", "uncovered", "uncaptioned", "misread_lines"], 0
    )
    for p in ok_pages(a):
        if p["resid"] > a.max_resid:
            n["resid"] += 1
            continue
        if p["uncovered"] > a.max_uncovered:
            n["uncovered"] += 1
            continue
        if p["key"] not in reads:
            n["unread"] += 1
            continue
        lines = [[t.strip(), b] for t, b in reads[p["key"]]]
        lines = [ln for ln in lines if ln[0]]
        owner = assign_lines(lines, p["regions"], a.pad_frac)
        misread = [
            k is not None and bool(JA_SCRIPT.search(t))
            for (t, _), k in zip(lines, owner)
        ]
        n["misread_lines"] += sum(misread)
        bad = {k for k, m in zip(owner, misread) if m}
        group_of = {k: gi for gi, g in enumerate(p["groups"]) for k in g["regions"]}
        has_line = {
            group_of[k] for k, m in zip(owner, misread) if k is not None and not m
        }
        clean = [
            gi
            for gi, g in enumerate(p["groups"])
            if gi in has_line
            and not bad & set(g["regions"])
            and g["seam"] <= a.max_seam
            and g["leftover"] <= a.max_leftover
        ]
        if not clean:
            n["uncaptioned"] += 1
            continue
        lines = [
            (ln, group_of[k])
            for ln, k, m in zip(lines, owner, misread)
            if k is not None and not m
        ]
        pages.append((p, clean, lines))

    n_eligible = sum(len(c) >= 2 for _, c, _ in pages)
    p_partial = min(1.0, a.partial_ratio * len(pages) / max(n_eligible, 1))
    for p, clean, lines in pages:
        stem = f"{p['anchor']}_{p['page']}_{p['lang']}"
        rng = random.Random(_stable_seed(f"{a.seed}:{stem}"))
        groups = clean
        if len(clean) >= 2 and rng.random() < p_partial:
            groups = sorted(rng.sample(clean, rng.randint(1, len(clean) - 1)))
        chosen = sorted(r for gi in groups for r in p["groups"][gi]["regions"])
        keep = [ln for ln, gi in lines if gi in groups]
        plans.append((p, stem, chosen, keep, len(groups), len(clean)))

    staging = a.root / "staging"
    for f in staging.glob("*/*"):
        f.unlink()
    (staging / "pairs.jsonl").unlink(missing_ok=True)
    jobs = [(p, chosen, a.root, a.dilate_frac, a.feather) for p, _, chosen, *_ in plans]
    t0 = time.time()
    with Pool(a.workers, initializer=_single_thread) as pool:
        for k, _ in enumerate(pool.imap_unordered(compose_job, jobs, chunksize=4), 1):
            if k % 500 == 0 or k == len(jobs):
                print(
                    f"  assemble {k}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True
                )

    rows = [
        {
            "stem": stem,
            "key": p["key"],
            "work": p["anchor"],
            "lang": p["lang"],
            "size": p["size"],
            "mode": "full" if n_swapped == n_clean else "partial",
            "groups": len(p["groups"]),
            "clean_groups": n_clean,
            "swapped_groups": n_swapped,
            "chosen": chosen,
            "lines": keep,
        }
        for p, stem, chosen, keep, n_swapped, n_clean in plans
    ]
    _write_text(
        staging / "plan.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stats = {
        "planned": len(rows),
        "partial_eligible": n_eligible,
        **{f"dropped_{k}": v for k, v in n.items() if k != "misread_lines"},
        "misread_lines": n["misread_lines"],
        "clean_group_frac": round(
            sum(len(c) for _, c, _ in pages)
            / max(sum(len(p["groups"]) for p, *_ in pages), 1),
            3,
        ),
    }
    _write_text(staging / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------------ caption


def stage_caption(a) -> None:
    staging = a.root / "staging"
    plan = [
        json.loads(ln)
        for ln in (staging / "plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    def key(r):
        return f"{r['key']}:{','.join(map(str, r['chosen']))}"

    if a.caption_ocr:
        reads = _read_into(
            a,
            a.root / "ocr_staged.json",
            [key(r) for r in plan],
            [staging / r["lang"] / f"{r['stem']}.tgt.webp" for r in plan],
            "caption ocr",
        )
    rows, n_ja, n_empty = [], 0, 0
    for r in plan:
        base = staging / r["lang"]
        read = reads[key(r)] if a.caption_ocr else r["lines"]
        lines = [[t.replace('"', "”").strip(), b] for t, b in read]
        lines = [ln for ln in lines if ln[0]]
        keep = [ln for ln in lines if not JA_SCRIPT.search(ln[0])]
        n_ja += len(lines) - len(keep)
        if not keep:
            n_empty += 1
            for ext in (".tgt.webp", ".src.webp", ".txt"):
                (base / f"{r['stem']}{ext}").unlink(missing_ok=True)
            continue
        _write_text(base / f"{r['stem']}.txt", caption(r["lang"], keep, r["size"]))
        rows.append(
            {
                "stem": r["stem"],
                "target": f"{r['lang']}/{r['stem']}.tgt.webp",
                "cond": f"{r['lang']}/{r['stem']}.src.webp",
                "caption": f"{r['lang']}/{r['stem']}.txt",
                "work": r["work"],
                "lang": r["lang"],
                "mode": r["mode"],
                "groups": r["groups"],
                "clean_groups": r["clean_groups"],
                "swapped_groups": r["swapped_groups"],
                "lines": len(keep),
            }
        )
    _write_text(
        staging / "pairs.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    stats = json.loads((staging / "stats.json").read_text(encoding="utf-8"))
    stats.update(
        {
            "pairs": len(rows),
            "by_lang": {lang: sum(r["lang"] == lang for r in rows) for lang in a.langs},
            "partial": sum(r["mode"] == "partial" for r in rows),
            "works": len({r["work"] for r in rows}),
            "dropped_unread_target": n_empty,
            "ja_lines_dropped": n_ja,
        }
    )
    _write_text(staging / "stats.json", json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["build", "ocr", "assemble", "caption"],
        choices=["build", "ocr", "assemble", "caption"],
    )
    ap.add_argument("--hito", type=Path, default=Path.home() / "hitocrawl/retrieved")
    ap.add_argument("--root", default="post_image_dataset/i2i/swap_text")
    ap.add_argument("--langs", nargs="+", default=["en", "ko"], choices=list(LANG_TAG))
    ap.add_argument("--works", nargs="*", help="anchor ids; default every work")
    ap.add_argument("--limit", type=int, default=0, help="works per language (build)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rebuild built works")
    ap.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="staged size: the MOSS cache's target_size at this resolution, so the "
        "cache encodes the pages as-is; the build diffs at this size too",
    )
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
        "--caption_ocr",
        action="store_true",
        help="caption from a second read of each staged target (GPU) instead of "
        "the twin read's lines in the pasted groups",
    )
    ap.add_argument(
        "--ocr_procs",
        type=int,
        default=2,
        help="OCR processes sharing the GPU (each loads its own reader)",
    )
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
        {
            "build": stage_build,
            "ocr": stage_ocr,
            "assemble": stage_assemble,
            "caption": stage_caption,
        }[stage](a)


if __name__ == "__main__":
    main()
