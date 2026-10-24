#!/usr/bin/env python3
"""Danbooru textless_version pairs → an existing ``_tags`` / ``_no_tags`` staging tree.

Reads gelcrawl's ``make crawl-dan-textless`` output — ``<src>/<artist>/dan_{id}.<ext>``
(the textless post, the target) beside ``dan_{id}_text.<ext>`` (its texted original,
the condition) — and adds each usable pair to ``--staging`` (default: the sanitize
tree) without touching what is already there, then rewrites ``pairs.jsonl`` from the
tree. The tree is hand-curated, so this merges; it never re-mines.

Per pair:
  * **size gate**: editions whose width or height differ by more than
    ``--max-size-diff`` (a rescaled or re-cropped edition) are dropped.
  * **register** the texted page onto the textless one (ORB partial affine + masked
    ECC, ``swap_text.align``); unregistered, or registered at a scale off 1 by more
    than ``--max-size-diff``, → dropped.
  * **slice** what is left (editions a few px apart) to the rectangle both cover, in
    the textless page's pixel grid: the target is cropped only, never resampled; the
    texted page is warped into it. A pair already pixel-aligned at equal size is
    symlinked as is.
  * **tone gate**: diff both pages at a blur that cancels sub-pixel line residual,
    mask the overlay (diff > 25, dilated) and drop the pair when more than
    ``--tone-max`` of the rest still differs by > 4 ΔE — a regraded edition
    (lens flare, colour pass), not a text overlay. Pairs with no overlay at all
    are dropped too.
  * **dedup** against the staged pairs and the rest of the crawl by 64-bit DCT
    phash, confirmed at 256 px (< 0.2 % of pixels off by > 16): a duplicate matches
    on *both* sides. A CG set reuses one base under different lettering and
    variants differ by a splash, so a phash match alone would fold distinct pairs.

Content edits beyond text (a clothed vs nude edition) pass every gate; review the
``textless_review/kept/`` (one textless | texted row per pair; the drops sit
beside it under their reason) and delete those by hand, then ``--from-review`` stages
exactly the rows left in ``kept/``.

    python -m i2i_tasks.tools.textless_pairs --src ~/gelcrawl/textless --dry-run
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from i2i_tasks.swap_text.align import phash, refine_ecc, register
from i2i_tasks.tools.near_twins.outputs import write_pair_manifest
from library.env import resolve_under_home

_IMAGE_SUFFIXES = {".png", ".webp", ".jpg", ".jpeg", ".gif"}


def _load(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        im.seek(0)
        return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def discover(src: Path) -> list[tuple[str, str, Path, Path]]:
    """``(artist, stem, textless, texted)`` for every complete pair under ``src``."""
    out = []
    for adir in sorted(p for p in src.iterdir() if p.is_dir()):
        imgs = {
            f.stem: f for f in adir.iterdir() if f.suffix.lower() in _IMAGE_SUFFIXES
        }
        for stem, f in sorted(imgs.items()):
            if not stem.endswith("_text") and (t := imgs.get(stem + "_text")):
                out.append((adir.name, stem, f, t))
    return out


def _hash(gray: np.ndarray) -> int:
    return int("".join("1" if b else "0" for b in phash(gray)), 2)


def _inner_rect(M: np.ndarray, src_hw, dst_hw, margin=2) -> tuple[int, int, int, int]:
    """Axis-aligned ``(x0, y0, x1, y1)`` inside both the ``M``-warped source and
    the destination frame."""
    h, w = src_hw
    c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64) @ M.T
    x0 = max(c[0, 0], c[2, 0], 0) + margin
    x1 = min(c[1, 0], c[3, 0], dst_hw[1]) - margin
    y0 = max(c[0, 1], c[1, 1], 0) + margin
    y1 = min(c[2, 1], c[3, 1], dst_hw[0]) - margin
    return int(np.ceil(x0)), int(np.ceil(y0)), int(np.floor(x1)), int(np.floor(y1))


def _thumb(a: np.ndarray, b: np.ndarray, px: int) -> bytes:
    row = np.hstack(
        [cv2.resize(x, None, fx=px / x.shape[0], fy=px / x.shape[0], interpolation=cv2.INTER_AREA) for x in (a, b)]
    )
    return cv2.imencode(".jpg", row, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def analyse(job) -> dict:
    """Register, slice and score one pair; returns its record (``crop`` + ``M``
    on success, ``drop`` otherwise)."""
    cv2.setNumThreads(1)
    artist, stem, tp, cp, a = job
    rec = {"artist": artist, "stem": stem, "textless": str(tp), "texted": str(cp)}
    T, C = _load(tp), _load(cp)
    tg, cg = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (T, C))
    rec["hash"] = [_hash(tg), _hash(cg)]
    h, w = tg.shape
    rec["size"] = [[w, h], list(cg.shape[::-1])]
    if max(abs(cg.shape[1] - w) / w, abs(cg.shape[0] - h) / h) > a["max_size_diff"]:
        return rec | {"drop": "size", "thumb": _thumb(T, C, a["thumb_px"])}
    M, inliers = register(tg, cg)
    if M is not None and inliers >= a["min_inliers"]:
        M = refine_ecc(tg, cg, M)
    if M is None or inliers < a["min_inliers"]:
        return rec | {"drop": "register", "inliers": inliers, "thumb": _thumb(T, C, a["thumb_px"])}
    if abs(np.sqrt(abs(np.linalg.det(M[:, :2]))) - 1) > a["max_size_diff"]:
        return rec | {"drop": "size", "thumb": _thumb(T, C, a["thumb_px"])}
    x0, y0, x1, y1 = _inner_rect(M, cg.shape, (h, w))
    cover_t = max(0, x1 - x0) * max(0, y1 - y0) / (h * w)
    cover_c = max(0, x1 - x0) * max(0, y1 - y0) / (cg.size * abs(np.linalg.det(M[:, :2])))
    rec |= {"inliers": inliers, "cover": round(min(cover_t, cover_c), 3)}
    if min(cover_t, cover_c) < a["min_cover"]:
        return rec | {"drop": "overlap", "thumb": _thumb(T, C, a["thumb_px"])}

    # Scored at <= work_px on the long side: the gate reads tone, not line detail.
    s = min(1.0, a["work_px"] / max(h, w))
    sw, sh = round(w * s), round(h * s)
    # Area-shrink C before the warp: warpAffine does not antialias, and an aliased C
    # against an area-shrunk T reads as faint diff everywhere.
    Ms = M.copy()
    Ms[:, 2] *= s
    Cs = cv2.resize(C, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else C
    W = cv2.warpAffine(Cs, Ms, (sw, sh), flags=cv2.INTER_LINEAR)
    Ts = cv2.resize(T, (sw, sh), interpolation=cv2.INTER_AREA) if s < 1 else T
    r = [round(v * s) for v in (x0, y0, x1, y1)]
    Tc, Wc = Ts[r[1] : r[3], r[0] : r[2]], W[r[1] : r[3], r[0] : r[2]]
    rec["thumb"] = _thumb(Tc, Wc, a["thumb_px"])
    sig = max(1.5, min(Tc.shape[:2]) / 600)
    lab = lambda x: cv2.cvtColor(  # noqa: E731
        cv2.GaussianBlur(x, (0, 0), sig), cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    d = np.linalg.norm(lab(Tc) - lab(Wc), axis=2)
    # Translucent balloons tint the art under them only faintly: the overlay takes in
    # the faint diff connected to the strong one, out to grow_frac of the short edge.
    seeds = (d > 25).astype(np.uint8)
    faint = ((d > 4) | (seeds > 0)).astype(np.uint8)
    n, cc = cv2.connectedComponents(faint, connectivity=8)
    touched = np.zeros(n, bool)
    touched[cc[seeds > 0]] = True
    touched[0] = False
    near = cv2.distanceTransform(1 - seeds, cv2.DIST_L2, 5) <= a["grow_frac"] * min(d.shape)
    k = max(15, int(min(d.shape) * 0.03)) | 1
    overlay = cv2.dilate(
        (seeds | (touched[cc] & near)).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    ).astype(bool)
    tone = float((d[~overlay] > 4).mean()) if (~overlay).any() else 1.0
    rec |= {"tone": round(tone, 4), "overlay": round(float(overlay.mean()), 4)}
    if tone > a["tone_max"]:
        return rec | {"drop": "tone"}
    if overlay.mean() < a["overlay_min"]:
        return rec | {"drop": "no_overlay"}
    exact = (
        T.shape == C.shape
        and np.abs(M - np.eye(2, 3)).max() * max(h, w) < 0.5
    )
    return rec | {"crop": [x0, y0, x1, y1], "M": M.tolist(), "exact": bool(exact)}


def staged_hashes(staging: Path) -> list[tuple[list[int], str, tuple[Path, Path]]]:
    """``([target phash, cond phash], pair, (target, cond))`` for every staged pair."""
    out = []
    for f in staging.rglob("*_no_tags.*"):
        if f.suffix.lower() not in _IMAGE_SUFFIXES or not f.exists():
            continue
        pid = f.name[: -len("_no_tags" + f.suffix)]
        cond = next((c for ext in _IMAGE_SUFFIXES if (c := f.with_name(f"{pid}_tags{ext}")).exists()), None)
        if cond is not None:
            h = [_hash(cv2.cvtColor(_load(x), cv2.COLOR_BGR2GRAY)) for x in (f, cond)]
            out.append((h, str(f.relative_to(staging)), (f, cond)))
    return out


def _small(p: Path, px=256) -> np.ndarray:
    g = cv2.cvtColor(_load(p), cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (px, px), interpolation=cv2.INTER_AREA).astype(np.int16)


def _same(a: tuple[Path, Path], b: tuple[Path, Path]) -> bool:
    return all((np.abs(_small(x) - _small(y)) > 16).mean() < 0.002 for x, y in zip(a, b))


def _near(h, paths, pool, thr: int) -> str | None:
    return next(
        (
            v
            for k, v, kp in pool
            if all((x ^ y).bit_count() <= thr for x, y in zip(h, k)) and _same(paths, kp)
        ),
        None,
    )


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_pair(job) -> None:
    cv2.setNumThreads(1)
    rec, staging = job
    adir = staging / rec["artist"]
    adir.mkdir(parents=True, exist_ok=True)
    tp, cp = Path(rec["textless"]), Path(rec["texted"])
    pid = rec["stem"]
    if rec["exact"]:
        _link(tp, adir / f"{pid}_no_tags{tp.suffix}")
        _link(cp, adir / f"{pid}_tags{cp.suffix}")
    else:
        T, C = _load(tp), _load(cp)
        x0, y0, x1, y1 = rec["crop"]
        W = cv2.warpAffine(
            C, np.array(rec["M"]), T.shape[1::-1], flags=cv2.INTER_LANCZOS4
        )
        cv2.imwrite(str(adir / f"{pid}_no_tags.png"), T[y0:y1, x0:x1])
        cv2.imwrite(str(adir / f"{pid}_tags.png"), W[y0:y1, x0:x1])
    for img, side in ((tp, "_no_tags"), (cp, "_tags")):
        cap = img.with_suffix(".txt")
        text = cap.read_text(encoding="utf-8").strip() if cap.is_file() else ""
        (adir / f"{pid}{side}.txt").write_text(text, encoding="utf-8")


def write_review(recs: list[dict], out: Path) -> None:
    """One ``textless | texted`` row per pair under ``out/<kept|drop reason>/``."""
    if out.is_dir():
        for f in out.rglob("*.jpg"):
            f.unlink()
    for r in recs:
        d = out / r.get("drop", "kept")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{r['artist']}__{r['stem']}.jpg").write_bytes(r["thumb"])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", type=Path, default=Path("~/gelcrawl/textless").expanduser())
    p.add_argument("--staging", default="post_image_dataset/i2i/sanitize/staging")
    p.add_argument("--tone-max", type=float, default=0.05, help="max fraction of non-overlay pixels > 4 ΔE")
    p.add_argument("--grow-frac", type=float, default=0.05, help="reach of the overlay into faint connected diff (translucent balloons), × short edge")
    p.add_argument("--overlay-min", type=float, default=0.003, help="min overlay fraction (else: nothing to remove)")
    p.add_argument("--max-size-diff", type=float, default=0.01, help="max relative width/height (and registered scale) difference")
    p.add_argument("--min-cover", type=float, default=0.6, help="min shared-rectangle fraction of either page")
    p.add_argument("--min-inliers", type=int, default=30)
    p.add_argument("--dup-hamming", type=int, default=8)
    p.add_argument("--work-px", type=int, default=1600, help="long side the tone gate scores at")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report", type=Path, default=None, help="default: <staging>/../textless_report.json")
    p.add_argument("--review-dir", type=Path, default=None, help="default: <staging>/../textless_review")
    p.add_argument("--dry-run", action="store_true", help="analyse + report only; write nothing to staging")
    p.add_argument("--from-review", action="store_true", help="stage only the pairs still in <review-dir>/kept (after hand review); no dedup, report or review rewrite")
    a = p.parse_args(argv)

    staging = resolve_under_home(a.staging)
    knobs = {"tone_max": a.tone_max, "overlay_min": a.overlay_min, "min_cover": a.min_cover, "min_inliers": a.min_inliers, "work_px": a.work_px, "max_size_diff": a.max_size_diff, "grow_frac": a.grow_frac, "thumb_px": 640}
    review = a.review_dir or staging.parent / "textless_review"
    jobs = [(*j, knobs) for j in discover(a.src)]
    if a.from_review:
        reviewed = {f.stem for f in (review / "kept").glob("*.jpg")}
        jobs = [j for j in jobs if f"{j[0]}__{j[1]}" in reviewed]
        print(f"{len(jobs)} of {len(reviewed)} reviewed pairs found under {a.src}")
    else:
        print(f"{len(jobs)} textless pairs under {a.src}")
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(analyse, jobs, chunksize=2))

    if a.from_review:
        kept = [r for r in recs if "drop" not in r]
        for r in recs:
            if "drop" in r:
                print(f"  {r['artist']}/{r['stem']}: now drops ({r['drop']}); skipped")
        if a.dry_run:
            print(f"dry run — would add {len(kept)} pairs")
            return
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(write_pair, [(r, staging) for r in kept]))
        n = write_pair_manifest(staging)
        print(f"added {len(kept)} pairs; {staging / 'pairs.jsonl'} now {n} rows")
        return

    pool = staged_hashes(staging)
    print(f"{len(pool)} staged pairs hashed")
    for r in recs:
        if "drop" in r:
            continue
        paths = (Path(r["textless"]), Path(r["texted"]))
        if hit := _near(r["hash"], paths, pool, a.dup_hamming):
            r["drop"], r["dup_of"] = "duplicate", hit
        else:
            pool.append((r["hash"], f"{r['artist']}/{r['stem']}", paths))

    kept = [r for r in recs if "drop" not in r]
    counts: dict[str, int] = {}
    for r in recs:
        counts[r.get("drop", "kept")] = counts.get(r.get("drop", "kept"), 0) + 1
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    report = a.report or staging.parent / "textless_report.json"
    report.write_text(json.dumps([{k: v for k, v in r.items() if k not in ("M", "thumb")} for r in recs], indent=1))
    write_review(recs, review)
    if a.dry_run:
        print(f"dry run — report {report}")
        return
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(write_pair, [(r, staging) for r in kept]))
    n = write_pair_manifest(staging)
    print(f"added {len(kept)} pairs; {staging / 'pairs.jsonl'} now {n} rows")


if __name__ == "__main__":
    main()
