"""Page matching, registration and lettering diff for hitocrawl edition twins.

Lifted from ``project/finished/ocr_reader/hito/build_crops.py`` (the recipe and
its thresholds are measured there); ``diff_regions`` additionally returns the
kept components as a label map, which the composite pastes by.
"""

from __future__ import annotations

import cv2
import numpy as np


def phash(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(small)[:8, :8]
    return (d > np.median(d)).flatten()


def align_pages(a: list[np.ndarray], b: list[np.ndarray], gap=16.0, thr=14):
    """[(i, j, hamming)] — monotonic Needleman-Wunsch over 64-bit phashes."""
    n, m = len(a), len(b)
    dist = np.array([[int((x != y).sum()) for y in b] for x in a]).reshape(n, m)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1) * gap
    d[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j - 1] + dist[i - 1, j - 1],
                d[i - 1, j] + gap,
                d[i, j - 1] + gap,
            )
    out, i, j = [], n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + dist[i - 1, j - 1]:
            if dist[i - 1, j - 1] <= thr:
                out.append((i - 1, j - 1, int(dist[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + gap:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def register(src: np.ndarray, tw: np.ndarray, work_px=1200):
    """2×3 affine mapping twin → src coordinates, and its RANSAC inlier count."""
    sa, sb = work_px / max(src.shape), work_px / max(tw.shape)
    a = cv2.resize(src, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    b = cv2.resize(tw, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(5000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(matches) < 20:
        return None, 0
    pb = np.float32([kb[mm.queryIdx].pt for mm in matches]) / sb
    pa = np.float32([ka[mm.trainIdx].pt for mm in matches]) / sa
    M, inl = cv2.estimateAffinePartial2D(
        pb, pa, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M, int(inl.sum()) if inl is not None else 0


def norm_gray(g: np.ndarray, valid: np.ndarray) -> np.ndarray:
    b = cv2.GaussianBlur(g, (3, 3), 0)
    lo, hi = np.percentile(b[valid > 0], [1, 99])
    return np.clip((b.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255)


def diff_regions(
    src: np.ndarray,
    warped: np.ndarray,
    valid: np.ndarray,
    slack_frac=0.004,
    close_frac=0.012,
    seed_frac=0.08,
    min_area_frac=0.0004,
):
    """Lettering regions of a registered pair: ``(regions, labels, changed_frac)``.

    A region is a closed component of the pixel diff kept when enough of it is
    *seed* — ink on one side with no ink within ``r`` px on the other, which a
    few-px misregistration of line art cannot produce. ``labels`` (uint8) holds
    region ``k`` as ``k + 1``, 0 elsewhere."""
    na, nb = norm_gray(src, valid), norm_gray(warped, valid)
    ia, ib = (na < 110).astype(np.uint8), (nb < 110).astype(np.uint8)
    r = max(3, int(round(min(src.shape) * slack_frac)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    only_a = ia & (1 - cv2.dilate(ib, ker)) & (valid > 0)
    only_b = ib & (1 - cv2.dilate(ia, ker)) & (valid > 0)
    seed = cv2.morphologyEx(only_a | only_b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    pix = ((np.abs(na - nb) > 70) & (valid > 0)).astype(np.uint8)
    k = max(9, int(round(min(src.shape) * close_frac)))
    mask = cv2.morphologyEx(
        pix | seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    seeds_per = np.bincount(lab[seed > 0], minlength=n)
    page_area = src.shape[0] * src.shape[1]
    regions, labels = [], np.zeros(src.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_frac * page_area or area > 0.35 * page_area:
            continue
        if seeds_per[i] < seed_frac * area or len(regions) == 254:
            continue
        labels[lab == i] = len(regions) + 1
        regions.append(
            {
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "seed": int(seeds_per[i]),
            }
        )
    return regions, labels, float(seed[valid > 0].mean())


def refine_ecc(src: np.ndarray, tw: np.ndarray, M: np.ndarray, work_px=1600):
    """``M`` refined by ECC: a full affine fit of the ``M``-warped twin to
    ``src`` at ``work_px``, composed onto ``M``. ORB's partial affine leaves a
    ~1 px misfit that lights every line in the diff; on 60 pages across the
    residual range ECC took the p90 residual from 10.0 to 6.5. Returns ``M``
    unchanged when ECC fails or wanders past 2 % of the page. A page already
    under ``work_px`` is fit at its own size.

    The fit is masked to where the warped twin covers the page: unmasked, a
    twin that leaves an uncovered strip drags the fit toward its black border
    (one such work: residual 6.6 before ECC, 8.5 after, 5.1 masked). ECC stops
    when the correlation moves < 1e-3; at 1e-6 some works oscillated to the
    iteration cap (~40 ms each) for < 0.03 px of change."""
    s = min(1.0, work_px / max(src.shape))
    size = src.shape[::-1]
    a = src
    b = cv2.warpAffine(tw, M, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full(tw.shape, 255, np.uint8), M, size)
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
    if s < 1:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, a.shape[::-1], interpolation=cv2.INTER_NEAREST)
    R = np.eye(2, 3, dtype=np.float32)
    try:
        _, R = cv2.findTransformECC(
            a.astype(np.float32),
            b.astype(np.float32),
            R,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3),
            mask,
            5,
        )
    except cv2.error:
        return M
    if (
        np.abs(R[:, 2]).max() > 0.02 * max(a.shape)
        or np.abs(R[:, :2] - np.eye(2)).max() > 0.02
    ):
        return M
    R[:, 2] /= s
    Ri = np.vstack([cv2.invertAffineTransform(R), [0, 0, 1]])
    return (Ri @ np.vstack([M, [0, 0, 1]]))[:2]


def grow_regions(
    labels: np.ndarray, diff: np.ndarray, thr=30.0, reach_frac=0.03
) -> np.ndarray:
    """``labels`` grown over the pixel diff connected to them: each pixel of a
    ``diff > thr`` component that touches a region joins its nearest region,
    up to ``reach_frac`` × the short edge away. The seed-based components stop
    short of a glyph's faint strokes and outline halo, and a paste by them
    leaves half-swapped glyphs."""
    seeds = (labels > 0) & (labels != 255)
    if not seeds.any():
        return labels
    d = cv2.morphologyEx(
        (diff > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
    )
    d = cv2.dilate(d, np.ones((3, 3), np.uint8)) & (labels != 255)
    n, cc = cv2.connectedComponents(((d > 0) | seeds).astype(np.uint8), connectivity=8)
    touched = np.zeros(n, bool)
    touched[cc[seeds]] = True
    touched[0] = False
    dist, idx = cv2.distanceTransformWithLabels(
        (~seeds).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros(idx.max() + 1, np.uint8)
    lut[idx[seeds]] = labels[seeds]
    grow = (
        touched[cc] & ~seeds & (labels == 0) & (dist <= reach_frac * min(labels.shape))
    )
    out = labels.copy()
    out[grow] = lut[idx[grow]]
    return out
