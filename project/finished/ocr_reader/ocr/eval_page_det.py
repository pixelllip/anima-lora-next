#!/usr/bin/env python3
"""Page-level text detection on Manga109-s COO test pages: OvisOCR3's boxed
prompts against the AnimeText YOLO detector.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/finished/ocr_reader/ocr/eval_page_det.py --reader ovis3"
    … --reader ovis3 --task document_structured --name ovis3_struct
    … --reader animetext
    .venv/bin/python project/finished/ocr_reader/ocr/eval_page_det.py --rescore animetext ovis3   # CPU

Ground truth is exhaustive on these pages: every Manga109-s ``<text>`` box
(``speech``) and every COO onomatopoeia polygon (``sfx``, as its bbox). The
three sources box at different grains — Manga109 a whole balloon, YOLO a
column (``inner``), Ovis ``word_spotting`` a word — so the scores are
coverage-based, not IoU:

* **recall** — a GT box is *found* when the union of predicted boxes covers
  ≥ ``COVER`` of its area;
* **hit** — looser: some predicted box has its centre inside the GT box
  (word boxes are tight, so a fully read balloon can sit under 0.5 cover);
* **precision** — a predicted box is *on text* when ≥ ``COVER`` of its area
  lies inside the union of GT boxes (each dilated by ``DILATE`` px; an
  unannotated scribble counts against it).

For a reader that also transcribes (Ovis), **read** pairs each found GT box
with the predicted texts whose box centre falls inside it, joined in output
order, and scores ``exact_key`` equality with the GT text (vertical reading
order inside a balloon is the model's).

AnimeText was trained on 735k anime / manga pages; whether Manga109 is among
them is not stated, so its numbers here may be in-domain.

Writes ``output/ocr/eval/det_<name>.jsonl`` (one row per page: boxes, texts,
raw) and ``reports/ocr_eval_det_<name>.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402
import textnorm  # noqa: E402

COVER = 0.5
DILATE = 8
N_PAGES = 200
_BOX_TEXT = re.compile(
    r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    r'(?:[^{}]*?"(?:text|content)"\s*:\s*"((?:[^"\\]|\\.)*)")?'
)


def gt_pages(n: int, seed: int = 0) -> dict[tuple[str, int], list[m109.Line]]:
    """``n`` annotated COO-test pages, fixed sample."""
    by_page: dict[tuple[str, int], list[m109.Line]] = defaultdict(list)
    for book in m109.load_split()["test"]:
        for ln in list(m109.iter_text(book)) + list(m109.iter_coo(book)):
            by_page[(ln.book, ln.page)].append(ln)
    keys = sorted(by_page)
    keys = sorted(random.Random(seed).sample(keys, min(n, len(keys))))
    return {k: by_page[k] for k in keys}


def bbox(poly) -> tuple[int, int, int, int]:
    xs, ys = poly[0::2], poly[1::2]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def parse_boxes(raw: str, w: int, h: int) -> list[dict]:
    """``[{"bbox": [0..1000) ×4, "text"|"content": …}]`` → pixel boxes. A
    truncated array is salvaged element by element."""
    try:
        items = json.loads(raw)
        items = [
            (it["bbox"], it.get("text", it.get("content", "")))
            for it in items
            if isinstance(it, dict) and len(it.get("bbox", [])) == 4
        ]
    except (json.JSONDecodeError, TypeError, KeyError):
        items = [
            ([int(m[i]) for i in range(1, 5)], json.loads(f'"{m[5]}"') if m[5] else "")
            for m in _BOX_TEXT.finditer(raw)
        ]
    out, seen = [], set()
    for b, t in items:
        if (tuple(b), t) in seen:  # a runaway repeats one element to the cap
            continue
        seen.add((tuple(b), t))
        x0, y0, x1, y1 = (int(v) for v in b)
        x0, x1 = sorted((x0 * w / 1000, x1 * w / 1000))
        y0, y1 = sorted((y0 * h / 1000, y1 * h / 1000))
        if x1 - x0 >= 1 and y1 - y0 >= 1:
            out.append(dict(box=[int(x0), int(y0), int(x1), int(y1)], text=str(t)))
    return out


def run_ovis(pages, a) -> list[dict]:
    import cv2

    reader = ev.READERS["ovis3"](a.ckpt, a.device)
    reader.prompt = (
        (Path(a.ckpt or m109.REPO / reader.DEFAULT) / "prompts" / f"{a.task}.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    reader.max_tokens = a.max_new_tokens
    reader.filter_imgtags = False
    rows = []
    for s in range(0, len(pages), a.bs):
        chunk = pages[s : s + a.bs]
        imgs = [cv2.imread(str(m109.page_path(b, p))) for b, p in chunk]
        raws = reader.read(imgs, [None] * len(imgs), a.bs)
        for (b, p), img, raw in zip(chunk, imgs, raws):
            h, w = img.shape[:2]
            n_tok = len(reader.proc.tokenizer(raw)["input_ids"])
            rows.append(
                dict(
                    book=b,
                    page=p,
                    w=w,
                    h=h,
                    raw=raw,
                    runaway=n_tok >= a.max_new_tokens - 8,
                    preds=parse_boxes(raw, w, h),
                )
            )
        print(f"{len(rows)} / {len(pages)} pages", flush=True)
    return rows


def run_animetext(pages, a) -> list[dict]:
    import cv2
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device=a.device)
    rows = []
    for s in range(0, len(pages), 16):
        chunk = pages[s : s + 16]
        imgs = [cv2.imread(str(m109.page_path(b, p))) for b, p in chunk]
        for (b, p), img, boxes in zip(chunk, imgs, det.detect_batch(imgs)):
            h, w = img.shape[:2]
            preds = [dict(box=[int(v) for v in bx], text="") for bx in boxes]
            rows.append(dict(book=b, page=p, w=w, h=h, raw="", preds=preds))
    return rows


def _mask(boxes, h, w, pad=0) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)] = 1
    return m


def score(rows: list[dict], gt) -> tuple[list[dict], list[dict]]:
    """Per-GT-line and per-pred rows."""
    lines, preds = [], []
    for r in rows:
        h, w = r["h"], r["w"]
        pb = [p["box"] for p in r["preds"]]
        pm = _mask(pb, h, w)
        gl = gt[(r["book"], r["page"])]
        gb = [bbox(ln.poly) for ln in gl]
        gm = _mask(gb, h, w, DILATE)
        for ln, (x0, y0, x1, y1) in zip(gl, gb):
            area = max(1, (x1 - x0) * (y1 - y0))
            cov = pm[y0:y1, x0:x1].sum() / area
            inside = [
                p["text"]
                for p in r["preds"]
                if x0 <= (p["box"][0] + p["box"][2]) / 2 <= x1
                and y0 <= (p["box"][1] + p["box"][3]) / 2 <= y1
            ]
            read = "".join(inside)
            hit = any(
                x0 <= (b[0] + b[2]) / 2 <= x1 and y0 <= (b[1] + b[3]) / 2 <= y1
                for b in pb
            )
            lines.append(
                dict(
                    kind=ln.kind,
                    found=bool(cov >= COVER),
                    hit=hit,
                    cov=float(cov),
                    gt=ln.text,
                    read=read,
                    exact=bool(read)
                    and textnorm.exact_key(read) == textnorm.exact_key(ln.text),
                )
            )
        for p in r["preds"]:
            x0, y0, x1, y1 = p["box"]
            area = max(1, (x1 - x0) * (y1 - y0))
            preds.append(dict(on_text=bool(gm[y0:y1, x0:x1].sum() / area >= COVER)))
    return lines, preds


def table(name: str, rows, gt, wall: float | None) -> str:
    lines, preds = score(rows, gt)
    n_pages = len(rows)
    md = [
        f"### `{name}` — {n_pages} pages"
        + (f", {wall / 60:.1f} min" if wall is not None else ""),
        "",
        f"pred boxes {len(preds)} ({len(preds) / max(1, n_pages):.1f} / page), "
        f"precision {100 * np.mean([p['on_text'] for p in preds] or [0]):.1f} %, "
        f"runaway pages {sum(r.get('runaway', False) for r in rows)}",
        "",
        "| kind | GT | found (cov ≥ 0.5) | hit (centre) | mean cov | read exact |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("speech", "sfx"):
        k = [x for x in lines if x["kind"] == kind]
        f = sum(x["found"] for x in k)
        e = sum(x["exact"] for x in k)
        c = sum(x["hit"] for x in k)
        md.append(
            f"| {kind} | {len(k)} | {f} ({100 * f / max(1, len(k)):.1f} %) | "
            f"{c} ({100 * c / max(1, len(k)):.1f} %) | {np.mean([x['cov'] for x in k]):.3f} | {e} ({100 * e / max(1, len(k)):.1f} %) |"
        )
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=["ovis3", "animetext"])
    ap.add_argument("--task", default="word_spotting")
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--pages", type=int, default=N_PAGES)
    ap.add_argument("--rescore", nargs="+", metavar="NAME")
    a = ap.parse_args()

    gt = gt_pages(a.pages)
    if a.rescore:
        out = [f"# page detection — Manga109-s COO test, {len(gt)} pages\n"]
        for name in a.rescore:
            rows = [
                json.loads(ln)
                for ln in (ev.OUT / f"det_{name}.jsonl").open(encoding="utf-8")
            ]
            out.append(table(name, rows, gt, None))
        print("\n".join(out))
        return
    if not a.reader:
        ap.error("--reader is required unless --rescore")
    name = a.name or (a.reader if a.task == "word_spotting" else f"{a.reader}_{a.task}")

    t0 = time.time()
    rows = (run_ovis if a.reader == "ovis3" else run_animetext)(list(gt), a)
    wall = time.time() - t0
    ev.OUT.mkdir(parents=True, exist_ok=True)
    with (ev.OUT / f"det_{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = table(name, rows, gt, wall)
    (ev.REPORTS / f"ocr_eval_det_{name}.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
