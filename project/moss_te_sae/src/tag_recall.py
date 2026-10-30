"""Anima Tagger recall of a reference appearance set over a folder of generations.

Images are grouped by the condition prefix of their name (``…_space<COND>_s<seed>``,
the ``example`` key ``generate.py`` writes). Per image: hits among ``--ref`` (kept tags)
and the mean tagger probability over ``--ref``; per condition: the means over seeds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from PIL import Image

from library.env import resolve_under_home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True, help="comma-separated reference tags")
    ap.add_argument("--tagger", default="models/captioners/anima-tagger-dbv4")
    args = ap.parse_args()
    from anime_tools.tagger.tagger import AnimaTagger

    ref = [t.strip().replace("_", " ") for t in args.ref.split(",") if t.strip()]
    tagger = AnimaTagger(resolve_under_home(args.tagger), device="cuda")
    root = resolve_under_home(args.dir)
    images = sorted(p for p in root.rglob("*.png"))
    rows, by_cond = {}, defaultdict(list)
    for p in images:
        out = tagger.predict(Image.open(p).convert("RGB"))
        scores, kept = out["scores"], out["kept"]
        in_vocab = [t for t in ref if t in scores]
        hits = [t for t in in_vocab if t in kept]
        m = re.search(r"space([A-Za-z]+_[A-Za-z]+)_s(\d+)", p.stem)
        cond = m.group(1) if m else p.stem
        row = {
            "cond": cond,
            "recall": len(hits) / max(len(in_vocab), 1),
            "mean_prob": sum(scores[t] for t in in_vocab) / max(len(in_vocab), 1),
            "missing": [t for t in in_vocab if t not in kept],
            "extra_top": [
                t
                for t, _ in sorted(kept.items(), key=lambda kv: -kv[1])
                if t not in ref
            ][:12],
        }
        rows[p.name] = row
        by_cond[cond].append(row)
        print(
            f"{p.name}: recall {row['recall']:.2f} prob {row['mean_prob']:.2f} missing {row['missing']}",
            flush=True,
        )
    print(f"\nref in vocab: {sum(t in scores for t in ref)}/{len(ref)}")
    summary = {}
    for cond, rs in sorted(by_cond.items()):
        summary[cond] = {
            "recall": sum(r["recall"] for r in rs) / len(rs),
            "mean_prob": sum(r["mean_prob"] for r in rs) / len(rs),
            "n": len(rs),
        }
        print(
            f"{cond:10s} n={len(rs)} recall {summary[cond]['recall']:.2f}  mean prob {summary[cond]['mean_prob']:.2f}"
        )
    (root / "tag_recall.json").write_text(
        json.dumps({"ref": ref, "summary": summary, "images": rows}, indent=1)
    )


if __name__ == "__main__":
    main()
