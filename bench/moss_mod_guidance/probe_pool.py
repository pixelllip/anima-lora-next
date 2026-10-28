"""Go/no-go probe for modulation guidance on MOSS-Image: does a pooled text vector
move more for quality tags than for content tags?

Anima's verdict (docs/anima/findings/mod_guidance_quality_tag_axis.md): the distilled
pooled head is a tag-agnostic ~3x amplifier, so the quality-selectivity it can carry
is fixed upstream by the pooled encoder output. Anima measured
``rel_dpool = ||pool(base+tag) - pool(base)|| / ||pool(base)||`` at 0.059 (quality)
vs 0.031 (content), ratio 1.89, and the quality direction's cross-content
consistency at cos 0.814. This probe takes the same measurements on MOSS before
anything is distilled.

Spaces: ``te`` = Qwen3.5-9B ``last_hidden_state`` (4096, what the cache stores),
``emb`` = after the DiT's ``text_embedder`` (3072, the analog of Anima's
post-LLM-adapter ``crossattn_emb``). Pools: ``max_all`` (every token, chat template
included), ``max_content`` / ``mean_content`` (the user text span only).
Formats: ``layout`` (the default MOSS prompt, tags spliced into the raw caption
before conversion), ``plain`` (the raw tag caption), ``nl`` (English prompts with
quality phrases appended, paper-style).

    make daemon-run ARGS="--stall-timeout 900 bench/moss_mod_guidance/probe_pool.py"
"""

import argparse
import glob
import itertools
import json
import random

import torch
from safetensors import safe_open

from bench._common import make_run_dir, write_result
from bench._moss import PROMPTS
from library.env import resolve_under_home

QUALITY_TAGS = [
    "score_9",
    "score_8_up",
    "masterpiece",
    "best quality",
    "absurdres",
    "highres",
]
CONTENT_TAGS = [
    "holding a sword",
    "red dress",
    "sitting on a bench",
    "wearing glasses",
    "blue sky",
    "a wooden table",
]
NL_QUALITY = [
    "high quality",
    "masterpiece",
    "highly detailed",
    "aesthetic",
    "professional illustration",
    "best quality",
]
AXIS_POS = ["masterpiece", "best quality", "score_9"]
AXIS_NEG = ["worst quality", "low quality", "score_1"]
NL_AXIS_POS = "Masterpiece, best quality, highly detailed."
NL_AXIS_NEG = "Worst quality, low quality, blurry."
NL_KEYS = ["anime_sunset", "ja_manga", "ko_neon", "zh_fridge"]
RATINGS = {"general", "sensitive", "questionable", "explicit", "nsfw", "safe"}
EPS = 1e-8


def load_captions(n, seed, min_chars, max_chars):
    files = sorted(
        f
        for f in glob.glob(
            str(resolve_under_home("post_image_dataset/resized")) + "/**/*.txt",
            recursive=True,
        )
        if not f.endswith((".variants.txt", ".layout.txt"))
    )
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            cap = fh.read().strip()
        first = cap.split(",", 1)[0].strip().lower()
        if min_chars <= len(cap) <= max_chars and first in RATINGS:
            out.append(cap)
        if len(out) >= n:
            break
    return out


def splice(caption, tags, quality):
    """Quality/meta tags go right after the rating (where the tagger emits them);
    content tags join the end of the flat bag, before any position clauses."""
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    flat = list(parsed.flat_tags)
    if quality and flat and flat[0].lower() in RATINGS:
        flat = flat[:1] + list(tags) + flat[1:]
    elif quality:
        flat = list(tags) + flat
    else:
        flat = flat + list(tags)
    return compose_caption(flat, parsed.clauses)


def nl_add(base, phrase):
    return f"{base} {phrase[0].upper()}{phrase[1:]}."


def build_specs(captions, content_front=False):
    """``{fmt: {"rows": [(base, tagged, tag, is_quality)], "axis": [(base, pos, neg)]}}``
    in each format's raw text (layout conversion happens later)."""
    specs = {}
    rows, axis = [], []
    for cap in captions:
        for tag in QUALITY_TAGS:
            rows.append((cap, splice(cap, [tag], True), tag, True))
        for tag in CONTENT_TAGS:
            rows.append((cap, splice(cap, [tag], content_front), tag, False))
        axis.append((cap, splice(cap, AXIS_POS, True), splice(cap, AXIS_NEG, True)))
    specs["plain"] = {"rows": rows, "axis": axis}
    rows, axis = [], []
    for key in NL_KEYS:
        base = PROMPTS[key]
        for tag in NL_QUALITY:
            rows.append((base, nl_add(base, tag), tag, True))
        for tag in CONTENT_TAGS:
            rows.append((base, nl_add(base, tag), tag, False))
        axis.append((base, f"{base} {NL_AXIS_POS}", f"{base} {NL_AXIS_NEG}"))
    specs["nl"] = {"rows": rows, "axis": axis}
    return specs


def content_span(tok, text):
    """``(n_prefix, n_suffix)`` tokens of the chat template around ``text``."""
    from library.moss.loader import prompt_ids

    sentinel = "⁣SPAN⁣"
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    head, tail = formatted.split(sentinel)
    n_head = len(tok(head, add_special_tokens=False).input_ids)
    n_tail = len(tok(tail, add_special_tokens=False).input_ids)
    n = prompt_ids(tok, text).input_ids.shape[1]
    return n_head, n - n_tail


def load_text_embedder(model_dir, device):
    from library.moss.model import _RoPE, _TextProjection
    from library.moss.requests import variant_dir

    root = variant_dir(model_dir, "base") / "transformer"
    cfg = json.loads((root / "config.json").read_text())
    emb = _TextProjection(cfg["text_dim"], cfg["dim"], cfg["ffn_dim"], cfg["num_heads"])
    index = json.loads(
        (root / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )
    shards = {
        f for k, f in index["weight_map"].items() if k.startswith("text_embedder.")
    }
    state = {}
    for shard in shards:
        with safe_open(str(root / shard), "pt") as s:
            for k in s.keys():
                if k.startswith("text_embedder."):
                    state[k[len("text_embedder.") :]] = s.get_tensor(k)
    emb.load_state_dict(state, strict=True)
    rope = _RoPE(cfg["dim"] // cfg["num_heads"], cfg["rope_base"])
    return emb.to(device, torch.bfloat16).eval(), rope.to(device)


@torch.no_grad()
def embed_text(emb, rope, hidden, device):
    n = hidden.shape[0]
    # Text rope depends on the text length only; a 1x1 image grid sits in front.
    r = rope(1, 1, n, device)[:, 1 : 1 + n].to(torch.bfloat16)
    return emb(hidden[None].to(device), r)[0].float().cpu()


def pools(x, span):
    a, b = span
    c = x[a:b].float()
    return {
        "max_all": x.float().max(0).values,
        "max_content": c.max(0).values,
        "mean_content": c.mean(0),
    }


def rel(a, b):
    return float((a - b).norm() / (b.norm() + EPS))


def mean_pairwise_cos(vecs):
    cs = [
        float(torch.nn.functional.cosine_similarity(u, v, dim=0))
        for u, v in itertools.combinations(vecs, 2)
    ]
    return sum(cs) / len(cs), min(cs)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n_captions", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_chars", type=int, default=120)
    p.add_argument("--max_chars", type=int, default=400)
    p.add_argument("--text_encoder", default="int8")
    p.add_argument("--model_dir", default="")
    p.add_argument(
        "--content_front",
        action="store_true",
        help="splice content tags after the rating too (position control: the TE is causal)",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    from library.moss.cache import caption_converter
    from library.moss.loader import encode_prompts, load_tokenizer

    device = "cuda"
    captions = load_captions(args.n_captions, args.seed, args.min_chars, args.max_chars)
    print(f"[probe] {len(captions)} captions")
    specs = build_specs(captions, args.content_front)
    to_layout = caption_converter("layout_json")
    specs["layout"] = {
        "rows": [
            (to_layout(b), to_layout(t), tag, q)
            for b, t, tag, q in specs["plain"]["rows"]
        ],
        "axis": [tuple(to_layout(x) for x in row) for row in specs["plain"]["axis"]],
    }

    texts = []
    for s in specs.values():
        for b, t, _, _ in s["rows"]:
            texts += [b, t]
        for row in s["axis"]:
            texts += list(row)
    hidden = encode_prompts(
        args.model_dir, texts, device, log=None, precision=args.text_encoder
    )
    print(f"[probe] encoded {len(hidden)} distinct prompts")
    tok = load_tokenizer(args.model_dir)
    spans = {t: content_span(tok, t) for t in hidden}

    emb, rope = load_text_embedder(args.model_dir, device)
    feats = {}
    for t, h in hidden.items():
        feats[t] = {
            "te": pools(h, spans[t]),
            "emb": pools(embed_text(emb, rope, h, device), spans[t]),
        }
    del emb
    torch.cuda.empty_cache()

    metrics, table = {}, []
    for fmt, s in specs.items():
        for space in ("te", "emb"):
            for pool in ("max_all", "max_content", "mean_content"):
                key = f"{fmt}.{space}.{pool}"
                by = {True: [], False: []}
                for b, t, tag, q in s["rows"]:
                    r = rel(feats[t][space][pool], feats[b][space][pool])
                    by[q].append(r)
                    table.append(
                        {
                            "fmt": fmt,
                            "space": space,
                            "pool": pool,
                            "tag": tag,
                            "is_quality": q,
                            "rel_dpool": r,
                        }
                    )
                qm = sum(by[True]) / len(by[True])
                cm = sum(by[False]) / len(by[False])
                dirs = [
                    feats[pos][space][pool] - feats[neg][space][pool]
                    for _, pos, neg in s["axis"]
                ]
                cos_mean, cos_min = mean_pairwise_cos(dirs)
                metrics[f"{key}.quality_rel_dpool"] = qm
                metrics[f"{key}.content_rel_dpool"] = cm
                metrics[f"{key}.ratio"] = qm / (cm + EPS)
                metrics[f"{key}.axis_cos_mean"] = cos_mean
                metrics[f"{key}.axis_cos_min"] = cos_min

    run_dir = make_run_dir("moss_mod_guidance", args.label)
    (run_dir / "rows.json").write_text(json.dumps(table, indent=1))
    (run_dir / "prompts.json").write_text(
        json.dumps(
            {f: s["axis"][:2] for f, s in specs.items()}, indent=1, ensure_ascii=False
        )
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["rows.json", "prompts.json"],
        extra={
            "anima_reference": {
                "quality_rel_dpool": 0.059,
                "content_rel_dpool": 0.031,
                "ratio": 1.89,
                "axis_cos_mean": 0.814,
            }
        },
    )
    print(
        f"\n{'format.space.pool':32s} {'qual':>7s} {'cont':>7s} {'ratio':>6s} {'axis cos':>9s}"
    )
    for fmt in specs:
        for space in ("te", "emb"):
            for pool in ("max_all", "max_content", "mean_content"):
                k = f"{fmt}.{space}.{pool}"
                print(
                    f"{k:32s} {metrics[k + '.quality_rel_dpool']:7.4f} "
                    f"{metrics[k + '.content_rel_dpool']:7.4f} {metrics[k + '.ratio']:6.2f} "
                    f"{metrics[k + '.axis_cos_mean']:6.3f}/{metrics[k + '.axis_cos_min']:.2f}"
                )
    print("anima (crossattn max pool): qual 0.059 cont 0.031 ratio 1.89 axis cos 0.814")
    print(f"[probe] {run_dir}")


if __name__ == "__main__":
    main()
