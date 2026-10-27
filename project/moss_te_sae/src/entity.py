"""Does the MOSS TE (Qwen3.5-9B, bf16) know a character?

Three reads per name, against known (hatsune miku, kasane teto) and invented controls:

- ``qa``: greedy chat answer (thinking off) to "who is X, what does X look like";
- ``next``: next-token top-k after attribute cloze sentences (raw text, no template);
- ``sae``: in a MOSS layout-JSON caption encoded exactly as MOSS does, the top SAE
  features at the name's last token (layers 15 / 31), each read through the logit
  lens ``W_U((1+w)⊙d)``, plus top-100 overlap between names.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from pathlib import Path

import torch

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader

QA = "Who is {name}? In 2-3 sentences: where the character comes from, and her appearance (hair colour, hairstyle, eye colour, outfit, signature item)."
CLOZE = (
    "{name} is an anime-style character whose hair colour is",
    "{name}'s hairstyle is a",
    "{name} is always holding her",
    "{name} is a derivative character of the",
)
FAKE = "okuda reni"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


@torch.no_grad()
def greedy(model, tok, text, max_new):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].cuda()
    out = model(input_ids=ids, use_cache=True)
    past, gen = out.past_key_values, []
    nxt = out.logits[0, -1].argmax()
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    while len(gen) < max_new and int(nxt) not in eos:
        gen.append(int(nxt))
        out = model(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1].argmax()
    return tok.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def next_tokens(model, tok, text, k=8):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    p = model(input_ids=ids, use_cache=False).logits[0, -1].float().softmax(-1)
    v, i = p.topk(k)
    return [f"{tok.decode([int(t)])!r}:{float(x):.2f}" for x, t in zip(v, i)]


def name_ends(tok, formatted, enc, name):
    """Token index of the last token of every occurrence of ``name``."""
    offs = enc.offset_mapping[0].tolist()
    ends, start = [], formatted.find(name)
    while start >= 0:
        end = start + len(name)
        ends.append(max(i for i, (a, b) in enumerate(offs) if a < end and b > start))
        start = formatted.find(name, end)
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inst_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_pe/inst")
    )
    ap.add_argument(
        "--sae_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_sae")
    )
    ap.add_argument(
        "--caption",
        default="post_image_dataset/resized/channel_(caststation)/channel6.layout.txt",
    )
    ap.add_argument("--target", default="akita neru")
    ap.add_argument("--known", default="hatsune miku,kasane teto")
    ap.add_argument("--layers", default="15,31")
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    layers = [int(x) for x in args.layers.split(",")]
    known = args.known.split(",")
    out_path = (
        args.out
        or args.sae_dir
        / "entity"
        / f"{args.target.replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = loader.load_tokenizer(default_moss_dir())
    model = AutoModelForCausalLM.from_pretrained(
        args.inst_dir, dtype=torch.bfloat16
    ).eval()
    model.lm_head.cuda()
    loader.offload_by_layer(
        SimpleNamespace(language_model=model.model),
        torch.device("cuda"),
        loader.te_reserve_gb(1024),
    )
    W_U = model.lm_head.weight.detach().cpu().float()
    gain = 1.0 + model.model.norm.weight.detach().float().cpu()
    result = {"target": args.target, "qa": {}, "next": {}, "sae": {}}

    names = [args.target, *known, FAKE]
    title = lambda n: n.title()  # noqa: E731
    for n in [*names, "亞北ネル"]:
        result["qa"][n] = greedy(model, tok, QA.format(name=title(n)), args.max_new)
        log(f"[qa] {n}: {result['qa'][n]}")
    result["qa"]["triple baka"] = greedy(
        model,
        tok,
        "What is 'Triple Baka' (vocaloid)? Who are the three characters?",
        args.max_new,
    )
    log(f"[qa] triple baka: {result['qa']['triple baka']}")
    for n in names:
        result["next"][n] = {
            c: next_tokens(model, tok, c.format(name=title(n))) for c in CLOZE
        }
        for c, v in result["next"][n].items():
            log(f"[next] {c.format(name=title(n))!r} → {' '.join(v)}")

    caption = resolve_under_home(args.caption).read_text(encoding="utf-8").strip()
    grabbed = {}
    hooks = [
        model.model.layers[li].register_forward_hook(
            lambda m, i, o, li=li: grabbed.__setitem__(
                li, (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
            )
        )
        for li in layers
    ]
    captured = {}
    for variant, text in (
        ("real", caption),
        ("fake", caption.replace(args.target, FAKE)),
    ):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        enc = tok(formatted, return_tensors="pt", return_offsets_mapping=True)
        with torch.no_grad():
            model.model(input_ids=enc.input_ids.cuda(), use_cache=False)
        for n in [args.target, *known] if variant == "real" else [FAKE]:
            for j, t in enumerate(name_ends(tok, formatted, enc, n)):
                ctx = tok.decode(enc.input_ids[0, max(0, t - 8) : t + 1])
                captured[f"{n}#{j}"] = {
                    "ctx": ctx,
                    **{li: grabbed[li][t] for li in layers},
                }
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    for li in layers:
        sae = torch.load(
            args.sae_dir / f"layer{li}.sae.pt", map_location="cpu", weights_only=True
        )
        sae = {k: v.float() for k, v in sae.items()}
        lens_cache = {}

        def lens(f):
            if f not in lens_cache:
                top = (W_U @ (gain * sae["W_dec"][:, f])).topk(8).indices
                lens_cache[f] = [tok.decode([int(t)]) for t in top]
            return lens_cache[f]

        sets = {}
        result["sae"][li] = {}
        for key, cap in captured.items():
            pre = cap[li] @ sae["W_enc"].T + sae["b_enc"]
            vals, idx = pre.topk(100)
            sets[key] = set(idx.tolist())
            top = [
                {"f": int(f), "act": round(float(a), 1), "lens": lens(int(f))}
                for a, f in zip(vals[:12], idx[:12])
            ]
            result["sae"][li][key] = {"ctx": cap["ctx"], "top": top}
            log(f"[sae L{li}] {key}  …{cap['ctx']!r}")
            for r in top:
                log(f"     f{r['f']:5d} {r['act']:7.1f}  {r['lens']}")
        keys = list(sets)
        overlap = {
            f"{a} ∩ {b}": len(sets[a] & sets[b])
            for i, a in enumerate(keys)
            for b in keys[i + 1 :]
        }
        result["sae"][li]["overlap_top100"] = overlap
        log(f"[sae L{li}] overlap {overlap}")
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
