"""Held-out teacher-forced comparison: PE vs Qwen3.5-9B + extracted LoRA variants.

One ``Qwen3_5ForCausalLM`` skeleton (INT8 ConvRot W8A8 decoder linears, bf16
lm_head) is rebuilt in place per variant from bf16 weights kept on the CPU:
``Q(W_inst + up[:, :r]·down[:r])`` against the teacher ``Q(W_pe)``, so both sides
carry the same round-to-nearest quantization. The teacher generates rewrites for the
held-out prompts; every variant is then scored on those tokens (response positions
only, thinking and answer split at ``</think>``):

  kl    Σ_{teacher top-64} p_t (log p_t − log p_s)
  nll   −log p_s(sampled token)
  top1  argmax agreement with the teacher

``--free_gen`` variants also generate the first 16 prompts themselves (JSON validity,
length). Variant names: ``inst``, ``r<k>`` (all linears rank k), ``r<k>d<j>``
(down_proj rank j), suffix ``+embed`` (PE's embed_tokens), ``-small`` (keep inst's
norms / conv1d).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from safetensors import safe_open

from analyze import PREFIX, Ckpt
from calib import generate, load_prompts
from library.moss import int8
from library.moss.fp8 import LINEARS

TOPK = 64
LM_RANK = 64


def runtime_to_ckpt(name: str) -> str:
    return name if name.startswith("lm_head") else PREFIX + name[len("model.") :]


def build(base_dir: Path, log):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    keep, quant = {}, {}
    for name, module in list(model.named_modules()):
        for cname, child in list(module.named_children()):
            full = f"{name}.{cname}"
            if not isinstance(child, torch.nn.Linear) or ".layers." not in full:
                continue
            if full.split(".layers.", 1)[1].split(".", 1)[1] not in LINEARS:
                continue
            keep[runtime_to_ckpt(full) + ".weight"] = child.weight.data
            q, scale = int8.quantize_weight(child.weight.data)
            mod = int8.Int8ConvRotLinear(q, scale, int8.GROUP, True)
            setattr(module, cname, mod)
            quant[runtime_to_ckpt(full) + ".weight"] = mod
    params = {
        runtime_to_ckpt(n): p
        for n, p in model.named_parameters()
        if not isinstance(_owner(model, n), int8.Int8ConvRotLinear)
    }
    for k, p in params.items():
        keep[k] = p.data.clone()
    log(f"[build] {len(quant)} int8 linears, {len(params)} other params")
    return model.to("cuda"), keep, quant, params


def _owner(model, pname):
    return model.get_submodule(pname.rsplit(".", 1)[0])


def parse_variant(v: str):
    if v in ("pe", "inst"):
        return {"kind": v}
    m = re.fullmatch(r"r(\d+)(?:d(\d+))?((?:[+-]\w+)*)", v)
    if not m:
        raise ValueError(v)
    flags = m.group(3)
    return {
        "kind": "lora",
        "rank": int(m.group(1)),
        "down": int(m.group(2) or m.group(1)),
        "embed": "+embed" in flags,
        "small": "-small" not in flags,
    }


@torch.no_grad()
def set_variant(spec, keep, quant, params, pe: Ckpt, lora, log):
    t0 = time.time()
    for key, mod in quant.items():
        if spec["kind"] == "pe":
            w = pe.get(key).cuda().float()
        else:
            w = keep[key].cuda().float()
            if spec["kind"] == "lora":
                r = (
                    spec["down"]
                    if key.endswith("mlp.down_proj.weight")
                    else spec["rank"]
                )
                base = key[: -len(".weight")]
                up = lora.get_tensor(f"{base}.lora_up")[:, :r].cuda().float()
                down = lora.get_tensor(f"{base}.lora_down")[:r].cuda().float()
                w = w + up @ down
        q, scale = int8.quantize_weight(w)
        mod.weight.data.copy_(q)
        mod.weight_scale.copy_(scale.float().reshape(-1, 1))
    for key, p in params.items():
        if spec["kind"] == "pe":
            v = pe.get(key)
        elif key == "lm_head.weight" and spec["kind"] == "lora":
            up = lora.get_tensor("lm_head.lora_up")[:, :LM_RANK].cuda().float()
            down = lora.get_tensor("lm_head.lora_down")[:LM_RANK].cuda().float()
            for a in range(0, p.shape[0], 16384):
                w = keep[key][a : a + 16384].cuda().float() + up[a : a + 16384] @ down
                p.data[a : a + 16384].copy_(w)
            continue
        elif "embed_tokens" in key:
            v = pe.get(key) if spec.get("embed") else keep[key]
        elif spec["kind"] == "lora" and spec["small"]:
            v = lora.get_tensor(f"small.{key}")
        else:
            v = keep[key]
        p.data.copy_(v.to(p.device, p.dtype))
    log(f"[variant] {spec} set in {time.time() - t0:.0f}s")


def think_split(tok, gen_ids):
    end = tok.convert_tokens_to_ids("</think>")
    return gen_ids.index(end) + 1 if end in gen_ids else len(gen_ids)


@torch.no_grad()
def response_logprobs(model, row, chunk=256):
    """Yields (log_softmax rows for response positions) chunk by chunk."""
    ids = row["prompt_ids"] + row["gen_ids"]
    p0 = len(row["prompt_ids"])
    h = model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    h = h.last_hidden_state[0, p0 - 1 : len(ids) - 1]
    for a in range(0, h.shape[0], chunk):
        yield a, model.lm_head(h[a : a + chunk]).float().log_softmax(-1)


@torch.no_grad()
def teacher_targets(model, rows):
    out = []
    for row in rows:
        tgt = torch.tensor(row["gen_ids"], device="cuda")
        lps, idxs, own = [], [], []
        for a, lp in response_logprobs(model, row):
            top = lp.topk(TOPK, -1)
            lps.append(top.values.cpu())
            idxs.append(top.indices.cpu())
            own.append(lp.gather(1, tgt[a : a + lp.shape[0], None])[:, 0].cpu())
        out.append((torch.cat(lps), torch.cat(idxs), torch.cat(own)))
    return out


@torch.no_grad()
def score(model, rows, targets, split_at):
    acc = {s: {"kl": 0.0, "nll": 0.0, "top1": 0.0, "n": 0} for s in ("think", "answer")}
    for row, (t_lp, t_idx, _), cut in zip(rows, targets, split_at):
        tgt = torch.tensor(row["gen_ids"], device="cuda")
        for a, lp in response_logprobs(model, row):
            n = lp.shape[0]
            tl, ti = t_lp[a : a + n].cuda(), t_idx[a : a + n].cuda()
            s_top = lp.gather(1, ti)
            kl = (tl.exp() * (tl - s_top)).sum(1)
            nll = -lp.gather(1, tgt[a : a + n, None])[:, 0]
            top1 = (lp.argmax(1) == ti[:, 0]).float()
            pos = torch.arange(a, a + n, device="cuda")
            for s, mask in (("think", pos < cut), ("answer", pos >= cut)):
                acc[s]["kl"] += float(kl[mask].sum())
                acc[s]["nll"] += float(nll[mask].sum())
                acc[s]["top1"] += float(top1[mask].sum())
                acc[s]["n"] += int(mask.sum())
    return {
        s: {k: v / max(d["n"], 1) for k, v in d.items() if k != "n"} | {"n": d["n"]}
        for s, d in acc.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--base_dir", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--calib_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--prompts", type=Path, default=here / "prompts_heldout.txt")
    ap.add_argument(
        "--captions_root", type=Path, default=Path("post_image_dataset/resized")
    )
    ap.add_argument("--n_captions", type=int, default=8)
    ap.add_argument(
        "--variants",
        default="inst,r64,r128,r128d256,r256,r512,r128d256+embed,r128d256-small",
    )
    ap.add_argument("--free_gen", default="inst,r128d256,r128d256+embed")
    ap.add_argument("--n_free", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=2048)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from library.env import resolve_under_home

    log = lambda s: print(s, flush=True)  # noqa: E731
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {
        json.loads(line)["prompt"]
        for line in (args.calib_dir / "generations.jsonl").read_text().splitlines()
    }
    prompts = [
        p
        for p in load_prompts(
            args.prompts, resolve_under_home(args.captions_root), args.n_captions, 1
        )
        if p not in seen
    ]
    log(f"[prompts] {len(prompts)} held-out")
    tok = AutoTokenizer.from_pretrained(args.pe_dir)
    system = (args.pe_dir / "system_prompt.txt").read_text().strip()
    pe = Ckpt(args.pe_dir)
    lora = safe_open(str(args.lora), "pt", device="cpu")
    model, keep, quant, params = build(args.base_dir, log)

    set_variant({"kind": "pe"}, keep, quant, params, pe, lora, log)
    gen_path = args.out_dir / "teacher_generations.jsonl"
    if gen_path.exists():
        rows = [json.loads(line) for line in gen_path.read_text().splitlines()]
    else:
        rows = generate(model, tok, system, prompts, 8, args.max_new, 1, log)
        gen_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    split_at = [think_split(tok, r["gen_ids"]) for r in rows]
    targets = teacher_targets(model, rows)
    t_self = score(model, rows, targets, split_at)
    log(f"[teacher] self-score {json.dumps(t_self)}")
    results = {"teacher": t_self, "variants": {}, "free_gen": {}}
    free = [p for p in args.free_gen.split(",") if p]
    for v in args.variants.split(","):
        spec = parse_variant(v)
        set_variant(spec, keep, quant, params, pe, lora, log)
        res = score(model, rows, targets, split_at)
        results["variants"][v] = res
        log(
            f"[score] {v:18s} "
            + " | ".join(
                f"{s}: kl={d['kl']:.4f} nll={d['nll']:.3f} top1={d['top1']:.3f}"
                for s, d in res.items()
            )
        )
        if v in free:
            g = generate(
                model, tok, system, prompts[: args.n_free], 8, args.max_new, 2, log
            )
            results["free_gen"][v] = {
                "json_ok": sum(r["json_ok"] for r in g),
                "finished": sum(r["finished"] for r in g),
                "n": len(g),
                "mean_tokens": sum(r["n_gen"] for r in g) / len(g),
                "answers": [{"prompt": r["prompt"], "answer": r["answer"]} for r in g],
            }
            log(f"[free_gen] {v}: json_ok={results['free_gen'][v]['json_ok']}/{len(g)}")
        (args.out_dir / "results.json").write_text(
            json.dumps(results, indent=1, ensure_ascii=False)
        )
    results["free_gen"]["pe"] = {
        "answers": [
            {"prompt": r["prompt"], "answer": r["answer"]} for r in rows[: args.n_free]
        ]
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False)
    )
    log("[done]")


if __name__ == "__main__":
    main()
