"""PE-mode decode speed at batch 1: dynamic vs static KV cache.

Loads the catalog INT8 TE with the saved PE-mode tensors (``eval_te.py --save``),
then greedy-generates a fixed ``--n`` tokens per arm. A logits processor stamps
each step (after a device sync), so prefill is the first step and decode is the
median of the rest. ``static`` passes ``cache_implementation="static"``, which
makes HF ``generate`` compile the decode forward; its warmup runs absorb that.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LogitsProcessor, LogitsProcessorList

from library.moss import int8
from pe_mode import PEMode


@torch.library.custom_op("pe_lora::int8_convrot_linear", mutates_args=())
def int8_convrot_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, group: int
) -> torch.Tensor:
    import comfy_kitchen

    return comfy_kitchen.int8_linear(
        x,
        weight,
        scale,
        out_dtype=x.dtype,
        convrot=group > 0,
        convrot_groupsize=group or int8.GROUP,
    )


@int8_convrot_linear.register_fake
def _(x, weight, scale, group):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _opaque_forward(self, x):
    # comfy_kitchen's W8A8 path reads data pointers (dlpack), which dynamo cannot trace
    return int8_convrot_linear(x, self.weight, self.weight_scale, self.group)


int8.Int8ConvRotLinear.forward = _opaque_forward


class Stamp(LogitsProcessor):
    def __init__(self):
        self.t = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.t.append(time.perf_counter())
        return scores


def run(causal, enc, n, cache, eos):
    stamp = Stamp()
    t0 = time.perf_counter()
    kw = {"cache_implementation": "static"} if cache == "static" else {}
    out = causal.generate(
        **enc,
        max_new_tokens=n,
        min_new_tokens=n,
        do_sample=False,
        logits_processor=LogitsProcessorList([stamp]),
        eos_token_id=eos,
        **kw,
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    steps = [b - a for a, b in zip(stamp.t, stamp.t[1:])]
    steps.sort()
    return {
        "prefill_s": stamp.t[0] - t0,
        "decode_ms_median": 1000 * steps[len(steps) // 2],
        "decode_ms_p90": 1000 * steps[int(len(steps) * 0.9)],
        "total_s": total,
        "tokens": out[0, enc["input_ids"].shape[1] :].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe_dir", type=Path, required=True)
    ap.add_argument("--tensors", type=Path, required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--arms", default="dynamic,static")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from library.moss import loader

    log = lambda s: print(s, flush=True)  # noqa: E731
    _, te = loader.load_text_encoder(None, precision="int8", log=log)
    pm = PEMode(te, load_file(str(args.tensors)))
    pm.set_mode("pe")
    tok = AutoTokenizer.from_pretrained(args.pe_dir)
    system = (args.pe_dir / "system_prompt.txt").read_text().strip()
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "해질녘 옥상에서 기타 치는 소년"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tok(text, return_tensors="pt").to("cuda")
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    log(f"[bench] prompt {enc['input_ids'].shape[1]} tokens, n={args.n}")
    res = {}
    for arm in args.arms.split(","):
        for w in range(args.warmup):
            t0 = time.perf_counter()
            run(pm.causal, enc, args.n, arm, eos)
            log(f"[warmup] {arm} {w}: {time.perf_counter() - t0:.1f}s")
        r = run(pm.causal, enc, args.n, arm, eos)
        res[arm] = r
        log(
            f"[bench] {arm:8s} prefill {r['prefill_s']:.2f}s | decode median "
            f"{r['decode_ms_median']:.1f} ms/tok (p90 {r['decode_ms_p90']:.1f}) | "
            f"{args.n} tok {r['total_s']:.1f}s | 1200 tok ≈ "
            f"{r['prefill_s'] + 1.2 * r['decode_ms_median']:.1f}s"
        )
    arms = list(res)
    for a in arms[1:]:
        x, y = res[arms[0]]["tokens"], res[a]["tokens"]
        same = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), len(x))
        log(f"[match] {arms[0]} vs {a}: first {same}/{len(x)} greedy tokens identical")
        res[a]["match_prefix"] = same
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "tokens"}
                for k, v in res.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
