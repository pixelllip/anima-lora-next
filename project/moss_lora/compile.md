# MOSS compile: FA2 + a range-marked token axis, Anima's way

Opened 2026-10-25. Goal: compile MOSS's blocks the way Anima's are compiled —
`dynamic=False`, only the token axis dynamic, strictly bounded to the range the run
measures, FA2 for attention — instead of today's `torch.compile(dynamic=True)` on SDPA
(`library/moss/loader.py::compile_blocks`).

## Trigger

Every i2i run stalls once in epoch 1 (sanitize, seed 0: step 32, 22.8 s cold / 6.9 s warm
vs 3.9 s/step). `TORCH_LOGS=recompiles` (job `20261025-162027-ac0dfd`): the blocks
recompile on

```
((ref_span[0] + x.size()[1] - ref_span[1]) % 8) != 0     # (n_img + n_text) % 8
(torch.sym_max(1, ref_span[1] - ref_span[0]) % 8) == 0   # n_ref % 8
```

Inductor's `sdpa_constraint` guards each SDPA call's query count on `% 8` (the backward's
`[B, H, Lq]` logsumexp stride). Under `ref_isolate` there are two SDPA calls, so two
residue checks; sanitize's 280 samples fall in three classes (142 / 133 / 5), and the
first of the 5-sample class arrives at step 32. Nine graph variants in all (3 classes ×
grad / no-grad / a `None`-arg variant), against a recompile limit of 16. Anima hit the
same guard on its SDPA path (GH #107, `library/anima/models.py::_make_dynamic_seq_forward`).

## How Anima got to a non-dynamic compile

| Step | When | What | Where |
|---|---|---|---|
| Per-block compile of `_forward` | — | not `forward`: `unsloth_checkpoint` is `@torch._disable_dynamo`, compiling it graph-breaks and guard-storms | `models.py::compile_blocks`, `docs/structure/anima-optimizations.md` §4.0 |
| `dynamic=False` | — | dynamic tracing "only buys recompile risk"; every dim static except what is marked | same, §4 |
| Static pad removed | 2026-05-24 | padding to one shape leaked pad tokens into real ones under flash (AdaLN shift / QKV bias, up to 6.5 % rel-L2) | §3 Removed modes |
| Native flatten | — | `(B,T,H,W,D)` → fake-5D `(B,1,seq,1,D)`: the graph keys on token count, not on H and W | §3, `_native_flatten` |
| Constant-token buckets removed → free-fit | 2026-06-19 | native aspect, token count anywhere in its tier band (`EDGE_TOKEN_BANDS`) | `library/datasets/buckets.py` |
| `compile_dynamic_seq` | 2026-06-09 (recompile fixes) | eager prologue `mark_dynamic(x, seq, min=lo, max=hi)` + RoPE tables, **inside** the checkpointed callable (the recompute's `detach_variable` drops the mark otherwise); range from the populated buckets (`derive_token_budget`) | `_make_dynamic_seq_forward` |
| FA2 as the attention | — | `flash_attn_func` on BLHD with `cu_seqlens=None`; no `sdpa_constraint`, so the strict range holds. SDPA (ROCm) keeps soft marks: GH #107 | `networks/attention_dispatch.py` |
| Text side shape-stable | — | TE max-padded to 512 (the pad tail is a load-bearing sink) | §3 Cross-attention |
| `mix_order_reduction` pinned off | 2026-07-15 / 07-17 | its `Ge(seq, 4096)` hint guard contradicts a strict range straddling 4096; `pin_inductor_flag`, not assignment (inductor config overrides are per-ContextVar, the backward compile reads the default) | `library/runtime/dynamo.py` |
| `pin_dynamo_limit` | — | same ContextVar trap for `recompile_limit`: a plain assignment reverts to 8 in the backward compile and spills to eager | same |
| Per-signature compile cache | — | persistent caches key on the graph, not on the mark's bounds; a cached narrower guard replays into a wider mark as `ConstraintViolationError` | `harness.py::isolate_compile_cache` |
| `requires_grad` normalised | — | one `requires_grad_()` on the block-stack input, so block 0 (frozen embed) and blocks 1+ share one graph | §4.3 |
| Per-band marks | 2026-08-19 | one tight range per token cluster instead of a union spanning the tier gaps; it/s neutral | `--compile_seq_bands`, `_archive/proposals/perband_dynamic_seq.md` |

The rule it ends in: **no shape enters the graph except the one marked axis, and nothing
in the graph may guard on that axis beyond its range.** No varying Python int reaches a
compiled block.

## MOSS today, against that rule

| Source of shape change | MOSS | Anima's answer | Carries over? |
|---|---|---|---|
| Image tokens | native aspect, sides ×16, `n_img` arbitrary | free-fit band + marked axis | yes: mark `x` / `rope` / `key_mask` dim 1 |
| Text length | **never padded** (Qwen3.5 GDN; RoPE puts the image at frame = text length) | pad to 512 | **no**. It folds into the same marked joint axis instead |
| Reference tokens | `ref_span = (n_img, n_img + n_ref)` as **Python ints** into the block (layout image, ref, text) | none | **the blocker**, P2 |
| `% 8` query guard | SDPA everywhere | FA2 | yes, P1 |
| Batch | B = 1 in training; generate batches cond + uncond with a `key_mask` only under `--batched` | drop the partial batch | training: nothing to do. `--batched` keeps SDPA (masked) |
| `requires_grad` block 0 vs 1+ | two graphs (`0/0`, `0/1` in the log) | `requires_grad_()` at the stack input | yes, P3 |
| Recompile limit | plain `torch._dynamo.config.recompile_limit = 16` | `pin_dynamo_limit` | yes, P3 |
| Range straddling 4096 | t2i caststation 4174–4656 (no); i2i swap_text 1.9k–8.9k (yes) | pin `mix_order_reduction` off | yes, P3 |
| Compile cache | shared default dir | per-signature dir | yes, P3, once ranges exist |

## Measured so far (2026-10-25, sanitize, seed 0, 40 steps, `blocks_to_swap 24`)

| Arm | Job | Result |
|---|---|---|
| today: `dynamic=True`, SDPA | `20261025-162027-ac0dfd` | 3.889 s/step median; 9 graphs; step 32 recompile (6.9 s) |
| soft marks, `dynamic=None`, SDPA | `20261025-163333-45a132` | 3.901 s/step: no step-time gain from static non-token dims; 11 graphs |
| strict range, `dynamic=None`, SDPA | `20261025-163241-5af865` | `ConstraintViolationError` at first compile: `(L − 4096) % 8 != 0` (the SDPA guard) and `8194 ≤ L ≤ 8299` (static `ref_span`) |
| strict range, `dynamic=None`, FA2 | `20261025-163814-f5ca56` | the `% 8` guard is **gone**; still `ConstraintViolationError` on `L − 8192 ∉ {0, 1}`, `8192 ≤ L ≤ 8299`: `ref_span`'s end (8192) specialised on the first call |
| **P1**: FA2 in `_attention`, `dynamic=True` (today's compile) | `20261025-164622-ff597b` | recompiles 9 → 3 graphs (left: `requires_grad` block 0 vs 1+, and the `None`-stride variant; no `% 8` guard). No stall: max step 4.3 s, step 32 4.1 s. Loss tracks SDPA (steps 1–2 equal to 4 d.p.; later drift ≤ 0.015, the same size as SDPA vs SDPA across compile modes, e.g. step 32 0.3386 / 0.3542 / FA2 0.3533). Median 4.07 s/step vs 3.89 (+5 %, not chased); peak 11.96 GB vs 12.66; step 1 37 s (cold inductor cache after the code change) |
| **P2 A**: P1 + strict mark `5901–8299` on `x` / `rope` dim 1, `dynamic=True` | `20261025-165247-96c7cb` | **holds**: no `ConstraintViolationError`; the same 3 graphs as P1 (`requires_grad`, `None`-stride); the symbolic `ref_span` ints raise no guard against the range. Loss matches P1 (step 40 0.3697 / 0.3698); median 4.01 s/step, max 4.4; step 1 22.5 s. Sanitize's text is 19 tokens on every sample |
| **P3** on P2 A: `pin_dynamo_limit`, grad leaf at block 0, range from the shortest text (`n_text_min`, empty prompt), 4096 pin, per-range compile cache | `20261025-165941-cea148` | **one graph** (`[0/0]` only, guard `5919 ≤ L ≤ 8299`), zero recompiles; the `None`-stride variant went with the block-0 `requires_grad` split. Loss tracks (step 40 0.3692); median 3.99 s/step, max 4.2; step 1 20.2 s (cold, new cache dir); peak 11.96 GB vs 12.66 SDPA |
| FA2 vs SDPA, one `_attention` call | `20261025-163800-9a8937` | L 8211 plain / 8269 isolated, bf16: out / dq / dk / dv rel-L2 to an fp32 reference identical for both (~1e-2, bf16 rounding); FA2 vs SDPA 4e-5. fwd+bwd 31.1 vs 32.0 ms |

### swap_text_600 (2026-10-25, 40 steps, `blocks_to_swap 28`, `--lr 1e-4`)

600 samples, text 67–542 tokens (192 lengths), `n_img = n_ref` 910–4160, joint 1896–8750
(straddles 4096), all eight `(n_img + n_text) % 8` classes. Sanitize's text is fixed at 19,
so this is the set that exercises a varying text length.

| Arm | Job | Result |
|---|---|---|
| **A** (`dynamic=True` + strict mark), budget 0.1 | `20261025-170847-3859ca` | **holds**: one graph, guard `1896 ≤ L ≤ 8750`; the `ref_span` guards are bounds and 0/1 checks only (`0 ≤ s ≤ e ≤ L`, `s ≠ 1`, `e − s ≠ 1`). Already guards `L · rope.size(2) ≤ 2³¹ − 1` (`can_use_32bit_indexing`), so int32 indexing is not what a range adds. Median 4.060 s/step, peak 11.38 GB |
| **C** (`dynamic=False` + strict mark + `dynamic_sources` on `ref_span`), budget 0.1 | `20261025-171656-c330b4` | one graph, no violation; the rope head-dim guards go, only the marked axis and the two ints stay symbolic. Median 4.048 s/step (A: 4.060), peak 11.38 GB, loss equal (step 40 0.1289 both) |
| C, budget 0.15 + `expandable_segments` | `20261025-172151-1caacb` | median 3.990 (−1.4 %), peak 13.38 GB (+2.0); the allocator OOM warning at budget 0.1 is gone |
| SDPA (pre-P1 code), budget 0.1 | `20261025-172626-062099` | median 3.928, peak 12.07 GB; 9 graphs in 40 steps, stalls at step 2 (17.7 s) and step 40 (12.1 s, a new residue class); epoch 200 s vs C 178 s |
| SDPA, budget 0.15 + expandable | `20261025-172625-3e77bd` | median 3.866, peak 14.11 GB; 9 graphs, step 40 20.3 s; epoch 219 s vs C 169 s |
| C, budget 0.2 + expandable, `--fit_report` | `20261025-173626-cf66b4` | **OOM** at step 1 (the largest sample) in the backward, 14.12 GiB allocated of 15.51 |
| C, budget 0.15 + expandable, `--fit_report` | `20261025-173720-229017` | largest sample (4104 image + 542 text): peak 13.35 GB allocated / 13.87 reserved, 0.53 GB free. Median 3.965, epoch 164 s |
| SDPA, budget 0.15 + expandable, `--fit_report` | `20261025-173905-00631c` | largest sample: 14.08 / 14.38 GB, one expandable mapping OOM (recovered). Median 3.929 (−0.9 % vs C), 5 graphs, step 2 12.0 s; epoch 184 s |

Budget 0.1 → 0.15 is worth ~1.5 % on both kernels for +2 GB. SDPA's steady state is
1–3 % faster than FA2 inside the compiled block (the one-call bench had FA2 faster, so it is
likely the graph around the call, not the kernel) and costs 0.5–0.7 GB more; its
recompiles still cost more wall time than that over a short run, and keep growing with the
dataset.

What the switch buys is therefore not step time (SDPA already lands on a flash kernel
here). It buys one graph per range instead of one per residue class × variant, no mid-run
stall, a recompile count that does not grow with the dataset, and no eager fallback when a
new dataset adds a residue class (the fourth class would take this run to 12 of 16).

## Plan

### P1 — FA2 for unmasked attention (`library/moss/model.py::_attention`)

`key_mask is None` and a CUDA bf16 / fp16 tensor → `flash_attn_func` on `(B, L, H, D)`
(MOSS's layout before the transpose, so no copy); `ref_isolate` = two calls, main queries
over all keys and ref queries over ref keys. Masked (`forward_batched`), CPU and fp32
keep SDPA. flash-attn 2.8.3 is already in the venv and runs on sm_120 (Anima's path).

- Gate (done): one call, fwd + bwd, FA2 ≡ SDPA within bf16 noise (table above).
- Gate: 40-step sanitize run under today's `dynamic=True`: loss per step matches the SDPA
  run to the third decimal, and the `% 8` recompiles are gone, so step 32 stops stalling.
  If this alone fixes the trigger, P2–P3 are still the compile design, not a fix.
- Pipeline invariant: same math, different kernel; `model.py`'s unbatched path stays
  op-for-op apart from the attention backend. Note it in the module docstring.

### P2 — get `ref_span` out of Python ints

Under `dynamic=False` a Python int argument is a constant, so `ref_span` specialises the
graph per sample, and with a strict mark it contradicts the range (measured). Two ways:

- **A (one probe run):** keep `torch.compile(dynamic=True)` so the ints are symbolic from
  the first call, and add the strict mark on top. Cheap to test; it leaves every other
  dim symbolic, which is what Anima set out to avoid, and relational guards between the
  int and the marked axis may still violate the range.
- **B (the design):** carry the segments in tensor shapes. `run_layers` cuts the joint
  sequence into `x_main` (image + text) and `x_ref` (+ their RoPE rows); the block takes
  both, applies the two modulations to the two tensors (today's `_affine` / `_gated`
  three-way cat goes away), runs each projection on `cat((x_main, x_ref), 1)` and splits
  by `x_main.shape[1]` (one GEMM, one LoRA call per linear, as now), and returns both.
  Both are marked, each with its own range. No int crosses the graph boundary. The t2i
  path (`x_ref is None`) is unchanged. Token order inside attention becomes (image, text,
  ref): attention is permutation-equivariant given RoPE, so this is a summation-order
  difference only. `_ref_only` (the skip-22 uncond under `ref_isolate`) becomes "run the
  block on `x_ref` alone".

Recommendation: run A once (it costs one 40-step job); build B either way if A passes
only with every dim dynamic.

Outcome: A held (`96c7cb`, and on swap_text's varying text `3859ca`). Landed instead of
B: **`dynamic=False` + `torch.compiler.config.dynamic_sources = "L['ref_span'][0],
L['ref_span'][1]"`** (`c330b4`). The source allow list makes the two ints symbolic under
`dynamic=False` (it takes precedence over it; `mark_dynamic` does not take ints), so only the
marked axis and `ref_span` are dynamic, with no change to the block. Pinned like the other
flags (`pin_compiler_flag`). Step time equals A's, so this buys the Anima rule, not speed.
B is not needed.

### P3 — Anima's infrastructure, ported

Done 2026-10-25 for training (`loader.py::compile_blocks(seq_range=…)`, range from
`train.py`), except per-band marks (sanitize is one cluster; revisit on a mixed-size set).

- `pin_dynamo_limit("recompile_limit", …)` instead of the plain assignment.
- `x.requires_grad_()` on the block-stack input when training, so block 0 shares a graph.
- Range: train from the samples (`min(n_img + n_ref) + shortest text` … `joint`); the
  shortest text has to cover the empty prompt (`caption_dropout`) and caption variants,
  which `Sample.n_text` (a max) does not record. Calls outside the range stay unmarked and
  warn once (Anima's out-of-band rule).
- `pin_inductor_flag("triton.mix_order_reduction", False)` when a range straddles 4096.
- `isolate_compile_cache(signature)` once ranges exist: the cost is a cold compile per new
  range (sanitize: first step 8 s warm → ~26 s cold).
- Per-band marks if a union range spans a dead zone (i2i sets mix 1k and 4k images).
- Find the `'NoneType' object has no attribute 'stride'` variant (`0/2`, `0/5`, `0/8` in
  the log): some tensor argument is `None` on some calls. `TORCH_LOGS=guards`.

### P4 — generate

Unbatched generate (the default) takes the FA2 path from P1. Its range comes from the
jobs (sizes × prompt lengths × ref), known before `compile_blocks` runs. `--batched`
(masked) stays on SDPA with today's dynamic compile, or moves to `flash_attn_varlen_func`
later. `tune.py` and the calibration / bench callers of `compile_blocks` keep the
range-free default.

### P5 — bench + tests (CONTRIBUTING Tier 1.5)

- Bench: s/step, first-step compile wall cold / warm, graph count, peak VRAM, loss per step;
  arms today / P1 / P1+P2+P3; sanitize and caststation t2i.
- Tests: CPU falls back to SDPA (existing `tests/test_moss_*` stay CPU); a CUDA-only
  FA2 ≡ SDPA test (skip without a GPU); a P2 equivalence test (split-segment block ≡
  joint block on CPU, SDPA).

## Open questions

- ~~Does P1 alone already end the trigger stall?~~ Yes (`ff597b`): the `% 8` recompiles and the step-32 stall are gone. Open: the +5 % median step time against SDPA (one run each; the one-call bench had FA2 3 % faster).
- ~~A vs B: does `dynamic=True` + strict mark hold?~~ It holds (`96c7cb`, `3859ca`); `dynamic=False` + `dynamic_sources` replaces both (`c330b4`).
- ~~Does an explicit range / static non-token dims make the step faster?~~ No: C vs A 4.048 vs 4.060 s/step (`c330b4`, `3859ca`), as `45a132` found on sanitize.
- Can SDPA's 1–3 % be had without its `% 8` guard? An SDPA call wrapped as an opaque `torch.library.custom_op` would not get `sdpa_constraint`. Untested; it would also give back 0.5–0.7 GB of headroom.
- ~~The `None`-stride variant: which argument?~~ Not identified; it disappeared with the block-0 grad leaf (`cea148`), so it was a branch of the no-grad block-0 graph.
- Is one union range enough for MOSS's sets, or does i2i with mixed sizes need bands?

## WIP

P1 + P2 A + P3 committed 2026-10-25 (`721466ab`); P2 as `dynamic=False` + `dynamic_sources`
on top. Budget 0.15 + `expandable_segments` fits swap_text's largest sample at swap 28 on
16 GB (0.2 does not); now the TrainRequest default (swap 28, budget 0.15, expandable).
Next: P4 (generate's range), P5 (bench + tests).
