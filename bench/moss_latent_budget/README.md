# bench/moss_latent_budget

How much of the FLUX.2 (MOSS) latent is high frequency, against Anima's: step 1 of
`docs/proposal/moss_latent_hf_budget.md` (measurements, gate and results live there).
VAE encodes only; `run_bench.py`'s docstring has the details.

```bash
make daemon-run ARGS="bench/moss_latent_budget/run_bench.py"
```

## Runs

| Run | Images | Verdict |
|---|---|---|
| `20261023-2321` | 200 originals, 91 folders, seed 0 | H holds: centred HF share above 0.5 Nyquist 2.55×, Haar detail share 2.33× |
