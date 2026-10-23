# bench/moss_vr_headroom

Step 2 of the MOSS latent HF line: `docs/proposal/moss_vr_headroom.md` (design, gates and
results live there). Phase A, `residual_bands.py`: the base DiTs' FM residual split by
frequency on content both lines cache. Phase B, `grad_diagnostics.py`: VR-loss vs standard
FM gradients at a trained MOSS LoRA (Gram-matrix metrics for any λ).

```bash
make daemon-run ARGS="bench/moss_vr_headroom/residual_bands.py"
make daemon-run ARGS="bench/moss_vr_headroom/grad_diagnostics.py"
```

## Runs

| Run | Script | Verdict |
|---|---|---|
| `20261023-2340` | `residual_bands.py`, 24 stems × 10 σ × 2 draws | gap 0.161 at Anima's density, 72 % above 0.5 Nyquist: Phase B |
| `20261023-2351` | `grad_diagnostics.py`, `caststation_ga4_lr1e4_s0`, K = 24 | FAIL: ρ²(y, z) ≈ 0.01, pooled λ −0.19 leaves g_vr = g_std (cos 0.993); line closed |
