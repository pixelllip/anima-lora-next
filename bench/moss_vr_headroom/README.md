# bench/moss_vr_headroom

Step 2 of the MOSS latent HF line: `docs/proposal/moss_vr_headroom.md` (design, gates and
results live there). Phase A, `residual_bands.py`: the base DiTs' FM residual split by
frequency on content both lines cache. Phase B (VR-loss gradients on a trained MOSS LoRA)
is not written yet.

```bash
make daemon-run ARGS="bench/moss_vr_headroom/residual_bands.py"
```

## Runs

| Run | Script | Verdict |
|---|---|---|
| `20261023-2340` | `residual_bands.py`, 24 stems × 10 σ × 2 draws | gap 0.161 at Anima's density, 72 % above 0.5 Nyquist: Phase B |
