# MOSS-Image findings

Measurements of MOSS-Image (`library/moss/`) that are not a shipped feature. Each doc
separates what was measured (with its run) from untested hypotheses. Run logs, probe code
and raw numbers live in the matching `bench/moss_*/` or `_archive/bench/moss_*/` folder.

| Doc | What it covers |
|---|---|
| [attention_sink_chat_template.md](attention_sink_chat_template.md) | The 10 chat-template tokens are MOSS's attention sink: an OFF-gate plus a per-head bias written into every image token. Trained LoRAs barely move it. |
| [vision_tower_reference.md](vision_tower_reference.md) | An image put through Qwen3.5's own vision tower conditions MOSS zero-shot: content and coarse layout come back, pixel alignment does not, and the DiT reads the image-token rows. |
| [sigma_staircase.md](sigma_staircase.md) | One forward's `x0_pred` resolves at Anima's σ knee (recognisable 0.74), but layout is set by σ 0.87–0.80 and shift 5 puts that at steps 22–28 of 50. Base and SFT resolve alike. |
| [lr_clip_sweep.md](lr_clip_sweep.md) | LR × grad clip × grad accum on one character set: the shipped 5e-5 / clip 1 beats clip 7, no-clip 2e-5 and no-clip at `grad_accum` 2 (1e-4 distorts; 5e-5 ties clip 7). The loss curve can't tell them apart. |
| [muon_lora.md](muon_lora.md) | Muon at AdamW's lr 5e-5 moves the LoRA by the same norm over ~3× the rank directions; with it `dc_isolate` becomes a bias channel (output DC shift 2.57 vs 1.24 without), so the default is Muon without `dc_isolate`. |
| [muon_channel_init.md](muon_channel_init.md) | Under Muon, channel scaling α 0.5 leaves the loud channel at 6.9× the mean channel's leverage (Muon's `down` step keeps per-channel proportions); 0.75 flattens it to 1.6×. The `weight_svd` basis feeds 4–6× more input DC; kaiming cuts the output DC shift 1.24 → 0.43. New default: α 0.75, kaiming. |
| [dcfree_svd_init.md](dcfree_svd_init.md) | A `weight_svd` init with x̄ projected out ends below kaiming's output DC shift (0.32 vs 0.43); also dropping ch 42, the one loud input channel (`svd_drop_loud 1`), zeroes its share and kept the no-LoRA composition best, with worse hands. Default stays kaiming. |
| [muon_no_channel_scaling.md](muon_no_channel_scaling.md) | Muon + kaiming with channel scaling off is the α 0.75 run to three decimals (loss, ‖ΔW‖, stable rank; ch 42 share 7.3 vs 6.8 %, DC shift 0.49 vs 0.43) and keeps the no-LoRA composition more often. New default: α 0; no calibration file read. |
| [ema_lr1e4.md](ema_lr1e4.md) | An adapter EMA at 0.97 (one epoch on 30 images) keeps lr 1e-4 on the prompt where the same run's raw weights drift (solid backgrounds, added outfit items); at 5e-5 it changes details, not strength. New default: `ema_decay` 0.97 (t2i), since `ema_window` 0.1 (t2i and i2i). |
| [spectrum_schedule.md](spectrum_schedule.md) | Spectrum's full-step schedule re-checked against the staircase: the shipped growing window beats log-SNR, step-even and layout-dense schedules at NFE 19 and an NFE-16 one. An open-loop skip-cost score ranks them backwards. |
