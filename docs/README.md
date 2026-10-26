# Documentation

Index of the `docs/` tree. Each model line has its own folder; docs that apply to more
than one line sit at this level. Read the linked doc before working on the thing it
describes.

## Lines

| Folder | Line |
|--------|------|
| [moss/](moss/README.md) | MOSS-Image (primary) — setup, GUI, CLI, defaults; i2i, findings, proposals, the model walkthrough ([moss/structure.md](moss/structure.md)) |
| [anima/](anima/README.md) | Anima (experimental) — methods, inference stacks, experimental methods, structure walkthroughs, findings, optimizations, proposals |

Qwen-Image-2.1 has no docs folder; its reference is `library/qwen21/CLAUDE.md`.

## MOSS-Image

| Doc | Description |
|-----|-------------|
| [moss/README.md](moss/README.md) | MOSS-Image LoRA (alpha) — setup, GUI, CLI, defaults and what to expect |
| [moss/structure.md](moss/structure.md) | The MOSS-Image model end to end — Qwen3.5 text pass, FLUX.2 latent boundary, single-stream DiT, training step and sampler |
| [moss/i2i.md](moss/i2i.md) | i2i — tasks, reference flags, status (design + probes: [../project/moss_lora/done/i2i.md](../project/moss_lora/done/i2i.md)) |
| [moss/channel_scaling.md](moss/channel_scaling.md) | Channel scaling + `dc_isolate` on MOSS (removed), and how the lr / alpha defaults got here |
| [moss/dave.md](moss/dave.md) | DAVE on MOSS — same-prompt seed variety (`--dave auto`), ComfyUI node |
| [moss/findings/README.md](moss/findings/README.md) | Findings — the chat template as attention sink, the vision tower as zero-shot reference |
| [moss/proposal/](moss/proposal/) | Active MOSS design docs |

## Shared

| Doc | Description |
|-----|-------------|
| [guidelines/guidebook.md](guidelines/guidebook.md) | Anima comprehensive guide (English); also [가이드북](guidelines/가이드북.md), [ガイドブック](guidelines/ガイドブック.md), [指南书](guidelines/指南书.md) |
| [guidelines/moss-guidebook.md](guidelines/moss-guidebook.md) | MOSS-Image GUI guidebook (English); also [가이드북](guidelines/moss-가이드북.md), [ガイドブック](guidelines/moss-ガイドブック.md), [指南书](guidelines/moss-指南书.md) |
| [guidelines/rocm.md](guidelines/rocm.md) | Windows ROCm — the AMD Radeon install path |
| [cns.md](cns.md) | CNS — colored-noise ER-SDE plug-in; Anima `--cns` and the MOSS γ files |

Curation docs — Anima Tagger, position captions, multiview audit, grouping,
masking — live with their code in the sibling
[`anime_tools`](https://github.com/sorryhyun/anime_tools) repo, under
[`docs/`](https://github.com/sorryhyun/anime_tools/tree/main/docs)
(`../anime_tools/docs/`).
