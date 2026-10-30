# Windows ROCm Guide

The AMD Radeon / ROCm path for Anima LoRA on Windows. The workflow is the same
as on NVIDIA: run the one-line installer, then use the GUI. You do not need to
choose PyTorch indexes or `uv` extras.

## Beginner install

Open PowerShell in the folder where you want Anima LoRA to be installed and run:

```powershell
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

The installer will:

1. detect the Windows GPU vendor;
2. select the CUDA or ROCm dependency set automatically;
3. install the locked Python / PyTorch environment;
4. save the selected backend for later updates;
5. verify the ROCm runtime when AMD is selected;
6. create the Anima LoRA GUI desktop shortcut; and
7. launch the GUI.

After this, use the GUI for model download, preprocessing, training, and
updates.

### Updating

Use the Update button in the GUI.

The installer stores the selected backend in `.anima_backend` and the updater
reuses it, so a ROCm install stays on ROCm.

## Current verified hardware scope

The locked ROCm environment packages and verifies RDNA 4 only:

| Architecture | Tested hardware | Status |
|---|---|---|
| `gfx1200` | Radeon RX 9060 XT | Verified |
| `gfx1201` | Radeon RX 9070 XT | Verified |

Both were tested on the ROCm 10.1 Windows package stack (tensor
allocation, `torch.compile`, bf16 SDPA, backward, finite-value checks).

> The installer detects the GPU vendor, not the AMD architecture. It may select
> ROCm for any Radeon, but GPUs other than `gfx1200` / `gfx1201` (including
> RDNA 3) are unverified by this project.

## Locked software stack

The supported Windows ROCm environment is pinned to:

- Python 3.13
- PyTorch `2.14.0+rocm10.1.0`
- torchvision `0.29.0a0+rocm10.1.0`
- ROCm 10.1 device packages for `gfx1200` and `gfx1201`
- ROCm SDK/devel `10.1.0` for the headers/tools used by Windows Triton
- `triton-windows` 3.8 for the Windows `torch.compile` runtime

ROCm 10.1 validates PyTorch 2.14 on Windows
([AMD release notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)).
The torchvision `a0` suffix is the build published on AMD's stable index; this
does not select a nightly index. The CUDA path still defaults to PyTorch 2.14
and CUDA 13.2 (2.13 / 2.12 opt-in).

The SDK/devel package is installed inside the Python environment; a separate
machine-wide ROCm SDK upgrade is not required for this setup.

## Attention behavior on ROCm

On ROCm, `attn_mode = "flash"` is switched to PyTorch SDPA (no AMD Flash
Attention package is installed); `torch_compile = true` stays enabled. CUDA
builds keep Flash Attention.

## Manual clone / advanced setup

If you install from a git clone on Windows, CUDA is the default backend (the
`cuda-windows` dependency group is on by default); ROCm swaps it out explicitly.

ROCm:

```powershell
uv sync --no-group cuda-windows --group rocm-windows
```

CUDA:

```powershell
uv sync
```

The two groups are mutually exclusive. Reuse the same ROCm flags for every
later manual sync — a flagless `uv sync` reverts to CUDA.

`uv run` also syncs the environment by default. After installing ROCm, use
`uv run --no-sync python tasks.py gui` to launch with the installed packages,
or pass `--no-group cuda-windows --group rocm-windows` to `uv run` as well.

To force ROCm when using the one-line installer:

```powershell
$env:ANIMA_BACKEND = 'rocm'
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

`ANIMA_BACKEND` accepts `auto`, `cuda`, or `rocm`. On a mixed NVIDIA + AMD
system, `auto` picks CUDA; set `rocm` to override.

## Runtime smoke test

For a manual ROCm install, run the installer's post-install check:

```powershell
uv run --no-sync python tasks.py daemon-run -- tests/rocm_smoke_test.py
```

It exercises allocation, `torch.compile`, BF16 SDPA, backward and finite-value
checks, not just `import torch`. Each visible GPU is tested in a separate child
process, avoiding device switches after compilation on mixed-GPU Windows hosts.

## Troubleshooting

### `Found no NVIDIA driver` on an AMD GPU

The environment contains CUDA PyTorch instead of ROCm PyTorch. Older Windows
installers could replace the verified ROCm build during shortcut creation or
GUI launch by running `uv run` without the ROCm group flags.

Close the GUI and stop active jobs, then repair the existing installation from
PowerShell in the install directory:

```powershell
uv sync --no-group cuda-windows --group rocm-windows
uv run --no-sync python tests/rocm_smoke_test.py
uv run --no-sync python tasks.py gui
```

ROCm still uses PyTorch's `torch.cuda` API and the `cuda` device name. Those
names alone do not indicate that the wrong PyTorch build is installed.

### The installer selected ROCm for an older or unverified AMD GPU

Only `gfx1200` / `gfx1201` are verified. To try another architecture (e.g.
RDNA 3), install the matching ROCm device package and run
`tests/rocm_smoke_test.py` on that GPU before training.

### `ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention`

Expected when the config requests `flash` on a ROCm build; the runtime uses
PyTorch SDPA instead. Nothing to fix.

### The ROCm smoke test fails

Don't train until it passes. Confirm the environment was installed with the
`rocm-windows` group and that the GPU is `gfx1200` / `gfx1201`. For a clean
reinstall, re-run the one-line installer into a new empty directory.
