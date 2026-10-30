"""Anima LoRA trainer (the ``train.py --method …`` path; Anima line only).

| module | role |
|---|---|
| ``cli`` | argparse surface, network-kwarg intake, crash reporter, ``main`` |
| ``trainer`` | ``AnimaTrainer``: ``train()`` orchestration + the loop's hooks |
| ``step`` | ``AnimaStep``: the per-step forward, aux losses, adapter dispatch |
| ``data`` | dataset groups, per-feature dataset switches, compile token budget, TE-cache pass |
| ``models`` | TE / VAE / DiT loading |
| ``network`` | grad-basis sketch, adapter build → apply → compile |
| ``optim`` | optimizer, dataloaders, scheduler, ``accelerator.prepare`` |
| ``sigma_route`` | ``--sigma_lowres`` σ-demote routing |

The per-epoch loop and validation live in ``library.training.loop`` /
``library.training.validation``. ``train.py`` at the repo root is the shim
that sets the process-level env (no-window, allocator) and calls ``main``.
"""

from library.anima.trainer.cli import (
    build_network_extras,
    main,
    resolve_network_kwargs,
    setup_parser,
)
from library.anima.trainer.trainer import AnimaTrainer

__all__ = [
    "AnimaTrainer",
    "build_network_extras",
    "main",
    "resolve_network_kwargs",
    "setup_parser",
]
