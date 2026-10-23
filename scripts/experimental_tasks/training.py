"""Experimental training entry-points (soft tokens).

Wired up under ``make exp-*`` / ``python tasks.py exp-*``. Each ``cmd_*`` is a
thin shim that translates env vars + extra argv into a ``train.py`` or script
call.
"""

from __future__ import annotations

from scripts.tasks._common import train


def cmd_soft_tokens(extra):
    train("soft_tokens", extra)
