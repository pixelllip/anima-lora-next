"""Krea-2 empty-prompt ("uncond") text-conditioning sidecar.

Whole-caption dropout needs an **in-distribution** replacement for a dropped
sample.  Zeroing ``hiddens`` and masking every text token is not that condition:
at inference the DiT only ever sees a text block produced by the ChatML
template, so an all-masked text block is out of distribution.  Anima solved the
same problem with a T5("") crossattn sidecar
(``library/training/uncond_sidecar.py``); the Krea-2 analogue is the *empty
prompt* encoded through the very same tokenizer + mRoPE contract that
``inference_runner.prepare_krea2_text`` uses for CFG-uncond.

The condition is caption- and dataset-independent, so one file serves every
run; it only needs rebuilding when the encoder weights or the template contract
change (``KREA2_TE_CONTRACT_VERSION``).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file as _save_safetensors

from library.models.krea2_raw.strategy import (
    KREA2_TE_CONTRACT_KEY,
    KREA2_TE_CONTRACT_VERSION,
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
)

logger = logging.getLogger(__name__)

KREA2_UNCOND_DIR = "post_image_dataset"
KREA2_UNCOND_FILENAME = "_krea2_uncond_te.safetensors"
# Consumed by the loader to report *which* null condition a run trained on.
KREA2_UNCOND_PROMPT_KEY = "uncond_prompt"


def default_krea2_uncond_path(root: Optional[str] = None) -> Path:
    """``post_image_dataset/_krea2_uncond_te.safetensors`` (anima sidecar parity)."""
    return Path(root or KREA2_UNCOND_DIR) / KREA2_UNCOND_FILENAME


def encode_krea2_uncond(
    text_encoder,
    prompt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode ``prompt`` (default the empty string) with the training contract.

    Mirrors ``inference_runner._encode_prompt``: same ``Krea2TokenizeStrategy``
    ChatML template, same cumulative-valid-token mRoPE ``position_ids``, same
    12-layer stack and prefix slice.
    """
    tokenize_strategy = Krea2TokenizeStrategy()
    encoding_strategy = Krea2TextEncodingStrategy()
    tokens = tokenize_strategy.tokenize([prompt])
    with torch.no_grad():
        hiddens, mask = encoding_strategy.encode_tokens(
            tokenize_strategy, [text_encoder], tokens
        )
    return hiddens, mask


def save_krea2_uncond(
    path: Path | str,
    hiddens: torch.Tensor,
    mask: torch.Tensor,
    *,
    prompt: str = "",
) -> Path:
    """Write one ``(hiddens, mask)`` pair under the Krea-2 cache contract."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _save_safetensors(
        {
            KREA2_TE_CONTRACT_KEY: torch.tensor(
                KREA2_TE_CONTRACT_VERSION, dtype=torch.int64
            ),
            "hiddens": hiddens.contiguous(),
            "mask": mask.to(torch.bool).contiguous(),
        },
        str(dest),
        metadata={
            KREA2_UNCOND_PROMPT_KEY: prompt,
            "producer": "library.models.krea2_raw.uncond",
        },
    )
    logger.info(
        "Krea-2 uncond sidecar written: %s (prompt=%r, hiddens=%s, valid_tokens=%d)",
        dest,
        prompt,
        tuple(hiddens.shape),
        int(mask.sum()),
    )
    return dest


def uncond_prompt_of(path: Path | str) -> str:
    """Read back the stored ``uncond_prompt`` metadata (``""`` when absent)."""
    with safe_open(str(path), framework="pt") as handle:
        return str((handle.metadata() or {}).get(KREA2_UNCOND_PROMPT_KEY, ""))


def load_krea2_uncond(
    path: Path | str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the sidecar, rejecting caches from an older encoding contract."""
    src = Path(path)
    with safe_open(str(src), framework="pt") as handle:
        keys = set(handle.keys())
        for required in (KREA2_TE_CONTRACT_KEY, "hiddens", "mask"):
            if required not in keys:
                raise ValueError(
                    f"Krea-2 uncond sidecar {src} is missing {required!r}; "
                    f"rebuild it with scripts.krea2.preprocess_te_cache "
                    f"--write_uncond_sidecar"
                )
        version = int(handle.get_tensor(KREA2_TE_CONTRACT_KEY))
        if version != KREA2_TE_CONTRACT_VERSION:
            raise ValueError(
                f"Krea-2 uncond sidecar {src} was written for encoding contract "
                f"v{version}, current is v{KREA2_TE_CONTRACT_VERSION}; rebuild it "
                f"with scripts.krea2.preprocess_te_cache --write_uncond_sidecar"
            )
        hiddens = handle.get_tensor("hiddens").to(device=device, dtype=dtype)
        mask = handle.get_tensor("mask").to(device=device, dtype=torch.bool)
    return hiddens, mask


def stage_krea2_uncond(
    out_path: Path | str,
    *,
    te_path: str,
    prompt: str = "",
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
) -> Path:
    """Load the Qwen3-VL text encoder, encode ``prompt`` once, save, free it.

    Kept here (not in the training loop) because the training run drops the
    text encoder before the first step; a run can therefore never stage this
    sidecar on demand.
    """
    from library.models.krea2_raw.strategy import load_krea2_text_encoder

    logger.info("Loading Krea-2 text encoder from %s to build the uncond sidecar", te_path)
    text_encoder, _ = load_krea2_text_encoder(te_path, dtype=dtype, device=str(device))
    text_encoder.eval()
    try:
        hiddens, mask = encode_krea2_uncond(text_encoder, prompt)
    finally:
        text_encoder.to("cpu")
        del text_encoder
        gc.collect()
    return save_krea2_uncond(out_path, hiddens, mask, prompt=prompt)
