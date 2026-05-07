"""Auto-pick an inference backend based on machine capabilities."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Optional

import torch

from dghard.inference.base import InferenceClient

logger = logging.getLogger(__name__)


def _vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def build_client(
    ckpt: Path,
    backend: str = "auto",
    batch_size: Optional[int] = None,
) -> InferenceClient:
    """Construct the appropriate InferenceClient.

    `auto`: pick vLLM if (CUDA available AND vllm importable), else HF.
    `vllm` / `hf`: force the named backend; raise if its prerequisites are
    missing.
    """
    backend = backend.lower()
    if backend == "auto":
        if torch.cuda.is_available() and _vllm_available():
            backend = "vllm"
        else:
            backend = "hf"
        logger.info("inference backend (auto-selected): %s", backend)

    if backend == "vllm":
        from dghard.inference.vllm_client import VLLMClient
        return VLLMClient(ckpt)
    if backend == "hf":
        from dghard.inference.hf_client import HFClient
        return HFClient(ckpt, batch_size=batch_size)
    raise ValueError(f"unknown inference backend: {backend!r}")
