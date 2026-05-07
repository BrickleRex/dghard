"""Device + dtype detection helpers. Pure stdlib + torch."""
from __future__ import annotations

import torch


def auto_device(preference: str = "auto") -> torch.device:
    """Resolve a device string into a `torch.device`.

    `auto` -> cuda if available else cpu. `cpu` / `cuda` -> as-is.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def auto_dtype(device: torch.device) -> torch.dtype:
    """Pick a sensible inference dtype for the device.

    bf16 on CUDA (matches the paper's vLLM `--dtype bfloat16`); fp32 on CPU
    because most ops have no bf16 kernels there.
    """
    return torch.bfloat16 if device.type == "cuda" else torch.float32
