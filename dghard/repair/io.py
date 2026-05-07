"""Checkpoint load + save helpers (HF-compatible safetensors directories)."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def load_state_dict(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from every *.safetensors shard in `ckpt_dir`.

    Returns a flat name -> tensor dict. Tensors stay on CPU; their dtype is
    whatever the shard stored.
    """
    ckpt_dir = Path(ckpt_dir)
    shards = sorted(ckpt_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors shards in {ckpt_dir}")
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
    logger.debug("loaded %d tensors from %d shards under %s",
                 len(sd), len(shards), ckpt_dir)
    return sd


# Files copied from the base checkpoint into the repaired output so the
# result loads via `AutoModelForCausalLM.from_pretrained(...)` without
# extra setup. Tokenizer files cover both slow and fast tokenizers; chat
# templates live in `tokenizer_config.json` / `chat_template.jinja`.
_SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def save_repaired(out_dir: Path, state_dict: dict[str, torch.Tensor],
                  base_ckpt: Path) -> None:
    """Write a repaired HF checkpoint to `out_dir`.

    - Weights go to `model.safetensors` (single shard for simplicity; HF can
      load any shard layout, and small-to-medium models fit comfortably).
    - Tokenizer + config files are copied from `base_ckpt` so the repaired
      model has the same vocab and chat template as the base. The repaired
      model's architecture is identical to the base — that's the whole
      point — so config copy is correct.
    """
    out_dir = Path(out_dir)
    base_ckpt = Path(base_ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make tensors safetensors-compatible: contiguous, on CPU.
    cleaned = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
    save_file(cleaned, str(out_dir / "model.safetensors"))
    logger.debug("wrote model.safetensors with %d tensors", len(cleaned))

    n_copied = 0
    for fname in _SIDECAR_FILES:
        src = base_ckpt / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)
            n_copied += 1
    logger.debug("copied %d sidecar files from %s", n_copied, base_ckpt)

    # Drop the legacy index json if it was copied — single-shard layout
    # makes it incorrect.
    stale_index = out_dir / "model.safetensors.index.json"
    if stale_index.is_file():
        stale_index.unlink()

    # Stamp a small marker so reviewers can tell at a glance that this is
    # a DG-Hard-repaired checkpoint (not the original FT).
    marker = {
        "method": "dg_hard",
        "base_ckpt": str(base_ckpt.resolve()),
        "n_tensors": len(cleaned),
    }
    (out_dir / "dghard_repair.json").write_text(json.dumps(marker, indent=2))
