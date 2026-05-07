"""Shared test fixtures: small random tensors only — no network, no model
download, no datasets.load_dataset. The slow integration test in
`tests/integration/` may pull a real tiny model; everything in `tests/unit/`
must run offline in seconds."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(scope="session")
def rng() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(0)
    return g


@pytest.fixture
def tiny_state_dicts(rng: torch.Generator) -> tuple[dict, dict]:
    """A 'base' and 'ft' state dict with the structural variety DG-Hard
    must handle: a 2D weight matrix that should be SVD-shrunk, a 1D bias
    that must pass through, a small 2D matrix below `min_numel` that must
    pass through, and an unchanged tensor that must come back identical.
    """
    base = {
        "transformer.layers.0.attn.q_proj.weight": torch.randn(64, 64, generator=rng),
        "transformer.layers.0.attn.q_proj.bias":   torch.randn(64, generator=rng),
        "transformer.layers.0.mlp.up_proj.weight": torch.randn(128, 64, generator=rng),
        "small_2d.weight":                         torch.randn(8, 8, generator=rng),
        "embeddings.weight":                       torch.randn(100, 64, generator=rng),
        "untouched.weight":                        torch.randn(32, 32, generator=rng),
    }
    # Build ft by adding a structured + noise delta, except untouched.
    ft = {k: v.clone() for k, v in base.items()}
    ft["transformer.layers.0.attn.q_proj.weight"] += 0.05 * torch.randn(64, 64, generator=rng)
    ft["transformer.layers.0.attn.q_proj.bias"]   += 0.1 * torch.randn(64, generator=rng)
    ft["transformer.layers.0.mlp.up_proj.weight"] += 0.05 * torch.randn(128, 64, generator=rng)
    ft["small_2d.weight"]                          += 0.1 * torch.randn(8, 8, generator=rng)
    ft["embeddings.weight"]                        += 0.01 * torch.randn(100, 64, generator=rng)
    return base, ft


@pytest.fixture
def tiny_ckpt_dirs(tmp_path: Path, tiny_state_dicts) -> tuple[Path, Path]:
    """Persist tiny_state_dicts to a pair of HF-style directories with
    real safetensors + a minimal config.json so validate_checkpoints can
    walk them."""
    base_sd, ft_sd = tiny_state_dicts
    base_dir = tmp_path / "base"
    ft_dir = tmp_path / "ft"
    base_dir.mkdir()
    ft_dir.mkdir()

    cfg = {
        "model_type": "tiny_test",
        "architectures": ["TinyTestForCausalLM"],
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "vocab_size": 100,
    }
    (base_dir / "config.json").write_text(json.dumps(cfg))
    (ft_dir / "config.json").write_text(json.dumps(cfg))

    save_file({k: v.contiguous() for k, v in base_sd.items()},
              str(base_dir / "model.safetensors"))
    save_file({k: v.contiguous() for k, v in ft_sd.items()},
              str(ft_dir / "model.safetensors"))
    return base_dir, ft_dir
