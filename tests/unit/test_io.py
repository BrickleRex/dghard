"""Tests for load_state_dict + save_repaired round-trip."""
from __future__ import annotations

import json

import pytest
import torch

from dghard.repair.io import load_state_dict, save_repaired


def test_load_state_dict_returns_all_keys(tiny_ckpt_dirs, tiny_state_dicts):
    base_dir, _ = tiny_ckpt_dirs
    base_sd, _ = tiny_state_dicts
    loaded = load_state_dict(base_dir)
    assert set(loaded.keys()) == set(base_sd.keys())
    for k in loaded:
        assert torch.equal(loaded[k], base_sd[k]), f"{k} mismatch"


def test_load_state_dict_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        load_state_dict(tmp_path / "nope")


def test_save_repaired_round_trips(tmp_path, tiny_ckpt_dirs, tiny_state_dicts):
    base_dir, _ = tiny_ckpt_dirs
    base_sd, _ = tiny_state_dicts
    out = tmp_path / "repaired"
    save_repaired(out, base_sd, base_ckpt=base_dir)

    # config.json copied
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "tiny_test"

    # weights round-trip exactly
    reloaded = load_state_dict(out)
    assert set(reloaded.keys()) == set(base_sd.keys())
    for k in base_sd:
        assert torch.equal(reloaded[k], base_sd[k]), f"{k} round-trip mismatch"

    # marker file written
    marker = json.loads((out / "dghard_repair.json").read_text())
    assert marker["method"] == "dg_hard"
    assert marker["n_tensors"] == len(base_sd)
