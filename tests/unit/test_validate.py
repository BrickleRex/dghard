"""Tests for the pre-flight checkpoint validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from dghard.repair.validate import ValidationError, validate_checkpoints


def test_happy_path(tiny_ckpt_dirs):
    base, ft = tiny_ckpt_dirs
    report = validate_checkpoints(base, ft)
    assert report.model_type == "tiny_test"
    assert report.architectures == "TinyTestForCausalLM"
    assert report.n_matched_2d >= 1
    # No shape mismatches in the happy fixture.
    assert report.n_shape_mismatched == 0


def test_missing_base_dir_raises(tmp_path):
    ft = tmp_path / "ft"
    ft.mkdir()
    with pytest.raises(ValidationError, match="base path"):
        validate_checkpoints(tmp_path / "nope", ft)


def test_missing_config_raises(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    with pytest.raises(ValidationError, match="config.json"):
        validate_checkpoints(a, b)


def test_no_safetensors_raises(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    cfg = json.dumps({"model_type": "x", "architectures": ["Y"]})
    (a / "config.json").write_text(cfg)
    (b / "config.json").write_text(cfg)
    with pytest.raises(ValidationError, match="safetensors"):
        validate_checkpoints(a, b)


def test_model_type_mismatch_raises(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "config.json").write_text(json.dumps(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    ))
    (b / "config.json").write_text(json.dumps(
        {"model_type": "qwen", "architectures": ["LlamaForCausalLM"]}
    ))
    save_file({"w": torch.randn(8, 8)}, str(a / "model.safetensors"))
    save_file({"w": torch.randn(8, 8)}, str(b / "model.safetensors"))
    with pytest.raises(ValidationError, match="model_type"):
        validate_checkpoints(a, b)


def test_architectures_mismatch_raises(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "config.json").write_text(json.dumps(
        {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    ))
    (b / "config.json").write_text(json.dumps(
        {"model_type": "llama", "architectures": ["LlamaForSequenceClassification"]}
    ))
    save_file({"w": torch.randn(8, 8)}, str(a / "model.safetensors"))
    save_file({"w": torch.randn(8, 8)}, str(b / "model.safetensors"))
    with pytest.raises(ValidationError, match="architectures"):
        validate_checkpoints(a, b)


def test_shape_mismatch_is_warning_not_error(tmp_path):
    """Shape drift on shared keys is a soft warning — repair will skip those
    keys, but a vocab-size change shouldn't kill the whole run."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    cfg = json.dumps(
        {"model_type": "x", "architectures": ["Y"], "hidden_size": 64}
    )
    (a / "config.json").write_text(cfg)
    (b / "config.json").write_text(cfg)
    save_file({"w": torch.randn(64, 64), "v": torch.randn(64, 64)},
              str(a / "model.safetensors"))
    save_file({"w": torch.randn(64, 32), "v": torch.randn(64, 64)},
              str(b / "model.safetensors"))
    report = validate_checkpoints(a, b)
    assert report.n_shape_mismatched == 1
    assert any("shape mismatch" in w for w in report.warnings)
    assert report.n_matched_2d == 1  # 'v' still matches


def test_no_matched_2d_raises(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    cfg = json.dumps({"model_type": "x", "architectures": ["Y"]})
    (a / "config.json").write_text(cfg)
    (b / "config.json").write_text(cfg)
    # Only a 1D tensor present.
    save_file({"bias": torch.randn(64)}, str(a / "model.safetensors"))
    save_file({"bias": torch.randn(64)}, str(b / "model.safetensors"))
    with pytest.raises(ValidationError, match="no shared 2D"):
        validate_checkpoints(a, b)
