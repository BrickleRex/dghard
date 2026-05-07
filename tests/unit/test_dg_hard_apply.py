"""Tests for the DG-Hard apply() pipeline: passthrough, masking, batch path."""
from __future__ import annotations

import pytest
import torch

from dghard.repair.dg_hard import DgHardRepair, apply_dg_hard


def test_apply_returns_full_state_dict_with_same_keys(tiny_state_dicts):
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, sigma_estimator="dg", scale=1.0)
    assert set(out.keys()) == set(base.keys())


def test_apply_does_not_mutate_inputs(tiny_state_dicts):
    base, ft = tiny_state_dicts
    base_copy = {k: v.clone() for k, v in base.items()}
    ft_copy = {k: v.clone() for k, v in ft.items()}
    apply_dg_hard(base, ft, scale=1.0)
    for k in base:
        assert torch.equal(base[k], base_copy[k]), f"base[{k}] mutated"
    for k in ft:
        assert torch.equal(ft[k], ft_copy[k]), f"ft[{k}] mutated"


def test_passthrough_for_1d_tensors(tiny_state_dicts):
    """1D biases must come back as base + raw_delta — no SVD."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1.0)
    bias_key = "transformer.layers.0.attn.q_proj.bias"
    expected = base[bias_key] + (ft[bias_key] - base[bias_key])
    assert torch.allclose(out[bias_key], expected, atol=1e-6)


def test_passthrough_for_below_min_numel(tiny_state_dicts):
    """small_2d.weight is 8x8 = 64 elements < min_numel=1024 default — passthrough."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1.0, min_numel=1024)
    expected = base["small_2d.weight"] + (ft["small_2d.weight"] - base["small_2d.weight"])
    assert torch.allclose(out["small_2d.weight"], expected, atol=1e-6)


def test_unchanged_tensor_returns_base(tiny_state_dicts):
    """A tensor with zero delta must come back exactly equal to base."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1.0)
    assert torch.equal(out["untouched.weight"], base["untouched.weight"])


def test_repaired_delta_has_smaller_norm_than_ft_delta(tiny_state_dicts):
    """The point of DG-Hard: ‖Δ*‖ ≤ ‖Δ_FT‖ (with equality only if no shrinkage)."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1.0)
    key = "transformer.layers.0.mlp.up_proj.weight"
    ft_delta_norm = (ft[key] - base[key]).norm().item()
    repaired_delta_norm = (out[key] - base[key]).norm().item()
    assert repaired_delta_norm <= ft_delta_norm + 1e-5


def test_scale_zero_preserves_full_ft_delta(tiny_state_dicts):
    """scale=0 -> threshold=0 -> every singular value kept -> Δ* == Δ_FT."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=0.0)
    key = "transformer.layers.0.mlp.up_proj.weight"
    assert torch.allclose(out[key], ft[key], atol=1e-4)


def test_scale_huge_returns_base_for_2d_weights(tiny_state_dicts):
    """Big enough scale -> threshold above every singular value -> Δ* == 0
    -> output equals base on every SVD-eligible 2D weight."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1e6)
    for key in ("transformer.layers.0.attn.q_proj.weight",
                "transformer.layers.0.mlp.up_proj.weight",
                "embeddings.weight"):
        assert torch.allclose(out[key], base[key], atol=1e-4), (
            f"{key} should equal base when scale is huge"
        )


def test_layer_mask_skips_unmatched_2d(tiny_state_dicts):
    """layer_mask=['mlp'] should leave non-mlp 2D weights at the raw FT delta."""
    base, ft = tiny_state_dicts
    out = apply_dg_hard(base, ft, scale=1.0, layer_mask=["mlp"])
    # q_proj.weight is 2D but doesn't match mask -> raw delta, equals ft
    assert torch.allclose(out["transformer.layers.0.attn.q_proj.weight"],
                          ft["transformer.layers.0.attn.q_proj.weight"],
                          atol=1e-4)


def test_last_stats_populated(tiny_state_dicts):
    base, ft = tiny_state_dicts
    repair = DgHardRepair(scale=1.0)
    repair.apply(base, ft)
    # Two SVD-eligible 2D matrices: q_proj.weight (64x64) and up_proj.weight
    # (128x64) and embeddings.weight (100x64). small_2d skipped (min_numel),
    # untouched skipped (zero delta).
    assert "transformer.layers.0.mlp.up_proj.weight" in repair.last_stats
    assert "transformer.layers.0.attn.q_proj.weight" in repair.last_stats
    assert "embeddings.weight" in repair.last_stats
    stats = repair.last_stats["transformer.layers.0.mlp.up_proj.weight"]
    assert stats["shape"] == [128, 64]
    assert 0 <= stats["rank_kept"] <= stats["rank_total"]
    assert stats["sigma_hat"] > 0
    assert stats["threshold"] > 0


def test_invalid_sigma_estimator_raises():
    base = {"w": torch.randn(64, 64)}
    ft = {"w": torch.randn(64, 64)}
    with pytest.raises(ValueError, match="sigma_estimator"):
        DgHardRepair(sigma_estimator="garbage").apply(base, ft)


def test_dg_vs_ours_estimators_differ_on_same_input(tiny_state_dicts):
    """Sanity: the two sigma_hat estimators are not aliases — they produce
    different repaired weights."""
    base, ft = tiny_state_dicts
    out_dg = apply_dg_hard(base, ft, sigma_estimator="dg", scale=1.0)
    out_ours = apply_dg_hard(base, ft, sigma_estimator="ours", scale=1.0)
    key = "transformer.layers.0.mlp.up_proj.weight"
    assert not torch.allclose(out_dg[key], out_ours[key], atol=1e-4)
