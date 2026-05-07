"""Tests for the closed-form DG-Hard math: omega(beta), sigma_hat, threshold."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dghard.repair.dg_hard import (
    dg_omega,
    dg_hard_shrink,
    mp_singular_median,
    sigma_hat_dg,
    sigma_hat_ours,
)


def test_dg_omega_square_is_4_over_sqrt3():
    """The eponymous Gavish-Donoho 2014 constant: omega(1) = 4/sqrt(3)."""
    assert dg_omega(1.0) == pytest.approx(4.0 / math.sqrt(3.0), rel=1e-12)


@pytest.mark.parametrize("beta", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_dg_omega_matches_paper_formula(beta: float):
    """omega(beta) = sqrt(2(b+1) + 8b / ((b+1) + sqrt(b^2 + 14b + 1)))."""
    expected = math.sqrt(
        2.0 * (beta + 1.0)
        + 8.0 * beta / ((beta + 1.0) + math.sqrt(beta * beta + 14.0 * beta + 1.0))
    )
    assert dg_omega(beta) == pytest.approx(expected, rel=1e-12)


def test_dg_omega_monotone_in_beta():
    """omega is monotone increasing in beta on (0, 1]."""
    betas = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    vals = [dg_omega(b) for b in betas]
    for i in range(len(vals) - 1):
        assert vals[i] < vals[i + 1], f"omega not monotone at idx {i}"


def test_mp_singular_median_beta_one():
    """At beta=1 the MP eigenvalue distribution lives on [0, 4] with
    eigenvalue median ~0.656; the singular-value median is sqrt of that,
    ~0.81. We just assert the value is in a sane window."""
    mu = mp_singular_median(1.0)
    assert 0.7 < mu < 0.9


def test_mp_singular_median_smaller_beta_pushes_mu_toward_1():
    """Aspect-ratio-free limit (beta -> 0): MP density concentrates near
    lambda=1, so mu(beta) -> 1 from below as beta shrinks."""
    assert mp_singular_median(0.1) > mp_singular_median(0.5) > mp_singular_median(1.0)


def test_mp_singular_median_caches():
    """lru_cache means a second call with the same beta hits the cache."""
    mp_singular_median.cache_clear()
    mp_singular_median(0.5)
    info_after_first = mp_singular_median.cache_info()
    mp_singular_median(0.5)
    info_after_second = mp_singular_median.cache_info()
    assert info_after_second.hits == info_after_first.hits + 1


def test_mp_singular_median_rejects_invalid_beta():
    with pytest.raises(ValueError):
        mp_singular_median(0.0)
    with pytest.raises(ValueError):
        mp_singular_median(1.5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sigma_hat_dg_recovers_noise_std(seed: int):
    """For an iid Gaussian m x n matrix with std sigma_n, sigma_hat_dg should
    return ~sigma_n * sqrt(n_large) within ~5%."""
    g = torch.Generator().manual_seed(seed)
    m, n = 256, 128
    sigma_n = 0.05
    X = sigma_n * torch.randn(m, n, generator=g)
    S = torch.linalg.svdvals(X)
    sigma_eff_true = sigma_n * math.sqrt(max(m, n))
    est = sigma_hat_dg(S, m, n)
    assert est == pytest.approx(sigma_eff_true, rel=0.05)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sigma_hat_ours_scales_with_noise(seed: int):
    """Q25-MAD is a *biased* estimator (the bottom-25% mean of SVs lies well
    below the bulk median), so we don't compare against sigma_n*sqrt(n_large)
    directly. We assert: it's positive, finite, and *scales linearly* with
    the noise std — the property that matters for thresholding."""
    g = torch.Generator().manual_seed(seed)
    m, n = 256, 128
    X1 = torch.randn(m, n, generator=g)
    g2 = torch.Generator().manual_seed(seed)
    X10 = 10.0 * torch.randn(m, n, generator=g2)
    e1 = sigma_hat_ours(torch.linalg.svdvals(X1))
    e10 = sigma_hat_ours(torch.linalg.svdvals(X10))
    assert e1 > 0 and e10 > 0
    assert e10 / e1 == pytest.approx(10.0, rel=1e-4)


def test_dg_hard_shrink_threshold_kills_below_keeps_above():
    """Hard-threshold semantics: S_new[i] = S[i] iff S[i] > scale * omega(beta) * sigma."""
    S = torch.tensor([0.1, 0.5, 1.0, 2.0, 3.0])
    sigma_eff = 0.5
    m, n = 100, 100  # beta = 1 -> omega = 4/sqrt(3) ~ 2.309
    threshold = 1.0 * dg_omega(1.0) * sigma_eff  # ~ 1.155
    S_new = dg_hard_shrink(S, sigma_eff, m, n, scale=1.0)
    expected_kept = S > threshold
    assert torch.equal(S_new > 0, expected_kept)
    assert torch.allclose(S_new[expected_kept], S[expected_kept])


def test_dg_hard_shrink_scale_zero_keeps_everything():
    """scale=0 -> threshold=0 -> every singular value > 0 is kept."""
    S = torch.tensor([0.01, 0.1, 1.0, 10.0])
    out = dg_hard_shrink(S, sigma_eff=1.0, m=10, n=10, scale=0.0)
    assert torch.allclose(out, S)


def test_dg_hard_shrink_scale_huge_kills_everything():
    """A huge scale pushes the threshold above any plausible singular value."""
    S = torch.tensor([0.01, 0.1, 1.0, 10.0])
    out = dg_hard_shrink(S, sigma_eff=1.0, m=10, n=10, scale=1000.0)
    assert torch.allclose(out, torch.zeros_like(out))


def test_dg_hard_recovers_low_rank_signal_under_noise():
    """End-to-end behavioral test of the math: a rank-3 signal plus iid noise
    should be cleanly separated by DG-Hard."""
    g = torch.Generator().manual_seed(42)
    m, n, k_true = 256, 128, 3
    # Strong rank-3 signal.
    U = torch.randn(m, k_true, generator=g)
    V = torch.randn(n, k_true, generator=g)
    signal = U @ V.T * 5.0
    noise = 0.05 * torch.randn(m, n, generator=g)
    delta = signal + noise

    U_, S, Vt = torch.linalg.svd(delta, full_matrices=False)
    sigma = sigma_hat_dg(S, m, n)
    S_new = dg_hard_shrink(S, sigma, m, n, scale=1.0)

    kept = int((S_new > 0).sum().item())
    # We should keep close to 3 components — DG-Hard is sample-dependent so
    # allow [k_true, k_true + 2].
    assert k_true <= kept <= k_true + 2, f"expected ~{k_true} kept, got {kept}"
