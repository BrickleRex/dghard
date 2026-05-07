"""DG-Hard: Donoho-Gavish optimal hard-threshold singular value shrinker.

For each 2D parameter delta ``ΔW = W_ft - W_base`` of size m×n:

    U, S, V^T = svd(ΔW)
    σ̂_eff    = sigma_estimator(S, m, n)            # 'dg' (default) or 'ours'
    τ*        = scale · ω(β) · σ̂_eff               # β = min(m,n) / max(m,n)
    S_new     = S · 𝟙[S > τ*]
    ΔW*       = U diag(S_new) V^T
    W_repair  = W_base + ΔW*

The σ̂ scale convention is the *fused* form ``σ̂_eff = σ_n · √n_large`` (algebraically
identical to the canonical Gavish-Donoho 2014 formulation, just with the √n folded
into σ̂ so both estimators land at the same scale).

References:
  - Gavish & Donoho, "The Optimal Hard Threshold for Singular Values is 4/√3",
    IEEE TIT 2014; arXiv:1305.5870.

Scope notes:
  - Tensors that are 1D (biases, layer norms) or smaller than ``min_numel`` are
    passed through unchanged — MP theory needs a 2D random matrix.
  - SVD is computed in fp32 and the result cast back to the base dtype.
  - On CUDA we group same-shape deltas and do one batched SVD per group
    (cuSOLVER gesvdjBatched), with an OOM-safe per-tensor fallback.
"""
from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Literal, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

StateDict = dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# σ̂ estimators
# ---------------------------------------------------------------------------

def sigma_hat_ours(S: torch.Tensor, n_tail_frac: float = 0.25) -> float:
    """Q25-MAD: ``σ̂_eff = mean(bottom n_tail_frac of S) / 0.6745``.

    A robust noise-floor estimator: averages the smallest 25% of singular
    values (which under MP noise hover around the bulk floor) and de-biases
    by the half-normal MAD constant 0.6745.
    """
    n_tail = max(int(len(S) * n_tail_frac), 1)
    return S[-n_tail:].float().mean().item() / 0.6745


@lru_cache(maxsize=1024)
def mp_singular_median(beta: float, n_grid: int = 200_000) -> float:
    """Median of the Marchenko-Pastur singular-value distribution at aspect β.

    The MP eigenvalue density on ``[β_-, β_+]`` with ``β_± = (1 ∓ √β)²`` is

        f(λ) = √((β_+ − λ)(λ − β_-)) / (2π β λ).

    We compute the median ``λ_med`` of this density via fine-grid trapezoidal
    integration of the CDF + linear interpolation at CDF=0.5, then return
    ``√λ_med`` — the median in singular-value space, i.e. the ``μ_β`` that
    appears in the DG σ̂ formula. ``beta`` must be in (0, 1].
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    sqrt_b = math.sqrt(beta)
    lam_minus = (1.0 - sqrt_b) ** 2
    lam_plus = (1.0 + sqrt_b) ** 2
    lam = np.linspace(lam_minus, lam_plus, n_grid + 1)
    radicand = np.maximum((lam_plus - lam) * (lam - lam_minus), 0.0)
    safe_lam = np.where(lam > 0, lam, 1.0)
    f = np.sqrt(radicand) / (2.0 * math.pi * beta * safe_lam)
    f[lam <= 0] = 0.0
    dx = (lam_plus - lam_minus) / n_grid
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[:-1] + f[1:]) * dx)])
    cdf = cdf / cdf[-1]
    idx = int(np.searchsorted(cdf, 0.5))
    if idx <= 0:
        med_eig = lam_minus
    elif idx >= len(cdf):
        med_eig = lam_plus
    else:
        c0, c1 = float(cdf[idx - 1]), float(cdf[idx])
        l0, l1 = float(lam[idx - 1]), float(lam[idx])
        med_eig = l0 + (0.5 - c0) * (l1 - l0) / max(c1 - c0, 1e-300)
    return math.sqrt(med_eig)


def sigma_hat_dg(S: torch.Tensor, m: int, n: int) -> float:
    """DG aspect-aware ``σ̂_eff = median(S) / μ_β``.

    β = min(m,n) / max(m,n). μ_β is cached. Algebraically identical to the
    canonical published formula ``median(S)/(μ_β·√n_large)·√n_large``, just
    with √n_large folded into σ̂ for scale consistency with the rest of the
    pipeline.
    """
    _ = n  # n_large is derived from (m, n)
    n_large = max(m, n)
    beta = min(m, n) / n_large
    mu = mp_singular_median(beta)
    s_med = float(torch.median(S).item())
    return s_med / mu


# ---------------------------------------------------------------------------
# DG hard-threshold math
# ---------------------------------------------------------------------------

def dg_omega(beta: float) -> float:
    """DG hard-threshold aspect coefficient (Gavish-Donoho 2014, eq. 1.1).

    ``ω(β) = √(2(β+1) + 8β / ((β+1) + √(β² + 14β + 1)))``.
    At β=1 this reduces to ``4/√3 ≈ 2.309`` — the eponymous constant.
    """
    return math.sqrt(
        2.0 * (beta + 1.0)
        + 8.0 * beta / ((beta + 1.0) + math.sqrt(beta * beta + 14.0 * beta + 1.0))
    )


def dg_hard_shrink(S: torch.Tensor, sigma_eff: float, m: int, n: int,
                   scale: float = 1.0) -> torch.Tensor:
    """Hard threshold S at ``scale · ω(β) · σ_eff``."""
    n_large = max(m, n)
    beta = min(m, n) / n_large
    threshold = scale * dg_omega(beta) * sigma_eff
    return torch.where(S > threshold, S, torch.zeros_like(S))


# ---------------------------------------------------------------------------
# Vectorized batch versions (for cuSOLVER's gesvdjBatched path on CUDA)
# ---------------------------------------------------------------------------

def _sigma_hat_ours_batch(S: torch.Tensor) -> torch.Tensor:
    n_tail = max(S.shape[-1] // 4, 1)
    return S[..., -n_tail:].mean(dim=-1) / 0.6745


def _sigma_hat_dg_batch(S: torch.Tensor, m: int, n: int) -> torch.Tensor:
    n_large = max(m, n)
    beta = min(m, n) / n_large
    mu = mp_singular_median(beta)
    s_med = torch.median(S, dim=-1).values
    return s_med / mu


def _dg_hard_shrink_batch(S: torch.Tensor, sigma_eff: torch.Tensor,
                          m: int, n: int, scale: float) -> torch.Tensor:
    n_large = max(m, n)
    beta = min(m, n) / n_large
    threshold = (scale * dg_omega(beta)) * sigma_eff
    return torch.where(S > threshold.unsqueeze(1), S, torch.zeros_like(S))


# ---------------------------------------------------------------------------
# Repair class
# ---------------------------------------------------------------------------

def _compute_deltas(base_sd: StateDict, ft_sd: StateDict,
                    eps: float = 1e-8) -> StateDict:
    """Return ``ft - base`` for every key present and meaningfully different.

    Skips: keys missing on either side, shape mismatches, deltas whose
    abs-max < eps (effectively zero — would just add noise to the SVD).
    """
    deltas: StateDict = {}
    for k, ft in ft_sd.items():
        base = base_sd.get(k)
        if base is None:
            continue
        if ft.shape != base.shape:
            continue
        d = ft.float() - base.float()
        if d.abs().max().item() <= eps:
            continue
        deltas[k] = d
    return deltas


class DgHardRepair:
    """Apply DG-Hard to a (base_state_dict, ft_state_dict) pair.

    Grouping: 2D deltas of identical shape are batched into one cuSOLVER SVD
    on CUDA. Singletons and OOM-falling groups go through the per-tensor
    path. CPU always uses per-tensor.
    """

    name: str = "dg_hard"

    def __init__(self, sigma_estimator: Literal["dg", "ours"] = "dg",
                 scale: float = 1.0, min_numel: int = 1024,
                 device: Optional[str] = None,
                 layer_mask: Optional[Sequence[str]] = None) -> None:
        if sigma_estimator not in ("dg", "ours"):
            raise ValueError(
                f"sigma_estimator must be 'dg' or 'ours', got {sigma_estimator!r}"
            )
        self.sigma_estimator = sigma_estimator
        self.scale = float(scale)
        self.min_numel = int(min_numel)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if layer_mask is None or (isinstance(layer_mask, (list, tuple))
                                  and len(layer_mask) == 0):
            self._mask_patterns: Optional[list[re.Pattern[str]]] = None
        else:
            self._mask_patterns = [re.compile(p) for p in layer_mask]
        self.last_stats: dict[str, dict] = {}

    def _matches_mask(self, key: str) -> bool:
        if self._mask_patterns is None:
            return True
        return any(p.search(key) for p in self._mask_patterns)

    def _sigma_per_matrix(self, S: torch.Tensor, m: int, n: int) -> float:
        if self.sigma_estimator == "dg":
            return sigma_hat_dg(S, m, n)
        return sigma_hat_ours(S)

    def _sigma_batch(self, S: torch.Tensor, m: int, n: int) -> torch.Tensor:
        if self.sigma_estimator == "dg":
            return _sigma_hat_dg_batch(S, m, n)
        return _sigma_hat_ours_batch(S)

    def apply(self, base_sd: StateDict, ft_sd: StateDict) -> StateDict:
        """Return the repaired state dict. Inputs are not mutated."""
        deltas = _compute_deltas(base_sd, ft_sd)
        repaired: StateDict = {}
        dev = self.device
        self.last_stats = {}

        # Pass 1: bucket each base param.
        passthrough_names: list[str] = []
        n_masked_out = 0
        by_shape: dict[tuple[int, int],
                       list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
        for name, base in base_sd.items():
            if name not in deltas:
                repaired[name] = base
                passthrough_names.append(name)
                continue
            delta = deltas[name]
            if delta.ndim != 2 or delta.numel() < self.min_numel:
                repaired[name] = base + delta.to(base.dtype)
                passthrough_names.append(name)
                continue
            if not self._matches_mask(name):
                repaired[name] = base + delta.to(base.dtype)
                passthrough_names.append(name)
                n_masked_out += 1
                continue
            by_shape.setdefault(tuple(delta.shape), []).append(
                (name, base, delta)
            )

        total_svd_keys = sum(len(v) for v in by_shape.values())
        n_svd = 0
        n_passthrough = len(passthrough_names)
        pbar = tqdm(total=total_svd_keys, desc=f"{self.name} [{dev.type}]",
                    unit="key", leave=False)

        # Pass 2: process each shape group.
        for shape, items in by_shape.items():
            m, n = shape
            ok = False
            if dev.type == "cuda" and len(items) >= 2:
                try:
                    self._apply_batch(items, m, n, repaired)
                    n_svd += len(items)
                    ok = True
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(
                        "[%s] batched SVD OOM on shape=(%d,%d) B=%d: %s"
                        " — falling back to per-tensor.",
                        self.name, m, n, len(items), e,
                    )
                    torch.cuda.empty_cache()
            if ok:
                pbar.update(len(items))
                continue
            for name, base, delta in items:
                try:
                    self._apply_single(name, base, delta, m, n, repaired)
                    n_svd += 1
                except RuntimeError as e:
                    logger.warning("SVD failed on %s (shape=%s, device=%s): %s",
                                   name, shape, dev.type, e)
                    repaired[name] = base + delta.to(base.dtype)
                    n_passthrough += 1
                pbar.update(1)

        pbar.close()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "[%s sigma=%s scale=%.2f device=%s] svd=%d passthrough=%d "
            "(masked_out_2d=%d) shape_groups=%d",
            self.name, self.sigma_estimator, self.scale, dev.type,
            n_svd, n_passthrough, n_masked_out, len(by_shape),
        )
        return repaired

    # ---- per-tensor and batched SVD paths ------------------------------

    def _apply_single(self, name: str, base: torch.Tensor,
                      delta: torch.Tensor, m: int, n: int,
                      repaired: StateDict) -> None:
        dev = self.device
        d = delta.to(device=dev, dtype=torch.float32, non_blocking=True)
        U, S, Vt = torch.linalg.svd(d, full_matrices=False)
        sigma = self._sigma_per_matrix(S, m, n)
        S_new = dg_hard_shrink(S, sigma, m, n, scale=self.scale)
        delta_new = (U * S_new.unsqueeze(0)) @ Vt
        repaired[name] = base + delta_new.to(device=base.device,
                                             dtype=base.dtype)
        kept = int((S_new > 0).sum().item())
        self.last_stats[name] = {
            "shape": [m, n],
            "rank_total": int(S.numel()),
            "rank_kept": kept,
            "sigma_hat": float(sigma),
            "threshold": float(self.scale * dg_omega(min(m, n) / max(m, n)) * sigma),
            "frob_delta_in": float(d.norm().item()),
            "frob_delta_out": float(delta_new.norm().item()),
        }
        del d, U, S, Vt, S_new, delta_new

    def _apply_batch(
        self,
        items: list[tuple[str, torch.Tensor, torch.Tensor]],
        m: int, n: int,
        repaired: StateDict,
    ) -> None:
        dev = self.device
        batch = torch.stack(
            [it[2].to(device=dev, dtype=torch.float32, non_blocking=True)
             for it in items],
            dim=0,
        )
        U, S, Vt = torch.linalg.svd(batch, full_matrices=False)
        sigma = self._sigma_batch(S, m, n)
        S_new = _dg_hard_shrink_batch(S, sigma, m, n, scale=self.scale)
        delta_new = (U * S_new.unsqueeze(1)) @ Vt
        thr_factor = self.scale * dg_omega(min(m, n) / max(m, n))
        for i, (name, base, _) in enumerate(items):
            repaired[name] = base + delta_new[i].to(device=base.device,
                                                    dtype=base.dtype)
            kept = int((S_new[i] > 0).sum().item())
            self.last_stats[name] = {
                "shape": [m, n],
                "rank_total": int(S.shape[-1]),
                "rank_kept": kept,
                "sigma_hat": float(sigma[i].item()),
                "threshold": float(thr_factor * sigma[i].item()),
                "frob_delta_in": float(batch[i].norm().item()),
                "frob_delta_out": float(delta_new[i].norm().item()),
            }
        del batch, U, S, Vt, S_new, delta_new


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------

def apply_dg_hard(
    base_sd: StateDict, ft_sd: StateDict,
    sigma_estimator: Literal["dg", "ours"] = "dg",
    scale: float = 1.0,
    min_numel: int = 1024,
    device: Optional[str] = None,
    layer_mask: Optional[Sequence[str]] = None,
) -> StateDict:
    """One-shot DG-Hard: returns the repaired state dict."""
    return DgHardRepair(
        sigma_estimator=sigma_estimator,
        scale=scale,
        min_numel=min_numel,
        device=device,
        layer_mask=layer_mask,
    ).apply(base_sd, ft_sd)
