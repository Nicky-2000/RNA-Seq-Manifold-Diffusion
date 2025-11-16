# src/metrics/basic.py

from __future__ import annotations

from typing import Dict

import torch

from manifold import Manifold  # for type hints / usage in manifold metrics


def mean_var_stats(real: torch.Tensor, gen: torch.Tensor) -> Dict[str, float]:
    """
    Compare basic distribution statistics between two sets of points.

    Args:
        real: (N, D) tensor of "real" data.
        gen:  (M, D) tensor of "generated" data.

    Returns:
        dict with:
          - mean_diff: L2 norm between means
          - var_diff:  L2 norm between per-dim variances
    """
    if real.ndim != 2 or gen.ndim != 2:
        raise ValueError(
            f"Expected real and gen to have shape (N, D) and (M, D), "
            f"got {real.shape} and {gen.shape}"
        )
    if real.shape[1] != gen.shape[1]:
        raise ValueError(
            f"real and gen must have same feature dim, got {real.shape[1]} and {gen.shape[1]}"
        )

    real_mean = real.mean(dim=0)
    gen_mean = gen.mean(dim=0)
    real_var = real.var(dim=0, unbiased=False)
    gen_var = gen.var(dim=0, unbiased=False)

    mean_diff = torch.norm(real_mean - gen_mean).item()
    var_diff = torch.norm(real_var - gen_var).item()

    return {
        "mean_diff": mean_diff,
        "var_diff": var_diff,
    }


def mse_between_sets(X1: torch.Tensor, X2: torch.Tensor) -> float:
    """
    Compute mean squared error between two sets of points.

    If the number of points differ, this compares the first min(N1, N2).

    Args:
        X1: (N1, D)
        X2: (N2, D)

    Returns:
        scalar MSE over matched points.
    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError(
            f"Expected X1 and X2 to have shape (N, D) and (M, D), "
            f"got {X1.shape} and {X2.shape}"
        )
    if X1.shape[1] != X2.shape[1]:
        raise ValueError(
            f"X1 and X2 must have same feature dim, got {X1.shape[1]} and {X2.shape[1]}"
        )

    n = min(X1.shape[0], X2.shape[0])
    if n == 0:
        raise ValueError("Cannot compute MSE between empty sets.")

    X1_m = X1[:n]
    X2_m = X2[:n]

    mse = torch.mean((X1_m - X2_m) ** 2).item()
    return mse


def manifold_projection_error(
    X: torch.Tensor,
    manifold: Manifold,
) -> float:
    """
    Measure how far points are from a given manifold, on average.

    For a given manifold with a `.project(X)` method, computes:

        mean ||X - project(X)||^2

    Args:
        X: (N, D) tensor of points.
        manifold: Manifold instance (e.g., IdentityManifold for baseline).

    Returns:
        scalar mean squared projection error.
    """
    if X.ndim != 2:
        raise ValueError(f"X must have shape (N, D), got {X.shape}")

    X_proj = manifold.project(X)
    if X_proj.shape != X.shape:
        raise ValueError(
            f"Projected points must have same shape as input, "
            f"got {X_proj.shape} vs {X.shape}"
        )

    err = torch.mean(torch.sum((X - X_proj) ** 2, dim=1)).item()
    return err
