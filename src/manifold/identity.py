# src/manifold/identity.py

import torch

from .base import Manifold


class IdentityManifold(Manifold):
    """
    Trivial manifold: does nothing.
    Useful as a placeholder and as a baseline manifold.
    """

    def __init__(self) -> None:
        self.dim: int | None = None

    def fit(self, X: torch.Tensor) -> None:
        """
        'Fit' the manifold. For identity, we just record dimensionality.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected X to have shape (N, D), got {X.shape}")
        self.dim = X.shape[1]

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Identity projection: return X unchanged.
        """
        return X
