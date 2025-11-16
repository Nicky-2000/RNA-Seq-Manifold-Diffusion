# src/manifold/base.py

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Manifold(ABC):
    """
    Abstract manifold interface.

    In the simplest case, this can just be an identity manifold
    that does nothing. Later, you can implement:
      - PCA manifolds
      - kNN + local tangent manifolds
      - AE/latent manifolds, etc.

    Everything operates on tensors of shape (N, D) or (B, D).
    """

    @abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        """
        Learn any manifold structure from data X.

        Args:
            X: (N, D) tensor of data points.
        """
        ...

    @abstractmethod
    def project(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project arbitrary points back onto the manifold.

        For an identity manifold, this is just `return X`.

        Args:
            X: (B, D) or (N, D) tensor.

        Returns:
            Tensor of same shape as X.
        """
        ...


class Noiser(ABC):
    """
    Abstract forward diffusion noise process.

    The DiffusionModel will only depend on this interface.
    Different Noiser implementations can:
      - ignore the manifold (standard Gaussian)
      - use manifold information (e.g., tangent-space noise)
    """

    @abstractmethod
    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data x0 at timesteps t.

        Args:
            x0: (B, D) clean data.
            t:  (B,) integer timesteps in [0, T).

        Returns:
            x_t:  (B, D) noised data at time t.
            eps:  (B, D) noise that was actually applied.
        """
        ...
