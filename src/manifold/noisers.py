# src/manifold/noisers.py

import torch

from .base import Noiser


class GaussianNoiser(Noiser):
    """
    Standard DDPM-style Gaussian forward diffusion.

    Given a beta schedule (length T), we precompute alpha_bar_t and use the
    closed-form q(x_t | x_0):

        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

    where eps ~ N(0, I).
    """

    def __init__(self, betas: torch.Tensor) -> None:
        """
        Args:
            betas: (T,) 1D tensor of beta_t values in (0, 1).
        """
        if betas.ndim != 1:
            raise ValueError(f"betas must be 1D (T,), got shape {betas.shape}")

        self.betas = betas.clone().detach()
        alphas = 1.0 - self.betas            # (T,)
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # (T,)

    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Args:
            x0: (B, D) clean data.
            t:  (B,) integer timesteps in [0, T).

        Returns:
            x_t: (B, D) noised data.
            eps: (B, D) noise used.
        """
        if x0.ndim != 2:
            raise ValueError(f"x0 must have shape (B, D), got {x0.shape}")
        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,), got {t.shape}")

        device = x0.device
        T = self.alpha_bar.size(0)

        t_max = int(t.max().item())
        if t_max >= T:
            raise ValueError(f"Found timestep t={t_max}, but T={T}.")

        alpha_bar_t = self.alpha_bar.to(device)[t]  # (B,)
        alpha_bar_t = alpha_bar_t.view(-1, 1)       # (B, 1)

        eps = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        return x_t, eps
