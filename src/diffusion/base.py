# src/diffusion/base.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from manifold import Noiser  # from your manifold/__init__.py


class DiffusionModel(nn.Module):
    """
    Generic diffusion model with a pluggable Noiser and denoiser network.

    The only thing this class assumes about the noiser is that it implements:

        x_t, eps = noiser.add_noise(x0, t)

    where:
        x0: (B, D) clean data
        t:  (B,) integer timesteps

    The denoiser must implement:

        eps_pred = denoiser(x_t, t)
    """

    def __init__(
        self,
        denoiser: nn.Module,
        noiser: Noiser,
        num_timesteps: int,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.noiser = noiser
        self.T = num_timesteps

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Alias for training_step so you can call model(x0) directly.
        """
        return self.training_step(x0)

    def training_step(self, x0: torch.Tensor) -> torch.Tensor:
        """
        One training step:

        1. Sample random timestep t for each batch element.
        2. Use the Noiser to generate (x_t, eps).
        3. Use the denoiser to predict eps from (x_t, t).
        4. Return MSE loss between eps_pred and eps.
        """
        if x0.ndim != 2:
            raise ValueError(f"x0 must have shape (B, D), got {x0.shape}")

        B = x0.size(0)
        device = x0.device

        # Sample timesteps uniformly from [0, T)
        t = torch.randint(0, self.T, (B,), device=device)  # (B,)

        # Forward diffusion step
        x_t, eps = self.noiser.add_noise(x0, t)  # both (B, D)

        # Predict noise
        eps_pred = self.denoiser(x_t, t)         # (B, D)

        # Standard DDPM loss: MSE between true and predicted noise
        loss = F.mse_loss(eps_pred, eps)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        dim: int,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """
        Placeholder for reverse sampling.

        Later you can implement a full reverse process:
          - start from x_T ~ N(0, I)
          - step backwards using the denoiser
          - follow DDPM / score-based updates.

        For now, left unimplemented so we can focus on forward training.
        """
        raise NotImplementedError("Sampling not implemented yet.")
