# src/diffusion/networks.py

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Minimal time embedding: maps scalar timesteps t to a feature vector.

    This is deliberately simple; later you can swap in sinusoidal or
    positional encodings if you want something fancier.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timesteps.

        Returns:
            (B, dim) embedding.
        """
        t = t.float().unsqueeze(-1)  # (B, 1)
        return torch.relu(self.lin(t))


class MLPDenoiser(nn.Module):
    """
    Simple MLP denoiser:

    Input:
        x_t: (B, D) noised data
        t:   (B,) timesteps

    Output:
        eps_pred: (B, D) predicted noise
    """

    def __init__(
        self,
        x_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape (B, D), got {x_t.shape}")
        if t.ndim != 1:
            raise ValueError(f"t must have shape (B,), got {t.shape}")

        t_emb = self.time_mlp(t)              # (B, time_dim)
        h = torch.cat([x_t, t_emb], dim=-1)   # (B, D + time_dim)
        return self.net(h)                    # (B, D)
