# EnergyMLP.py

from typing import Sequence, Optional
import torch
from torch import nn


class EnergyMLP(nn.Module):
    """
    E(x) = <E_theta(x), x> where E_theta is an MLP with nonlinearities.

    x is HVG, log-normalized expression (optionally mean-centered).

    We also expose a latent representation z(x) from the last hidden layer,
    which can be used as a geometry for manifold learning.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: Sequence[int] = (512, 512, 512, 512),
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.Softplus()

        layers = []
        in_dim = n_genes
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation)
            in_dim = h

        # encoder: maps x → z in R^{hidden_dims[-1]}
        self.hidden = nn.Sequential(*layers)

        # head: maps z → v(x) in R^{D} (D = n_genes)
        self.vector_head = nn.Linear(in_dim, n_genes)

        # store for convenience
        self.n_genes = n_genes
        self.latent_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward used in training:
          x: (B, D)
          returns: energy (B,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        z = self.hidden(x)               # (B, latent_dim)
        v = self.vector_head(z)          # (B, D)
        energy = (v * x).sum(dim=-1)     # <v(x), x>
        return energy

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return latent representation z(x) from the last hidden layer.
        x: (B, D)
        returns: z (B, latent_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.hidden(x)
        return z

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        score(x) ≈ ∇_x log p(x) = -∇_x E(x)
        """
        x = x.clone().detach().requires_grad_(True)
        energy = self.forward(x)  # (B,)
        energy_sum = energy.sum()
        (grad,) = torch.autograd.grad(
            energy_sum,
            x,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )
        score = -grad
        return score
