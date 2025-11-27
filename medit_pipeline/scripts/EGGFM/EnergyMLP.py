# EnergyMLP.py

from typing import Sequence, Optional
import torch
from torch import nn


class EnergyMLP(nn.Module):
    """
    E(x) = <E_theta(x), x> where E_theta is an MLP with nonlinearities.

    x is HVG, log-normalized expression (optionally mean-centered).
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
        # final layer outputs a vector in R^D
        layers.append(nn.Linear(in_dim, n_genes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D)
        returns: energy (B,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        v = self.net(x)  # (B, D) = E_theta(x)
        energy = (v * x).sum(dim=-1)  # <E_theta(x), x>
        return energy

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
