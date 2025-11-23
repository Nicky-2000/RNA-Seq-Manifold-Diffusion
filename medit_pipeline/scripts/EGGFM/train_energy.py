# train_energy.py

from typing import Dict, Any
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .energy_model import EnergyMLP
from .AnnDataPyTorch import AnnDataExpressionDataset


def train_energy_model(
    ad_prep,                      # output of prep(ad, params)
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> EnergyMLP:
    """
    Train an energy-based model on preprocessed AnnData using denoising score matching.

    model_cfg:
        {
            "hidden_dims": [512, 512],
        }

    train_cfg:
        {
            "batch_size": 256,
            "num_epochs": 50,
            "lr": 1e-3,
            "sigma": 0.1,
            "device": "cuda" or "cpu",
        }
    """

    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    dataset = AnnDataExpressionDataset(ad_prep)
    n_genes = dataset.X.shape[1]

    model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=model_cfg.get("hidden_dims", (512, 512)),
    ).to(device)

    batch_size = train_cfg.get("batch_size", 256)
    num_epochs = train_cfg.get("num_epochs", 50)
    lr = train_cfg.get("lr", 1e-3)
    sigma = train_cfg.get("sigma", 0.1)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in loader:
            x = batch.to(device)  # (B, D)
            # Sample Gaussian noise
            eps = torch.randn_like(x)
            y = x + sigma * eps
            y.requires_grad_(True)

            # Predicted score at y: s_theta(y) = -∇_y E(y)
            energy = model(y)                # (B,)
            energy_sum = energy.sum()
            grad_y, = torch.autograd.grad(
                energy_sum,
                y,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )
            s_theta = -grad_y                # (B, D)

            # DSM target: -(y - x) / sigma^2
            target = -(y - x) / (sigma ** 2)

            # MSE over batch and dimensions
            loss = ((s_theta - target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.4f}", flush=True)

    return model
