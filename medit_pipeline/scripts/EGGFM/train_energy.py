from typing import Dict, Any
import torch
from torch import optim
from torch.utils.data import DataLoader

from .energy_model import EnergyMLP
from .AnnDataPyTorch import AnnDataExpressionDataset
import scanpy as sc

def train_energy_model(
    ad_prep,  # output of prep(ad, params)
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> EnergyMLP:
    """
    Train an energy-based model on preprocessed AnnData using denoising score matching.

    model_cfg (from params['eggfm_model']), e.g.:
        hidden_dims: [512, 512, 512, 512]

    train_cfg (from params['eggfm_train']), e.g.:
        batch_size: 2048
        num_epochs: 50
        lr: 1e-4
        sigma: 0.1
        device: "cuda"

        # optional early stopping:
        early_stop_patience: 0   # 0 => disable early stopping (default)
        early_stop_min_delta: 0.0
    """
    sc.pp.pca(ad_prep, n_comps=50)

    # Device
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = AnnDataExpressionDataset(ad_prep)
    n_genes = ad_prep.o["X_pca"]

    # Model
    hidden_dims = model_cfg.get("hidden_dims", (512, 512, 512, 512))
    model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=hidden_dims,
    ).to(device)

    # Training hyperparameters (YAML overrides defaults)
    batch_size = int(train_cfg.get("batch_size", 2048))
    num_epochs = int(train_cfg.get("num_epochs", 50))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.1))

    # Early stopping hyperparams
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))  # 0 = off
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improve = 0

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
            energy = model(y)  # (B,)
            energy_sum = energy.sum()
            (grad_y,) = torch.autograd.grad(
                energy_sum,
                y,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )
            s_theta = -grad_y  # (B, D)

            # DSM target: -(y - x) / sigma^2
            target = -(y - x) / (sigma**2)

            # MSE over batch and dimensions
            loss = ((s_theta - target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataset)
        print(
            f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  loss={epoch_loss:.4f}",
            flush=True,
        )

        # ---- early stopping bookkeeping ----
        if epoch_loss + early_stop_min_delta < best_loss:
            best_loss = epoch_loss
            best_state_dict = model.state_dict()
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(
                f"[Energy DSM] Early stopping at epoch {epoch+1} "
                f"(best_loss={best_loss:.4f})",
                flush=True,
            )
            break

    # Restore best weights if we tracked them
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model
