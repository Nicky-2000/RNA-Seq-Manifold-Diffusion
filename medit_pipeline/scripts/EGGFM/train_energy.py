# train_energy.py
from typing import Dict, Any
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .EnergyMLP import EnergyMLP
from .AnnDataPyTorch import AnnDataExpressionDataset
import scanpy as sc


def train_energy_model(
    ad_prep,  # output of prep(ad, params)
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
) -> EnergyMLP:
    """
    Train an energy-based model on preprocessed AnnData usjing denoising score matching.
    """

    # -------- device --------
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset: HVG or PCA --------
    latent_space = train_cfg.get("latent_space", "hvg")
    if latent_space == "hvg":
        X = ad_prep.X
    else:
        if "X_pca" not in ad_prep.obsm:
            sc.pp.pca(ad_prep, n_comps=50)
        X = ad_prep.obsm["X_pca"]

    dataset = AnnDataExpressionDataset(X)
    n_genes = dataset.X.shape[1]

    # -------- model --------
    hidden_dims = model_cfg.get("hidden_dims", (512, 512, 512, 512))
    model = EnergyMLP(
        n_genes=n_genes,
        hidden_dims=hidden_dims,
    ).to(device)

    # -------- training hyperparams --------
    batch_size = int(train_cfg.get("batch_size", 2048))
    num_epochs = int(train_cfg.get("num_epochs", 50))
    lr = float(train_cfg.get("lr", 1e-4))
    sigma = float(train_cfg.get("sigma", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))  # 0 = off
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer,T_max = num_epochs,eta_min=lr/10.0)
    best_loss = float("inf")
    best_state_dict = None
    epochs_without_improve = 0

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for xb in loader:
            xb = xb.to(device)  # (B, D)

            # Sample Gaussian noise
            eps = torch.randn_like(xb)
            y = xb + sigma * eps
            y.requires_grad_(True)

            # Energy and score
            energy = model(y)          # (B,)
            energy_sum = energy.sum()  # scalar

            # IMPORTANT: create_graph=True so grad flows back to model params
            (grad_y,) = torch.autograd.grad(
                energy_sum,
                y,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            s_theta = -grad_y  # (B, D)

            # DSM target: -(y - x) / sigma^2
            target = -(y - xb) / (sigma**2)

            # MSE over batch and dimensions
            loss = ((s_theta - target) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        curr_lr = scheduler.get_last_lr()[0]
        # more precision so we can see if it's actually tiny, not exactly zero
        print(
            f"[Energy DSM] Epoch {epoch+1}/{num_epochs}  "
            f"loss={epoch_loss:.6e}  lr={curr_lr:.2e}",
            flush=True,
        )
        
        scheduler.step()
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
                f"(best_loss={best_loss:.6e})",
                flush=True,
            )
            break

    # Restore best weights if we tracked them
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model
