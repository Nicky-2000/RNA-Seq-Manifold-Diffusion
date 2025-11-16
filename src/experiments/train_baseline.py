# src/experiments/train_baseline.py

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataset, identity_preprocess
from manifold import IdentityManifold, GaussianNoiser
from diffusion import DiffusionModel, MLPDenoiser
from utils.logging import set_seed


def main():
    # ----- Setup -----
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- 1. Load + (identity) preprocess data -----
    X = load_dataset("swiss_roll", n_samples=5000, noise=0.1, n_dims=3)
    X = identity_preprocess(X)  # currently a no-op, but keeps the hook

    print(f"Loaded data with shape: {X.shape}")  # (N, D)

    # ----- 2. Fit manifold (trivial for IdentityManifold) -----
    manifold = IdentityManifold()
    manifold.fit(X)  # does nothing but record dim
    print("Fitted IdentityManifold.")

    # ----- 3. Create noiser (Gaussian DDPM forward process) -----
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T)
    noiser = GaussianNoiser(betas=betas)
    print(f"Created GaussianNoiser with T={T} timesteps.")

    # ----- 4. Create denoiser network + diffusion model -----
    x_dim = X.shape[1]
    denoiser = MLPDenoiser(x_dim=x_dim).to(device)
    model = DiffusionModel(denoiser=denoiser, noiser=noiser, num_timesteps=T).to(device)

    # ----- 5. DataLoader -----
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

    # ----- 6. Optimizer -----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ----- 7. Training loop (tiny, just to test pipeline) -----
    num_epochs = 3
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for (x0_batch,) in dl:
            x0_batch = x0_batch.to(device)

            loss = model.training_step(x0_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"[baseline] epoch={epoch+1}/{num_epochs}  loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
