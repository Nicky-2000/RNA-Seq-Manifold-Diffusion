# src/data/loaders.py

from typing import Literal
import torch

from sklearn.datasets import make_swiss_roll

DatasetName = Literal["swiss_roll"]  # add "rna" or "rna_h5ad" later

# -------- SWISS ROLL -------- #

def load_swiss_roll(
    n_samples: int = 2000,
    noise: float = 0.1,
    n_dims: int = 3,
) -> torch.Tensor:
    """
    Generate a swiss roll toy dataset.

    Returns:
        X: tensor (N, n_dims)
    """
    if n_dims < 1 or n_dims > 3:
        raise ValueError("n_dims must be between 1 and 3 for swiss roll (max 3).")

    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)

    # Keep only first n_dims
    X = X[:, :n_dims]

    return torch.from_numpy(X).float()


# -------- UNIFIED ENTRYPOINT -------- #

def load_dataset(
    name: DatasetName = "swiss_roll",
    **kwargs,
) -> torch.Tensor:
    """
    Unified loader for any dataset added to the project.
    Currently only swiss roll.
    """
    if name == "swiss_roll":
        return load_swiss_roll(**kwargs)

    raise ValueError(f"Unknown dataset name: {name}")
