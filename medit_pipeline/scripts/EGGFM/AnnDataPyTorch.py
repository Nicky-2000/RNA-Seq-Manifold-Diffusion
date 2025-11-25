# dataset_energy.py

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import scanpy as sc  # only for type hints; not strictly necessary


class AnnDataExpressionDataset(Dataset):
    """
    Wraps an AnnData object's X matrix (after prep()) as a PyTorch dataset.
    Uses HVG, log-normalized expression directly.
    """

    def __init__(self, X: sc.AnnData.X, float_dtype: np.dtype = np.float32):
        X = X
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float_dtype)
        # Optional: mean-center features for numerical stability
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.X = X - self.mean_

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[idx])
