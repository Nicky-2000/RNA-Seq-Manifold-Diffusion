# diffmap_eggfm.py

from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp_sparse

import torch
from torch import Tensor


def hessian_quadratic_form_batched(
    energy_model,
    X_batch: np.ndarray,  # (B, D) points x_b
    V_batch: np.ndarray,  # (B, D) directions v_b
    device: str,
    mode: str = "Hv_norm2",
) -> np.ndarray:
    ...
    # unchanged from your version
    ...


def build_eggfm_diffmap(
    ad_prep,
    energy_model,
    diff_cfg: Dict[str, Any],
):
    """
    Build an EGGFM-aware Diffusion Map embedding using a metric induced by
    the Hessian of the energy model via batched Hessian-vector products.
    """
    X = ad_prep.X
    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_cells, D = X.shape

    n_neighbors = diff_cfg.get("n_neighbors", 30)
    n_comps = diff_cfg.get("n_comps", 30)
    device = diff_cfg.get("device", "cuda")
    device = device if torch.cuda.is_available() else "cpu"
    eps_mode = diff_cfg.get("eps_mode", "median")
    eps_value = diff_cfg.get("eps_value", 1.0)
    hvp_mode = diff_cfg.get("hvp_mode", "Hv_norm2")
    hvp_batch_size = diff_cfg.get("hvp_batch_size", 1024)
    t = diff_cfg.get("t", 1.0)

    # Move model to device once
    energy_model = energy_model.to(device)
    energy_model.eval()

    print(f"[EGGFM DiffMap] X shape: {X.shape}", flush=True)

    # 1) kNN for neighbor selection (Euclidean, only for neighbor indices)
    print(
        "[EGGFM DiffMap] building kNN graph (euclidean for neighbor selection)...",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # neighbors per cell (excluding self)
    k = indices.shape[1] - 1
    assert k == n_neighbors, "indices second dimension should be n_neighbors+1"

    # 2) Flatten edges
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    n_edges = rows.shape[0]

    print(f"[EGGFM DiffMap] total edges (directed): {n_edges}", flush=True)

    # 3) Compute metric-aware edge lengths ℓ_ij^2 via batched HVPs
    l2_vals = np.empty(n_edges, dtype=np.float64)

    print(
        "[EGGFM DiffMap] computing Hessian-based edge lengths in batches...", flush=True
    )
    n_batches = (n_edges + hvp_batch_size - 1) // hvp_batch_size

    for b in range(n_batches):
        start = b * hvp_batch_size
        end = min((b + 1) * hvp_batch_size, n_edges)
        if start >= end:
            break

        i_batch = rows[start:end]
        j_batch = cols[start:end]

        Xi_batch = X[i_batch]  # (B, D)
        Xj_batch = X[j_batch]  # (B, D)
        V_batch = Xj_batch - Xi_batch  # (B, D)

        q_batch = hessian_quadratic_form_batched(
            energy_model,
            Xi_batch,
            V_batch,
            device=device,
            mode=hvp_mode,
        )
        l2_vals[start:end] = q_batch

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(
                f"  processed batch {b+1}/{n_batches} ({end} / {n_edges} edges)",
                flush=True,
            )

    # 3.5) Clip extreme metric values (robust metric quantiles)
    q_low = np.quantile(l2_vals, 0.05)
    q_hi = np.quantile(l2_vals, 0.98)
    l2_vals = np.clip(l2_vals, q_low, q_hi)

    # 4) Choose kernel bandwidth ε
    if eps_mode == "median":
        eps = np.median(l2_vals)
    elif eps_mode == "fixed":
        eps = float(eps_value)
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")
    print(f"[EGGFM DiffMap] using eps = {eps:.4g}", flush=True)

    # 5) Build kernel W_ij = exp(-ℓ_ij^2 / eps)
    W_vals = np.exp(-l2_vals / eps)
    W = sparse.csr_matrix((W_vals, (rows, cols)), shape=(n_cells, n_cells))
    # Symmetrize for robustness
    W = 0.5 * (W + W.T)

    # 6) Normalize to Markov matrix P (row-stochastic)
    d = np.array(W.sum(axis=1)).ravel()
    d_safe = np.maximum(d, 1e-12)
    D_inv = sparse.diags(1.0 / d_safe)
    P = D_inv @ W

    # 7) Eigendecompose P^T for diffusion map
    k_eigs = n_comps + 1  # include trivial eigenpair
    print("[EGGFM DiffMap] computing eigenvectors...", flush=True)
    eigvals, eigvecs = eigs(P.T, k=k_eigs, which="LR")  # largest real parts

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # sort by eigenvalue magnitude descending
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # drop trivial eigenvector (λ≈1)
    lambdas = eigvals[1 : n_comps + 1]
    phis = eigvecs[:, 1 : n_comps + 1]  # (n_cells, n_comps)

    # Diffusion map coordinates Ψ_t(x_i) = (λ_1^t φ_1(i), ..., λ_m^t φ_m(i))
    diff_coords = phis * (lambdas**t)

    print("[EGGFM DiffMap] finished. Embedding shape:", diff_coords.shape, flush=True)
    return diff_coords.astype(np.float32)
