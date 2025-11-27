# EGGFM/diffusion_map.py

from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


def build_diffusion_map_from_l2(
    n_cells: int,
    rows: np.ndarray,
    cols: np.ndarray,
    l2_vals: np.ndarray,
    diff_cfg: Dict[str, Any],
):
    """
    Given squared edge lengths ℓ_ij^2 on a kNN graph, build a diffusion map embedding.
    This is the lower half of your current build_eggfm_diffmap.
    """

    eps_mode = diff_cfg.get("eps_mode", "median")
    eps_value = diff_cfg.get("eps_value", 1.0)
    n_comps = diff_cfg.get("n_comps", 30)
    t = diff_cfg.get("t", 1.0)

    if eps_mode == "median":
        eps = np.median(l2_vals)
    elif eps_mode == "fixed":
        eps = float(eps_value)
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")
    print(f"[DiffusionMap] using eps = {eps:.4g}", flush=True)

    if diff_cfg.get("eps_trunc") == "yes":
        q_low = np.quantile(l2_vals, 0.05)
        q_hi = np.quantile(l2_vals, 0.98)
        l2_vals = np.clip(l2_vals, q_low, q_hi)
        print(
            f"[DiffusionMap] eps_trunc=yes, clipped l2_vals to [{q_low:.4g}, {q_hi:.4g}]",
            flush=True,
        )

    W_vals = np.exp(-l2_vals / eps)
    W = sparse.csr_matrix((W_vals, (rows, cols)), shape=(n_cells, n_cells))
    W = 0.5 * (W + W.T)

    d = np.array(W.sum(axis=1)).ravel()
    d_safe = np.maximum(d, 1e-12)
    D_inv = sparse.diags(1.0 / d_safe)
    P = D_inv @ W

    k_eigs = n_comps + 1
    print("[DiffusionMap] computing eigenvectors...", flush=True)
    eigvals, eigvecs = eigs(P.T, k=k_eigs, which="LR")

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lambdas = eigvals[1 : n_comps + 1]
    phis = eigvecs[:, 1 : n_comps + 1]

    diff_coords = phis * (lambdas**t)
    print("[DiffusionMap] finished. Embedding shape:", diff_coords.shape, flush=True)
    return diff_coords.astype(np.float32)
