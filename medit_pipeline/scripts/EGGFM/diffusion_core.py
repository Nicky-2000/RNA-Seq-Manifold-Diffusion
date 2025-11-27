# EGGFM/diffusion_core.py

from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


class DiffusionMapBuilder:
    """
    Metric- and norm-agnostic diffusion map constructor.
    """

    def __init__(self, diff_cfg: Dict[str, Any]):
        self.diff_cfg = diff_cfg

    def build_from_distances(
        self,
        n_cells: int,
        rows: np.ndarray,
        cols: np.ndarray,
        dist_vals: np.ndarray,
    ) -> np.ndarray:
        diff_cfg = self.diff_cfg
        eps_mode = diff_cfg.get("eps_mode", "median")
        eps_value = diff_cfg.get("eps_value", 1.0)
        n_comps = diff_cfg.get("n_comps", 30)
        t = diff_cfg.get("t", 1.0)
        p = float(diff_cfg.get("distance_power", 1.0))

        dist_p = dist_vals**p

        if eps_mode == "median":
            eps = np.median(dist_p)
        elif eps_mode == "fixed":
            eps = float(eps_value)
        else:
            raise ValueError(f"Unknown eps_mode: {eps_mode}")
        print(f"[DiffusionMap] using eps = {eps:.4g} (power p={p})", flush=True)

        if diff_cfg.get("eps_trunc") == "yes":
            q_low = np.quantile(dist_p, 0.05)
            q_hi = np.quantile(dist_p, 0.98)
            dist_p = np.clip(dist_p, q_low, q_hi)
            print(
                f"[DiffusionMap] eps_trunc=yes, clipped d^p to [{q_low:.4g}, {q_hi:.4g}]",
                flush=True,
            )

        W_vals = np.exp(-dist_p / eps)
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
