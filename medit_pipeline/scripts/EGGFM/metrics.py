# EGGFM/metrics.py

from typing import Dict, Any
import numpy as np
import torch
from scipy import sparse as sp_sparse


def hessian_quadratic_form_batched(
    energy_model,
    X_batch: np.ndarray,  # (B, D)
    V_batch: np.ndarray,  # (B, D)
    device: str = "cuda",
) -> np.ndarray:
    """
    Your existing v^T H(x) v computation, just moved here.
    """
    energy_model.eval()
    X = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
    V = torch.from_numpy(V_batch).to(device=device, dtype=torch.float32)
    X.requires_grad_(True)

    E = energy_model(X)  # (B,)
    E_sum = E.sum()
    (grad_x,) = torch.autograd.grad(
        E_sum,
        X,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    Hv = torch.autograd.grad(
        grad_x,
        X,
        grad_outputs=V,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]

    q = (Hv * V).sum(dim=1)
    return q.detach().cpu().numpy()


def compute_edge_lengths_hessian_mixed(
    energy_model,
    X_energy: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    diff_cfg: Dict[str, Any],
    device: str = "cuda",
) -> np.ndarray:
    """
    This is basically your current batch loop from build_eggfm_diffmap,
    but isolated as a pure metric function.

    Returns:
        l2_vals: (n_edges,) array of ℓ_ij^2
    """
    edge_batch_size = diff_cfg.get("hvp_batch_size", 1024)

    hessian_mix_mode = diff_cfg.get("hessian_mix_mode", "additive")
    hessian_mix_alpha = float(diff_cfg.get("hessian_mix_alpha", 0.3))
    hessian_beta = float(diff_cfg.get("hessian_beta", 0.3))
    hessian_clip_std = float(diff_cfg.get("hessian_clip_std", 2.0))
    hessian_use_neg = bool(diff_cfg.get("hessian_use_neg", True))

    n_edges = rows.shape[0]
    l2_vals = np.empty(n_edges, dtype=np.float64)

    n_batches = (n_edges + edge_batch_size - 1) // edge_batch_size
    print("[EGGFM Metrics] computing Hessian-mixed edge lengths...", flush=True)

    for b in range(n_batches):
        start = b * edge_batch_size
        end = min((b + 1) * edge_batch_size, n_edges)
        if start >= end:
            break

        i_batch = rows[start:end]
        j_batch = cols[start:end]

        Xi_batch = X_energy[i_batch]  # (B, D)
        Xj_batch = X_energy[j_batch]  # (B, D)
        V_batch = Xj_batch - Xi_batch

        norms = np.linalg.norm(V_batch, axis=1, keepdims=True) + 1e-8
        eucl2 = (norms.squeeze(-1)) ** 2

        if hessian_mix_mode == "none":
            l2_vals[start:end] = eucl2
        else:
            V_unit = V_batch / norms
            q_dir = hessian_quadratic_form_batched(
                energy_model, Xi_batch, V_unit, device
            )
            q_dir = np.asarray(q_dir, dtype=np.float64)

            if hessian_use_neg:
                q_dir = -q_dir

            q_dir[np.isnan(q_dir)] = 0.0
            q_dir[q_dir < 1e-12] = 1e-12

            if hessian_mix_mode == "additive":
                med_e = np.median(eucl2)
                med_q = np.median(q_dir)
                scale = med_e / (med_q + 1e-8)
                q_rescaled = q_dir * scale
                q_rescaled[q_rescaled < 1e-12] = 1e-12

                alpha = max(0.0, min(1.0, float(diff_cfg.get("hessian_mix_alpha", 0.3))))
                l2_vals[start:end] = (1.0 - alpha) * eucl2 + alpha * q_rescaled

            elif hessian_mix_mode == "multiplicative":
                med_q = np.median(q_dir)
                mad_q = np.median(np.abs(q_dir - med_q)) + 1e-8
                q_std = (q_dir - med_q) / mad_q

                c = hessian_clip_std
                q_std = np.clip(q_std, -c, c)

                beta = hessian_beta
                factor = np.exp(beta * q_std)
                l2_vals[start:end] = eucl2 * factor

            else:
                raise ValueError(f"Unknown hessian_mix_mode: {hessian_mix_mode!r}")

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(
                f"  [EGGFM Metrics] batch {b+1}/{n_batches} ({end}/{n_edges} edges)",
                flush=True,
            )

    return l2_vals


def compute_edge_lengths_euclidean(
    X_geom: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """
    Simple Euclidean squared distances for each edge (i, j).
    """
    Xi = X_geom[rows]
    Xj = X_geom[cols]
    V = Xj - Xi
    return np.sum(V * V, axis=1)


# (Future) scalar conformal metric:
def compute_edge_lengths_scm(
    X_geom: np.ndarray,
    G: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """
    Placeholder for your SCM metric:
        ℓ_ij^2 = 0.5 * (G_i + G_j) * ||x_i - x_j||^2
    """
    Xi = X_geom[rows]
    Xj = X_geom[cols]
    V = Xj - Xi
    eucl2 = np.sum(V * V, axis=1)

    Gi = G[rows]
    Gj = G[cols]
    G_edge = 0.5 * (Gi + Gj)
    return G_edge * eucl2
