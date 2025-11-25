# diffmap_eggfm.py

from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp_sparse
import torch
import scanpy as sc


def compute_scores_batched(
    energy_model,
    X_energy: np.ndarray,  # (N, D)
    energy_batch_size: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Approximate score(x) = -∇_x E(x) for all cells, in batches.

    Returns: scores of shape (N, D)
    """
    energy_model.eval()
    scores = []

    N = X_energy.shape[0]
    X_tensor_full = torch.from_numpy(X_energy).to(device=device, dtype=torch.float32)

    for start in range(0, N, energy_batch_size):
        end = min(start + energy_batch_size, N)
        xb = X_tensor_full[start:end]  # (B, D)
        xb = xb.clone().detach().requires_grad_(True)

        energy = energy_model(xb)  # (B,)
        energy_sum = energy.sum()
        (grad_x,) = torch.autograd.grad(
            energy_sum,
            xb,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )
        s_batch = -grad_x.detach().cpu().numpy()  # (B, D)

        scores.append(s_batch)

    S = np.concatenate(scores, axis=0).astype(np.float32)
    return S


def build_eggfm_diffmap(
    ad_prep,
    energy_model,
    diff_cfg: Dict[str, Any],
):
    """
    Build an EGGFM-aware Diffusion Map embedding using a *score-tangent metric*.

    For each neighbor edge i->j, we compute:
        v_ij = x_j - x_i
        n_i  = score(x_i) / ||score(x_i)||

        v_par = (v_ij · n_i) n_i
        v_tan = v_ij - v_par

        ℓ_ij^2 = ||v_tan||^2 + alpha * ||v_par||^2

    where alpha ∈ [0,1] is a hyperparameter (alpha=0 => pure tangent distance).

    Then we build a standard diffusion map with ℓ_ij^2 as squared distances.
    """

    # Geometry / energy space: log-normalized HVG expression
    X_energy = ad_prep.X
    if sp_sparse.issparse(X_energy):
        X_energy = X_energy.toarray()
    X_energy = np.asarray(X_energy, dtype=np.float32)

    n_cells, D = X_energy.shape
    print(
        f"[EGGFM DiffMap] geometry/energy X shape: {X_energy.shape}",
        flush=True,
    )

    n_neighbors = diff_cfg.get("n_neighbors", 30)
    n_comps = diff_cfg.get("n_comps", 30)

    device = diff_cfg.get("device", "cuda")
    device = device if torch.cuda.is_available() else "cpu"

    eps_mode = diff_cfg.get("eps_mode", "median")
    eps_value = diff_cfg.get("eps_value", 1.0)

    edge_batch_size = diff_cfg.get("hvp_batch_size", 1024)  # reused as edge batch size
    energy_batch_size = int(diff_cfg.get("energy_batch_size", 2048))

    t = diff_cfg.get("t", 1.0)

    # New hyperparameter: how much to penalize motion along the score (normal)
    alpha_normal = float(diff_cfg.get("alpha_normal", 0.2))

    # Move model to device once
    energy_model = energy_model.to(device)

    # ------------------------------------------------------------
    # 0) Compute scores s(x) ≈ ∇ log p(x) = -∇E(x) for all cells
    # ------------------------------------------------------------
    print("[EGGFM DiffMap] computing score(x) for all cells...", flush=True)
    S = compute_scores_batched(
        energy_model,
        X_energy,
        energy_batch_size=energy_batch_size,
        device=device,
    )  # (N, D)

    # Optional: print some score stats
    norms = np.linalg.norm(S, axis=1)
    print(
        "[EGGFM DiffMap] score stats: "
        f"||s|| min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 1) kNN in geometry space (here: HVG space)
    # ------------------------------------------------------------
    print(
        "[EGGFM DiffMap] building kNN graph (euclidean in HVG space)...",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X_energy)
    distances, indices = nn.kneighbors(X_energy)

    k = indices.shape[1] - 1
    assert k == n_neighbors, "indices second dimension should be n_neighbors+1"

    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    n_edges = rows.shape[0]

    print(f"[EGGFM DiffMap] total edges (directed): {n_edges}", flush=True)

    # ------------------------------------------------------------
    # 2) Compute score-tangent edge lengths ℓ_ij^2
    # ------------------------------------------------------------
    l2_vals = np.empty(n_edges, dtype=np.float64)

    print(
        "[EGGFM DiffMap] computing score-tangent edge lengths...",
        flush=True,
    )
    n_batches = (n_edges + edge_batch_size - 1) // edge_batch_size

    for b in range(n_batches):
        start = b * edge_batch_size
        end = min((b + 1) * edge_batch_size, n_edges)
        if start >= end:
            break

        i_batch = rows[start:end]
        j_batch = cols[start:end]

        Xi = X_energy[i_batch]  # (B, D)
        Xj = X_energy[j_batch]  # (B, D)
        V = Xj - Xi  # (B, D)

        # Scores at the base points
        S_i = S[i_batch]  # (B, D)
        # Normalize scores to unit vectors (with eps)
        S_norm = np.linalg.norm(S_i, axis=1, keepdims=True) + 1e-8
        n_i = S_i / S_norm  # (B, D)

        # Decompose displacement into parallel and tangent components
        v_par_scalar = np.sum(V * n_i, axis=1, keepdims=True)  # (B, 1)
        v_par = v_par_scalar * n_i  # (B, D)
        v_tan = V - v_par  # (B, D)

        # Score-tangent distance
        tan_sq = np.sum(v_tan * v_tan, axis=1)  # (B,)
        par_sq = np.sum(v_par * v_par, axis=1)  # (B,)

        q_batch = tan_sq + alpha_normal * par_sq

        # Numerical floor
        q_batch[q_batch < 1e-12] = 1e-12

        l2_vals[start:end] = q_batch

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(
                f"  processed batch {b+1}/{n_batches} ({end} / {n_edges} edges)",
                flush=True,
            )

    # ------------------------------------------------------------
    # 3) Choose kernel bandwidth ε
    # ------------------------------------------------------------
    if eps_mode == "median":
        eps = np.median(l2_vals)
    elif eps_mode == "fixed":
        eps = float(eps_value)
    else:
        raise ValueError(f"Unknown eps_mode: {eps_mode}")
    print(f"[EGGFM DiffMap] using eps = {eps:.4g}", flush=True)

    # Optional: clip l2_vals quantiles
    if diff_cfg.get("eps_trunc") == "yes":
        q_low = np.quantile(l2_vals, 0.05)
        q_hi = np.quantile(l2_vals, 0.98)
        l2_vals = np.clip(l2_vals, q_low, q_hi)
        print(
            f"[EGGFM DiffMap] eps_trunc=yes, clipped l2_vals to [{q_low:.4g}, {q_hi:.4g}]",
            flush=True,
        )

    # 4) Build kernel W_ij = exp(-ℓ_ij^2 / eps)
    W_vals = np.exp(-l2_vals / eps)
    W = sparse.csr_matrix((W_vals, (rows, cols)), shape=(n_cells, n_cells))
    W = 0.5 * (W + W.T)

    # 5) Normalize to Markov matrix P (row-stochastic)
    d = np.array(W.sum(axis=1)).ravel()
    d_safe = np.maximum(d, 1e-12)
    D_inv = sparse.diags(1.0 / d_safe)
    P = D_inv @ W

    # 6) Eigendecompose P^T for diffusion map
    k_eigs = n_comps + 1  # include trivial eigenpair
    print("[EGGFM DiffMap] computing eigenvectors...", flush=True)
    eigvals, eigvecs = eigs(P.T, k=k_eigs, which="LR")

    eigvals = eigvals.real
    eigvecs = eigvecs.real

    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    lambdas = eigvals[1 : n_comps + 1]
    phis = eigvecs[:, 1 : n_comps + 1]

    diff_coords = phis * (lambdas**t)

    print("[EGGFM DiffMap] finished. Embedding shape:", diff_coords.shape, flush=True)
    return diff_coords.astype(np.float32)
