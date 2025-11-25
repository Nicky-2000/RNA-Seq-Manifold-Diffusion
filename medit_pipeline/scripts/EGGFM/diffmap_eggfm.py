# diffmap_eggfm.py
from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp_sparse
import torch
import scanpy as sc

def build_eggfm_diffmap(
    ad_prep,
    energy_model,
    diff_cfg: Dict[str, Any],
):
    """
    Build an EGGFM-aware Diffusion Map embedding using a *conformal
    energy-based metric*:

        G(x) = gamma + lambda * exp(clip(E(x)))

    and edge lengths

        ℓ_ij^2 = 0.5 * (G(x_i) + G(x_j)) * ||x_i - x_j||^2

    The rest of the pipeline (kernel -> Markov -> eigendecomposition)
    is unchanged.
    """
    use_pca = diff_cfg.get("use_pca", True)
    # Geometry space: used for kNN + Euclidean distances
    if use_pca and "X_pca" in ad_prep.obsm:
        X_geom = ad_prep.obsm["X_pca"]
        print(
            "[EGGFM DiffMap] using X_pca for geometry with shape",
            X_geom.shape,
            flush=True,
        )
    else:
        X_geom = ad_prep.X
        if sp_sparse.issparse(X_geom):
            X_geom = X_geom.toarray()
        X_geom = np.asarray(X_geom, dtype=np.float32)
        print(
            "[EGGFM DiffMap] using raw X for geometry with shape",
            X_geom.shape,
            flush=True,
        )

    # Energy space: ALWAYS use the original HVG expression matrix which the energy model was trained on.
    X_energy = ad_prep.X
    if sp_sparse.issparse(X_energy):
        X_energy = X_energy.toarray()
    X_energy = np.asarray(X_energy, dtype=np.float32)

    n_cells_geom, D_geom = X_geom.shape
    n_cells_energy, D_energy = X_energy.shape
    assert n_cells_geom == n_cells_energy, (
        "Geometry and energy spaces must have the same number of cells, "
        f"got {n_cells_geom} vs {n_cells_energy}"
    )
    n_cells = n_cells_geom

    n_neighbors = diff_cfg.get("n_neighbors", 30)
    n_comps = diff_cfg.get("n_comps", 30)

    device = diff_cfg.get("device", "cuda")
    device = device if torch.cuda.is_available() else "cpu"

    eps_mode = diff_cfg.get("eps_mode", "median")
    eps_value = diff_cfg.get("eps_value", 1.0)

    edge_batch_size = diff_cfg.get("hvp_batch_size", 1024)
    t = diff_cfg.get("t", 1.0)

    metric_gamma = float(diff_cfg.get("metric_gamma", 0.2))
    metric_lambda = float(diff_cfg.get("metric_lambda", 4.0))
    energy_clip_low = float(diff_cfg.get("energy_clip_low", 0.05))
    energy_clip_high = float(diff_cfg.get("energy_clip_high", 0.95))
    energy_batch_size = int(diff_cfg.get("energy_batch_size", 2048))

    # Move model to device once
    energy_model = energy_model.to(device)
    energy_model.eval()

    print(
        f"[EGGFM DiffMap] geometry X shape: {X_geom.shape}, "
        f"energy X shape: {X_energy.shape}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 0) Precompute energy E(x) and scalar metric G(x) in ENERGY SPACE
    # ------------------------------------------------------------
    print("[EGGFM DiffMap] computing energies E(x) for all cells...", flush=True)
    with torch.no_grad():
        X_energy_tensor = torch.from_numpy(X_energy).to(
            device=device, dtype=torch.float32
        )
        E_list = []
        for start in range(0, n_cells, energy_batch_size):
            end = min(start + energy_batch_size, n_cells)
            xb = X_energy_tensor[start:end]  # (B, D_energy)
            Eb = energy_model(xb)  # (B,)
            E_list.append(Eb.detach().cpu().numpy())
        E_vals = np.concatenate(E_list, axis=0).astype(np.float64)

    clip_mode = diff_cfg.get("clip_mode", "baseline")

    if clip_mode == "legacy":
        # Clip energies to avoid extreme tails
        q_low = np.quantile(E_vals, energy_clip_low)
        q_hi = np.quantile(E_vals, energy_clip_high)
        E_clip = np.clip(E_vals, q_low, q_hi)
        # Scalar conformal metric field
        G = metric_gamma + metric_lambda * np.exp(E_clip)  # shape (n_cells,)
        print(
            "[EGGFM DiffMap] metric G stats: "
            f"min={G.min():.4f}, max={G.max():.4f}, mean={G.mean():.4f}",
            flush=True,
        )

    elif clip_mode == "current":
        # Robust center & scale energies
        med = np.median(E_vals)
        mad = np.median(np.abs(E_vals - med)) + 1e-8  # robust scale
        E_norm = (E_vals - med) / mad

        # Clip normalized energies to a small range, e.g. [-3, 3]
        max_abs = float(diff_cfg.get("energy_clip_abs", 3.0))
        E_clip = np.clip(E_norm, -max_abs, max_abs)

        # Scalar conformal metric field
        G = metric_gamma + metric_lambda * np.exp(
            E_clip
        )  # E_clip ∈ [-3,3] → exp ∈ [~0.05, ~20]
        if not np.isfinite(G).all():
            raise ValueError(
                "[EGGFM DiffMap] non-finite values in G after exp; "
                "check energy normalization / clipping."
            )
        print(
            "[EGGFM DiffMap] energy stats: "
            f"raw_min={E_vals.min():.4f}, raw_max={E_vals.max():.4f}, "
            f"norm_min={E_norm.min():.4f}, norm_max={E_norm.max():.4f}, "
            f"clip=[{-max_abs:.1f}, {max_abs:.1f}]",
            flush=True,
        )

    # ------------------------------------------------------------
    # 1) kNN for neighbor selection in GEOMETRY SPACE
    # ------------------------------------------------------------
    print(
        "[EGGFM DiffMap] building kNN graph (euclidean in geometry space)...",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X_geom)
    distances, indices = nn.kneighbors(X_geom)

    # neighbors per cell (excluding self)
    k = indices.shape[1] - 1
    assert k == n_neighbors, "indices second dimension should be n_neighbors+1"

    # 2) Flatten edges
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    n_edges = rows.shape[0]

    print(f"[EGGFM DiffMap] total edges (directed): {n_edges}", flush=True)

    # ------------------------------------------------------------
    # 2) Compute conformal energy-based edge lengths ℓ_ij^2
    #    using geometry space for distances and G from energy space
    # ------------------------------------------------------------
    l2_vals = np.empty(n_edges, dtype=np.float64)

    print(
        "[EGGFM DiffMap] computing conformal energy-based edge lengths...",
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

        Xi_batch = X_geom[i_batch]  # (B, D_geom)
        Xj_batch = X_geom[j_batch]  # (B, D_geom)
        V_batch = Xj_batch - Xi_batch

        eucl2 = np.sum(V_batch * V_batch, axis=1)  # (B,)

        Gi = G[i_batch]
        Gj = G[j_batch]
        G_edge = 0.5 * (Gi + Gj)

        q_batch = G_edge * eucl2
        q_batch[q_batch < 1e-12] = 1e-12

        l2_vals[start:end] = q_batch

        if (b + 1) % 50 == 0 or b == n_batches - 1:
            print(
                f"  processed batch {b+1}/{n_batches} ({end} / {n_edges} edges)",
                flush=True,
            )

    # Optional: clip extreme metric values (robust quantiles)
    if diff_cfg.get("eps_trunc") == "upper":
        q_hi = np.quantile(l2_vals, 0.99)
        l2_vals = np.minimum(l2_vals, q_hi)
        print(
            f"[EGGFM DiffMap] eps_trunc=upper, clipped l2_vals to <= {q_hi:.4g}",
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

    # ------------------------------------------------------------
    # 3.5) Clip
    # ------------------------------------------------------------
    if diff_cfg.get("eps_trunc") == "yes":
        q_low = np.quantile(l2_vals, 0.05)
        q_hi = np.quantile(l2_vals, 0.98)
        l2_vals = np.clip(l2_vals, q_low, q_hi)

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
