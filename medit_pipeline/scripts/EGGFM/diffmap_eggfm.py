# diffmap_eggfm.py
from typing import Dict, Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp_sparse
import torch
import scanpy as sc


def hessian_quadratic_form_batched(
    energy_model,
    X_batch: np.ndarray,  # (B, D) base points x_i
    V_batch: np.ndarray,  # (B, D) directions v_ij
    device: str = "cuda",
) -> np.ndarray:
    """
    Returns q_b = v_b^T H(x_b) v_b for each (x_b, v_b).
    """
    energy_model.eval()
    X = torch.from_numpy(X_batch).to(device=device, dtype=torch.float32)
    V = torch.from_numpy(V_batch).to(device=device, dtype=torch.float32)

    X.requires_grad_(True)

    # First gradient: ∇E(x)
    E = energy_model(X)  # (B,)
    E_sum = E.sum()
    (grad_x,) = torch.autograd.grad(
        E_sum,
        X,
        create_graph=True,  # we will differentiate again
        retain_graph=True,
        only_inputs=True,
    )  # grad_x: (B, D)

    # Directional derivative of grad along V: v^T H(x)
    Hv = torch.autograd.grad(
        grad_x,
        X,
        grad_outputs=V,  # each row of V is v_b
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[
        0
    ]  # Hv: (B, D)

    # v^T H v  = <Hv, v>
    q = (Hv * V).sum(dim=1)  # (B,)
    return q.detach().cpu().numpy()


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
    # Energy space: ALWAYS use the original HVG expression matrix which the energy model was trained on.
    X_energy = ad_prep.obsm["X_pca"]
    if sp_sparse.issparse(X_energy):
        X_energy = X_energy.toarray()
    X_energy = np.asarray(X_energy, dtype=np.float32)

    n_cells_geom, D_geom = X_energy.shape
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

    # Move model to device once
    energy_model = energy_model.to(device)
    energy_model.eval()

    print(
        f"[EGGFM DiffMap] geometry X shape: {X_energy.shape}, "
        f"energy X shape: {X_energy.shape}",
        flush=True,
    )

    # Hessian mixing hyperparams
    hessian_mix_mode = diff_cfg.get(
        "hessian_mix_mode", "additive"
    )  # "additive", "multiplicative", or "none"
    hessian_mix_alpha = float(
        diff_cfg.get("hessian_mix_alpha", 0.3)
    )  # strength of Hessian in additive mode
    hessian_beta = float(
        diff_cfg.get("hessian_beta", 0.3)
    )  # strength of anisotropy in multiplicative mode
    hessian_clip_std = float(
        diff_cfg.get("hessian_clip_std", 2.0)
    )  # |z| <= this after standardization
    hessian_use_neg = bool(
        diff_cfg.get("hessian_use_neg", True)
    )  # flip sign to use curvature of -log p

    # ------------------------------------------------------------
    # 1) kNN for neighbor selection in GEOMETRY SPACE
    # ------------------------------------------------------------
    print(
        "[EGGFM DiffMap] building kNN graph (euclidean in geometry space)...",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X_energy)
    distances, indices = nn.kneighbors(X_energy)

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
    print(
        "[EGGFM DiffMap] computing Hessian-based edge lengths...",
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

        # Work entirely in energy / HVG space
        Xi_batch = X_energy[i_batch]  # (B, D)
        Xj_batch = X_energy[j_batch]  # (B, D)
        V_batch = Xj_batch - Xi_batch  # (B, D)

        # Euclidean baseline: ||v||^2
        norms = np.linalg.norm(V_batch, axis=1, keepdims=True) + 1e-8  # (B,1)
        norm_sq = norms.squeeze(-1) ** 2  # (B,)
        eucl2 = norm_sq.copy()

        if hessian_mix_mode == "none":
            # Pure Euclidean edge lengths
            l2_vals[start:end] = eucl2
        else:
            # Unit directions for Hessian quadratic form
            V_unit = V_batch / norms  # (B, D)

            # q_dir = v^T H(x) v
            q_dir = hessian_quadratic_form_batched(
                energy_model, Xi_batch, V_unit, device
            )
            q_dir = np.asarray(q_dir, dtype=np.float64)

            # Optionally interpret curvature of -log p instead of E directly
            if hessian_use_neg:
                q_dir = -q_dir

            # Basic positivity / stability
            q_dir[np.isnan(q_dir)] = 0.0
            q_dir[q_dir < 1e-12] = 1e-12

            if hessian_mix_mode == "additive":
                # Rescale q_dir to live on the same scale as eucl2 using medians
                med_e = np.median(eucl2)
                med_q = np.median(q_dir)
                scale = med_e / (med_q + 1e-8)
                q_rescaled = q_dir * scale
                q_rescaled[q_rescaled < 1e-12] = 1e-12

                alpha = hessian_mix_alpha
                alpha = max(0.0, min(1.0, alpha))  # clamp to [0,1]

                l2_vals[start:end] = (1.0 - alpha) * eucl2 + alpha * q_rescaled

            elif hessian_mix_mode == "multiplicative":
                # Standardize q_dir in this batch, then exponentiate to get a bounded factor
                med_q = np.median(q_dir)
                mad_q = np.median(np.abs(q_dir - med_q)) + 1e-8
                q_std = (q_dir - med_q) / mad_q  # (B,)

                # Clip to control extremes
                c = hessian_clip_std
                q_std = np.clip(q_std, -c, c)

                beta = hessian_beta
                # Factor in [exp(-beta*c), exp(beta*c)]
                factor = np.exp(beta * q_std)

                l2_vals[start:end] = eucl2 * factor

            else:
                raise ValueError(f"Unknown hessian_mix_mode: {hessian_mix_mode!r}")

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
