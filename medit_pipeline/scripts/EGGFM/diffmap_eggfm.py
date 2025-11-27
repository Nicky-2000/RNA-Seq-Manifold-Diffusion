# EGGFM/diffmap_eggfm.py

from typing import Dict, Any
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.neighbors import NearestNeighbors
import torch
import scanpy as sc

from .EnergyMLP import EnergyMLP
from .metrics import (
    compute_edge_lengths_hessian_mixed,
    compute_edge_lengths_euclidean,
    compute_edge_lengths_scm,
)
from .diffusion_map import build_diffusion_map_from_l2


def _compute_scalar_conformal_field(
    ad_prep,
    energy_model: EnergyMLP,
    diff_cfg: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """
    Your old SCM precomputation:

        G(x) = gamma + lambda * exp(clip(E(x)))

    computed in ENERGY SPACE (always ad_prep.X HVG).
    """
    X_energy = ad_prep.X
    if sp_sparse.issparse(X_energy):
        X_energy = X_energy.toarray()
    X_energy = np.asarray(X_energy, dtype=np.float32)

    n_cells = X_energy.shape[0]
    metric_gamma = float(diff_cfg.get("metric_gamma", 0.2))
    metric_lambda = float(diff_cfg.get("metric_lambda", 4.0))
    energy_clip_low = float(diff_cfg.get("energy_clip_low", 0.05))
    energy_clip_high = float(diff_cfg.get("energy_clip_high", 0.95))
    energy_batch_size = int(diff_cfg.get("energy_batch_size", 2048))

    energy_model = energy_model.to(device)
    energy_model.eval()

    print("[EGGFM SCM] computing energies E(x) for all cells...", flush=True)
    with torch.no_grad():
        X_energy_tensor = torch.from_numpy(X_energy).to(
            device=device, dtype=torch.float32
        )
        E_list = []
        for start in range(0, n_cells, energy_batch_size):
            end = min(start + energy_batch_size, n_cells)
            xb = X_energy_tensor[start:end]
            Eb = energy_model(xb)  # (B,)
            E_list.append(Eb.detach().cpu().numpy())
        E_vals = np.concatenate(E_list, axis=0).astype(np.float64)

    clip_mode = diff_cfg.get("clip_mode", "baseline")

    if clip_mode == "legacy":
        q_low = np.quantile(E_vals, energy_clip_low)
        q_hi = np.quantile(E_vals, energy_clip_high)
        E_clip = np.clip(E_vals, q_low, q_hi)
        G = metric_gamma + metric_lambda * np.exp(E_clip)

    elif clip_mode == "current":
        med = np.median(E_vals)
        mad = np.median(np.abs(E_vals - med)) + 1e-8
        E_norm = (E_vals - med) / mad
        max_abs = float(diff_cfg.get("energy_clip_abs", 3.0))
        E_clip = np.clip(E_norm, -max_abs, max_abs)
        G = metric_gamma + metric_lambda * np.exp(E_clip)
        if not np.isfinite(G).all():
            raise ValueError(
                "[EGGFM SCM] non-finite values in G after exp; "
                "check energy normalization / clipping."
            )
        print(
            "[EGGFM SCM] energy stats: "
            f"raw_min={E_vals.min():.4f}, raw_max={E_vals.max():.4f}, "
            f"norm_min={E_norm.min():.4f}, norm_max={E_norm.max():.4f}, "
            f"clip=±{max_abs:.1f}",
            flush=True,
        )
    else:
        raise ValueError(f"Unknown clip_mode: {clip_mode}")

    print(
        "[EGGFM SCM] metric G stats: "
        f"min={G.min():.4f}, max={G.max():.4f}, mean={G.mean():.4f}",
        flush=True,
    )
    return G


def build_eggfm_diffmap(
    ad_prep,
    energy_model: EnergyMLP,
    diff_cfg: Dict[str, Any],
):
    """
    Refactored EGGFM DiffMap with pluggable metrics:

    diff_cfg["metric_mode"] in:
      - "hessian_mixed" (your DSM/Hessian metric)
      - "euclidean"
      - "scm" (scalar conformal metric from old code)
    """

    # --- 1) Choose geometry space X_geom ---
    use_pca = diff_cfg.get("use_pca", True)
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

    n_cells_geom, _ = X_geom.shape
    n_neighbors = diff_cfg.get("n_neighbors", 30)
    device = diff_cfg.get("device", "cuda")
    device = device if torch.cuda.is_available() else "cpu"

    # --- 2) kNN in geometry space ---
    print(
        "[EGGFM DiffMap] building kNN graph (euclidean in geometry space)...",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X_geom)
    distances, indices = nn.kneighbors(X_geom)

    k = indices.shape[1] - 1
    assert k == n_neighbors, "indices second dimension should be n_neighbors+1"

    rows = np.repeat(np.arange(n_cells_geom, dtype=np.int64), k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    n_edges = rows.shape[0]
    print(f"[EGGFM DiffMap] total edges (directed): {n_edges}", flush=True)

    # --- 3) Compute ℓ_ij^2 according to metric_mode ---
    metric_mode = diff_cfg.get("metric_mode", "hessian_mixed")

    if metric_mode == "hessian_mixed":
        # Hessian metric uses energy/latent space; here we use HVG (ad_prep.X)
        X_energy = ad_prep.X
        if sp_sparse.issparse(X_energy):
            X_energy = X_energy.toarray()
        X_energy = np.asarray(X_energy, dtype=np.float32)

        energy_model = energy_model.to(device)
        l2_vals = compute_edge_lengths_hessian_mixed(
            energy_model=energy_model,
            X_energy=X_energy,
            rows=rows,
            cols=cols,
            diff_cfg=diff_cfg,
            device=device,
        )

    elif metric_mode == "euclidean":
        l2_vals = compute_edge_lengths_euclidean(
            X_geom=X_geom,
            rows=rows,
            cols=cols,
        )

    elif metric_mode == "scm":
        # Compute scalar conformal field G(x) from energies on HVG
        G = _compute_scalar_conformal_field(
            ad_prep=ad_prep,
            energy_model=energy_model,
            diff_cfg=diff_cfg,
            device=device,
        )
        l2_vals = compute_edge_lengths_scm(
            X_geom=X_geom,
            G=G,
            rows=rows,
            cols=cols,
        )

    else:
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    # --- 4) Build diffusion map embedding ---
    diff_coords = build_diffusion_map_from_l2(
        n_cells=n_cells_geom,
        rows=rows,
        cols=cols,
        l2_vals=l2_vals,
        diff_cfg=diff_cfg,
    )

    print("[EGGFM DiffMap] finished. Embedding shape:", diff_coords.shape, flush=True)
    return diff_coords.astype(np.float32)
