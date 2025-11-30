# EGGFM/eggfm.py

from typing import Dict, Any, Tuple
import scanpy as sc

from .train_energy import train_energy_model
from .engine import EGGFMDiffusionEngine
from .data_sources import AnnDataViewProvider
from .EnergyMLP import EnergyMLP

import numpy as np
from scipy import sparse as sp_sparse
import torch
import scanpy as sc
from typing import Dict, Any

def compute_eggfm_latent(
    ad: sc.AnnData,
    energy_model: EnergyMLP,
    train_cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Compute EGGFM latent representation z(x) for all cells,
    using the same input space the energy model was trained on.
    """
    latent_space = train_cfg.get("latent_space", "hvg").lower()
    batch_size = int(train_cfg.get("latent_batch_size", 2048))
    device = train_cfg.get("device", "cuda")
    device = device if torch.cuda.is_available() else "cpu"

    # choose input matrix
    if latent_space == "hvg":
        X = ad.X
    elif latent_space == "pca":
        if "X_pca" not in ad.obsm:
            raise ValueError("latent_space='pca' but 'X_pca' not found in ad.obsm")
        X = ad.obsm["X_pca"]
    else:
        raise ValueError(f"Unknown latent_space: {latent_space}")

    if sp_sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n_cells, _ = X.shape
    latent_dim = energy_model.latent_dim
    Z = np.empty((n_cells, latent_dim), dtype=np.float32)

    energy_model.eval()
    energy_model = energy_model.to(device)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            xb = torch.from_numpy(X[start:end]).to(device=device, dtype=torch.float32)
            zb = energy_model.encode(xb)      # (B, latent_dim)
            Z[start:end] = zb.cpu().numpy()

    return Z

def run_eggfm_dimred(
    qc_ad: sc.AnnData,
    params: Dict[str, Any],
) -> Tuple[sc.AnnData, object]:
    """
    Run EGGFM-based dimension reduction on a preprocessed AnnData.
    """

    model_cfg = params.get("eggfm_model", {})
    train_cfg = params.get("eggfm_train", {})
    diff_cfg = params.get("eggfm_diffmap", {})

    # 1) Train energy model
    energy_model = train_energy_model(qc_ad, model_cfg, train_cfg)

    # 2) View provider
    view_provider = AnnDataViewProvider(
        geometry_source=diff_cfg.get("geometry_source", "pca"),
        energy_source=diff_cfg.get("energy_source", "hvg"),
    )
    
    
    
    # 3) Engine
    engine = EGGFMDiffusionEngine(
        energy_model=energy_model,
        diff_cfg=diff_cfg,
        view_provider=view_provider,
    )

    # 4) Single-pass embedding for now
    metric_mode = diff_cfg.get("metric_mode", "hessian_mixed")
    
    Z_latent_20 = None
    if train_cfg.get("latent_dim"):
            # 3) compute latent
        Z_latent = compute_eggfm_latent(qc_ad, energy_model, train_cfg)
        qc_ad.obsm["X_eggfm_latent"] = Z_latent
        
        # compress latent → 20D before giving to engine
        from sklearn.decomposition import PCA
        pca_latent = PCA(n_components=20)
        Z_latent_20 = pca_latent.fit_transform(Z_latent)
        qc_ad.obsm["X_eggfm_latent_pca20"] = Z_latent_20    
    
    # If X_geo_override is none, recovers PCA geo
    X_eggfm = engine.build_embedding(
            qc_ad,
            metric_mode=metric_mode,
            X_geom_override=Z_latent_20
        )
    

        
    qc_ad.obsm["X_eggfm"] = X_eggfm

    qc_ad.uns["eggfm_meta"] = {
        "hidden_dims": model_cfg.get("hidden_dims"),
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "sigma": train_cfg.get("sigma"),
        "n_neighbors": diff_cfg.get("n_neighbors"),
        "metric_mode": metric_mode,
        "geometry_source": diff_cfg.get("geometry_source", "pca"),
        "energy_source": diff_cfg.get("energy_source", "hvg"),
    }

    return qc_ad, energy_model
