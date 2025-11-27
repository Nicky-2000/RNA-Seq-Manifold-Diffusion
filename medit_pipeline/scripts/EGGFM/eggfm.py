# EGGFM/eggfm.py

from typing import Dict, Any, Tuple
import scanpy as sc

from .train_energy import train_energy_model
from .engine import EGGFMDiffusionEngine
from .data_sources import AnnDataViewProvider


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
    X_eggfm = engine.build_embedding(qc_ad, metric_mode=metric_mode)
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
