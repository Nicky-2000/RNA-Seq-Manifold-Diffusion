# eggfm.py

from typing import Dict, Any, Tuple
import scanpy as sc

from .train_energy import train_energy_model
from .diffmap_eggfm import build_eggfm_diffmap


def run_eggfm_dimred(
    qc_ad: sc.AnnData,
    params: Dict[str, Any],
) -> Tuple[sc.AnnData, object]:
    """
    Run EGGFM-based dimension reduction on a preprocessed AnnData.

    Assumes qc_ad is already:
      - gene-filtered
      - HVG-selected
      - normalized + log1p
    """

    # 1) Train energy model
    model_cfg = params.get(
        "eggfm_model",
        {
            "hidden_dims": [512, 512, 512, 512],
        },
    )
    train_cfg = params.get(
        "eggfm_train",
        {
            "batch_size": 256,
            "num_epochs": 50,
            "lr": 1e-3,
            "sigma": 0.1,
            "device": "cuda",
        },
    )
    energy_model = train_energy_model(qc_ad, model_cfg, train_cfg)

    # 2) Build EGGFM DiffMap embedding (no subsampling)
    diff_cfg = params.get(
        "eggfm_diffmap",
        {
            "n_neighbors": 30,
            "n_comps": 30,
            "device": "cuda",
            "eps_mode": "median",  # or "fixed"
            "eps_value": 1.0,
            "hvp_mode": "Hv_norm2",  # or "vHv"
            "hvp_batch_size": 1024,  # edges per HVP batch
            "t": 1.0,
        },
    )
    X_eggfm = build_eggfm_diffmap(qc_ad, energy_model, diff_cfg)
    qc_ad.obsm["X_eggfm"] = X_eggfm

    return qc_ad, energy_model
