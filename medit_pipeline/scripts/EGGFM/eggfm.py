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

    Config sections (from params.yml):
      eggfm_model:
        hidden_dims: [512, 512, 512, 512]

      eggfm_train:
        batch_size: 2048
        num_epochs: 50
        lr: 1e-4
        sigma: 0.1
        device: "cuda"

      eggfm_diffmap:
        n_neighbors: 10
        n_comps: 30
        device: "cuda"
        hvp_mode: "vHv"
        hvp_batch_size: 2048
        eps_mode: "median"
        eps_value: 1.0
        t: 1.0
    """

    # Let the train/diffmap functions handle their own defaults.
    model_cfg = params.get("eggfm_model", {})
    train_cfg = params.get("eggfm_train", {})
    diff_cfg = params.get("eggfm_diffmap", {})

    # 1) Train energy model
    energy_model = train_energy_model(qc_ad, model_cfg, train_cfg)

    # 2) Build EGGFM DiffMap embedding (no subsampling)
    X_eggfm = build_eggfm_diffmap(qc_ad, energy_model, diff_cfg)
    qc_ad.obsm["X_eggfm"] = X_eggfm  # make sure clustering sees this
    qc_ad.uns["eggfm_meta"] = {
        "hidden_dims": model_cfg.get("hidden_dims"),
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "sigma": train_cfg.get("sigma"),
        "n_neighbors": diff_cfg.get("n_neighbors"),
        "hvp_mode": diff_cfg.get("hvp_mode"),
    }
    return qc_ad, energy_model
