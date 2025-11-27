# EGGFM/admr.py

from typing import List, Dict, Any
import scanpy as sc

from .engine import EGGFMDiffusionEngine


class AlternatingDiffuser:
    """
    Alternating Diffusion Metric Refinement (ADMR).

    layers: list of dicts, e.g.
      [
        {"metric_mode": "scm", "t": 2.0},
        {"metric_mode": "euclidean", "t": 1.0},
        {"metric_mode": "scm", "t": 1.0},
      ]

    Strategy:
      - Layer 1 uses whatever geometry is configured (HVG/PCA).
      - After each layer, we write the embedding into adata.obsm["X_pca"]
        so that subsequent layers (with geometry_source="pca") operate on
        the "massaged" embedding.
    """

    def __init__(self, engine: EGGFMDiffusionEngine, layers: List[Dict[str, Any]]):
        self.engine = engine
        self.layers = layers

    def run(self, adata: sc.AnnData):
        ad_current = adata
        X_last = None

        for idx, layer in enumerate(self.layers):
            print(f"[ADMR] Layer {idx+1}/{len(self.layers)}: {layer}", flush=True)

            metric_mode = layer.get("metric_mode", self.engine.diff_cfg.get("metric_mode", "hessian_mixed"))
            t = layer.get("t", None)

            # After first layer, geometry = previous embedding via X_pca
            if idx > 0:
                ad_current.obsm["X_pca"] = X_last

            X_last = self.engine.build_embedding(
                ad_current,
                metric_mode=metric_mode,
                t_override=t,
            )

        return X_last
