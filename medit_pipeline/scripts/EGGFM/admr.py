# EGGFM/admr.py

from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import numpy as np
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
      
def run_admr_layers(
    ad_prep: sc.AnnData,
    engine: EGGFMDiffusionEngine,
    n_layers: int = 3,
    metric_sequence: Optional[List[str]] = None,
    t_sequence: Optional[List[Optional[float]]] = None,
    base_geometry_source: str = "pca",  # "pca" or "hvg"
    store_prefix: str = "X_admr_layer",
    # diagnostics:
    labels: Optional[np.ndarray] = None,
    label_key: Optional[str] = None,
    n_clusters: Optional[int] = None,
    k_overlap: Optional[int] = 30,
    ari_random_state: int = 0,
) -> Tuple[sc.AnnData, Dict[int, np.ndarray], List[Dict[str, Any]]]:
    """
    Alternating Diffusion Metric Regularization (ADMR) with diagnostics.

    - Start from a base geometry (PCA or HVG).
    - For each layer ℓ:
        * run diffusion with metric_sequence[ℓ] and t_sequence[ℓ],
        * use that embedding as geometry for the next layer,
        * optionally compute ARI vs labels,
        * optionally compute mean kNN Jaccard overlap vs previous geometry.

    Returns
    -------
    ad_prep:
        AnnData with .obsm[f"{store_prefix}{ℓ}"] for each layer.
    layer_embeddings:
        Dict[ℓ -> np.ndarray] of embeddings.
    metrics_log:
        List of per-layer metrics dicts, easy to turn into a DataFrame.
    """
    # --- base geometry for layer 0 ---
    if base_geometry_source.lower() == "pca":
        if "X_pca" not in ad_prep.obsm:
            raise ValueError("[ADMR] base_geometry_source='pca' but 'X_pca' missing.")
        X_geom = np.asarray(ad_prep.obsm["X_pca"], dtype=np.float32)
        base_geom_name = "pca"
        print("[ADMR] Using PCA as base geometry with shape", X_geom.shape, flush=True)
    elif base_geometry_source.lower() == "hvg":
        X_raw = ad_prep.X
        if hasattr(X_raw, "toarray"):
            X_raw = X_raw.toarray()
        X_geom = np.asarray(X_raw, dtype=np.float32)
        base_geom_name = "hvg"
        print("[ADMR] Using HVG (ad.X) as base geometry with shape", X_geom.shape, flush=True)
    else:
        raise ValueError(f"[ADMR] Unknown base_geometry_source: {base_geometry_source}")

    # --- labels, if provided ---
    if labels is None and label_key is not None:
        if label_key not in ad_prep.obs:
            raise KeyError(f"[ADMR] label_key '{label_key}' not in ad_prep.obs")
        labels = ad_prep.obs[label_key].to_numpy()

    # --- default metric / t sequences ---
    if metric_sequence is None:
        # Alternate SCM, Euclidean, SCM, Euclidean, ...
        metric_sequence = ["scm" if (i % 2 == 0) else "euclidean" for i in range(n_layers)]
    if len(metric_sequence) != n_layers:
        raise ValueError("[ADMR] metric_sequence length must equal n_layers")

    if t_sequence is None:
        t_sequence = [None] * n_layers
    if len(t_sequence) != n_layers:
        raise ValueError("[ADMR] t_sequence length must equal n_layers")

    diff_cfg = engine.diff_cfg  # same for all layers
    metrics_log: List[Dict[str, Any]] = []
    layer_embeddings: Dict[int, np.ndarray] = {}

    # --- iterate layers ---
    for ell in range(n_layers):
        metric_mode = metric_sequence[ell]
        t_val = t_sequence[ell]

        print(
            f"[ADMR] Layer {ell}: metric_mode='{metric_mode}', "
            f"t={'default' if t_val is None else t_val}, "
            f"geometry shape={X_geom.shape}",
            flush=True,
        )

        # Run diffusion with current geometry
        X_emb = engine.build_embedding(
            ad_prep,
            metric_mode=metric_mode,
            t_override=t_val,
            X_geom_override=X_geom,
        )

        key = f"{store_prefix}{ell}"
        ad_prep.obsm[key] = X_emb
        layer_embeddings[ell] = X_emb

        # --- diagnostics ---
        # 1) ARI (if labels provided)
        ari_val = None
        if labels is not None:
            ari_val = kmeans_ari(
                X_emb,
                labels,
                n_clusters=n_clusters,
                random_state=ari_random_state,
            )

        # 2) neighbor overlap vs previous geometry
        overlap_val = None
        if k_overlap is not None and k_overlap > 0:
            overlap_val = mean_jaccard_knn(
                X_ref=X_geom,
                X_new=X_emb,
                k=k_overlap,
            )

        # t effective (fall back to diff_cfg if None)
        t_eff = t_val if t_val is not None else diff_cfg.get("t", 1.0)

        metrics_log.append(
            {
                "layer": ell,
                "metric_mode": metric_mode,
                "t": float(t_eff),
                "n_cells": int(X_emb.shape[0]),
                "geometry_dim": int(X_geom.shape[1]),
                "embedding_dim": int(X_emb.shape[1]),
                "base_geometry_source": base_geom_name if ell == 0 else "admr_prev",
                "n_neighbors": int(diff_cfg.get("n_neighbors", 30)),
                "distance_power": float(diff_cfg.get("distance_power", 1.0)),
                "norm_type": diff_cfg.get("norm_type", "l2"),
                "ari": None if ari_val is None else float(ari_val),
                "label_key": label_key,
                "n_clusters": None if n_clusters is None else int(n_clusters),
                "k_overlap": None if k_overlap is None else int(k_overlap),
                "overlap_prev_mean": None if overlap_val is None else float(overlap_val),
            }
        )

        # Use this embedding as geometry for the next layer
        X_geom = X_emb

    return ad_prep, layer_embeddings, metrics_log