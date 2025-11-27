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
    metric_sequence: List[str] | None = None,
    t_sequence: List[float] | None = None,
    base_geometry_source: str = "pca",
    store_prefix: str = "X_admr_layer",
) -> Tuple[sc.AnnData, Dict[int, np.ndarray]]:
    """
    Alternating Diffusion Metric Regularization (ADMR):

    - Start from a base geometry (e.g., PCA).
    - For each layer ℓ:
        * run diffusion with given metric_mode & t,
        * use that embedding as geometry for the next layer.

    Parameters
    ----------
    ad_prep
        Preprocessed AnnData (HVG/log1p/etc.).
    engine
        EGGFMDiffusionEngine built with your energy_model + diff_cfg.
    n_layers
        Number of diffusion layers to apply.
    metric_sequence
        List of metric_modes per layer, e.g. ["scm", "euclidean", "scm"].
        If None, we use a default alternating pattern ["scm", "euclidean", ...].
    t_sequence
        Optional list of diffusion times per layer; if None, all use diff_cfg["t"].
    base_geometry_source
        "pca" or "hvg" – how to initialize geometry for the first layer.
    store_prefix
        Prefix for keys in ad_prep.obsm where we store each layer embedding.

    Returns
    -------
    (ad_prep, layer_embeddings)
        - ad_prep with new .obsm entries: f"{store_prefix}{ℓ}"
        - dict mapping layer index -> embedding array.
    """
    # 1) base geometry for layer 0
    if base_geometry_source.lower() == "pca":
        X_geom = np.asarray(ad_prep.obsm["X_pca"], dtype=np.float32)
        print("[ADMR] Using PCA as base geometry with shape", X_geom.shape, flush=True)
    elif base_geometry_source.lower() == "hvg":
        X_geom = np.asarray(ad_prep.X.toarray() if hasattr(ad_prep.X, "toarray") else ad_prep.X,
                            dtype=np.float32)
        print("[ADMR] Using HVG (ad.X) as base geometry with shape", X_geom.shape, flush=True)
    else:
        raise ValueError(f"[ADMR] Unknown base_geometry_source: {base_geometry_source}")

    # 2) default sequences if not provided
    if metric_sequence is None:
        # alternate SCM and Euclidean, starting with SCM
        metric_sequence = ["scm" if (i % 2 == 0) else "euclidean" for i in range(n_layers)]
    if len(metric_sequence) != n_layers:
        raise ValueError("[ADMR] metric_sequence length must equal n_layers")

    if t_sequence is None:
        t_sequence = [None] * n_layers
    if len(t_sequence) != n_layers:
        raise ValueError("[ADMR] t_sequence length must equal n_layers")

    layer_embeddings: Dict[int, np.ndarray] = {}

    # 3) iterate layers
    for ell in range(n_layers):
        metric_mode = metric_sequence[ell]
        t_val = t_sequence[ell]

        print(
            f"[ADMR] Layer {ell}: metric_mode='{metric_mode}', "
            f"t={'default' if t_val is None else t_val}, "
            f"geometry shape={X_geom.shape}",
            flush=True,
        )

        X_emb = engine.build_embedding(
            ad_prep,
            metric_mode=metric_mode,
            t_override=t_val,
            X_geom_override=X_geom,
        )

        key = f"{store_prefix}{ell}"
        ad_prep.obsm[key] = X_emb
        layer_embeddings[ell] = X_emb

        # Use this embedding as geometry for next layer
        X_geom = X_emb

    return ad_prep, layer_embeddings
  
  
def kmeans_ari(
    X: np.ndarray,
    labels: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 0,
) -> float:
    """
    Cluster X with k-means and compute ARI vs. labels.
    """
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    preds = km.fit_predict(X)
    return adjusted_rand_score(labels, preds)


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """
    Return neighbor indices (excluding self) for each point.
    Shape: (n_cells, k)
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # drop self


def mean_jaccard_knn(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    k: int = 30,
) -> float:
    """
    Mean Jaccard similarity between k-NN sets in X_ref vs X_new,
    computed per point and then averaged.
    """
    idx_ref = _knn_indices(X_ref, k=k)
    idx_new = _knn_indices(X_new, k=k)

    n = idx_ref.shape[0]
    scores = np.empty(n, dtype=np.float32)

    for i in range(n):
        s1 = set(idx_ref[i])
        s2 = set(idx_new[i])
        inter = len(s1 & s2)
        union = len(s1 | s2)
        scores[i] = inter / union if union > 0 else 0.0

    return float(scores.mean())
