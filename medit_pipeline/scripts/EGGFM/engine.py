# EGGFM/engine.py

from typing import Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import scanpy as sc

from .EnergyMLP import EnergyMLP
from .data_sources import AnnDataViewProvider
from .metrics import (
    BaseMetric,
    EuclideanMetric,
    SCMMetric,
    HessianMixedMetric,
)
from .diffusion_core import DiffusionMapBuilder

_METRIC_REGISTRY = {
    "euclidean": EuclideanMetric,
    "scm": SCMMetric,
    "hessian_mixed": HessianMixedMetric,
}


class EGGFMDiffusionEngine:
    """
    - Chooses geometry space (via AnnDataViewProvider)
    - Builds kNN graph
    - Delegates distances to a metric strategy
    - Delegates diffusion to DiffusionMapBuilder
    """

    def __init__(
        self,
        energy_model: EnergyMLP,
        diff_cfg: Dict[str, Any],
        view_provider: AnnDataViewProvider | None = None,
    ):
        self.energy_model = energy_model
        self.diff_cfg = diff_cfg

        if view_provider is None:
            geom_src = diff_cfg.get("geometry_source", "pca")
            energy_src = diff_cfg.get("energy_source", "hvg")
            self.view_provider = AnnDataViewProvider(
                geometry_source=geom_src,
                energy_source=energy_src,
            )
        else:
            self.view_provider = view_provider

    def _get_geometry_matrix(self, ad_prep: sc.AnnData) -> np.ndarray:
        return self.view_provider.get_geometry_matrix(ad_prep)

    def _build_knn_graph(self, X_geom: np.ndarray):
        n_cells, _ = X_geom.shape
        n_neighbors = self.diff_cfg.get("n_neighbors", 30)
        print(
            "[EGGFM Engine] building kNN graph (euclidean in geometry space)...",
            flush=True,
        )
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
        nn.fit(X_geom)
        distances, indices = nn.kneighbors(X_geom)

        k = indices.shape[1] - 1
        assert k == n_neighbors, "indices second dimension should be n_neighbors+1"

        rows = np.repeat(np.arange(n_cells, dtype=np.int64), k)
        cols = indices[:, 1:].reshape(-1).astype(np.int64)
        print(f"[EGGFM Engine] total edges (directed): {rows.shape[0]}", flush=True)
        return rows, cols

    def _get_metric(self, metric_mode: str) -> BaseMetric:
        if metric_mode not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric_mode: {metric_mode}")
        return _METRIC_REGISTRY[metric_mode](self.diff_cfg)

    def build_embedding(
        self,
        ad_prep: sc.AnnData,
        metric_mode: str | None = None,
        t_override: float | None = None,
    ) -> np.ndarray:
        diff_cfg = dict(self.diff_cfg)
        if t_override is not None:
            diff_cfg["t"] = t_override
        metric_mode = metric_mode or diff_cfg.get("metric_mode", "hessian_mixed")

        norm_type = diff_cfg.get("norm_type", "l2")
        device = diff_cfg.get("device", "cuda")
        device = device if torch.cuda.is_available() else "cpu"

        # geometry + graph
        X_geom = self._get_geometry_matrix(ad_prep)
        n_cells = X_geom.shape[0]
        rows, cols = self._build_knn_graph(X_geom)

        # metric strategy
        metric = self._get_metric(metric_mode)
        print("[EGGFMDiffusion Engine] metric", metric)
        return
        state = metric.prepare(
            ad_prep=ad_prep,
            energy_model=self.energy_model,
            device=device,
        )
        dist_vals = metric.edge_distances(
            X_geom=X_geom,
            rows=rows,
            cols=cols,
            state=state,
            norm_type=norm_type,
        )

        # diffusion map
        builder = DiffusionMapBuilder(diff_cfg)
        diff_coords = builder.build_from_distances(
            n_cells=n_cells,
            rows=rows,
            cols=cols,
            dist_vals=dist_vals,
        )
        return diff_coords
