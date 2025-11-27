# EGGFM/data_sources.py

from typing import Literal
import numpy as np
from scipy import sparse as sp_sparse
import scanpy as sc

GeometrySource = Literal["pca", "hvg"]
EnergySource = Literal["hvg", "pca"]


class AnnDataViewProvider:
    """
    Standardizes which view of AnnData we use for:
      - geometry (kNN graph)
      - energy / metric computation
    """

    def __init__(
        self,
        geometry_source: GeometrySource = "pca",
        energy_source: EnergySource = "hvg",
        pca_key: str = "X_pca",
    ):
        self.geometry_source = geometry_source
        self.energy_source = energy_source
        self.pca_key = pca_key

    @staticmethod
    def _to_dense(X):
        if sp_sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def get_geometry_matrix(self, ad: sc.AnnData) -> np.ndarray:
        if self.geometry_source == "pca":
            if self.pca_key not in ad.obsm:
                raise ValueError(
                    f"[AnnDataViewProvider] geometry_source='pca' but '{self.pca_key}' "
                    "not found in ad.obsm. Run PCA first."
                )
            X_geom = ad.obsm[self.pca_key]
            print(
                "[AnnDataViewProvider] using PCA for geometry with shape",
                X_geom.shape,
                flush=True,
            )
        elif self.geometry_source == "hvg":
            X_geom = ad.X
            print(
                "[AnnDataViewProvider] using HVG (ad.X) for geometry with shape",
                X_geom.shape,
                flush=True,
            )
        else:
            raise ValueError(f"Unknown geometry_source: {self.geometry_source}")
        return self._to_dense(X_geom)

    def get_energy_matrix(self, ad: sc.AnnData) -> np.ndarray:
        if self.energy_source == "hvg":
            X_energy = ad.X
            print(
                "[AnnDataViewProvider] using HVG (ad.X) for energy space with shape",
                X_energy.shape,
                flush=True,
            )
        elif self.energy_source == "pca":
            if self.pca_key not in ad.obsm:
                raise ValueError(
                    f"[AnnDataViewProvider] energy_source='pca' but '{self.pca_key}' "
                    "not found in ad.obsm. Run PCA first."
                )
            X_energy = ad.obsm[self.pca_key]
            print(
                "[AnnDataViewProvider] using PCA for energy space with shape",
                X_energy.shape,
                flush=True,
            )
        else:
            raise ValueError(f"Unknown energy_source: {self.energy_source}")
        return self._to_dense(X_energy)
