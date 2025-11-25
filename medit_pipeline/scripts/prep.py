import numpy as np
import scanpy as sc
from scipy import sparse


def prep_for_manifolds(
    ad: sc.AnnData,
    min_genes: int = 200,
    hvg_n_top_genes: int = 2000,
    min_cells_frac: float = 0.001,
) -> sc.AnnData:
    """
    Preprocessing that mirrors your real `prep` function:

      - QC metrics
      - filter genes by min_cells_frac * n_cells
      - filter cells by min_genes
      - drop zero-total cells
      - HVG selection (Seurat v3, subset=False)
      - subset to HVGs
      - normalize_total + log1p

    No PCA here; we keep the data nonlinear for downstream DR.
    """
    n_cells = ad.n_obs
    print("[prep_for_manifolds] running Scanpy QC metrics", flush=True)
    sc.pp.calculate_qc_metrics(ad, inplace=True)

    # Remove genes that are not statistically relevant (< min_cells_frac of cells)
    min_cells = max(3, int(min_cells_frac * n_cells))
    sc.pp.filter_genes(ad, min_cells=min_cells)

    # Remove empty droplets / low-complexity cells
    sc.pp.filter_cells(ad, min_genes=min_genes)

    # Drop zero-count cells
    totals = np.ravel(ad.X.sum(axis=1))
    ad = ad[totals > 0, :].copy()

    print("n_obs, n_vars (pre-HVG):", ad.n_obs, ad.n_vars, flush=True)

    # Explicit mean check, like in your script
    X = ad.X
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
    else:
        means = np.nanmean(X, axis=0)

    print("Means finite?", np.all(np.isfinite(means)), flush=True)
    print("Means min/max:", np.nanmin(means), np.nanmax(means), flush=True)
    print("# non-finite means:", np.sum(~np.isfinite(means)), flush=True)

    # HVG selection on raw X (no raw layer here)
    sc.pp.highly_variable_genes(
        ad,
        n_top_genes=int(hvg_n_top_genes),
        flavor="seurat_v3",
        subset=False,
    )

    ad = ad[:, ad.var["highly_variable"]].copy()
    print("n_obs, n_vars (post-HVG):", ad.n_obs, ad.n_vars, flush=True)

    # Now normalize/log on X
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.pca(ad, n_comps=50)

    return ad
