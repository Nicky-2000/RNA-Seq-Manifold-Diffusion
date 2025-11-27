# EGGFM/utils.py

from typing import Optional
import numpy as np
import scanpy as sc


def subsample_adata(
    ad: sc.AnnData,
    max_cells: Optional[int] = None,
    seed: int = 0,
) -> sc.AnnData:
    """
    Return a (possibly) subsampled AnnData for quick experiments.

    - If max_cells is None or >= n_obs, returns ad unchanged (no copy).
    - Otherwise, randomly sample max_cells cells without replacement and copy().
    """
    n = ad.n_obs
    if max_cells is None or max_cells >= n:
        print(f"[subsample_adata] Using all {n} cells (max_cells={max_cells})", flush=True)
        return ad

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_cells, replace=False)
    idx.sort()
    print(
        f"[subsample_adata] Subsampling {max_cells} / {n} cells "
        f"(seed={seed})",
        flush=True,
    )
    return ad[idx].copy()
