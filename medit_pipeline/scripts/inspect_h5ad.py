#!/usr/bin/env python
"""
Inspect an .h5ad file to understand its structure for QC ingestion.

Usage:
  conda run -n venv python scripts/inspect_h5ad.py \
      --h5ad data/raw/weinreb/stateFate_inVitro/stateFate_inVitro_normed_counts.h5ad
"""

import argparse
import numpy as np
import scanpy as sc
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype


def inspect_h5ad(path: str) -> None:
    print(f"Loading {path}")
    ad = sc.read_h5ad(path, backed=None)

    print("\n=== AnnData overview ===")
    print(ad)
    print("shape (n_cells, n_genes):", ad.shape)
    print("X class:", type(ad.X))

    # ---------- OBS (cell metadata) ----------
    print("\n=== OBS (cell metadata) ===")
    print("obs columns:", list(ad.obs.columns))
    print("\nobs.head():")
    print(ad.obs.head())

    print("\n[obs summary by column]")
    for col in ad.obs.columns:
        s = ad.obs[col]
        nunique = s.nunique()
        print(f"\n---- {col} ----")
        print("dtype:", s.dtype)
        print("n_unique:", nunique)

        if nunique <= 20:
            # categorical-ish: show value counts
            print("value_counts():")
            print(s.value_counts().head(20))
        else:
            # high-cardinality: branch by dtype
            if is_numeric_dtype(s):
                print(
                    "min/mean/max:",
                    float(s.min()),
                    float(s.mean()),
                    float(s.max()),
                )
            elif is_categorical_dtype(s):
                cats = s.cat.categories
                print(f"categorical with {len(cats)} categories")
                print("categories (first 20):", list(cats[:20]))
            else:
                print("example values:", s.iloc[:10].tolist())

    # Highlight likely-important columns for QC / modeling if present
    interesting_obs = [
        "timepoint",
        "day",
        "treatment",
        "condition",
        "sample",
        "batch",
        "clone",
        "clone_id",
        "lineage",
        "cell_type",
        "state",
    ]
    print("\n=== Selected interesting obs columns (if present) ===")
    for col in interesting_obs:
        if col in ad.obs:
            print(f"\n---- {col} ----")
            s = ad.obs[col]
            print("dtype:", s.dtype)
            print("n_unique:", s.nunique())
            print(s.value_counts().head(20))

    # ---------- VAR (gene metadata) ----------
    print("\n=== VAR (gene metadata) ===")
    print("var columns:", list(ad.var.columns))
    print("\nvar.head():")
    print(ad.var.head())
    print("\nvar_names (first 10):")
    print(ad.var_names[:10].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--h5ad",
        type=str,
        default="data/raw/stateFate_inVitro_normed_counts.h5ad",
        help="Path to .h5ad file to inspect",
    )
    args = parser.parse_args()
    inspect_h5ad(args.h5ad)


if __name__ == "__main__":
    main()
