#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scanpy as sc  # type: ignore
from scipy import sparse
import subprocess
import yaml
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import scvi
import phate
from dcol_pca import dcol_pca0
from EGGFM.eggfm import run_eggfm_dimred

# Example use
# conda run -n venv python scripts/dim_red.py   --params configs/params.yml   --out out   --ad data/prep/qc.h5ad   --plot-max-cells 10000 --report-to-gcs gs://medit-uml-prod-uscentral1-8e7a/out

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--ad", required=True, help="path to unperturbed .h5ad")
    ap.add_argument(
        "--report", action="store_true", help="emit qc_summary, plots, manifest"
    )
    ap.add_argument(
        "--report-to-gcs",
        metavar="GS_PREFIX",
        default=None,
        help="If set (e.g., gs://BUCKET/out/interim), upload report files there",
    )
    ap.add_argument(
        "--plot-max-cells",
        type=int,
        default=10000,
        help="Max cells to plot (subsample if larger)",
    )
    return ap


def _try_gsutil_cp(paths: List[Path], gs_prefix: str) -> Dict[str, List[str]]:
    """
    Try to 'gsutil cp' each file. Always keep local copies.
    Returns a dict with 'uploaded' and 'failed' lists of basenames.
    """
    results: dict[str, list[str]] = {"uploaded": [], "failed": []}
    gs_prefix = gs_prefix.rstrip("/")

    for p in paths:
        try:
            # Use -n so we don't overwrite; capture output for debugging
            proc = subprocess.run(
                ["gsutil", "-m", "cp", str(p), f"{gs_prefix}/"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                results["uploaded"].append(p.name)
            else:
                # Do not raise; we want to keep going and keep files local.
                results["failed"].append(p.name)
                print(f"[report] upload failed for {p.name}:\n{proc.stderr.strip()}")
        except FileNotFoundError:
            # gsutil not installed
            results["failed"].append(p.name)
            print("[report] 'gsutil' not found on PATH; keeping file locally:", p.name)
        except Exception as e:
            results["failed"].append(p.name)
            print(f"[report] unexpected error uploading {p.name}: {e}")
    return results


def dim_red(
    qc_ad: sc.AnnData,
    params: Dict[str, Any],
) -> tuple[Path, Path, Path]:
    """
    Run multiple dimension reduction methods, compute ARI for each, and
    generate:
      - DCOL-PCA scree plot
      - regular PCA scree plot
      - ARI comparison bar plot across methods

    Notes
    -----
    - DCOL-PCA and regular PCA are both fitted on the same subsample
      (up to dcol_max_cells cells) for an apples-to-apples comparison.
    - ARI requires a label column in qc_ad.obs; we read it from
      params["spec"]["ari_label_key"] (e.g., "gene" or "label").
    """    
    spec = params.get("spec")
    n_pcs = int(spec.get("n_pcs", 10))
    max_cells_dcol = int(spec.get("dcol_max_cells", 3000))

    # =========================
    # 1. DCOL-PCA (fit on subset, project all cells)
    # =========================

    if qc_ad.n_obs > max_cells_dcol:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(qc_ad.n_obs, size=max_cells_dcol, replace=False))
        qc_dcol = qc_ad[idx, :].copy()
        print(
            f"[dcol_pca] subsampled {max_cells_dcol}/{qc_ad.n_obs} cells for DCOL-PCA",
            flush=True,
        )
    else:
        qc_dcol = qc_ad
        print(f"[dcol_pca] using all {qc_ad.n_obs} cells for DCOL-PCA", flush=True)

    X_sub = qc_dcol.X
    print(
        "[dcol_pca] subset shape:",
        qc_dcol.shape,
        "sparse?",
        sparse.issparse(X_sub),
        flush=True,
    )
    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()

    K_sub = dcol_pca0(X_sub, nPC_max=n_pcs, Scale=True)
    vecs_dcol = K_sub["vecs"]  # shape (n_genes x n_pcs)
    vals_dcol = np.asarray(K_sub["vals"], float)

    # --- Inspect DCOL spectrum numerically (positive eigenvalues only) ---
    pos = vals_dcol[vals_dcol > 0]
    var_ratio_dcol = pos / pos.sum()
    cum_ratio_dcol = np.cumsum(var_ratio_dcol)

    print("[dcol_pca] eigenvalues + variance fractions (positive eigs only):")
    for i, (lam, vr, cr) in enumerate(
        zip(pos, var_ratio_dcol, cum_ratio_dcol), start=1
    ):
        print(f"  PC{i:2d}: eig={lam:9.4f}, frac={vr:7.4f}, cum={cr:7.4f}")

    # Project all QC cells using same gene loadings
    X_full = qc_ad.X
    X_proj_full_dcol = X_full @ vecs_dcol
    qc_ad.obsm["X_dcolpca"] = X_proj_full_dcol

    # =========================
    # 2. Regular PCA (fit on the same subset as DCOL)
    # =========================
    # Fit PCA on qc_dcol for apples-to-apples comparison
    sc.tl.pca(qc_dcol, n_comps=n_pcs, use_highly_variable=False, zero_center=False)

    pca_vals = np.asarray(qc_dcol.uns["pca"]["variance"], float)
    pca_var = pca_vals / pca_vals.sum()
    pca_cum = np.cumsum(pca_var)
    print("[pca] eigenvalues + variance fractions (subset fit):")
    for i, (lam, vr, cr) in enumerate(zip(pca_vals, pca_var, pca_cum), start=1):
        print(f"  PC{i:2d}: eig={lam:9.4f}, frac={vr:7.4f}, cum={cr:7.4f}")

    # Project ALL qc_ad cells using loadings learned on qc_dcol
    # Scanpy stores loadings in varm["PCs"] (genes x n_pcs)
    pca_loadings = qc_dcol.varm["PCs"][:, :n_pcs]  # (n_genes x n_pcs)
    X_full = qc_ad.X
    X_proj_full_pca = X_full @ pca_loadings
    qc_ad.obsm["X_pca"] = X_proj_full_pca

    # =========================
    # 3. Diffusion maps (PCA prior & DCOL prior)
    # =========================
    # (a) Diffmap with PCA prior
    sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_pca")
    sc.tl.diffmap(qc_ad, n_comps=n_pcs)
    X_diff_pca = qc_ad.obsm["X_diffmap"][:, :n_pcs]
    qc_ad.obsm["X_diff_pca"] = X_diff_pca

    # (b) Diffmap with DCOL prior
    sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_dcolpca")
    sc.tl.diffmap(qc_ad, n_comps=n_pcs)
    X_diff_dcol = qc_ad.obsm["X_diffmap"][:, :n_pcs]
    qc_ad.obsm["X_diff_dcol"] = X_diff_dcol

    # (c) Diffmap with EGGFM
    # run EGGFM
    run_eggfm_dimred(qc_ad, params)
    sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_eggfm")
    sc.tl.diffmap(qc_ad, n_comps=n_pcs)
    X_diff_dcol = qc_ad.obsm["X_diffmap"][:, :n_pcs]
    qc_ad.obsm["X_diff_eggfm"] = X_diff_dcol

    # =========================
    # 4. scVI latent space
    # =========================
    scvi.model.SCVI.setup_anndata(qc_ad, layer=None)
    scvi_model = scvi.model.SCVI(qc_ad, n_latent=n_pcs)
    scvi_model.train()
    X_scvi = scvi_model.get_latent_representation()
    qc_ad.obsm["X_scvi"] = X_scvi  # (n_cells x n_pcs)

    # =========================
    # 5. PHATE embedding
    # =========================
    X = qc_ad.X
    if sparse.issparse(X):
        X = X.toarray()

    phate_op = phate.PHATE(n_components=n_pcs)
    X_phate = phate_op.fit_transform(X)
    qc_ad.obsm["X_phate"] = X_phate  # (n_cells x n_pcs)

    return


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    qc_ad = sc.read_h5ad(args.ad)
    d_plot, pca_plot, ari_plot = dim_red(qc_ad, params, out_dir)
    qc_ad.write_h5ad(args.ad)
    #  Upload if requested
    if args.report_to_gcs:
        _try_gsutil_cp([ari_plot, d_plot, pca_plot], args.report_to_gcs)


if __name__ == "__main__":
    main()
