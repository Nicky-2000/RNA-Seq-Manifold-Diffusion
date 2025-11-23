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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import scvi
import phate
from dcol_pca import dcol_pca0, plot_spectral
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


def report(ad: sc.AnnData, args: Dict[str], params: Dict[str]) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_files: List[Path] = []

    # QC summary CSV
    obs_cols = [
        c
        for c in ["n_counts", "n_genes_by_counts", "mitopercent", "pct_counts_mt"]
        if c in ad.obs
    ]
    if obs_cols:
        qc_summary = ad.obs[obs_cols].apply(pd.to_numeric, errors="coerce").describe()
        qc_csv = out_dir / "qc_summary.csv"
        qc_summary.to_csv(qc_csv)
        report_files.append(qc_csv)

    # Manifest JSON
    manifest = {
        "git": os.popen("git rev-parse --short HEAD").read().strip(),
        "input": os.path.abspath(args.ad),
        "params": {
            "min_genes": params["qc"]["min_genes"],
            "max_pct_mt": params["qc"]["max_pct_mt"],
            "hvg_n_top_genes": int(params["hvg_n_top_genes"]),
        },
        "n_cells": int(ad.n_obs),
        "n_genes": int(ad.n_vars),
        "obs_cols": list(ad.obs.columns)[:25],
        "var_cols": list(ad.var.columns)[:25],
    }
    man_json = out_dir / "manifest_qc.json"
    man_json.write_text(json.dumps(manifest, indent=2))
    report_files.append(man_json)

    # Subsample for plotting
    nmax = int(args.plot_max_cells)
    if ad.n_obs > nmax:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(ad.n_obs, size=nmax, replace=False))
        ad_plot = ad[idx, :].copy()
        print(f"[plot] subsampled {nmax}/{ad.n_obs} for speed")
    else:
        ad_plot = ad

    # PCA/Neighbors/UMAP if needed for nicer violins ordering later (optional)
    pca_png = out_dir / "weinreb_pca.png"
    try:
        if "X_pca" not in ad_plot.obsm:
            sc.pp.scale(ad_plot, max_value=10)
            sc.pp.pca(ad_plot)

        sc.pl.pca(ad_plot, show=False, save=None)
        plt.savefig(pca_png, bbox_inches="tight", dpi=160)
        plt.close()
        report_files.append(pca_png)
    except Exception as e:
        print(f"[plot] PCA plot failed: {e}")

    qc_png = out_dir / "qc_violin.png"
    try:
        sc.pl.violin(
            ad_plot,
            keys=["n_counts", "n_genes_by_counts", "mitopercent"],
            jitter=0.4,
            multi_panel=True,
            show=False,
            save=None,
        )
        plt.savefig(qc_png, bbox_inches="tight", dpi=160)
        plt.close()
        report_files.append(qc_png)
    except Exception as e:
        print(f"[plot] violin failed: {e}")

    # 2) HVG overview
    hvg_png = out_dir / "hvg.png"
    try:
        sc.pl.highly_variable_genes(ad_plot, show=False, save=None)
        plt.savefig(hvg_png, bbox_inches="tight", dpi=160)
        plt.close()
        report_files.append(hvg_png)
    except Exception as e:
        print(f"[plot] hvg plot failed: {e}")

    print(f"[report] wrote locally: {[p.name for p in report_files]}")

    # ---- Optional GCS upload (AFTER local writes) ----
    if args.report_to_gcs:
        results = _try_gsutil_cp(report_files, args.report_to_gcs)
        if results["uploaded"]:
            print("[report] uploaded to GCS:", ", ".join(results["uploaded"]))
        if results["failed"]:
            print(
                "[report] kept local copies for (upload failed):",
                ", ".join(results["failed"]),
            )
    else:
        print("[report] no --report-to-gcs provided; keeping local files only.")


def dim_red(
    qc_ad: sc.AnnData,
    params: Dict[str, Any],
    out_dir: str | Path,
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

    dcol_scree_path = plot_spectral(vals_dcol, out_dir, "dcol-pca")

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

    pca_scree_path = plot_spectral(pca_vals, out_dir, "reg-pca")

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

    ari_plot_path = ari(qc_ad, spec, out_dir)

    return dcol_scree_path, pca_scree_path, ari_plot_path


def ari(qc_ad, spec, out_dir):
    # ----- ARI label setup -----
    label_key = spec.get("ari_label_key", None)
    if label_key is None or label_key not in qc_ad.obs:
        raise ValueError(
            "ARI requested but 'ari_label_key' not set in params['spec'] "
            f"or not found in qc_ad.obs. Got ari_label_key={label_key!r}."
        )

    labels = qc_ad.obs[label_key].to_numpy()
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"Need at least 2 unique labels for ARI; got {unique_labels.size} "
            f"for label_key={label_key!r}."
        )

    n_clusters = unique_labels.size
    n_pcs = int(spec.get("n_pcs", 10))
    ari_k = int(spec.get("ari_n_dims", min(n_pcs, 10)))  # dims for ARI
    embeddings: Dict[str, np.ndarray] = {
        "dcol_pca": qc_ad.obsm["X_dcolpca"],
        "pca": qc_ad.obsm["X_pca"],
        "diffmap_pca": qc_ad.obsm["X_diff_pca"],
        "diffmap_eggfm": qc_ad.obsm["X_diff_eggfm"],
        "diffmap_dcol": qc_ad.obsm["X_diff_dcol"],
        "scvi": qc_ad.obsm["X_scvi"],
        "phate": qc_ad.obsm["X_phate"],
    }

    ari_scores: Dict[str, float] = {}

    def _compute_ari(emb: np.ndarray, name: str) -> float:
        if emb.ndim != 2:
            raise ValueError(
                f"[ARI] embedding {name} is not 2D, got shape={emb.shape}."
            )
        k_eff = min(ari_k, emb.shape[1])
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        km.fit(emb[:, :k_eff])
        ari = adjusted_rand_score(labels, km.labels_)
        return float(ari)

    for name, emb in embeddings.items():
        try:
            ari = _compute_ari(emb, name)
            ari_scores[name] = ari
        except Exception as e:
            print(f"[ARI] failed for {name}: {e}", flush=True)

    print("[ARI] scores (k dimensions per embedding):")
    for name, ari in ari_scores.items():
        print(f"  {name:12s}: ARI = {ari:0.4f}")

    # ----- ARI bar plot -----
    if len(ari_scores) == 0:
        raise RuntimeError("[ARI] no ARI scores computed; cannot make ARI plot.")

    methods = list(ari_scores.keys())
    values = [ari_scores[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, values)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(f"Clustering ARI across DR methods (k={ari_k})")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ari_plot_path = out_dir / "dimred_ari_comparison.png"
    plt.savefig(ari_plot_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return ari_plot_path


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
