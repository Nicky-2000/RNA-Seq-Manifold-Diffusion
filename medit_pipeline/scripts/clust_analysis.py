from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import subprocess
import argparse
import yaml
from scipy import sparse


def ari_stability(
    qc_ad: sc.AnnData,
    spec: Dict[str, Any],
    out_dir: Path,
    n_repeats: int = 10,
) -> Path:
    label_key = spec.get("ari_label_key", None)
    if label_key is None or label_key not in qc_ad.obs:
        raise ValueError("ARI label_key missing or not in qc_ad.obs")

    labels = qc_ad.obs[label_key].to_numpy()
    unique_labels = np.unique(labels)
    n_clusters = unique_labels.size
    n_pcs = int(spec.get("n_pcs", 10))
    ari_k = int(spec.get("ari_n_dims", min(n_pcs, 10)))

    embeddings: Dict[str, np.ndarray] = {
        "dcol_pca": qc_ad.obsm["X_dcolpca"],
        "pca": qc_ad.obsm["X_pca"],
        "diffmap_pca": qc_ad.obsm["X_diff_pca"],
        "diffmap_eggfm": qc_ad.obsm["X_diff_eggfm"],
        "diffmap_dcol": qc_ad.obsm["X_diff_dcol"],
        "scvi": qc_ad.obsm["X_scvi"],
        "phate": qc_ad.obsm["X_phate"],
    }

    summary = []

    for name, emb in embeddings.items():
        k_eff = min(ari_k, emb.shape[1])
        X_use = emb[:, :k_eff]

        scores = []
        for seed in range(n_repeats):
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            km.fit(X_use)
            ari = adjusted_rand_score(labels, km.labels_)
            scores.append(ari)

        scores = np.array(scores, float)
        mean = scores.mean()
        std = scores.std()
        print(f"[ARI-stab] {name:12s}: mean={mean:0.4f}, std={std:0.4f}")
        summary.append((name, mean, std))

    # Plot means with error bars
    methods = [s[0] for s in summary]
    means = [s[1] for s in summary]
    stds = [s[2] for s in summary]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, means, yerr=stds, capsize=4)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(f"ARI mean ± std across {n_repeats} KMeans seeds")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = out_dir / "dimred_ari_stability.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return out_path


def plot_umaps(
    qc_ad: sc.AnnData,
    label_key: str,
    out_dir: Path,
    methods: Dict[str, str] | None = None,
) -> List[Path]:
    """
    methods: mapping from embedding key in obsm -> pretty name
    """
    if methods is None:
        methods = {
            "X_pca": "PCA",
            "X_dcolpca": "DCOL-PCA",
            "X_diff_pca": "Diffmap (PCA)",
            "X_diff_dcol": "Diffmap (DCOL)",
            "X_diff_eggfm": "Diffmap (EGGFM)",
            "X_scvi": "scVI",
            "X_phate": "PHATE",
        }

    out_paths = []
    ad_tmp = qc_ad.copy()

    for obsm_key, title in methods.items():
        if obsm_key not in qc_ad.obsm:
            print(f"[UMAP] skipping {obsm_key}, not in qc_ad.obsm")
            continue

        # Work on a copy to avoid overwriting neighbors/UMAP repeatedly
        ad_tmp.obsm["X_tmp"] = ad_tmp.obsm[obsm_key]

        sc.pp.neighbors(ad_tmp, n_neighbors=30, use_rep="X_tmp")
        sc.tl.umap(ad_tmp, min_dist=0.3)

        fig = sc.pl.umap(
            ad_tmp,
            color=label_key,
            show=False,
            return_fig=True,
            title=f"{title} → UMAP ({label_key})",
        )
        out_path = out_dir / f"umap_{obsm_key}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        out_paths.append(out_path)
        print(f"[UMAP] wrote {out_path}")

    return out_paths

from sklearn.neighbors import NearestNeighbors


def neighbor_overlap(X1, X2, k=30, n_subsample=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = X1.shape[0]
    idx = np.sort(rng.choice(n, size=min(n_subsample, n), replace=False))

    nn1 = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(
        X1
    )
    nn2 = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(
        X2
    )

    _, inds1 = nn1.kneighbors(X1[idx])
    _, inds2 = nn2.kneighbors(X2[idx])

    # drop self
    inds1 = inds1[:, 1:]
    inds2 = inds2[:, 1:]

    overlaps = []
    for a, b in zip(inds1, inds2):
        overlaps.append(len(set(a).intersection(b)) / k)
    return np.mean(overlaps)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", default=None, help="out/interim")
    ap.add_argument("--pca", default=None, help="path to unperturbed .h5ad")
    ap.add_argument("--ari", default=None, help="path to unperturbed .h5ad")
    ap.add_argument("--umap", default=None, help="path to unperturbed .h5ad")
    ap.add_argument("--ad", default=None, help="path to unperturbed .h5ad")
    ap.add_argument("--ari-stab", default=None, help="emit qc_summary, plots, manifest")
    ap.add_argument(
        "--report-to-gcs",
        metavar="GS_PREFIX",
        default=None,
        help="If set (e.g., gs://BUCKET/out/interim), upload report files there",
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

def ari_plot(
    qc_ad,
    spec: Dict[str, object],
    out_dir: Path,
    embeddings: Dict[str, np.ndarray],
) -> Path:
    """
    Compute ARI for a dict of embeddings and make a barplot.

    embeddings: dict mapping method name -> embedding (n_cells, d)
    """
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

    ari_scores: Dict[str, float] = {}

    def _compute_ari(emb: np.ndarray, name: str) -> float:
        if emb.ndim != 2:
            raise ValueError(
                f"[ARI] embedding {name} is not 2D, got shape={emb.shape}."
            )
        if emb.shape[0] != labels.shape[0]:
            raise ValueError(
                f"[ARI] embedding {name} has n_cells={emb.shape[0]} "
                f"but labels have n_cells={labels.shape[0]}."
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

    if len(ari_scores) == 0:
        raise RuntimeError("[ARI] no ARI scores computed; cannot make ARI plot.")

    # ----- ARI bar plot -----
    methods_plotted = list(ari_scores.keys())
    values = [ari_scores[m] for m in methods_plotted]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods_plotted, values)
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(f"Clustering ARI across DR methods (k={ari_k})")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ari_plot_path = out_dir / "dimred_ari_comparison.png"
    plt.savefig(ari_plot_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return ari_plot_path


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


def main() -> None:
    print("[main] starting", flush=True)
    args = build_argparser().parse_args()
    print("[main] parsed args", flush=True)

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    spec = params.get("spec")
    print("[main] loaded params", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[main] reading AnnData...", flush=True)
    if args.ad:
        qc_ad = sc.read_h5ad(args.ad)
    else:
        ad = sc.datasets.paul15()
        qc_ad = prep_for_manifolds(ad)

    print("[main] AnnData loaded, computing neighbor_overlap...", flush=True)

    spec = params.get("spec")
    n_pcs = int(spec.get("n_pcs", 10))

    # EGGFM → Diffmap
    # Nothing, X_eggfm is already a Diffmap

    # EGGFM → Diffmap → Diffmap
    sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_eggfm")
    sc.tl.diffmap(qc_ad, n_comps=n_pcs)
    X_diff_eggfm = qc_ad.obsm["X_diffmap"][:, :n_pcs]
    qc_ad.obsm["X_diff_eggfm"] = X_diff_eggfm

    if args.pca:
        # PCA → Diffmap
        sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_pca")
        sc.tl.diffmap(qc_ad, n_comps=n_pcs)
        X_diff_pca = qc_ad.obsm["X_diffmap"][:, :n_pcs]
        qc_ad.obsm["X_diff_pca"] = X_diff_pca

        # PCA → Diffmap → Diffmap
        sc.pp.neighbors(qc_ad, n_neighbors=30, use_rep="X_diff_pca")
        sc.tl.diffmap(qc_ad, n_comps=n_pcs)
        X_diff_pca_double = qc_ad.obsm["X_diffmap"][:, :n_pcs]
        qc_ad.obsm["X_diff_pca_x2"] = X_diff_pca_double

    qc_ad.write_h5ad(args.ad)
    gcs_paths = []

    if args.umap:
        if args.pca:
            umap_paths = plot_umaps(
                qc_ad,
                spec["ari_label_key"],
                out_dir,
                {
                    "X_eggfm": "Diffmap (EGGFM)",
                    "X_diff_eggfm": "Diffmap (EGGFMx2)",
                    "X_diff_pca": "Diffmap (PCA)",
                    "X_diff_pca_x2": "Diffmap (PCAx2)",
                },
            )
        else:
            umap_paths = plot_umaps(
                qc_ad,
                spec["ari_label_key"],
                out_dir,
                {
                    "X_eggfm": "Diffmap (EGGFM)",
                    "X_diff_eggfm": "Diffmap (EGGFMx2)",
                },
            )
        gcs_paths += umap_paths

    if args.ari:
        embeddings = {
            "diffmap_eggfm": qc_ad.obsm["X_eggfm"],
            "diffmap_eggfm_x2": qc_ad.obsm["X_diff_eggfm"],
            "diffmap_pca": qc_ad.obsm["X_diff_pca"],
            "diffmap_pca_x2": qc_ad.obsm["X_diff_pca_x2"],
        }
        ari_path = ari_plot(qc_ad, spec, out_dir, embeddings)
        gcs_paths.append(ari_path)

    if args.ari_stab:
        ov = neighbor_overlap(X_diff_pca, X_diff_eggfm, k=30)
        ari_stab_path = ari_stability(qc_ad, spec, out_dir)
        gcs_paths.append(ari_stab_path)
        print("Mean neighbor overlap (Diffmap PCA vs EGGFM):", ov)

    #  Upload if requested
    if args.report_to_gcs:
        _try_gsutil_cp(gcs_paths, args.report_to_gcs)


if __name__ == "__main__":
    main()
