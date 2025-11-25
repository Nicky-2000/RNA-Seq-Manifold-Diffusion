from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import scanpy as sc
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import subprocess

from EGGFM.eggfm import run_eggfm_dimred
from EGGFM.prep import prep_for_manifolds


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument(
        "--ad", default=None, help="path to .h5ad (if omitted, uses paul15)"
    )
    ap.add_argument(
        "--out_txt",
        default="out/eggfm_ablation_results.txt",
        help="Where to write ablation results",
    )
    ap.add_argument(
        "--gcs_path",
        default=None,
        help="Optional GCS path (e.g. gs://bucket/dir/) to copy the txt file to",
    )
    return ap


def compute_ari(
    qc_ad: sc.AnnData,
    emb_key: str,
    label_key: str,
    n_dims: int,
) -> float:
    """Compute ARI from an embedding in qc_ad.obsm[emb_key]."""
    if emb_key not in qc_ad.obsm:
        raise ValueError(f"Embedding {emb_key!r} not found in qc_ad.obsm")

    X = qc_ad.obsm[emb_key]
    if X.ndim != 2:
        raise ValueError(f"Embedding {emb_key!r} must be 2D, got shape={X.shape}")

    if label_key not in qc_ad.obs:
        raise ValueError(f"Label key {label_key!r} not found in qc_ad.obs")

    labels = qc_ad.obs[label_key].to_numpy()
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(
            f"Need at least 2 unique labels for ARI; got {unique_labels.size}"
        )

    n_clusters = unique_labels.size
    k_eff = min(n_dims, X.shape[1])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km.fit(X[:, :k_eff])
    ari = adjusted_rand_score(labels, km.labels_)
    return float(ari)


def main() -> None:
    print("[main] starting", flush=True)
    args = build_argparser().parse_args()
    print("[main] parsed args", flush=True)

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    print("[main] loaded params", flush=True)

    # ---- load data ----
    print("[main] reading AnnData...", flush=True)
    if args.ad:
        qc_ad = sc.read_h5ad(args.ad)
        dataset_name = Path(args.ad).stem
    else:
        ad = sc.datasets.paul15()
        qc_ad = prep_for_manifolds(ad)
        dataset_name = "paul15"
    print(f"[main] AnnData loaded for dataset={dataset_name}", flush=True)

    spec = params.get("spec", {})
    label_key = spec.get("ari_label_key", None)
    if label_key is None:
        raise ValueError("spec.ari_label_key must be set in params.yml")
    ari_n_dims = int(spec.get("ari_n_dims", spec.get("n_pcs", 10)))

    out_txt_path = Path(args.out_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- define ablation grid (edit these lists as you like) ----
    hidden_dims_grid: List[List[int]] = [
        [256],
        [256, 256],
    ]
    sigma_grid = [0.05, 0.1]
    metric_lambda_grid = [2.0, 4.0]
    eps_trunc_grid = ["no", "yes"]

    # keep metric_gamma from base params, or default if missing
    base_metric_gamma = params.get("eggfm_diffmap", {}).get("metric_gamma", 0.2)

    # ---- write header ----
    if not out_txt_path.exists():
        with out_txt_path.open("w") as f:
            f.write(
                "dataset\thidden_dims\tsigma\tmetric_gamma\tmetric_lambda\t"
                "eps_trunc\tari_diffmap_eggfm\tari_diffmap_eggfm_x2\n"
            )

    # ---- ablation loop ----
    combo_idx = 0
    for hd in hidden_dims_grid:
        for sigma in sigma_grid:
            for lam in metric_lambda_grid:
                for eps_trunc in eps_trunc_grid:
                    combo_idx += 1
                    print(
                        f"[main] === combo {combo_idx}: "
                        f"hidden_dims={hd}, sigma={sigma}, "
                        f"metric_lambda={lam}, eps_trunc={eps_trunc} ===",
                        flush=True,
                    )

                    # Deep copy base params and apply overrides
                    cfg = copy.deepcopy(params)
                    cfg.setdefault("eggfm_model", {})
                    cfg.setdefault("eggfm_train", {})
                    cfg.setdefault("eggfm_diffmap", {})

                    cfg["eggfm_model"]["hidden_dims"] = hd
                    cfg["eggfm_train"]["sigma"] = float(sigma)
                    cfg["eggfm_diffmap"]["metric_gamma"] = float(base_metric_gamma)
                    cfg["eggfm_diffmap"]["metric_lambda"] = float(lam)
                    cfg["eggfm_diffmap"]["eps_trunc"] = str(eps_trunc)

                    # Work on a copy of qc_ad so we don't keep stacking obsm keys
                    qc_run = qc_ad.copy()

                    # ---- 1) Run EGGFM + pure EGGFM Diffmap ----
                    qc_run, _ = run_eggfm_dimred(qc_run, cfg)
                    # At this point qc_run.obsm["X_eggfm"] should be the pure EGGFM DM

                    # ---- 2) Build double diffusion on top of X_eggfm ----
                    n_pcs = int(spec.get("n_pcs", 10))
                    sc.pp.neighbors(qc_run, n_neighbors=30, use_rep="X_eggfm")
                    sc.tl.diffmap(qc_run, n_comps=n_pcs)
                    qc_run.obsm["X_diff_eggfm_x2"] = qc_run.obsm["X_diffmap"][:, :n_pcs]

                    # ---- 3) Compute ARIs ----
                    try:
                        ari_eggfm = compute_ari(
                            qc_run, "X_eggfm", label_key, ari_n_dims
                        )
                    except Exception as e:
                        print(f"[main] ARI failed for X_eggfm: {e}", flush=True)
                        ari_eggfm = float("nan")

                    try:
                        ari_eggfm_x2 = compute_ari(
                            qc_run, "X_diff_eggfm_x2", label_key, ari_n_dims
                        )
                    except Exception as e:
                        print(f"[main] ARI failed for X_diff_eggfm_x2: {e}", flush=True)
                        ari_eggfm_x2 = float("nan")

                    print(
                        f"[main] combo {combo_idx} ARIs: "
                        f"eggfm={ari_eggfm:.4f}, eggfm_x2={ari_eggfm_x2:.4f}",
                        flush=True,
                    )

                    # ---- 4) Append to txt file ----
                    with out_txt_path.open("a") as f:
                        f.write(
                            f"{dataset_name}\t{hd}\t{sigma}\t{base_metric_gamma}\t{lam}\t"
                            f"{eps_trunc}\t{ari_eggfm:.6f}\t{ari_eggfm_x2:.6f}\n"
                        )

    print(f"[main] ablation done, results at {out_txt_path}", flush=True)

    # ---- Optional: copy to GCS ----
    if args.gcs_path:
        print(
            f"[main] copying {out_txt_path} to {args.gcs_path} via gsutil cp",
            flush=True,
        )
        try:
            subprocess.run(
                ["gsutil", "cp", str(out_txt_path), args.gcs_path],
                check=False,
            )
        except Exception as e:
            print(f"[main] gsutil cp failed: {e}", flush=True)


if __name__ == "__main__":
    main()
