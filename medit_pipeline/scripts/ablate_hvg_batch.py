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
        "--out_txt",
        default="out/eggfm_hvg_batch_ablation_paul15.txt",
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
    print("[hvg_batch_ablation] starting", flush=True)
    args = build_argparser().parse_args()
    print("[hvg_batch_ablation] parsed args", flush=True)

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    print("[hvg_batch_ablation] loaded params", flush=True)

    spec = params.get("spec", {})
    label_key = spec.get("ari_label_key", None)
    if label_key is None:
        raise ValueError("spec.ari_label_key must be set in params.yml")
    ari_n_dims = int(spec.get("ari_n_dims", spec.get("n_pcs", 10)))

    out_txt_path = Path(args.out_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- best-config hyperparams (from your ablation) ----
    best_hidden_dims = [256]
    best_sigma = 0.05
    best_metric_gamma = params.get("eggfm_diffmap", {}).get("metric_gamma", 0.2)
    best_metric_lambda = 4.0
    best_eps_trunc = "no"
    best_t = 2.0
    best_n_neighbors = 15
    best_num_epochs = params.get("eggfm_train", {}).get("num_epochs", 50)

    # ---- ablation grids: HVGs and batch size only ----
    hvg_grid: List[int] = [1000, 2000, 3000]
    batch_size_grid: List[int] = [512, 1024, 2048]

    # ---- write header ----
    if not out_txt_path.exists():
        with out_txt_path.open("w") as f:
            f.write(
                "dataset\thvg_n_top_genes\tbatch_size\t"
                "hidden_dims\tsigma\tmetric_gamma\tmetric_lambda\t"
                "eps_trunc\tt\tn_neighbors\tnum_epochs\t"
                "ari_diffmap_eggfm\tari_diffmap_eggfm_x2\n"
            )

    combo_idx = 0

    for hvg_n in hvg_grid:
        # ---- load raw paul15 and re-run prep for each HVG setting ----
        print(
            f"[hvg_batch_ablation] loading paul15 and prepping for HVG={hvg_n}...",
            flush=True,
        )
        ad = sc.datasets.paul15()

        # NOTE: this assumes prep_for_manifolds can accept hvg_n_top_genes.
        # If your actual signature differs, adjust this call accordingly.
        qc_base = prep_for_manifolds(ad, hvg_n_top_genes=hvg_n)
        dataset_name = f"paul15_hvg{hvg_n}"

        print(
            f"[hvg_batch_ablation] qc_base shape={qc_base.shape} for HVG={hvg_n}",
            flush=True,
        )

        for batch_size in batch_size_grid:
            combo_idx += 1
            print(
                f"[hvg_batch_ablation] === combo {combo_idx}: "
                f"hvg={hvg_n}, batch_size={batch_size} ===",
                flush=True,
            )

            # Build cfg with best hyperparams + current batch_size
            cfg = copy.deepcopy(params)
            cfg.setdefault("eggfm_model", {})
            cfg.setdefault("eggfm_train", {})
            cfg.setdefault("eggfm_diffmap", {})

            cfg["eggfm_model"]["hidden_dims"] = best_hidden_dims
            cfg["eggfm_train"]["sigma"] = float(best_sigma)
            cfg["eggfm_train"]["batch_size"] = int(batch_size)
            cfg["eggfm_train"]["num_epochs"] = int(best_num_epochs)

            cfg["eggfm_diffmap"]["metric_gamma"] = float(best_metric_gamma)
            cfg["eggfm_diffmap"]["metric_lambda"] = float(best_metric_lambda)
            cfg["eggfm_diffmap"]["eps_trunc"] = str(best_eps_trunc)
            cfg["eggfm_diffmap"]["t"] = float(best_t)
            cfg["eggfm_diffmap"]["n_neighbors"] = int(best_n_neighbors)

            # copy qc_base so we don't mutate across combos
            qc_run = qc_base.copy()

            # ---- 1) Run EGGFM + pure EGGFM Diffmap ----
            qc_run, _ = run_eggfm_dimred(qc_run, cfg)
            # qc_run.obsm["X_eggfm"] is pure EGGFM DM under this config.

            # ---- 2) Double diffusion on top of X_eggfm ----
            n_pcs = int(spec.get("n_pcs", 10))
            sc.pp.neighbors(
                qc_run,
                n_neighbors=best_n_neighbors,
                use_rep="X_eggfm",
            )
            sc.tl.diffmap(qc_run, n_comps=n_pcs)
            qc_run.obsm["X_diff_eggfm_x2"] = qc_run.obsm["X_diffmap"][:, :n_pcs]

            # ---- 3) Compute ARIs ----
            try:
                ari_eggfm = compute_ari(qc_run, "X_eggfm", label_key, ari_n_dims)
            except Exception as e:
                print(
                    f"[hvg_batch_ablation] ARI failed for X_eggfm: {e}",
                    flush=True,
                )
                ari_eggfm = float("nan")

            try:
                ari_eggfm_x2 = compute_ari(
                    qc_run, "X_diff_eggfm_x2", label_key, ari_n_dims
                )
            except Exception as e:
                print(
                    f"[hvg_batch_ablation] ARI failed for X_diff_eggfm_x2: {e}",
                    flush=True,
                )
                ari_eggfm_x2 = float("nan")

            print(
                f"[hvg_batch_ablation] combo {combo_idx} ARIs: "
                f"eggfm={ari_eggfm:.4f}, eggfm_x2={ari_eggfm_x2:.4f}",
                flush=True,
            )

            # ---- 4) Append to txt file ----
            with out_txt_path.open("a") as f:
                f.write(
                    f"{dataset_name}\t{hvg_n}\t{batch_size}\t"
                    f"{best_hidden_dims}\t{best_sigma}\t{best_metric_gamma}\t{best_metric_lambda}\t"
                    f"{best_eps_trunc}\t{best_t}\t{best_n_neighbors}\t{best_num_epochs}\t"
                    f"{ari_eggfm:.6f}\t{ari_eggfm_x2:.6f}\n"
                )

    print(f"[hvg_batch_ablation] done, results at {out_txt_path}", flush=True)

    # ---- Optional: copy to GCS ----
    if args.gcs_path:
        print(
            f"[hvg_batch_ablation] copying {out_txt_path} to {args.gcs_path} via gsutil cp",
            flush=True,
        )
        try:
            subprocess.run(
                ["gsutil", "cp", str(out_txt_path), args.gcs_path],
                check=False,
            )
        except Exception as e:
            print(f"[hvg_batch_ablation] gsutil cp failed: {e}", flush=True)


if __name__ == "__main__":
    main()
