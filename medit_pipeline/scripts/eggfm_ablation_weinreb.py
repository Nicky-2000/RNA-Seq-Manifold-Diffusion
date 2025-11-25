from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
        "--ad",
        default=None,
        help="path to .h5ad; if omitted, uses a built-in dataset (e.g. paul15)",
    )
    ap.add_argument(
        "--out_txt",
        default="out/eggfm_single_ablation.txt",
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


def apply_overrides(
    base: Dict[str, Any], overrides: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Shallow nested update: overrides[section][key] -> base[section][key].
    Sections like 'eggfm_diffmap', 'eggfm_train', etc.
    """
    cfg = copy.deepcopy(base)
    for section, sub in overrides.items():
        cfg.setdefault(section, {})
        cfg[section].update(sub)
    return cfg


def main() -> None:
    print("[single_ablation] starting", flush=True)
    args = build_argparser().parse_args()
    print("[single_ablation] parsed args", flush=True)

    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    print("[single_ablation] loaded params", flush=True)

    # ---- load data ----
    print("[single_ablation] reading AnnData...", flush=True)
    if args.ad:
        qc_base = sc.read_h5ad(args.ad)
        dataset_name = Path(args.ad).stem
    else:
        # You can swap this out to Weinreb / your own loader if desired
        ad = sc.datasets.paul15()
        qc_base = prep_for_manifolds(ad)
        dataset_name = "paul15"
    print(f"[single_ablation] AnnData loaded for dataset={dataset_name}", flush=True)

    spec = params.get("spec", {})
    label_key = spec.get("ari_label_key", None)
    if label_key is None:
        raise ValueError("spec.ari_label_key must be set in params.yml")
    ari_n_dims = int(spec.get("ari_n_dims", spec.get("n_pcs", 10)))
    n_pcs = int(spec.get("n_pcs", 10))

    out_txt_path = Path(args.out_txt)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- baseline values from params (your “previous best run”) ----
    base_diff = params.get("eggfm_diffmap", {})
    base_metric_gamma = float(base_diff.get("metric_gamma", 0.2))
    base_metric_lambda = float(base_diff.get("metric_lambda", 4.0))
    base_n_neighbors = int(base_diff.get("n_neighbors", 30))
    base_t = float(base_diff.get("t", 1.0))
    base_clip_low = base_diff.get("energy_clip_low", 0.05)
    base_clip_high = base_diff.get("energy_clip_high", 0.95)
    base_clip_mode = base_diff.get("clip_mode", "baseline")  # optional flag

    # ---- define single-parameter ablation ranges around baseline ----
    metric_lambda_values = [2.0, 8.0]  # around baseline 4.0
    metric_gamma_values = [0.1, 0.5]  # around baseline 0.2
    t_values = [0.5, 2.0]  # baseline 1.0
    n_neighbors_values = [15, 50]  # baseline 30
    clip_low_values = [0.01, 0.10]  # around baseline 0.05 quantile
    clip_high_values = [0.90, 0.99]  # around baseline 0.95 quantile

    # --- SLOT for "clipping vs no clipping" ablation ---
    # You can use this param (or rename it) inside diffmap_eggfm.build_eggfm_diffmap
    # to switch between:
    #   - your current robust-normalized clipped metric
    #   - your legacy/no-clipping implementation
    #
    # For now, we only run with "current" so this script works immediately.
    # You can add e.g. "legacy" here and handle it in your EGGFM code.
    clip_mode_values = ["current"]  # TODO: add "legacy" or "no_clip" etc.

    # ---- set up experiment list: (name, overrides) ----
    experiments: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []

    def add_experiment(name: str, overrides: Dict[str, Dict[str, Any]]) -> None:
        experiments.append((name, overrides))

    # Baseline: exactly params.yml
    add_experiment("baseline", {})

    # Metric λ ablation
    for lam in metric_lambda_values:
        add_experiment(
            f"metric_lambda={lam}",
            {"eggfm_diffmap": {"metric_lambda": float(lam)}},
        )

    # Metric γ ablation
    for gam in metric_gamma_values:
        add_experiment(
            f"metric_gamma={gam}",
            {"eggfm_diffmap": {"metric_gamma": float(gam)}},
        )

    # Diffusion time t ablation
    for t_val in t_values:
        add_experiment(
            f"t={t_val}",
            {"eggfm_diffmap": {"t": float(t_val)}},
        )

    # Neighborhood size ablation
    for k in n_neighbors_values:
        add_experiment(
            f"n_neighbors={k}",
            {"eggfm_diffmap": {"n_neighbors": int(k)}},
        )

    # Energy clip low ablation (quantile)
    for q_low in clip_low_values:
        add_experiment(
            f"energy_clip_low={q_low}",
            {"eggfm_diffmap": {"energy_clip_low": float(q_low)}},
        )

    # Energy clip high ablation (quantile)
    for q_hi in clip_high_values:
        add_experiment(
            f"energy_clip_high={q_hi}",
            {"eggfm_diffmap": {"energy_clip_high": float(q_hi)}},
        )

    # Clipping-mode ablation (hook for your legacy vs new implementation)
    for clip_mode in clip_mode_values:
        add_experiment(
            f"clip_mode={clip_mode}",
            {"eggfm_diffmap": {"clip_mode": clip_mode}},
        )

    # ---- write header ----
    if not out_txt_path.exists():
        with out_txt_path.open("w") as f:
            f.write(
                "dataset\texp_name\t"
                "metric_gamma\tmetric_lambda\tenergy_clip_low\tenergy_clip_high\t"
                "t\tn_neighbors\tclip_mode\t"
                "ari_diffmap_eggfm\tari_diffmap_eggfm_x2\n"
            )

    # ---- run all experiments ----
    for idx, (exp_name, overrides) in enumerate(experiments, start=1):
        print(
            f"[single_ablation] === experiment {idx}/{len(experiments)}: {exp_name} ===",
            flush=True,
        )

        cfg = apply_overrides(params, overrides)
        diff_cfg = cfg.get("eggfm_diffmap", {})

        metric_gamma = float(diff_cfg.get("metric_gamma", base_metric_gamma))
        metric_lambda = float(diff_cfg.get("metric_lambda", base_metric_lambda))
        n_neighbors = int(diff_cfg.get("n_neighbors", base_n_neighbors))
        t_val = float(diff_cfg.get("t", base_t))
        energy_clip_low = diff_cfg.get("energy_clip_low", base_clip_low)
        energy_clip_high = diff_cfg.get("energy_clip_high", base_clip_high)
        clip_mode = diff_cfg.get("clip_mode", base_clip_mode)

        # fresh copy of data
        qc_run = qc_base.copy()

        # 1) Run EGGFM + pure EGGFM Diffmap
        qc_run, _ = run_eggfm_dimred(qc_run, cfg)
        # qc_run.obsm["X_eggfm"] is the pure EGGFM DM under this config.

        # 2) Double diffusion on top of X_eggfm
        sc.pp.neighbors(
            qc_run,
            n_neighbors=n_neighbors,
            use_rep="X_eggfm",
        )
        sc.tl.diffmap(qc_run, n_comps=n_pcs)
        qc_run.obsm["X_diff_eggfm_x2"] = qc_run.obsm["X_diffmap"][:, :n_pcs]

        # 3) Compute ARIs
        try:
            ari_eggfm = compute_ari(qc_run, "X_eggfm", label_key, ari_n_dims)
        except Exception as e:
            print(
                f"[single_ablation] ARI failed for X_eggfm in {exp_name}: {e}",
                flush=True,
            )
            ari_eggfm = float("nan")

        try:
            ari_eggfm_x2 = compute_ari(qc_run, "X_diff_eggfm_x2", label_key, ari_n_dims)
        except Exception as e:
            print(
                f"[single_ablation] ARI failed for X_diff_eggfm_x2 in {exp_name}: {e}",
                flush=True,
            )
            ari_eggfm_x2 = float("nan")

        print(
            f"[single_ablation] {exp_name}: "
            f"eggfm={ari_eggfm:.4f}, eggfm_x2={ari_eggfm_x2:.4f}",
            flush=True,
        )

        # 4) Append to txt file
        with out_txt_path.open("a") as f:
            f.write(
                f"{dataset_name}\t{exp_name}\t"
                f"{metric_gamma}\t{metric_lambda}\t{energy_clip_low}\t{energy_clip_high}\t"
                f"{t_val}\t{n_neighbors}\t{clip_mode}\t"
                f"{ari_eggfm:.6f}\t{ari_eggfm_x2:.6f}\n"
            )

    print(f"[single_ablation] done, results at {out_txt_path}", flush=True)

    # ---- Optional: copy to GCS ----
    if args.gcs_path:
        print(
            f"[single_ablation] copying {out_txt_path} to {args.gcs_path} via gsutil cp",
            flush=True,
        )
        try:
            subprocess.run(
                ["gsutil", "cp", str(out_txt_path), args.gcs_path],
                check=False,
            )
        except Exception as e:
            print(f"[single_ablation] gsutil cp failed: {e}", flush=True)


if __name__ == "__main__":
    main()
