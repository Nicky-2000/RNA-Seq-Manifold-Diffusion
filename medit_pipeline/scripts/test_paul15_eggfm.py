# scripts/test_paul15_eggfm.py

from pathlib import Path
from typing import Any, Dict, List
import argparse
import yaml
import scanpy as sc

from EGGFM.prep import prep_for_manifolds
from EGGFM.train_energy import train_energy_model
from EGGFM.engine import EGGFMDiffusionEngine
from EGGFM.data_sources import AnnDataViewProvider
from EGGFM.utils import subsample_adata


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument(
        "--max-cells",
        type=int,
        default=2000,
        help="Subsample this many cells for quick tests (default: 2000).",
    )
    ap.add_argument(
        "--subsample-seed",
        type=int,
        default=0,
        help="Random seed for subsampling.",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())

    diff_cfg: Dict[str, Any] = params.get("eggfm_diffmap", {})
    model_cfg: Dict[str, Any] = params.get("eggfm_model", {})
    train_cfg: Dict[str, Any] = params.get("eggfm_train", {})

    print("[test_paul15] loading paul15...", flush=True)
    ad = sc.datasets.paul15()

    print("[test_paul15] running prep_for_manifolds...", flush=True)
    qc_ad = prep_for_manifolds(ad)

    qc_ad = subsample_adata(
        qc_ad,
        max_cells=args.max_cells,
        seed=args.subsample_seed,
    )

    print("[test_paul15] training energy model...", flush=True)
    energy_model = train_energy_model(qc_ad, model_cfg, train_cfg)

    view_provider = AnnDataViewProvider(
        geometry_source=diff_cfg.get("geometry_source", "pca"),
        energy_source=diff_cfg.get("energy_source", "hvg"),
    )

    engine = EGGFMDiffusionEngine(
        energy_model=energy_model,
        diff_cfg=diff_cfg,
        view_provider=view_provider,
    )

    metric_modes: List[str] = ["euclidean", "scm", "hessian_mixed"]

    for mode in metric_modes:
        print(f"[test_paul15] Building embedding for metric_mode='{mode}'", flush=True)
        X_emb = engine.build_embedding(qc_ad, metric_mode=mode)
        key = f"X_eggfm_{mode}"
        qc_ad.obsm[key] = X_emb
        print(f"[test_paul15] Stored embedding in .obsm['{key}'] with shape {X_emb.shape}", flush=True)

    out_path = f"data/paul15/paul15_eggfm_test_{args.max_cells}cells.h5ad"
    Path("data/paul15").mkdir(parents=True, exist_ok=True)
    print(f"[test_paul15] writing result to {out_path}", flush=True)
    qc_ad.write_h5ad(out_path)


if __name__ == "__main__":
    main()
