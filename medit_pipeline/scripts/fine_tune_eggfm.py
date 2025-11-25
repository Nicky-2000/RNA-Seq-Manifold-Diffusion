from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import scanpy as sc
import argparse
import yaml
from EGGFM.eggfm import run_eggfm_dimred
from EGGFM.prep import prep_for_manifolds

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--ad", default=None, help="path to unperturbed .h5ad")
    return ap


def main() -> None:
    print("[main] starting", flush=True)
    args = build_argparser().parse_args()
    print("[main] parsed args", flush=True)
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    print("[main] loaded params", flush=True)
    print("[main] reading AnnData...", flush=True)
    if args.ad:
        qc_ad = sc.read_h5ad(args.ad)
    else:
        ad = sc.datasets.paul15()
        qc_ad = prep_for_manifolds(ad)
    print("[main] AnnData loaded, computing neighbor_overlap...", flush=True)
    qc_ad, _ = run_eggfm_dimred(qc_ad, params)
    qc_ad.write_h5ad(args.ad)

if __name__ == "__main__":
    main()
