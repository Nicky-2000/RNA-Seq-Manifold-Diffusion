from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import scanpy as sc
import argparse
import yaml
from .eggfm import run_eggfm_dimred

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="QC + EDA for unperturbed cells.")
    ap.add_argument("--params", required=True, help="configs/params.yml")
    ap.add_argument("--out", required=True, help="out/interim")
    ap.add_argument("--ad", required=True, help="path to unperturbed .h5ad")


def main() -> None:
    print("[main] starting", flush=True)
    args = build_argparser().parse_args()
    print("[main] parsed args", flush=True)
    params: Dict[str, Any] = yaml.safe_load(Path(args.params).read_text())
    print("[main] loaded params", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[main] reading AnnData...", flush=True)
    qc_ad = sc.read_h5ad(args.ad)
    print("[main] AnnData loaded, computing neighbor_overlap...", flush=True)
    qc_ad = run_eggfm_dimred(qc_ad,params)    
    qc_ad.write_h5ad(args.ad)
