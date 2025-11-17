# MEDIT – Euclidean vs Manifold Diffusion for Single-Cell Trajectories

MEDIT is the reproducible codebase for our UML course project.

**Goal.** Build and compare two diffusion models for single-cell differentiation data:

- **Euclidean baseline** – standard DDPM / SDE trained on PCA embeddings.
- **Manifold-aware model** – diffusion in intrinsic coordinates (e.g. Laplace–Beltrami / heat kernel on a kNN graph).

This repo focuses on **data ingestion, QC, and preprocessing** (plus GCP/GCS + VM plumbing).  
The preprocessed outputs (`data/prep`) are then consumed by the diffusion model code.

---

## Project Scope

We focus on three datasets:

1. **Weinreb et al. 2020** – lineage-traced hematopoiesis with clonal barcodes and scRNA-seq.
2. **Waddington-OT (Schiebinger et al. 2019)** – time-course reprogramming used in the WOT paper.
3. **K562 GWPS (optional extension)** – genome-wide Perturb-seq screen for perturbation → state / fate benchmarking.

Each dataset goes through a shared preprocessing stack  
(Scanpy-based QC → log1p → HVG selection → PCA), then is written to `data/prep/` for downstream diffusion experiments.

---

## Repo Layout

Top level:

- `env.yml` – Conda environment for MEDIT.
- `configs/`
  - `paths.yml` – Local/GCS layout and external data source URLs.
  - `params.yml` – QC / preprocessing hyperparameters.
  - `/manifest` – Manifest of files to fetch into `data/raw/`.
- `scripts/`
  - `qc_eda.py` – QC + preprocessing; writes preprocessed data to `data/prep/` and reports to `out/`.
- `tools/`
  - `bootstrap_vm.sh` – One-time VM setup (conda env, packages, etc.).
  - `download_data.sh` – GCS-aware downloader used on the VM.
  - `init_gcs.sh` – Create GCS prefixes (`data/raw`, `data/prep`, `out`) with `.keep` files.
  - `cleanup.sh` – Helpers to clean staging directories on the VM.
- `Makefile` – Runner for different tasks.
- `start.sh` – Orchestrates the full VM pipeline via `make`.
- `data/`
  - `raw/` – Raw downloaded single-cell data (from Figshare/Zenodo/etc.).
  - `prep/` – Preprocessed outputs from `qc_eda.py` (what the diffusion code consumes).
- `out/`  
  - QC plots, logs.

When using GCS, `data/raw`, `data/prep`, and `out` are mirrored under `gs://$BUCKET/`.

---

## Setup

### 1. Create the Conda environment

```bash
conda env create -f env.yml
conda activate medit
```

---

## How to Run Things

### VM / cloud usage (recommended for heavy runs)

Use **`start.sh`** as the one-button entry point:

```bash
./start.sh
```

This script:

1. **Bootstraps (or updates) the VM:**

   ```bash
   make bootstrap
   ```

2. **Ensures the local workspace directories exist on the VM:**

   ```bash
   make vm.ensure_dirs
   ```

   (Creates `~/Medit/data/raw`, `~/Medit/data/prep`, and `~/Medit/out`.)

3. **Ensures GCS prefixes exist (`data/raw`, `data/prep`, `out`):**

   ```bash
   make gcs.init
   ```

4. **Downloads raw data on the VM and uploads it to GCS:**

   ```bash
   make data.download
   ```

5. **Runs a CUDA/Torch sanity check on the VM:**

   ```bash
   make vm.cuda
   ```

6. **Runs the MEDIT pipeline (`make all`) on the VM:**

   ```bash
   make vm.run PIPELINE=all
   ```

7. **Cleans staging downloads on the VM:**

   ```bash
   make cleanup.staging
   ```

You can override project/zone/VM/bucket when calling `start.sh`:

```bash
PROJECT=medit-478122 ZONE=us-west4-a VM=medit-g2 BUCKET=medit-uml-prod-uscentral1-8e7a ./start.sh
```

---

## Status

- **Stable:** environment setup, raw data download, QC + preprocessing, and GCP/VM plumbing.
- **Next:** hook `data/prep/*.h5ad` into the diffusion modeling code (Euclidean vs manifold) so both tracks consume the same preprocessed datasets.
