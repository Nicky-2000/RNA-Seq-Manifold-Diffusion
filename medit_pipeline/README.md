# MEDIT (Manifold-Embedded Diffusion for Inferred Trajectories) – Single-Cell Preprocessing Pipeline

MEDIT is the reproducible preprocessing and infrastructure layer for our UML course project.

Goal: take raw single-cell RNA-seq data (e.g. Weinreb in-vitro hematopoiesis), run a shared QC + preprocessing stack, and produce clean `.h5ad` files in `data/prep/` that the main diffusion repo can consume.

Pipeline:

    Raw .h5ad → QC / filtering → HVGs → PCA → data/prep → Diffusion models

---

## Quickstart

### 1. Clone the Repository (locally)

From your laptop (this repo is typically embedded inside the main diffusion project):

    git clone <MAIN_REPO_URL> MEDIT
    cd MEDIT/medit_pipeline

Adjust the path if your folder layout is slightly different.

---

## GCP / gcloud Setup (Columbia Account)

You must use your **Columbia-affiliated Google account** (e.g. `uni@columbia.edu`) that has access to the `medit-478122` project and its billing account.

1. Install the gcloud CLI (once, on your laptop):

    https://cloud.google.com/sdk/docs/install

2. Log in with your Columbia account:

    gcloud auth login

   In the browser window that opens, pick your `@columbia.edu` account.

3. (Recommended) Set up Application Default Credentials:

    gcloud auth application-default login

   This lets code on your laptop or VM pick up credentials automatically.

4. Set the active GCP project:

    gcloud config set project medit-478122
    gcloud config list project

   You should see:

    project = medit-478122

5. SSH into the VM instance:

    gcloud compute ssh medit-g2 --project=medit-478122 --zone=us-west4-a

---

## Git / Repo Setup on the VM (first time only)

These steps are run **inside the VM shell**, after the SSH command above.

1. Install git (if it isn't already installed):

    sudo apt-get update -y
    sudo apt-get install -y git

2. Clone the main repo into `~/MEDIT`:

    cd ~
    git clone https://github.com/Nicky-2000/RNA-Seq-Manifold-Diffusion.git MEDIT
3. Enter the MEDIT pipeline folder on the VM:

    cd ~/MEDIT/medit_pipeline

From here you can set up the conda environment and run the pipeline as below.
---

## Run the Preprocessing Pipeline on the VM

The VM is where downloads + heavy QC runs happen. The entry point is `start.sh`. This script will also setup the conda env.

From inside `~/MEDIT/medit_pipeline` on the VM:

    ./tools/start.sh

This script will:

1. Bootstrap or update the VM

       make bootstrap

   Uploads and runs `tools/bootstrap_vm.sh` to create the conda env, install Python deps, etc.

2. Ensure local workspace dirs on the VM

       make vm.ensure_dirs

   Creates:

       ~/MEDIT/data/raw
       ~/MEDIT/data/prep
       ~/MEDIT/out

3. Ensure GCS prefixes exist

       make gcs.init

   Ensures the bucket has:

       gs://$BUCKET/data/raw/
       gs://$BUCKET/data/prep/
       gs://$BUCKET/out/

4. Download raw data on the VM and upload to GCS

       make data.download

   Uses `configs/manifest/weinreb_manifest.csv` to fetch files (e.g. `stateFate_inVitro.h5ad`) and place them under `data/raw/`.

5. Run a CUDA / Torch sanity check

       make vm.cuda

6. Run the MEDIT preprocessing pipeline on the VM

       make vm.run PIPELINE=all

   Internally this runs:

       make qc

   which calls:

       python scripts/qc_eda.py \
         --params configs/params.yml \
         --out data/prep \
         --adata data/raw/stateFate_inVitro.h5ad

   (Update the `--adata` path if your raw file lives in a subfolder, e.g. `data/raw/weinreb/stateFate_inVitro.h5ad`.)

7. Clean up staging downloads on the VM

       make cleanup.staging

You can override project/zone/VM/bucket when calling `start.sh`:

    PROJECT=medit-478122 \
    ZONE=us-west4-a \
    VM=medit-g2 \
    BUCKET=medit-uml-prod-uscentral1-8e7a \
    ./start.sh

---

## Local Usage (Optional)

If you already have the raw `.h5ad` locally (for debugging or small runs):

    cd medit_pipeline
    conda activate medit

    # Put stateFate_inVitro.h5ad into ../data/raw/ first
    make qc          # or: make all

This will:

- Read the raw file from `../data/raw/…`
- Write preprocessed AnnData to `../data/prep/`
- Write QC plots / logs to `../out/`

Your teammate’s diffusion code should then read from `data/prep/*.h5ad`.

---

## 📁 Project Structure (MEDIT Layer)

    medit_pipeline/
      README.md
      env.yml
      Makefile
      start.sh
      configs/
        params.yml              # QC / preprocessing hyperparameters
        download_manifest.csv   # Raw data download manifest (URL, dst_relpath, sha256)
      scripts/
        qc_eda.py               # Scanpy-based QC + preprocessing
      tools/
        bootstrap_vm.sh         # VM bootstrap (conda env, packages)
        download_data.sh        # GCS-aware downloader (used by data.download)
        init_gcs.sh             # Create data/raw, data/prep, out prefixes in GCS
        cleanup.sh              # Remove staging / scratch data on VM

Shared data + outputs at repo root:

    data/
      raw/                      # Raw .h5ad (Weinreb, WOT, K562, …)
      prep/                     # Preprocessed .h5ad used by diffusion models
    out/
      ...                       # QC plots, logs, small manifests

---

## Status

- Stable: gcloud + project config, VM bootstrap, git + repo setup, raw data download, QC + preprocessing, and GCS mirroring.
- Next: hook `data/prep/*.h5ad` into the main diffusion codebase (Euclidean vs manifold), so both tracks share the same preprocessed inputs.
