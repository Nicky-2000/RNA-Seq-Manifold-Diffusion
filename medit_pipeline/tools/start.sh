#!/usr/bin/env bash
set -euo pipefail

# ---- config you can edit or export beforehand ----
PROJECT="${PROJECT:-medit-478122}"
ZONE="${ZONE:-us-west4-a}"
VM="${VM:-medit-g2}"
BUCKET="${BUCKET:-medit-uml-prod-uscentral1-8e7a}"

export PROJECT ZONE VM BUCKET

echo "[*] Bootstrapping MEDIT VM…"
make bootstrap PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] Ensuring local workspace dirs on VM (data/raw, data/prep, out)…"
make vm.ensure_dirs PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM"

echo "[*] Ensuring GCS prefixes exist (data/raw, data/prep, out)…"
make gcs.init PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] Downloading raw data to GCS from VM…"
make data.download PROJECT="$PROJECT" ZONE="$ZONE" VM="$VM" BUCKET="$BUCKET"

echo "[*] Done."
