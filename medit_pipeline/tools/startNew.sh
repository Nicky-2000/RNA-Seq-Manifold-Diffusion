#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-medit-478122}"
ZONE="${ZONE:-us-west4-a}"
VM="${VM:-medit-g2}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-medit}"

echo "[*] Ensuring VM has Docker + GPU support..."
gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" -- \
  "sudo apt-get update -y && \
   sudo apt-get install -y docker.io && \
   sudo usermod -aG docker \$USER || true"

echo "[*] Pulling image on VM..."
make vm.pull PROJECT="$PROJECT" VM="$VM" ZONE="$ZONE" REGION="$REGION" REPO="$REPO"

echo "[*] Syncing local data → GCS..."
make sync.up

echo "[*] Ready. Run pipeline on VM via:"
echo "    make vm.run SCRIPT=scripts/dim_red.py ARGS='--params configs/params.yml ...'"
