#!/usr/bin/env bash
set -euo pipefail

log() { printf "\n[%s] %s\n" "$(date +'%F %T')" "$*"; }

# -------- 0) GPU sanity (non-fatal) --------
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  # might require sudo path on some images
  sudo nvidia-smi 2>/dev/null || true
fi

# -------- 1) Miniconda install (idempotent) --------
if [ ! -d "$HOME/miniconda3" ]; then
  log "Installing Miniconda..."
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$HOME/miniconda.sh"
  bash "$HOME/miniconda.sh" -b -p "$HOME/miniconda3"
  rm -f "$HOME/miniconda.sh"
  "$HOME/miniconda3/bin/conda" init bash || true
fi

# Load conda shell funcs for THIS process
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi

# -------- 2) Configure conda + mamba --------
log "Preparing conda (mamba + strict channel priority)…"

# Only use the channels we actually care about
conda config --remove-key channels || true
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels conda-forge
conda config --set channel_priority strict

# Ensure mamba is available in base
conda install -y -n base -c conda-forge mamba || true

# Optional but recommended: avoid host CUDA 12.8 fighting pytorch-cuda=12.1
export CONDA_OVERRIDE_CUDA=12.1

# -------- 3) Create/Update env from lockfile (env name: venv) --------
ENV_NAME=venv
LOCKFILE="configs/env.lock.yml"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Updating existing env '$ENV_NAME' from $LOCKFILE…"
  mamba env update \
    -n "$ENV_NAME" \
    -f "$LOCKFILE" \
    --prune \
    --override-channels \
    -c pytorch -c nvidia -c conda-forge
else
  log "Creating env '$ENV_NAME' from $LOCKFILE…"
  mamba env create \
    -n "$ENV_NAME" \
    -f "$LOCKFILE" \
    --override-channels \
    -c pytorch -c nvidia -c conda-forge
fi

# -------- 3) Sanity imports via conda-run (no activation assumptions) --------
log "Verifying core packages in env 'venv'…"
conda run -n venv python - <<'PY'
import sys, importlib
mods = ["scanpy","numpy","scipy","pandas"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"[OK] {m} {getattr(mod,'__version__','?')}")
print("Python:", sys.version)
PY

# -------- 4) Optional CUDA/Torch probe (non-fatal) --------
log "Torch/CUDA check (non-fatal)…"
conda run -n venv python - <<'PY'
try:
    import torch
    print("torch:", getattr(torch,"__version__","?"),
          "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        print("device_0:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch check skipped/failed:", e)
PY

# -------- 5) System deps for downloads (idempotent) --------
if ! command -v aria2c >/dev/null 2>&1; then
  log "Installing aria2 for robust downloads…"
  sudo apt-get update -y && sudo apt-get install -y aria2
fi

log "✅ Bootstrap complete."