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

# -------- 2) Accept Anaconda ToS once (safe if not needed) --------
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    || true

# -------- 3) Create/Update env from configs/env.yml (env name: venv) --------
log "Preparing conda (mamba + strict channel priority)…"
conda config --set channel_priority strict || true
conda install -y -n base -c conda-forge mamba || true

ENV_FILE="$HOME/MEDIT/medit_pipeline/configs/env.yml"
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found" >&2
  exit 1
fi
# normalize CRLF just in case
sed -i 's/\r$//' "$ENV_FILE"

if conda env list | awk '{print $1}' | grep -qx venv; then
  log "Env 'venv' exists → updating with --prune"
  (command -v mamba >/dev/null && mamba env update -n venv -f "$ENV_FILE" --prune) || \
  conda env update -n venv -f "$ENV_FILE" --prune
else
  log "Creating env 'venv' from $ENV_FILE"
  (command -v mamba >/dev/null && mamba env create -n venv -f "$ENV_FILE") || \
  conda env create -n venv -f "$ENV_FILE"
fi

# -------- 5) Sanity imports via conda-run (no activation assumptions) --------
log "Verifying core packages in env 'venv'…"
conda run -n venv python - <<'PY'
import sys, importlib
mods = ["scanpy","numpy","scipy","pandas"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"[OK] {m} {getattr(mod,'__version__','?')}")
print("Python:", sys.version)
PY

# -------- 6) Optional CUDA/Torch probe (non-fatal) --------
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

# -------- 7) System deps for downloads (idempotent) --------
if ! command -v aria2c >/dev/null 2>&1; then
  log "Installing aria2 for robust downloads…"
  sudo apt-get update -y && sudo apt-get install -y aria2
fi

log "✅ Bootstrap complete."
