#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
# Default to HTTPS URL so you can just paste a PAT/token if GitHub prompts.
# You can override on the command line:
#   REPO_URL=git@github.com:USER/REPO.git ./setup_git_repo.sh
REPO_URL="${REPO_URL:-https://github.com/Nicky-2000/RNA-Seq-Manifold-Diffusion.git}"

# This MUST match what your Makefile/start.sh expect on the VM
TARGET_DIR="${TARGET_DIR:-$HOME/MEDIT}"
BRANCH="${BRANCH:-main}"

echo "[*] Repo URL    : $REPO_URL"
echo "[*] Target dir  : $TARGET_DIR"
echo "[*] Branch      : $BRANCH"
echo

# ---- install git if needed ----
if ! command -v git >/dev/null 2>&1; then
  echo "[*] git not found, installing..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y git
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y git
  else
    echo "[-] Could not detect package manager (no apt-get or yum)."
    echo "    Please install git manually and re-run this script."
    exit 1
  fi
else
  echo "[ok] git already installed: $(git --version)"
fi

echo

# ---- clone or update repo ----
if [ ! -d "$TARGET_DIR/.git" ]; then
  echo "[*] Cloning repo into $TARGET_DIR ..."
  git clone "$REPO_URL" "$TARGET_DIR"
  echo "[ok] Clone complete."
else
  echo "[*] Repo already exists at $TARGET_DIR, updating..."
  cd "$TARGET_DIR"
  git remote -v
  git fetch origin
  git checkout "$BRANCH"
  git pull --rebase origin "$BRANCH"
  echo "[ok] Repo updated."
fi

echo
echo "[*] Done. Repo is at: $TARGET_DIR"
echo "    Next steps (on VM):"
echo "      cd $TARGET_DIR/medit_pipeline"
echo "      conda env create -f env.yml  # first time only"
echo "      conda activate medit"
echo "      ./start.sh"
