#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
DO_ALL=0
SWAP_GB=""
RED="$(tput setaf 1 || true)"; GRN="$(tput setaf 2 || true)"; YLW="$(tput setaf 3 || true)"; DIM="$(tput dim || true)"; RST="$(tput sgr0 || true)"

run() {
  if [[ $DRY_RUN -eq 1 ]]; then echo "${DIM}[dry-run]$RST $*"; else echo "${GRN}→$RST $*"; eval "$@"; fi
}

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--all] [--swap <GB>]

  --dry-run     Show actions without executing
  --all         Do deeper cleanup (logs, caches, docker system prune)
  --swap <GB>   Create/enable a swapfile of size <GB> (e.g., --swap 8)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --all) DO_ALL=1; shift ;;
    --swap) SWAP_GB="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "$RED[error]$RST Unknown arg: $1"; usage; exit 1 ;;
  esac
done

header(){ echo; echo "${YLW}== $* ==$RST"; }

human_mem() { free -h || true; }
human_disk() { df -hT | awk 'NR==1 || /(^\/dev|Filesystem)/' || true; }

# ----- Prep: snap baseline -----
header "Baseline (RAM + Disk)"
human_mem
human_disk

# ----- RAM cleanup -----
header "RAM cleanup"
# 1) Kill obvious hogs interactively (skip if dry-run)
echo "${DIM}Top memory processes:${RST}"
ps aux --sort=-%mem | awk 'NR==1, NR<=11 {print}'
echo

# 2) Stop stray Jupyter servers (non-interactive)
if command -v jupyter >/dev/null 2>&1; then
  JLIST="$(jupyter notebook list 2>/dev/null || true)"
  if [[ -n "$JLIST" ]]; then
    echo "${DIM}Jupyter notebooks running:${RST}"
    echo "$JLIST"
  fi
fi

# 3) Drop filesystem caches (safe; requires sudo)
if [[ $DRY_RUN -eq 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    echo "${DIM}Dropping FS caches (pagecache,dentries,inodes)${RST}"
    sudo sync || true
    echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true
  fi
fi

# 4) GPU VRAM (if applicable)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "${DIM}nvidia-smi:${RST}"
  nvidia-smi || true
fi

# ----- Optional: create swap -----
if [[ -n "$SWAP_GB" ]]; then
  header "Swap setup (${SWAP_GB}G)"
  if ! swapon --show | grep -q '/swapfile'; then
    run "sudo fallocate -l ${SWAP_GB}G /swapfile"
    run "sudo chmod 600 /swapfile"
    run "sudo mkswap /swapfile"
    run "sudo swapon /swapfile"
    run "grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null"
  else
    echo "${DIM}Swapfile already active:${RST}"
    swapon --show || true
  fi
fi

# ----- Disk cleanup: quick wins -----
header "Disk: quick wins"

# 1) Conda caches
if command -v conda >/dev/null 2>&1; then
  run "conda clean -a -y"
  if command -v mamba >/dev/null 2>&1; then
    run "mamba clean --all -y"
  fi
fi

# 2) Pip cache
if command -v pip >/dev/null 2>&1; then
  run "pip cache purge"
fi

# 3) APT cache (Debian/Ubuntu)
if command -v apt-get >/dev/null 2>&1; then
  run "sudo apt-get clean"
  run "sudo apt-get autoremove -y"
fi

# 4) User caches & temp
run "rm -rf \$HOME/.cache/*"
run "sudo rm -rf /tmp/* /var/tmp/*"

# 5) Notebook checkpoints
run "find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} +"

# ----- Disk cleanup: deeper (optional) -----
if [[ $DO_ALL -eq 1 ]]; then
  header "Disk: deeper cleanup (--all)"
  # Journald logs
  if command -v journalctl >/dev/null 2>&1; then
    run "sudo journalctl --vacuum-time=7d"
  fi
  # Truncate large logs (safe-ish)
  run "sudo bash -c 'for f in /var/log/syslog /var/log/kern.log; do [ -f \$f ] && : > \$f; done'"

  # Docker system prune
  if command -v docker >/dev/null 2>&1; then
    run "docker system prune -a --volumes -f"
  fi
fi

# ----- Largest directories & files report -----
header "Largest directories under \$HOME"
du -xh --max-depth=1 "$HOME" 2>/dev/null | sort -h | tail -n 20 || true

header "Largest items under repo (data/ and out/ if present)"
for d in data out; do
  if [[ -d $d ]]; then
    echo "${DIM}# $d${RST}"
    du -xh --max-depth=1 "$d" 2>/dev/null | sort -h | tail -n 20 || true
  fi
done

header "Top 20 large files (≥ 500M, top-level filesystems only)"
sudo find / -xdev -type f -size +500M 2>/dev/null \
 | xargs -r ls -lh \
 | sort -k5 -h \
 | tail -n 20 || true

# ----- Final stats -----
header "After (RAM + Disk)"
human_mem
human_disk

echo
echo "${GRN}Done.$RST If you need to reclaim more, consider moving big artifacts to GCS and symlinking back."
echo "Example:"
echo "  gsutil -m rsync -r data gs://YOUR_BUCKET/data"
echo "  rm -rf data/huge_dir && ln -s /path/mounted/from/gcs huge_dir"