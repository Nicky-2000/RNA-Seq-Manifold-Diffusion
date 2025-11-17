#!/usr/bin/env bash
set -euo pipefail

# Download from GCS to local.
# Works for a single file (cp) or a folder/prefix (rsync).
#
# Usage:
#   get_gcs.sh gs://BUCKET/path/to/file.h5ad  /local/dir [--only-missing] [--dry-run]
#   get_gcs.sh gs://BUCKET/path/prefix/       /local/dir --sync [--include "*.h5ad"] [--exclude "*.tmp"] [--dry-run]
#
# Notes:
# - Add trailing slash on the GCS side to treat it as a "folder/prefix".
# - --only-missing uses `cp -n` to avoid overwriting existing files.
# - --sync uses `rsync -r` (mirrors remote->local; does not delete local extra files).
# - `gsutil` does integrity checks by default.

SRC="${1:?need GCS src (e.g., gs://bucket/path or gs://bucket/prefix/) }"
DEST="${2:?need local dest dir (e.g., ./data/raw)}"
shift 2 || {
  }

ONLY_MISSING=0
SYNC=0
DRY_RUN=0
INCLUDE=""
EXCLUDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-missing) ONLY_MISSING=1; shift;;
    --sync)         SYNC=1; shift;;
    --dry-run)      DRY_RUN=1; shift;;
    --include)      INCLUDE="$2"; shift 2;;
    --exclude)      EXCLUDE="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

command -v gsutil >/dev/null 2>&1 || { echo "missing gsutil" >&2; exit 1; }

mkdir -p "$DEST"

# Common opts: parallel uploads/downloads, big-file composite
GSOPTS=(-m -o "GSUtil:parallel_composite_upload_threshold=150M")
[[ $DRY_RUN -eq 1 ]] && GSOPTS+=(-n)  # don't overwrite; acts like dry-ish for cp/rsync

# If SRC ends with '/', treat as prefix sync
if [[ "$SYNC" -eq 1 || "$SRC" == */ ]]; then
  # Build rsync args
  RSARGS=(-r)
  [[ -n "$INCLUDE" ]] && RSARGS+=( -x "(?i)^((?=.*$INCLUDE).*)$" )
  [[ -n "$EXCLUDE" ]] && RSARGS+=( -x "^(?=.*$EXCLUDE).*" )
  # gsutil rsync doesn't have rich include/exclude; the -x regex excludes.
  # If INCLUDE given, emulate include by excluding everything not matching.
  if [[ -n "$INCLUDE" ]]; then
    RSARGS=(-r -x "^(?!.*${INCLUDE//\./\\.}).*$")  # include pattern, exclude others
  fi
  if [[ -n "$EXCLUDE" && -z "$INCLUDE" ]]; then
    RSARGS=(-r -x "${EXCLUDE//\./\\.}$")
  fi

  echo "==> rsync: $SRC -> $DEST"
  gsutil "${GSOPTS[@]}" rsync "${RSARGS[@]}" "$SRC" "$DEST"
else
  # Single file copy
  CPCMD=(cp)
  [[ $ONLY_MISSING -eq 1 || $DRY_RUN -eq 1 ]] && CPCMD+=(-n)
  echo "==> cp: $SRC -> $DEST/"
  gsutil "${GSOPTS[@]}" "${CPCMD[@]}" "$SRC" "$DEST/"
fi

echo "[ok] fetched to $DEST"
