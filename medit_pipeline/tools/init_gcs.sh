#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:?set BUCKET}"

# Create core prefixes in the GCS bucket by dropping a .keep file.
for p in data/raw data/prep out; do
  gsutil -m cp -n /etc/hosts "gs://${BUCKET}/${p}/.keep" >/dev/null 2>&1 || true
done

echo "GCS prefixes ensured in gs://${BUCKET}/{data/raw,data/prep,out}"
