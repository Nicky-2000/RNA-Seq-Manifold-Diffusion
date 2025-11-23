#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 PROJECT INSTANCE_NAME MACHINE_TYPE IMAGE_FAMILY [ACCEL_TYPE]"
  echo "  PROJECT       GCP project id (e.g. medit-478122)"
  echo "  INSTANCE_NAME Name for the VM (e.g. medit-a2)"
  echo "  MACHINE_TYPE  GCE machine type (e.g. a2-highgpu-1g or g2-standard-8)"
  echo "  IMAGE_FAMILY  DL VM image family"
  echo "  ACCEL_TYPE    Optional. e.g. nvidia-l4; use 'none' for A2/A100."
  exit 1
fi

PROJECT="$1"
INSTANCE_NAME="$2"
MACHINE_TYPE="$3"
IMAGE_FAM="$4"
ACCEL_TYPE="${5:-none}"
  
IMAGE_PROJECT="deeplearning-platform-release"

# If MACHINE_TYPE is A2, restrict to A2-supported US zones
if [[ "$MACHINE_TYPE" == a2-* ]]; then
  ZONES="${ZONES:-"\
us-central1-a us-central1-b us-central1-c us-central1-f \
us-east1-b    us-east1-c    us-east1-d \
us-east4-a    us-east4-b    us-east4-c"}"
else
  # Fallback: all US zones for other types (you can expand if you like)
  ZONES="${ZONES:-"\
us-west1-a us-west1-b us-west1-c \
us-west2-a us-west2-b us-west2-c \
us-west3-a us-west3-b us-west3-c \
us-west4-a us-west4-b us-west4-c \
us-central1-a us-central1-b us-central1-c us-central1-f \
us-east1-b us-east1-c us-east1-d \
us-east4-a us-east4-b us-east4-c \
us-south1-a us-south1-b us-south1-c"}"
fi

BOOT_DISK_SIZE="200GB"

for Z in $ZONES; do
  echo ">>> Trying $Z"

  CMD=(
    gcloud compute instances create "$INSTANCE_NAME"
      --project="$PROJECT"
      --zone="$Z"
      --machine-type="$MACHINE_TYPE"
      --maintenance-policy=TERMINATE
      --image-project="$IMAGE_PROJECT"
      --image-family="$IMAGE_FAM"
      --boot-disk-type=pd-ssd
      --boot-disk-size="$BOOT_DISK_SIZE"
      --no-address
      --scopes=cloud-platform
  )

  if [ "$ACCEL_TYPE" != "none" ]; then
    CMD+=(--accelerator="count=1,type=$ACCEL_TYPE")
  fi

  if "${CMD[@]}"; then
    echo "✅ Landed in $Z"
    exit 0
  else
    echo "✗ $Z failed (no capacity / unsupported combo / org policy)"
  fi
done

echo "❌ Failed to create instance in any candidate zone" >&2
exit 1
