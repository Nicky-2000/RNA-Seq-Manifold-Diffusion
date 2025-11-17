PROJECT=medit-478122
IMAGE_FAM=pytorch-2-7-cu128-ubuntu-2204-nvidia-570

for Z in us-west4-b us-west4-a us-west4-c; do
  echo ">>> Trying $Z"
  gcloud compute instances create medit-g2 \
    --project=$PROJECT \
    --zone=$Z \
    --machine-type=g2-standard-8 \
    --accelerator=count=1,type=nvidia-l4 \
    --maintenance-policy=TERMINATE \
    --image-project=deeplearning-platform-release \
    --image-family=$IMAGE_FAM \
    --boot-disk-type=pd-ssd --boot-disk-size=200GB \
    --scopes=cloud-platform \
  && { echo "✅ Landed in $Z"; break; } \
  || echo "✗ $Z had no capacity"
done
