#!/usr/bin/env bash
# 01_create_l4_vm.sh — run on YOUR LAPTOP. Creates the "Kaggle twin": g2-standard-48 = 4x NVIDIA L4 (96 GB),
# the same accelerator the ARC-AGI-2 scoring run uses.
#
#   PROJECT=my-arc-project bash 01_create_l4_vm.sh            # on-demand (reliable 12h rehearsals)
#   PROJECT=my-arc-project SPOT=1 bash 01_create_l4_vm.sh     # spot (~60-70% cheaper, can be preempted)
#   PROJECT=my-arc-project SPOT=1 NGPU=2 bash 01_create_l4_vm.sh   # 2xL4 half-scale twin (fits a 3-GPU quota)
#
# Cost control: the VM bills while RUNNING. Always stop it when idle:
#   gcloud compute instances stop arc-l4x4 --zone <zone>
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT=<your gcp project id>}"
NAME="${NAME:-arc-l4x4}"
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-west1-a us-west1-b europe-west4-a}"
DISK_GB="${DISK_GB:-400}"          # Kaggle GPU docker image is huge; models + caches need room
SPOT="${SPOT:-0}"
NGPU="${NGPU:-4}"                  # 4 -> g2-standard-48 (exact twin); 2 -> g2-standard-24 (half-scale twin)

gcloud config set project "$PROJECT" >/dev/null

# Pick the newest Deep Learning VM image with CUDA 12 on Ubuntu (driver preinstalled via metadata flag).
IMAGE_FAMILY="${IMAGE_FAMILY:-$(gcloud compute images list --project=deeplearning-platform-release \
   --filter='family~^common-cu12 AND family~ubuntu' --format='value(family)' | sort -u | tail -1)}"
if [[ -z "$IMAGE_FAMILY" ]]; then
  echo "Could not auto-detect a Deep Learning VM image family. List them with:"
  echo "  gcloud compute images list --project=deeplearning-platform-release --format='value(family)' | sort -u"
  echo "and re-run with IMAGE_FAMILY=<family>"; exit 1
fi
echo ">> image family: $IMAGE_FAMILY"

SPOT_FLAGS=()
if [[ "$SPOT" == "1" ]]; then
  SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
  echo ">> SPOT instance requested"
fi

for Z in $ZONES; do
  echo ">> trying zone $Z ..."
  if gcloud compute instances create "$NAME" --zone="$Z" \
      --machine-type="$([[ $NGPU == 2 ]] && echo g2-standard-24 || echo g2-standard-48)" \
      --accelerator=type=nvidia-l4,count="$NGPU" \
      --maintenance-policy=TERMINATE \
      --image-project=deeplearning-platform-release --image-family="$IMAGE_FAMILY" \
      --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
      --metadata=install-nvidia-driver=True \
      --scopes=cloud-platform \
      "${SPOT_FLAGS[@]}"; then
    echo; echo ">> created $NAME in $Z"
    echo "   copy the kit:   gcloud compute scp --zone $Z --recurse ./infra $NAME:~/"
    echo "   copy creds:     gcloud compute scp --zone $Z ~/.kaggle/kaggle.json $NAME:~/"
    echo "   ssh:            gcloud compute ssh --zone $Z $NAME"
    echo "   STOP WHEN IDLE: gcloud compute instances stop $NAME --zone $Z"
    exit 0
  fi
  echo "   (no capacity/quota in $Z, trying next)"
done
echo "!! Could not create the VM in any zone. Check quota (00_gcp_quota_check.sh) or try SPOT=1 / other ZONES."
exit 1
