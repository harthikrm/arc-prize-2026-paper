#!/usr/bin/env bash
# 00_gcp_quota_check.sh — run on YOUR LAPTOP (needs gcloud CLI, logged in).
# Enables Compute Engine and prints the GPU quotas that gate everything else.
# New projects usually have 0 GPU quota -> request increases today (approval can take 1-3 days).
#
#   PROJECT=my-arc-project REGION=us-central1 bash 00_gcp_quota_check.sh
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT=<your gcp project id>}"
REGION="${REGION:-us-central1}"

gcloud config set project "$PROJECT" >/dev/null
echo ">> enabling compute API (idempotent)"
gcloud services enable compute.googleapis.com

TMP=$(mktemp -d)
gcloud compute project-info describe --format=json > "$TMP/proj.json"
gcloud compute regions describe "$REGION" --format=json > "$TMP/region.json"
python3 - "$TMP" "$REGION" <<'PY'
import json, sys
d, region = sys.argv[1], sys.argv[2]
print("\n>> GLOBAL quota (need >= 4 for a 4xL4 box, >= 8 for an 8xH100 node):")
for x in json.load(open(d + "/proj.json"))["quotas"]:
    if x["metric"] == "GPUS_ALL_REGIONS":
        print("   %-40s limit=%d usage=%d" % (x["metric"], x["limit"], x["usage"]))
print("\n>> REGION %s GPU quotas (on-demand and spot/preemptible):" % region)
for x in json.load(open(d + "/region.json"))["quotas"]:
    if any(w in x["metric"] for w in ("NVIDIA_L4", "NVIDIA_H100", "NVIDIA_A100")):
        print("   %-40s limit=%d usage=%d" % (x["metric"], x["limit"], x["usage"]))
PY
rm -rf "$TMP"

cat <<EOF

>> What to request (Console -> IAM & Admin -> Quotas & System Limits, filter by metric name):
   GPUS_ALL_REGIONS                         -> 12   (global cap, needed for anything)
   NVIDIA_L4_GPUS / PREEMPTIBLE_NVIDIA_L4_GPUS ($REGION)   -> 4   (rehearsal box = exact Kaggle hardware)
   NVIDIA_H100_GPUS or PREEMPTIBLE_NVIDIA_H100_GPUS ($REGION) -> 8 (training node, a3-highgpu-8g)
   Justification text that tends to get approved:
   "Research/competition workload (ARC Prize 2026): LLM fine-tuning and batch inference, short-lived VMs."
   Link: https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT
EOF
