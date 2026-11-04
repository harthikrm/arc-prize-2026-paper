#!/usr/bin/env bash
# 02_vm_bootstrap.sh — run ON THE VM, once. Makes the VM behave like a Kaggle notebook:
#   * NVIDIA driver + docker + nvidia-container-toolkit
#   * the official Kaggle GPU docker image (same Python/torch/CUDA stack as the scoring environment)
#   * Kaggle credentials + CLI + kagglehub
#   * competition data mounted at /kaggle/input/arc-prize-2026-arc-agi-2 (the path notebooks expect)
#
#   bash ~/infra/02_vm_bootstrap.sh
set -euo pipefail
COMP=arc-prize-2026-arc-agi-2
KAGGLE_IMAGE="${KAGGLE_IMAGE:-gcr.io/kaggle-gpu-images/python:latest}"

echo ">> 1/6 GPUs"
if ! nvidia-smi >/dev/null 2>&1; then
  echo "   driver not ready yet (DLVM installs it on first boot, can take ~5 min). Retry in a few minutes."; exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l); [[ "$NGPU" -eq 4 ]] || echo "   WARNING: expected 4 GPUs, found $NGPU"

echo ">> 2/6 docker + nvidia container toolkit"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo usermod -aG docker "$USER" || true

echo ">> 3/6 pulling Kaggle GPU image (large: expect 10-20 min the first time)"
sudo docker pull "$KAGGLE_IMAGE"
sudo docker run --rm --gpus all "$KAGGLE_IMAGE" python -c \
  "import torch; print('   torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

echo ">> 4/6 kaggle credentials + tools (host side)"
mkdir -p ~/.kaggle
[[ -f ~/kaggle.json ]] && mv ~/kaggle.json ~/.kaggle/kaggle.json
[[ -f ~/.kaggle/kaggle.json ]] || { echo "   put your API token at ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token)"; exit 1; }
chmod 600 ~/.kaggle/kaggle.json
python3 -m pip install -q --user --upgrade kaggle kagglehub
export PATH="$HOME/.local/bin:$PATH"

echo ">> 5/6 competition data -> /kaggle/input/$COMP"
sudo mkdir -p /kaggle/input /kaggle/runs && sudo chown -R "$USER" /kaggle
if [[ ! -f /kaggle/input/$COMP/arc-agi_evaluation_challenges.json ]]; then
  echo "   (you must have clicked 'Join Competition' + accepted rules on kaggle.com, or this 403s)"
  kaggle competitions download -c "$COMP" -p /tmp/$COMP
  mkdir -p /kaggle/input/$COMP && unzip -o -q /tmp/$COMP/*.zip -d /kaggle/input/$COMP
fi
ls -la /kaggle/input/$COMP

echo ">> 6/6 sanity: harness on the data"
python3 ~/infra/arc_eval.py --check-placeholder \
  /kaggle/input/$COMP/arc-agi_test_challenges.json \
  /kaggle/input/$COMP/arc-agi_training_challenges.json \
  /kaggle/input/$COMP/arc-agi_evaluation_challenges.json

echo; echo "Bootstrap done. Log out and back in once (docker group), then run 03_fetch_notebook.py."
