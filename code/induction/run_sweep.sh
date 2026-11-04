#!/usr/bin/env bash
# run_sweep.sh — one command on a fresh GPU pod: install, then run the model sweep in the background.
#
#   bash run_sweep.sh                      # all configs in sweep_configs.json, 80 dev tasks
#   bash run_sweep.sh --only qwen36_think  # one config
#   bash run_sweep.sh --limit 10           # quick smoke test (10 tasks per config)
#   bash run_sweep.sh --tp 4               # on a 4xL4 box
#
# Follow progress:  tail -f runs/sweep1.log        per-task lines: tail -f runs/sweep1/<label>/solver.log
# Results:          runs/sweep1/summary.md  summary.png  <label>/submission.json  <label>/transcripts/
set -euo pipefail
cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-/workspace/hf}"            # model cache on the persistent volume
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME" runs

if [ ! -x /workspace/venv/bin/python ]; then
  echo ">> creating venv and installing vLLM (5-10 min, once per volume)"
  python3 -m venv /workspace/venv
  /workspace/venv/bin/pip install -q -U pip
  /workspace/venv/bin/pip install -q -U "vllm>=0.19" scipy numpy matplotlib "huggingface_hub>=0.30"
fi
source /workspace/venv/bin/activate
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN is not set: the Nemotron config (gated repo) will fail; the others will run."
fi

OUT=runs/sweep1
NVARC=()
[ -f nvarc_eval_submission.json ] && NVARC=(--nvarc nvarc_eval_submission.json)   # fills "only ours vs NVARC"
nohup python sweep.py "${NVARC[@]}" \
  --challenges ../data/arc-agi_evaluation_challenges.json \
  --solutions  ../data/arc-agi_evaluation_solutions.json \
  --split dev --out "$OUT" "$@" > "$OUT.log" 2>&1 &
echo ">> sweep started (pid $!). Follow with:  tail -f $OUT.log"
