#!/usr/bin/env bash
# 04_rehearse.sh — run ON THE VM. Executes a Kaggle notebook exactly like the scoring rerun would:
#   Kaggle GPU image, 4 GPUs, /kaggle/input read-only, NO network, 12h wall clock,
# but with the hidden test file swapped for a set whose answers you have, then scores it.
#
#   bash ~/infra/04_rehearse.sh <notebook-slug> eval         # 120 eval tasks (honest local proxy)
#   bash ~/infra/04_rehearse.sh <notebook-slug> eval 20      # first 20 eval tasks only (quick check)
#   bash ~/infra/04_rehearse.sh <notebook-slug> placeholder  # what Kaggle's "Save & Run All" commit sees
#
#   GPUS=2 bash ~/infra/04_rehearse.sh <slug> eval 60      # half-scale twin on a 2xL4 box:
#        same per-GPU workload as Kaggle's 4 GPUs x 120 tasks; patches a COPY of the notebook
#        (nprocs=4 -> 2, range(4) -> range(2)) and exposes only GPUs 0..GPUS-1 to the container.
#
# Output: /kaggle/runs/<slug>_<mode>_<timestamp>/{submission.json, executed.ipynb, run.log, score.txt, rows.csv}
set -euo pipefail
SLUG="${1:?usage: 04_rehearse.sh <notebook-slug> eval|placeholder [n_tasks]}"
MODE="${2:-eval}"
NTASKS="${3:-0}"
COMP=arc-prize-2026-arc-agi-2
CDIR=/kaggle/input/$COMP
KAGGLE_IMAGE="${KAGGLE_IMAGE:-gcr.io/kaggle-gpu-images/python:latest}"
WALL="${WALL:-12h}"
GPUS="${GPUS:-4}"

NB_DIR=/kaggle/notebooks/$SLUG
NB=$(ls "$NB_DIR"/*.ipynb 2>/dev/null | head -1 || true)
[[ -n "$NB" ]] || { echo "no .ipynb in $NB_DIR — run 03_fetch_notebook.py first"; exit 1; }
NB_MOUNT_DIR="$NB_DIR"
if [[ "$GPUS" != "4" ]]; then
  NB_MOUNT_DIR=$(mktemp -d)
  python3 - "$NB" "$NB_MOUNT_DIR/$(basename "$NB")" "$GPUS" <<'PY'
import json, re, sys
src, dst, g = sys.argv[1], sys.argv[2], sys.argv[3]
nb = json.load(open(src)); n = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    text = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new = re.sub(r"nprocs\s*=\s*4\b", f"nprocs={g}", text)
    new = re.sub(r"range\(\s*4\s*\)", f"range({g})", new)
    if new != text:
        n += 1
        c["source"] = new
json.dump(nb, open(dst, "w"))
print(f">> GPUS={g}: patched {n} cell(s) (nprocs/range). CHECK the notebook has no other hardcoded 4-GPU assumptions.")
PY
fi

SUFFIX=""; [[ "$NTASKS" -gt 0 ]] && SUFFIX="_n$NTASKS"
RUN=/kaggle/runs/${SLUG}_${MODE}${SUFFIX}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN/working" "$RUN/stage"

# ---- build the test file the notebook will see as arc-agi_test_challenges.json
case "$MODE" in
  eval)        SRC_CH=$CDIR/arc-agi_evaluation_challenges.json; SRC_SOL=$CDIR/arc-agi_evaluation_solutions.json ;;
  placeholder) SRC_CH=$CDIR/arc-agi_test_challenges.json;       SRC_SOL=$CDIR/arc-agi_training_solutions.json ;;
  *) echo "mode must be eval|placeholder"; exit 1 ;;
esac
python3 - "$SRC_CH" "$SRC_SOL" "$RUN/stage" "$NTASKS" <<'EOF'
import json, sys
ch, sol, out, n = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3], int(sys.argv[4])
ids = sorted(ch)[:n] if n > 0 else sorted(ch)
json.dump({k: ch[k] for k in ids}, open(f"{out}/test_challenges.json", "w"))
json.dump({k: sol[k] for k in ids}, open(f"{out}/solutions.json", "w"))
print(f">> staged {len(ids)} tasks as the hidden test file")
EOF

echo ">> run dir: $RUN"
echo ">> notebook: $NB   wall clock: $WALL   network: none   GPUs: $GPUS"
START=$(date +%s)
set +e
GPU_DEV=$(seq -s, 0 $((GPUS-1)))
sudo docker run --rm --gpus "\"device=$GPU_DEV\"" --ipc=host --shm-size=32g --network none \
  -v /kaggle/input:/kaggle/input:ro \
  -v "$RUN/stage/test_challenges.json":/kaggle/input/$COMP/arc-agi_test_challenges.json:ro \
  -v "$RUN/working":/kaggle/working \
  -v "$NB_MOUNT_DIR":/nb:ro \
  -w /kaggle/working \
  "$KAGGLE_IMAGE" \
  timeout "$WALL" jupyter nbconvert --to notebook --execute "/nb/$(basename "$NB")" \
      --output /kaggle/working/executed.ipynb --ExecutePreprocessor.timeout=-1 \
  > "$RUN/run.log" 2>&1
RC=$?
set -e
END=$(date +%s); ELAPSED=$(( (END-START)/60 ))
echo ">> exit code $RC after ${ELAPSED} min (log: $RUN/run.log)"
[[ $RC -eq 124 ]] && echo "!! HIT THE WALL CLOCK — on Kaggle this is a failed submission."

if [[ -f "$RUN/working/submission.json" ]]; then
  cp "$RUN/working/submission.json" "$RUN/submission.json"
  python3 ~/infra/arc_eval.py --challenges "$RUN/stage/test_challenges.json" \
      --solutions "$RUN/stage/solutions.json" --submission "$RUN/submission.json" \
      --csv "$RUN/rows.csv" | tee "$RUN/score.txt"
  echo "elapsed_min=$ELAPSED exit=$RC" >> "$RUN/score.txt"
else
  echo "!! no submission.json produced — on Kaggle this scores as an error. Tail of log:"
  tail -40 "$RUN/run.log"
fi
echo; echo "Reminder: stop the VM when idle ->  gcloud compute instances stop \$(hostname) --zone <zone>"
