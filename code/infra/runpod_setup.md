# RunPod: the model sweep in one evening

Goal: run the zero-shot solver with 4 model configs on the 80 dev tasks, get `summary.md` +
`summary.png` (accuracy vs. projected Kaggle hours). About 3–4 hours of pod time, roughly $10–15.

## 0. Before renting (5 minutes, free)

1. **RunPod account** at runpod.io → *Billing* → add **$25** (enough for this and the L4 timing run).
2. **Hugging Face token** (for the gated Nemotron model):
   - huggingface.co → *Settings* → *Access Tokens* → *New token* (Read) → copy it.
   - Open https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 and click the
     "agree / access repository" button. Qwen and gpt-oss need no approval.
3. Have **arc2026_kit.zip** on your Mac.

## 1. Create the pod (GUI)

RunPod → **Pods** → **Deploy**:

| Setting | Choose |
|---|---|
| Cloud | *Secure Cloud* (fewer interruptions than Community) |
| GPU | **H100 80GB** (SXM or PCIe, either is fine). 1 GPU. |
| Template | **RunPod PyTorch 2.x** (any recent CUDA 12.x) |
| *Edit Template* → Container disk | 40 GB |
| *Edit Template* → Volume disk | **150 GB**, mount path `/workspace` |
| Expose HTTP ports | leave `8888` (JupyterLab) — we don't need 8000 |
| Pricing | On-Demand (not Spot — a spot kill mid-sweep wastes the run) |

Click **Deploy On-Demand**. When the pod shows *Running*, click **Connect** → **Start Web Terminal**
(or **Connect to Jupyter Lab** — we use Jupyter for the file upload).

## 2. Upload the kit

In **Jupyter Lab**: the file browser opens at `/workspace`. Click the **upload** icon (↑) and pick
`arc2026_kit.zip`. Wait for the progress bar to finish.

## 3. Run (web terminal)

```bash
cd /workspace && unzip -o arc2026_kit.zip -d arc2026 && cd arc2026/induction
export HF_TOKEN=hf_xxxxxxxx            # paste your token (skip = Nemotron config fails, others run)
bash run_sweep.sh --limit 5            # smoke test: 5 tasks per config, ~15 min incl. downloads
tail -f runs/sweep1.log                # Ctrl-C stops the tail, not the run
```

What a healthy log looks like:

```
[21:02:10] sweep of 4 configs -> runs/sweep1
[21:02:10]    starting: vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --port 8000 ...
[21:02:10]    waiting for server (0 min) ...          <- model download + load, 5-10 min first time
[21:09:41]    server ready after 451s
[21:09:41] >> qwen36_think: running solver: ...
[21:14:02] >> qwen36_think: solver finished rc=0 in 4.3 min
[21:14:05] >> qwen36_think: done
```

If the smoke test ends with a `summary.md` table, run the real thing:

```bash
rm -rf runs/sweep1                      # the smoke test used --limit; start clean
bash run_sweep.sh                       # 80 dev tasks x 4 configs, ~2-3 h
```

You can close the laptop: the run is detached (`nohup`). Come back and `tail -f runs/sweep1.log`.

## 4. Read the results

```bash
cat runs/sweep1/summary.md
```

Then in Jupyter Lab, navigate to `arc2026/induction/runs/sweep1/`, right-click `summary.png` →
Download, and the same for `summary.md`. Also zip the transcripts for the writeup/SFT:

```bash
cd /workspace/arc2026/induction && zip -qr /workspace/sweep1_results.zip runs/sweep1 -x "*/server.log"
```

Download `/workspace/sweep1_results.zip` from Jupyter (it's a few MB).

## 5. Stop paying

Pods → your pod → **Stop** (keeps the volume: models + venv persist, ~$0.10/GB/month ≈ $15/month
for 150 GB) or **Terminate** (deletes everything, $0/month). Terminate once the results are downloaded;
the models re-download in minutes next time.

## Troubleshooting

| Symptom (in `runs/sweep1/<label>/server.log`) | Fix |
|---|---|
| `401` / `403` / `GatedRepoError` | Accept the licence on the model page; `export HF_TOKEN=...`; re-run `bash run_sweep.sh --only nemotron_nano_think` |
| `invalid choice: 'xyz' (choose from ...)` for a parser | The sweep already tries the listed fallbacks. Pick a name from the `choose from` list and edit `sweep_configs.json` |
| `CUDA out of memory` at start-up | Add `"--max-num-seqs","64"` to that config's `vllm_extra`, or run with `--max-model-len 49152` |
| `Address already in use` | An old server is still up: `pkill -f "vllm serve"`; wait 10 s; re-run |
| solver log shows `backend unreachable` repeatedly | The server crashed mid-run: check `server.log` tail; re-run that config with `--only <label>` (finished configs are skipped) |
| `NVCC version must be at least 12.9` / DeepGEMM error in `fp8.py` | The pod's CUDA toolkit is older than DeepGEMM needs. `export VLLM_USE_DEEP_GEMM=0` and re-run that config (the Qwen configs already set this) |
| `Disk quota exceeded (os error 122)` while a model downloads | The 150 GB volume is full. Models live in `$HF_HOME/hub` (the RunPod template sets `HF_HOME=/workspace/.cache/huggingface`). `du -sh $HF_HOME/hub/* | sort -h`, then `rm -rf $HF_HOME/hub/models--<org>--<name>` for models no longer in the sweep. Each 30B FP8 model is ~31-35 GB; the venv is ~18 GB |
| `vllm: command not found` | `source /workspace/venv/bin/activate` |

## Next: the L4 timing run (turns tokens into Kaggle hours)

Same steps on a **4× L4** pod (community cloud is fine, ~$1.60/h), with the winning config:

```bash
cd /workspace/arc2026/induction && source /workspace/venv/bin/activate   # venv is on the volume if you kept it
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name m --tensor-parallel-size 4 \
     --max-model-len 65536 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --reasoning-parser qwen3 --language-model-only &
python bench_speed.py --model m --challenges ../data/arc-agi_evaluation_challenges.json \
     --concurrency 16,64 --thinking on --note "4xL4 qwen36 fp8"
python sweep.py --summary-only --out runs/sweep1 --l4-tps <gen_tps from the 64 line> \
     --challenges ../data/arc-agi_evaluation_challenges.json --solutions ../data/arc-agi_evaluation_solutions.json
```

(The venv lives on the volume, so attach the same volume or re-run `run_sweep.sh` once to recreate it.)
