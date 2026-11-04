# Inductive solver: how to run it

## Files

| File | What it does |
|---|---|
| `arc_tools.py` | Copies the training data's environment: the system prompt (2 variants), the user prompt, the 4 tool schemas, the executor's reply text and metrics, and the round messages |
| `solver.py` | Runs multi-round tool conversations with any OpenAI-compatible model server, sends every program through the verifier, and writes `submission.json`, candidates and transcripts |
| `verifier.py` | Applies L0–L3 trust checks: train pass, leave-one-out, hygiene and consensus |
| `sandbox.py` | Runs model code safely (numpy, scipy and torch allowed; stdout discarded; CPU, memory and time limits) |
| `check_format.py` | Replays real dataset conversations through our executor and reports, line type by line type, how exactly we reproduce the training replies |
| `sweep.py` + `sweep_configs.json` | Runs the solver across model configs (starts/stops vLLM for each), writes `summary.md` / `summary.png`: accuracy vs projected Kaggle hours |
| `run_sweep.sh` | One command on a fresh GPU pod: venv + vLLM install, then the sweep in the background |
| `bench_speed.py` | Measures generated tokens/s of a model server under solver-like load (run on 4xL4 to convert tokens into Kaggle hours) |
| `fake_server.py` | OpenAI-compatible stand-in for vLLM; tests the whole pipeline without a GPU |
| `test_solver.py`, `test_verifier.py` | Offline tests; no GPU or server needed |

## Step 1: Check format fidelity on the prep VM (CPU, about 5 minutes)

```bash
cd ~/arc2026/induction && pip install scipy numpy
python check_format.py --data ~/sft_train.jsonl --n 300 > ~/format_report.txt; tail -60 ~/format_report.txt
```

Send `format_report.txt` back. Parts marked `APPROX` in `arc_tools.py` were not visible in the samples; this report shows what they really say:

- the test-input header
- the commit reply
- the reply to a turn with no tool call
- the name of colour 8

## Step 2: Weekend gate — model sweep on the 80 dev tasks (one H100)

Follow `../infra/runpod_setup.md`. In short, on the pod:

```bash
cd /workspace/arc2026/induction && export HF_TOKEN=hf_...
bash run_sweep.sh --limit 5      # smoke test
bash run_sweep.sh                # full: 4 configs x 80 tasks x 8 samples x 5 rounds
cat runs/sweep1/summary.md
```

Offline dry run of the same pipeline (no GPU), useful to check a pod before paying for the real thing:

```bash
# add  "server_cmd": ["python","fake_server.py","--port","{port}"]  to a config copy, then:
python sweep.py --configs my_fake_configs.json --challenges ../data/arc-agi_training_challenges.json \
       --solutions ../data/arc-agi_training_solutions.json --split all --limit 3 --samples 2 --out runs/dry
```

## Step 2b: Manual single-model run (any OpenAI-compatible server)

```bash
# terminal 1: model server (one H100 80GB, or 4xL4 with --tensor-parallel-size 4)
pip install vllm
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name m --max-model-len 65536 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
     --language-model-only --gpu-memory-utilization 0.92

# terminal 2: solver
python solver.py --challenges ../data/arc-agi_evaluation_challenges.json \
   --solutions ../data/arc-agi_evaluation_solutions.json --split dev \
   --base-url http://localhost:8000/v1 --model m --thinking on \
   --samples 8 --rounds 5 --max-tokens 6000 --parallel-tasks 8 --out runs/zs_qwen36

# score + complementarity against the NVARC baseline submission
python ../infra/arc_eval.py --challenges ../data/arc-agi_evaluation_challenges.json \
   --solutions ../data/arc-agi_evaluation_solutions.json --split dev \
   --submission runs/zs_qwen36/submission.json --compare nvarc_dev_submission.json
```

Ablation for the writeup, at no extra GPU cost: re-select from the same candidates with `--verifier-level L0` and compare false positives against `full`.

## What to read in a run

- `report.json` → per task:
  - `n_pass_train` vs `n_trusted` (the hygiene rejections)
  - `n_heldout_ok` (leave-one-out wins)
  - `stops` (committed, rounds, or context)
- `transcripts/*.jsonl` are in the training-data format. Correct ones on *training* tasks can be reused as new SFT data (expert iteration). Never use transcripts from eval tasks for training.
