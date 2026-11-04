# NVARC continued training (LoRA): hop-anywhere training kit

Continue-trains `sorokin/qwen3_4b_grids15_sft139` on new ARC puzzles. It's designed to move between providers on small
credits: Lightning, Modal, RunPod, GCP, Kaggle.

## Files
- `arc_format.py`: NVARC's exact prompt format, NVARC-style augmentation (D8 transforms + 10-colour permutation + example shuffle), loss on outputs only. Unit-tested.
- `train_lora.py`: training, auto-resume, timed full-state checkpoints, Hugging Face hand-off, `--merge`.
- `requirements.txt`

## Set up any GPU box (A100/H100/L4/A10, CUDA 12)
```bash
pip install -r requirements.txt
# model: download once (Kaggle model sorokin/qwen3_4b_grids15_sft139, bfloat16 v1)
pip install kaggle && kaggle models instances versions download sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1 -p model && tar -xzf model/*.tar.gz -C model
export HF_TOKEN=...            # NEW token, write access, never pasted in chat
```

## Data (from the Mega scoring step)
- `train.json`: the hard Mega tasks (challenges with test outputs), or `challenges.json` + `--train-solutions` + `--train-keys mega_hard_keys.json`
- `replay.json`: tasks NVARC already solves (public training + easy Mega), so it doesn't forget them

## Run
```bash
# 1. dry run: data stats, token count, steps (CPU is fine)
python train_lora.py --model-dir model --out runs/r1 --train train.json --replay replay.json --dry-run

# 2. go/no-go (Lightning, 1x A100, ~3-5 h)
python train_lora.py --model-dir model --out runs/r1 --train train.json --replay replay.json \
  --arc-eval ../data/arc-agi_evaluation_challenges.json --arc-eval-solutions ../data/arc-agi_evaluation_solutions.json \
  --n-aug 4 --hub-repo <hf-user>/arc-lora-r1 --max-hours 4.5

# 3. continue elsewhere: same command (+ torchrun for multi-GPU). It pulls the newest checkpoint from the hub.
torchrun --nproc_per_node 8 train_lora.py ...same args...

# 4. merge for Kaggle: upload runs/r1/merged as a Kaggle model, point the NVARC notebook's model glob at it
python train_lora.py --merge --model-dir model --out runs/r1
```

## Reading the eval logs
- `eval_mega_heldout_loss` should fall: the model is learning the new rule families.
- `eval_arc_eval_loss` must NOT rise: the public eval is contaminated for NVARC, so its level means little, but a rise means forgetting. If it rises, lower `--lr` or raise `--replay-frac`.

## Rules
- Global batch is fixed (`--global-batch 32`). Gradient accumulation adapts to the GPU count, so hopping doesn't change the training.
- Full state (adapter + optimizer + scheduler + RNG + step) is saved every 30 min and pushed to the private HF repo.
- `--max-hours` stops cleanly with a save before credits run out. Set it about 30 min under your remaining credit.
- **Terminate the machine** after the final push. Idle GPUs still bill.
