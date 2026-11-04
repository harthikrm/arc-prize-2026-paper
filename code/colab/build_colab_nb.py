#!/usr/bin/env python3
"""Builds arc_train_colab.ipynb: continue-train NVARC on the Mega band set, on a Colab GPU, with auto-resume.

Expects in Google Drive folder  MyDrive/arc/ :
  arc_train_kit.zip      (train/ folder: train_lora.py, arc_format.py, requirements.txt)
  train_data_big.zip     (train_data_big/: train.json, val.json, replay.json, arc-agi_evaluation_*.json)
Colab secrets: HF_TOKEN (write), KAGGLE_USERNAME, KAGGLE_API_TOKEN
"""
import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


md("""# NVARC + Mega band set: training run r2 (Colab)

**Before running:** Runtime → Change runtime type → **G4 GPU** (or A100). Secrets (key icon on the left, "Notebook access" on):
`HF_TOKEN` (write), `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`. Drive folder `MyDrive/arc/` must contain
`arc_train_kit.zip` and `train_data_big.zip`.

**Resume after a disconnect:** Runtime → Run all. Cell 5 pulls the newest checkpoint from the Hugging Face repo and continues.
Nothing is lost except the minutes since the last checkpoint.""")

code("""# ---- 1. settings --------------------------------------------------------------
HF_USER   = "CHANGE_ME"          # your Hugging Face username (checkpoints go to HF_USER/arc-lora-r2, private)
RUN       = "r2"
DRIVE_DIR = "/content/drive/MyDrive/arc"
WORK      = "/content/arc"
SAVE_MIN  = 60                   # checkpoint + push every N minutes (each push ~3 GB)
EVAL_STEPS = 200
assert HF_USER != "CHANGE_ME", "set HF_USER first"
""")

code("""# ---- 2. drive, secrets, GPU ------------------------------------------------------
import os, subprocess, time, glob, json
from google.colab import drive, userdata
drive.mount("/content/drive", force_remount=False)
for k in ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN"):
    os.environ[k] = userdata.get(k)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print("GPU:", gpu)
mem_gb = int(gpu.split(",")[-1].strip().split()[0]) / 1024
NO_GRAD_CKPT = mem_gb >= 60          # G4 (96 GB) / A100-80: skip checkpointing, ~1.3x faster; A100-40/L4: keep it
print(f"{mem_gb:.0f} GB -> gradient checkpointing {'OFF' if NO_GRAD_CKPT else 'ON'}")
for f in ("arc_train_kit.zip", "train_data_big.zip"):
    assert os.path.exists(f"{DRIVE_DIR}/{f}"), f"missing {DRIVE_DIR}/{f}"
""")

code("""# ---- 3. unpack kit + data, install deps (skipped when already done in this runtime) ----
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
if not os.path.exists("train/train_lora.py"):
    subprocess.run(["unzip", "-oq", f"{DRIVE_DIR}/arc_train_kit.zip"], check=True)
if not os.path.exists("train_data_big/train.json"):
    print("unzipping data (~1 GB)..."); subprocess.run(["unzip", "-oq", f"{DRIVE_DIR}/train_data_big.zip"], check=True)
for f in ("train.json", "val.json", "replay.json", "arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    assert os.path.exists(f"train_data_big/{f}"), f"train_data_big/{f} missing (copy the two arc-agi_evaluation files in before zipping)"
subprocess.run(["pip", "install", "-q", "transformers>=4.55,<5", "peft>=0.13", "accelerate>=1.0", "huggingface_hub>=0.25", "kagglehub"], check=True)
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "| cuda", torch.cuda.is_available())
""")

code("""# ---- 4. base model (NVARC) from Kaggle, once per runtime -------------------------
import kagglehub
if not os.path.exists("model/config.json"):
    p = kagglehub.model_download("sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    if os.path.islink("model"): os.remove("model")
    os.symlink(p, "model")
print("model ->", os.path.realpath("model"), os.listdir("model"))
""")

code("""# ---- 5. train (resumes from the newest Hugging Face checkpoint if the runtime restarted) ----
args = ["python", "train/train_lora.py", "--model-dir", "model", "--out", f"runs/{RUN}",
        "--train", "train_data_big/train.json", "--val", "train_data_big/val.json", "--replay", "train_data_big/replay.json",
        "--arc-eval", "train_data_big/arc-agi_evaluation_challenges.json",
        "--arc-eval-solutions", "train_data_big/arc-agi_evaluation_solutions.json",
        "--n-aug", "1", "--eval-steps", str(EVAL_STEPS), "--save-minutes", str(SAVE_MIN),
        "--hub-repo", f"{HF_USER}/arc-lora-{RUN}"] + (["--no-grad-ckpt"] if NO_GRAD_CKPT else [])
print(" ".join(args))
log = open("train.log", "a")
p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
t0 = time.time(); shown = set()
def lines():
    try:
        with open("train.log", "rb") as f:
            return [l[-150:] for l in f.read().decode("utf-8", "replace").replace("\\r", "\\n").splitlines() if l.strip()]
    except FileNotFoundError:
        return []
while p.poll() is None:
    time.sleep(300)
    ls = lines()
    prog = [l for l in ls if "it/s]" in l or "s/it]" in l]
    print(f"[{(time.time()-t0)/3600:.2f} h] {prog[-1] if prog else (ls[-1] if ls else 'starting')}", flush=True)
    for l in ls:
        if ("eval_mega_heldout_loss" in l or "eval_arc_eval_loss" in l) and l not in shown:
            shown.add(l); print("   ", l[:120], flush=True)
print("exit code", p.returncode)
print("\\n".join(lines()[-15:]))
""")

code("""# ---- 6. status: every held-out / forgetting check so far -------------------------
import re
txt = open("train.log", "rb").read().decode("utf-8", "replace")
for m in re.finditer(r"'eval_(mega_heldout|arc_eval)_loss': ([0-9.]+).*?'epoch': ([0-9.]+)", txt):
    print(f"{m.group(1):<13} loss {float(m.group(2)):.4f}   epoch {float(m.group(3)):.2f}")
print(sorted(glob.glob(f"runs/{RUN}/*")))
""")

code("""# ---- 7. AFTER training finished: merge and upload to Kaggle as a private model -------
assert os.path.exists(f"runs/{RUN}/final_adapter"), "training has not finished (no final_adapter)"
subprocess.run(["python", "train/train_lora.py", "--merge", "--model-dir", "model", "--out", f"runs/{RUN}"], check=True)
print(os.listdir(f"runs/{RUN}/merged"))
import kagglehub
kagglehub.model_upload(f"{os.environ['KAGGLE_USERNAME']}/nvarc-mega-{RUN}/transformers/bfloat16", f"runs/{RUN}/merged",
                       version_notes=f"{RUN}: NVARC + Mega band set (47k tasks) LoRA merged", license_name="Apache 2.0")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("arc_train_colab.ipynb", "w"), indent=1)
print("wrote arc_train_colab.ipynb with", len(cells), "cells")
