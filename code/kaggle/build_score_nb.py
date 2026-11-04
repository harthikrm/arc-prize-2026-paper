import json
src = json.load(open("nvarc-v5-fallback-full.ipynb"))
loader = "".join(src["cells"][2]["source"])
assert loader.startswith("%%writefile arc_loader.py")
worker = open("mega_score_worker.py").read()

c0 = r'''import os, json, glob, random, time, subprocess, sys
# Needs only the image's own torch + transformers. The NVARC utility script ships a Python-3.11 torch
# that shadows the image's torch on Python 3.13 -> strip it from every import path (kernel + workers).
sys.path[:] = [p for p in sys.path if "pip_install_unsloth" not in p]
os.environ["PYTHONPATH"] = ":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p and "pip_install_unsloth" not in p)
print("python", sys.version.split()[0])
NOTEBOOK_START = time.time()
# ---- settings ---------------------------------------------------------------
MAX_MEGA     = 0        # 0 = all tasks in the uploaded sample; e.g. 400 for a quick test
N_PUBTRAIN   = 200      # public-training reference tasks (NVARC trained on these -> should score high)
DEADLINE_H   = 11.0     # workers stop here; results written so far are kept
# -----------------------------------------------------------------------------
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
if not os.path.isdir(COMP):
    COMP = os.path.dirname(sorted(glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True))[0])
_cands = sorted(glob.glob("/kaggle/input/**/qwen3_4b_grids15_sft139/transformers/bfloat16/1", recursive=True))
_cands = [c for c in _cands if os.path.exists(os.path.join(c, "config.json"))]
if not _cands:
    raise FileNotFoundError("NVARC model (qwen3_4b_grids15_sft139 bfloat16 v1) is not attached")
os.environ["NVARC_MODEL_DIR"] = _cands[0]
mega_ch = [p for p in glob.glob("/kaggle/input/**/challenges.json", recursive=True)
           if os.path.exists(os.path.join(os.path.dirname(p), "solutions.json"))]
if not mega_ch:
    raise FileNotFoundError("arc-mega-sample dataset (challenges.json + solutions.json) is not attached")
MEGA_DIR = os.path.dirname(mega_ch[0])
print("competition:", COMP); print("model:", _cands[0]); print("mega sample:", MEGA_DIR)

items = []
ch = json.load(open(f"{MEGA_DIR}/challenges.json")); so = json.load(open(f"{MEGA_DIR}/solutions.json"))
keys = sorted(ch); random.Random(0).shuffle(keys)
if MAX_MEGA: keys = keys[:MAX_MEGA]
for k in keys:
    t = ch[k]
    if not t["test"] or so.get(k) is None or not so[k] or so[k][0] is None: continue
    items.append({"key": k, "source": "mega", "train": t["train"], "test_input": t["test"][0]["input"], "test_output": so[k][0]})
ev = json.load(open(f"{COMP}/arc-agi_evaluation_challenges.json")); evs = json.load(open(f"{COMP}/arc-agi_evaluation_solutions.json"))
for k, t in ev.items():
    for i, x in enumerate(t["test"]):
        items.append({"key": f"{k}_{i}", "source": "eval", "train": t["train"], "test_input": x["input"], "test_output": evs[k][i]})
tr = json.load(open(f"{COMP}/arc-agi_training_challenges.json")); trs = json.load(open(f"{COMP}/arc-agi_training_solutions.json"))
for k in random.Random(1).sample(sorted(tr), N_PUBTRAIN):
    items.append({"key": f"{k}_0", "source": "pubtrain", "train": tr[k]["train"], "test_input": tr[k]["test"][0]["input"], "test_output": trs[k][0]})
json.dump(items, open("/kaggle/working/score_items.json", "w"))
from collections import Counter
print("items:", dict(Counter(i["source"] for i in items)))
'''

c3 = r'''# ---- run 4 workers (one per L4), heartbeat every 5 min ----------------------
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.version.cuda)
n_gpu = torch.cuda.device_count(); print("GPUs:", n_gpu)
if n_gpu == 0:
    raise RuntimeError("no GPU visible - set the accelerator to GPU L4x4")
end_time = NOTEBOOK_START + DEADLINE_H * 3600
procs = []
for r in range(n_gpu):
    log = open(f"/kaggle/working/worker{r}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "mega_score_worker.py", str(r), str(n_gpu), "/kaggle/working/score_items.json",
                                   f"/kaggle/working/scores_{r}.jsonl", str(end_time)], stdout=log, stderr=subprocess.STDOUT))
def _count():
    return sum(sum(1 for _ in open(p)) for p in glob.glob("/kaggle/working/scores_*.jsonl"))
while any(p.poll() is None for p in procs):
    time.sleep(300)
    tails = []
    for r in range(n_gpu):
        try: tails.append(open(f"/kaggle/working/worker{r}.log").read().strip().splitlines()[-1][-110:])
        except Exception: pass
    print(f"[{(time.time() - NOTEBOOK_START) / 3600:.2f} h] scored {_count()}/{len(items)} | " + " || ".join(tails), flush=True)
print("exit codes:", [p.returncode for p in procs])
for r in range(n_gpu):
    print(f"--- worker{r}.log (tail) ---"); print("\n".join(open(f"/kaggle/working/worker{r}.log").read().splitlines()[-8:]))
'''

c4 = r'''# ---- summary ----------------------------------------------------------------
import csv, numpy as np
rows = [json.loads(l) for p in sorted(glob.glob("/kaggle/working/scores_*.jsonl")) for l in open(p) if l.strip()]
print(f"{len(rows)} scored")
for src in ("pubtrain", "eval", "mega"):
    r = [x for x in rows if x["source"] == src]
    if not r: continue
    g = np.mean([x["greedy_exact"] for x in r]); nm = np.median([x["nll_mean"] for x in r])
    p50 = np.mean([x["seq_prob"] > 0.5 for x in r]); p10 = np.mean([x["seq_prob"] < 0.1 for x in r])
    print(f"{src:<9} n={len(r):>6}  greedy-exact {100 * g:5.1f}%   seq_prob>0.5 {100 * p50:5.1f}%   seq_prob<0.1 {100 * p10:5.1f}%   median per-token NLL {nm:.4f}")
with open("/kaggle/working/mega_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
hard = sorted([x for x in rows if x["source"] == "mega" and not x["greedy_exact"]], key=lambda x: -x["nll_mean"])
json.dump([x["key"] for x in hard], open("/kaggle/working/mega_hard_keys.json", "w"))
print(f"mega tasks NVARC gets wrong zero-shot: {len(hard)} -> mega_hard_keys.json; all scores -> mega_scores.csv")
print("Reading: pubtrain should be high (NVARC trained on it). If mega greedy-exact is close to pubtrain, Mega adds little;")
print("if it is far lower, there is headroom and mega_hard_keys.json is the candidate training set.")
'''

def cell(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}
nb = {"cells": [cell(c0), cell(loader), cell("%%writefile mega_score_worker.py\n" + worker), cell(c3), cell(c4)],
      "metadata": src["metadata"] | {"papermill": {}}, "nbformat": 4, "nbformat_minor": 5}
nb["metadata"].pop("papermill", None)
json.dump(nb, open("nvarc-mega-score.ipynb", "w"), indent=1)
print("written")
