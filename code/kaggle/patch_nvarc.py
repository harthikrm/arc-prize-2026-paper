"""Patch the NVARC 2025 notebook: fast commit, interruptible TTT, deadline, placeholder submission, fixed hash seed."""
import json, sys, copy
src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
C = nb["cells"]
S = lambda i: "".join(C[i]["source"])
assert S(0).startswith("import time\nglobal_end_time") and "pip uninstall" in S(1)
assert S(4).startswith("%%writefile arc_solver.py") and S(6).startswith("!UNSLOTH") and "get_submission" in S(7)

def setsrc(i, text):
    C[i]["source"] = text.splitlines(keepends=True)

setsrc(0, '''import os, json, time, subprocess, sys
NOTEBOOK_START = time.time()

# ---- settings -------------------------------------------------------------
COMMIT_MODE = "full" # what the COMMIT run does (the scoring rerun always runs in full):
                      #   "smoke": 4 eval tasks, 20-min deadline -> tests every code path in ~22 min
                      #   "skip" : placeholder only, ~1 min (use once a version has passed smoke)
                      #   "full" : all 120 eval tasks, ~5.5 h of quota (re-measure the eval score)
                      #   "simrerun": behaves EXACTLY like the scoring rerun (test file, rerun code
                      #              paths) but stops at 75 min -> ~2.8 h of quota
DFS_MIN_PROB = 0.20   # DFS keeps answers whose whole-sequence probability exceeds this (NVARC: 0.20)
DFS_FALLBACK_PROB = 0.10  # test outputs that end the normal search with <2 distinct answers get ONE
                          # extra pass over 4 views at this looser floor (0 = off). Targets the empty slots
                          # (21 of 172 eval outputs) without widening the search everywhere.
DEADLINE_H  = 11.75   # solver stops at 11h45m after this cell; TTT is interruptible, so the
                      # remaining 15 min cover kernel boot before this cell + final decoding.
# ---------------------------------------------------------------------------

RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
MODE = "full" if RERUN else COMMIT_MODE
assert MODE in ("smoke", "skip", "full", "simrerun"), MODE
if MODE == "simrerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"   # every later cell and the solver take the rerun path
    DEADLINE_H = 75 / 60
STRICT = RERUN or MODE == "simrerun"       # scoring-like run: fail loudly instead of submitting zeros
RUN_FULL = MODE != "skip"                  # (name kept: "run the solver at all")
if MODE == "smoke":
    DEADLINE_H = 20 / 60                   # model loading takes ~9 min; 20 min leaves time to decode
                                           # a little and still makes the deadline stop fire
    os.environ["NVARC_MAX_TASKS"] = "4"
global_end_time = NOTEBOOK_START + DEADLINE_H * 3600
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"

# Locate the model wherever Kaggle mounts it (old layout /kaggle/input/<slug>/..., new layout
# /kaggle/input/models/<owner>/<slug>/...). Fail loudly here if it is missing.
import glob
_cands = sorted(glob.glob("/kaggle/input/**/qwen3_4b_grids15_sft139/transformers/bfloat16/1", recursive=True))
_cands = [c for c in _cands if os.path.exists(os.path.join(c, "config.json"))]
if not _cands:
    raise FileNotFoundError("qwen3_4b_grids15_sft139 bfloat16 v1 not found under /kaggle/input")
os.environ["NVARC_MODEL_DIR"] = _cands[0]
print("model dir:", _cands[0])

# A valid submission exists from the first minute; the decoder overwrites it at the end.
_tasks = json.load(open(f"{COMP}/arc-agi_test_challenges.json"))
_placeholder = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in _tasks.items()}
json.dump(_placeholder, open("/kaggle/working/submission.json", "w"))
print(f"dfs_min_prob={DFS_MIN_PROB} dfs_fallback_prob={DFS_FALLBACK_PROB} rerun={RERUN} mode={MODE} strict={STRICT} placeholder_tasks={len(_placeholder)} deadline={DEADLINE_H}h")
''')

setsrc(1, '''if RUN_FULL:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
''')

s4 = S(4)
old_imp = "from transformers import DataCollatorForLanguageModeling\n"
assert s4.count(old_imp) == 1
s4 = s4.replace(old_imp, old_imp + '''from transformers import TrainerCallback


class StopAtDeadline(TrainerCallback):
    """Ends test-time training at the global deadline. Without it one TTT run (up to ~14 min
    measured on the public eval set) can start just before the deadline and push the notebook
    past 12 h, which scores 0."""

    def __init__(self, end_time):
        self.end_time = end_time

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.end_time:
            control.should_training_stop = True
        return control
''')
old_ms = "    max_score = -np.log(0.2)"
assert s4.count(old_ms) == 1
s4 = s4.replace(old_ms, "    max_score = -np.log(float(os.getenv(\"DFS_MIN_PROB\", \"0.2\")))\n"
                        "    print(f\"[Rank {rank}] DFS min sequence probability {np.exp(-max_score):.3f}\")")
old_tr = "                args=UnslothTrainingArguments(**train_args),\n            )"
assert s4.count(old_tr) == 1
s4 = s4.replace(old_tr, "                args=UnslothTrainingArguments(**train_args),\n"
                         "                callbacks=[StopAtDeadline(end_time)],\n            )")
assert s4.count('            for subkeys in batches:\n\n                spend_time = time.time() - start_time\n                if spend_time > 1200 or time.time() > end_time:\n                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")\n                    break\n\n                print(f"[Rank {rank}] decoding {subkeys}")\n') == 1
s4 = s4.replace('            for subkeys in batches:\n\n                spend_time = time.time() - start_time\n                if spend_time > 1200 or time.time() > end_time:\n                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")\n                    break\n\n                print(f"[Rank {rank}] decoding {subkeys}")\n', '            # Work list: the normal batches at the normal floor; when they are done, every test output\n            # with fewer than 2 distinct answers gets one extra batch (4 views) at a looser floor.\n            found = defaultdict(set)\n            fb_prob = float(os.getenv("DFS_FALLBACK_PROB", "0"))\n            work = [(sk, max_score, "") for sk in batches]\n            wi = 0\n            while wi < len(work):\n                subkeys, cur_max_score, suffix = work[wi]\n                wi += 1\n\n                spend_time = time.time() - start_time\n                if spend_time > 1200 or time.time() > end_time:\n                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")\n                    break\n\n                print(f"[Rank {rank}] decoding{suffix} {subkeys}")\n')
assert s4.count('                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)\n') == 1
s4 = s4.replace('                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)\n', '                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, cur_max_score, end_time)\n')
assert s4.count('                        grid_id = (bk, tuple(map(tuple, solution)))\n') == 1
s4 = s4.replace('                        grid_id = (bk, tuple(map(tuple, solution)))\n', '                        grid_id = (bk, tuple(map(tuple, solution)))\n                        found[bk].add(grid_id)\n')
assert s4.count('                    if len(decoded_result):\n                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:\n                            pickle.dump(decoded_result, f)\n') == 1
s4 = s4.replace('                    if len(decoded_result):\n                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:\n                            pickle.dump(decoded_result, f)\n', '                    if len(decoded_result):\n                        with bz2.BZ2File(os.path.join(dir_outputs, subkey + suffix), "w") as f:\n                            pickle.dump(decoded_result, f)\n\n                if wi == len(work) and suffix == "" and fb_prob > 0:\n                    for test_id, sks in test_id_to_subkeys.items():\n                        if len(found[f"{key}_{test_id}"]) < 2:\n                            work.append((sks[0:2] + sks[4:6], -np.log(fb_prob), ".fb"))\n                    if len(work) > wi:\n                        print(f"[Rank {rank}] fallback pass for {len(work) - wi} output(s) of {key} at floor {fb_prob}")\n')
old_mp = '        model_name="/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\n'
assert s4.count(old_mp) == 1
s4 = s4.replace(old_mp, '        model_name=os.environ["NVARC_MODEL_DIR"],\n')
setsrc(4, s4)

s5 = S(5)
old_q = "    for key in sorted(data.keys()):\n        queue.put(key)"
assert s5.count(old_q) == 1
s5 = s5.replace(old_q, "    # smallest tasks first: a deadline cut then loses a few big tasks rather than many small ones\n"
    "    def _cost(k):\n"
    "        t = data[k]\n"
    "        grids = [g for p in t['train'] for g in (p['input'], p['output'])] + [p['input'] for p in t['test']]\n"
    "        return sum(len(g) * (len(g[0]) + 1) for g in grids) * len(t['test'])\n"
    "    keys = sorted(data.keys(), key=lambda k: (_cost(k), k))\n"
    "    if os.getenv(\"NVARC_MAX_TASKS\") and not rerun_mode:\n"
    "        keys = keys[:int(os.environ[\"NVARC_MAX_TASKS\"])]\n"
    "    print(f\"queued {len(keys)} tasks\")\n"
    "    for key in keys:\n        queue.put(key)")
setsrc(5, s5)

setsrc(6, '''import glob, collections
SOLVER_LOG = "/kaggle/working/solver.log"
OUT_DIR = "/kaggle/inference_outputs"

def _tail(n):
    try:
        with open(SOLVER_LOG, "rb") as f:
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except FileNotFoundError:
        return []

if RUN_FULL:
    env = dict(os.environ, UNSLOTH_DISABLE_STATISTICS="1", TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
               OMP_NUM_THREADS="12", PYTHONHASHSEED="0",
               DFS_MIN_PROB=str(DFS_MIN_PROB), DFS_FALLBACK_PROB=str(DFS_FALLBACK_PROB))   # fixed seed: hash(bk) picks the scoring augmentations
    # Solver output goes to a file (not a pipe into the notebook): nothing in the kernel can block or
    # crash on it, and the full log is kept in /kaggle/working for the commit run.
    with open(SOLVER_LOG, "wb") as lf:
        p = subprocess.Popen([sys.executable, "starter.py", "--end-time", str(global_end_time)],
                             env=env, stdout=lf, stderr=subprocess.STDOUT)
        last = 0
        while p.poll() is None:
            time.sleep(30)
            if time.time() - last > 600:
                last = time.time()
                n_out = len(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else 0
                print(f"[{(time.time() - NOTEBOOK_START) / 3600:.2f} h] solver running | outputs {n_out} | "
                      + " / ".join(l[:120] for l in _tail(2)), flush=True)
    log = _tail(10**9)
    counts = collections.Counter()
    for l in log:
        for key in ("finished", "stop!", "timeout after", "Traceback", "Error", "Killed", "out of memory"):
            if key in l:
                counts[key] += 1
    print(f"solver exit code {p.returncode} | elapsed {(time.time() - NOTEBOOK_START) / 3600:.2f} h | {dict(counts)}")
    print("---- solver log tail ----")
    print("\\n".join(log[-40:]))
else:
    print("commit mode skip: solver skipped, placeholder submission.json kept")
''')

s7 = S(7)
assert s7.count("    decoder.benchmark_selection_algos()") == 1
s7 = s7.replace("    decoder.benchmark_selection_algos()",
    "    try:\n        decoder.benchmark_selection_algos()\n"
    "    except Exception as e:\n        print(\"benchmark skipped:\", e)")
body = "\n".join("        " + l if l.strip() else l for l in s7.splitlines())
setsrc(7, '''import traceback
decode_error = None
if RUN_FULL:
    try:
''' + body + '''
    except Exception as e:
        traceback.print_exc()
        decode_error = e
        print("decoding failed: placeholder submission.json kept")
    if not STRICT and os.path.isdir("/kaggle/inference_outputs"):
        # commit runs only: keep every candidate + its scores for offline selection-rule experiments
        import shutil
        shutil.make_archive("/kaggle/working/nvarc_candidates", "zip", "/kaggle/inference_outputs")
        print("saved /kaggle/working/nvarc_candidates.zip")
    sub = json.load(open("/kaggle/working/submission.json"))
    n_slots = sum(len(v) for v in sub.values())
    n_answered = sum(1 for v in sub.values() for a in v if a["attempt_1"] != [[0]])
    print(f"submission: {len(sub)} tasks, {n_answered}/{n_slots} test outputs answered")
    if STRICT and (decode_error is not None or n_answered == 0):
        # A scoring run that answered nothing must show as an error, not as a silent 0.00.
        raise RuntimeError(f"scoring run produced no answers (decode_error={decode_error!r})")
print(f"total elapsed {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
''')

for c in C:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(dst, "w"), indent=1)
print("written", dst)
