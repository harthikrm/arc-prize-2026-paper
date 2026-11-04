"""
kaggle_scaffold.py — drop-in top/bottom cells for your competition notebook.

Rules this enforces (learned the hard way by the 2025 field):
  1. A complete, valid submission.json exists on disk within the first minute,
     BEFORE any model loads. Whatever happens later, Kaggle finds a scoreable file.
  2. Placeholder detection: when the test file is the public placeholder (its ids
     overlap the training set), run only a tiny smoke subset so a "Save & Run All"
     commit costs minutes of quota, not 12 hours. The full pipeline runs only in the
     hidden rerun.
  3. A global deadline (default 11h 20m) with per-task budgets; solvers are called
     through `run_task()` and results are flushed to submission.json after EVERY task,
     so a crash or a hard kill at 12h still leaves your best partial work.
  4. Every prediction passes through `sanitize()` -> always a rectangular int grid,
     never None, never >30x30, attempt_2 defaults to attempt_1 (or the fallback).

Wire your solver(s) in `solve_task()` and keep everything else.
"""
import json, os, time, signal, traceback
from pathlib import Path

T0 = time.time()
DATA_DIR = Path("/kaggle/input/arc-prize-2026-arc-agi-2")
TEST_PATH = DATA_DIR / "arc-agi_test_challenges.json"
TRAIN_PATH = DATA_DIR / "arc-agi_training_challenges.json"
OUT_PATH = Path("/kaggle/working/submission.json")

TOTAL_BUDGET_S = 11 * 3600 + 20 * 60      # leave ~40 min for teardown/slack
SMOKE_N_TASKS = 3                          # tasks to run when on the placeholder
DEBUG_FULL_RUN_ON_PLACEHOLDER = False      # flip to True once, to time a real full run

# ------------------------------------------------------------------ 1. fallback first
def fallback_grid(task):
    """Cheap, never-fails guess: test input itself (16/172 eval outputs are >=95% identical
    to the input), which at least has a plausible shape for same-shape tasks."""
    return task["test"][0]["input"] if task["test"] else [[0]]

def sanitize(g, fb):
    try:
        if not isinstance(g, list) or not g or not isinstance(g[0], list):
            return fb
        w = len(g[0])
        if w == 0 or len(g) > 30 or w > 30:
            return fb
        out = []
        for r in g:
            if len(r) != w:
                return fb
            out.append([int(c) if isinstance(c, (int, float)) and 0 <= int(c) <= 9 else 0 for c in r])
        return out
    except Exception:
        return fb

test = json.load(open(TEST_PATH))
train_ids = set(json.load(open(TRAIN_PATH)).keys())
# Kaggle sets KAGGLE_IS_COMPETITION_RERUN only in the hidden scoring rerun; the id-overlap
# check is a second guard (and the only one that works on our GCP rehearsal box).
IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
IS_PLACEHOLDER = (not IS_RERUN) and len(set(test) & train_ids) > 0
print(f"[scaffold] {len(test)} tasks; placeholder={IS_PLACEHOLDER}")

submission = {}
for tid, task in test.items():
    submission[tid] = []
    for t in task["test"]:
        fb = sanitize(t["input"], [[0]])
        submission[tid].append({"attempt_1": fb, "attempt_2": fb})

def flush():
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(submission, f)
    os.replace(tmp, OUT_PATH)          # atomic: never a half-written file

flush()
print(f"[scaffold] fallback submission written at t={time.time()-T0:.0f}s")

# ------------------------------------------------------------------ 2. budgets
def time_left():
    return TOTAL_BUDGET_S - (time.time() - T0)

class TaskTimeout(Exception):
    pass

def _alarm(signum, frame):
    raise TaskTimeout()

def run_with_timeout(fn, seconds, *a, **kw):
    """Hard per-task cap. Works for CPU/GPU python code on the main thread."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(max(1, int(seconds)))
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)

# ------------------------------------------------------------------ 3. your solver(s)
def solve_task(tid, task, budget_s):
    """
    Return a list (one per test input) of (attempt_1, attempt_2) grids, or None to keep
    the fallback. Plug solvers in here, in priority order, e.g.:
        1) inductive: program that passes the (hardened) verifier on all train pairs
        2) transductive: NVARC-style TTT + DFS candidates
    `budget_s` is this task's share of the remaining clock.
    """
    raise NotImplementedError

def run_task(tid, task, budget_s):
    try:
        preds = run_with_timeout(solve_task, budget_s, tid, task, budget_s)
    except TaskTimeout:
        print(f"[{tid}] timeout after {budget_s:.0f}s -> keeping fallback"); return
    except NotImplementedError:
        return
    except Exception:
        print(f"[{tid}] solver error -> keeping fallback\n{traceback.format_exc()[-800:]}"); return
    if not preds:
        return
    for k, t in enumerate(task["test"]):
        if k >= len(preds):
            break
        fb = submission[tid][k]["attempt_1"]
        a1, a2 = preds[k]
        a1 = sanitize(a1, fb)
        a2 = sanitize(a2, a1)
        if a2 == a1:                       # never waste the second try if you have any alternative
            pass                           # (leave as-is; better arbitration comes later)
        submission[tid][k] = {"attempt_1": a1, "attempt_2": a2}

# ------------------------------------------------------------------ 4. main loop
def main():
    ids = list(test.keys())
    if IS_PLACEHOLDER and not DEBUG_FULL_RUN_ON_PLACEHOLDER:
        ids = ids[:SMOKE_N_TASKS]
        print(f"[scaffold] placeholder run -> smoke test on {len(ids)} tasks only")
    # largest grids first: they are slowest and most likely to time out; better to
    # discover that early while there is slack (Adarsh's load-balancing observation)
    ids.sort(key=lambda t: -max(len(p["input"]) * len(p["input"][0]) for p in test[t]["train"]))
    for i, tid in enumerate(ids):
        remaining = len(ids) - i
        budget = time_left() / remaining
        if budget < 20:
            print(f"[scaffold] out of time at task {i}/{len(ids)}; stopping"); break
        run_task(tid, test[tid], budget)
        flush()
        if i % 10 == 0:
            print(f"[scaffold] {i+1}/{len(ids)} done, {time_left()/60:.0f} min left")
    flush()
    print(f"[scaffold] finished at t={(time.time()-T0)/3600:.2f}h; submission.json has {len(submission)} tasks")

if __name__ == "__main__":
    main()
