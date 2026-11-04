"""Run model-written generator / transform code in a separate, resource-limited process.

Not a security boundary against a hostile author - it is a crash/timeout/infinite-loop guard plus a
static ban on the obviously dangerous modules. Good enough for code from an API we chose to call.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

BANNED = re.compile(
    r"^\s*(?:import|from)\s+(os|sys|subprocess|socket|shutil|pathlib|importlib|ctypes|signal|threading|"
    r"multiprocessing|requests|urllib|http|builtins|pickle|marshal|code|inspect)\b", re.M)
BANNED_CALLS = re.compile(r"\b(open|exec|eval|__import__|compile|input|breakpoint|exit|quit)\s*\(")

_RUNNER = r'''
import sys, json, random, resource, signal
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT, CPU_LIMIT))
job = json.load(sys.stdin)
ns = {}
try:
    exec(job["code"], ns)
except Exception as e:
    print(json.dumps({"ok": False, "stage": "exec", "error": repr(e)[:500]})); sys.exit(0)
fn = ns.get(job["func"])
if fn is None:
    print(json.dumps({"ok": False, "stage": "exec", "error": "function %s not defined" % job["func"]})); sys.exit(0)
results = []
for call in job["calls"]:
    try:
        if job["func"] == "generate_input":
            out = fn(random.Random(call))
        else:
            out = fn([list(r) for r in call])
        # normalise numpy -> list of lists of int
        try:
            import numpy as np
            if isinstance(out, np.ndarray):
                out = out.tolist()
        except Exception:
            pass
        out = [[int(c) for c in row] for row in out]
        results.append({"ok": True, "grid": out})
    except Exception as e:
        results.append({"ok": False, "error": repr(e)[:300]})
print(json.dumps({"ok": True, "results": results}))
'''


def static_check(code):
    """Return a reason string if the code uses banned modules/calls, else None."""
    m = BANNED.search(code)
    if m:
        return f"banned import: {m.group(1)}"
    m = BANNED_CALLS.search(code)
    if m:
        return f"banned call: {m.group(1)}("
    return None


def run(code, func, calls, timeout=20.0, cpu_limit=15):
    """Execute `func` from `code` on each item in `calls`.

    func == "generate_input": each call is an int seed -> rng.
    func == "transform":      each call is a grid.
    Returns dict: {"ok": bool, "results": [{"ok", "grid"|"error"}], "error"?: str}
    """
    reason = static_check(code)
    if reason:
        return {"ok": False, "stage": "static", "error": reason}
    job = json.dumps({"code": code, "func": func, "calls": calls})
    runner = _RUNNER.replace("CPU_LIMIT", str(int(cpu_limit)))
    with tempfile.NamedTemporaryFile("w", suffix="_runner.py", delete=False) as f:
        f.write(runner)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], input=job, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "timeout", "error": f"exceeded {timeout}s"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if p.returncode != 0 and not p.stdout.strip():
        err = (p.stderr or "").strip()[-500:] or f"killed (exit {p.returncode}; CPU or memory limit)"
        return {"ok": False, "stage": "crash", "error": err}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"ok": False, "stage": "parse", "error": (p.stdout + p.stderr)[-500:]}


def valid_grid(g):
    if not isinstance(g, list) or not (1 <= len(g) <= 30):
        return False
    w = len(g[0]) if isinstance(g[0], list) else -1
    if not (1 <= w <= 30):
        return False
    for row in g:
        if not isinstance(row, list) or len(row) != w:
            return False
        for c in row:
            if not isinstance(c, int) or not (0 <= c <= 9):
                return False
    return True
