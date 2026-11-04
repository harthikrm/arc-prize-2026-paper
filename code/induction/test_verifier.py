import json, sys, time
sys.path.insert(0, ".")
from verifier import Candidate, verify, select_attempts

ch = json.load(open("../data/arc-agi_training_challenges.json"))
sol = json.load(open("../data/arc-agi_training_solutions.json"))
tid = "007bbfb7"; task = ch[tid]; truth = sol[tid]
print("task", tid, "train pairs", len(task["train"]), "test", len(task["test"]))

CORRECT = '''
import numpy as np
def transform(grid):
    g = np.array(grid); h, w = g.shape
    out = np.zeros((h*h, w*w), dtype=int)
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0:
                out[i*h:(i+1)*h, j*w:(j+1)*w] = g
    return out.tolist()
'''
CORRECT_V2 = '''
def transform(grid):
    n = len(grid)
    return [[grid[r//n][c//n] and grid[r%n][c%n] for c in range(n*n)] for r in range(n*n)]
'''
# memorises the demo answers: passes every demo, knows nothing about the rule
outs = [p["output"] for p in task["train"]]; ins = [p["input"] for p in task["train"]]
CHEAT = "TABLE = " + repr(list(zip(ins, outs))) + '''
def transform(grid):
    for i, o in TABLE:
        if i == grid:
            return o
    return [row * 3 for row in grid] * 3
'''
IDENTITY = "def transform(grid):\n    return grid\n"
EVIL = "import os\ndef transform(grid):\n    os.system('echo hacked')\n    return grid\n"
LOOP = "def transform(grid):\n    while True:\n        pass\n"
NOFN = "x = 1\n"

cands = [Candidate(CORRECT, heldout_index=0), Candidate(CORRECT_V2, heldout_index=2), Candidate(CHEAT),
         Candidate(IDENTITY), Candidate(EVIL), Candidate(LOOP), Candidate(NOFN)]
names = ["correct", "correct_v2", "cheat", "identity", "evil", "loop", "no_transform"]
t = time.time()
for n, c in zip(names, cands):
    verify(c, task, timeout_s=2)
    print(f"{n:13s} pass_train={c.pass_train!s:5s} heldout_ok={c.heldout_ok!s:5s} "
          f"hygiene={c.hygiene_fail or '-'} | {c.feedback.splitlines()[0][:90]}")
print(f"verified {len(cands)} programs in {time.time()-t:.1f}s")

for level in ("L0", "full"):
    att, rep = select_attempts(cands, task, level=level)
    solved = any(a == truth[0] for a in att[0] if a is not None)
    print(f"[{level}] report={rep} -> test solved: {solved}")

# the cheat alone: L0 trusts it (false positive), full rejects it
att, rep = select_attempts([cands[2]], task, level="L0")
print("cheat alone, L0  -> trusted:", rep["n_trusted"], "| solves test:", any(a == truth[0] for a in att[0] if a))
att, rep = select_attempts([cands[2]], task, level="full")
print("cheat alone, full-> trusted:", rep["n_trusted"], "| hygiene:", cands[2].hygiene_fail)

# ------------------------------------------------------------------ solution extraction (2026-10-01)
# Real case from the gpt-oss smoke test (eval 1818057f): a correct transform() followed by the
# model's own test code with literal demo grids. Old verifier rejected it; it must be trusted now.
# Every cheat variant below must still be caught, wherever the memorised grid hides.
lit_out = repr(outs[0]); lit_in = repr(ins[0])
CORRECT_WITH_TESTCODE = CORRECT + f'''
import os, sys                      # disallowed in the sandbox, but transform() never uses them
res = transform({lit_in})
ex_out = {lit_out}
print(res == ex_out, undefined_name)   # would raise NameError at import
'''
pairs = repr(list(zip(ins, outs)))
CHEATS = {      # each memorises EVERY demo, so it passes train and only hygiene can catch it
    "grid inside transform": f'''
def transform(grid):
    known = {pairs}
    for i, o in known:
        if i == grid:
            return o
    return [row * 3 for row in grid] * 3
''',
    "module constant read by transform": CHEAT,
    "grid inside a helper": f'''
def helper():
    return {pairs}
def transform(grid):
    for i, o in helper():
        if i == grid:
            return o
    return [row * 3 for row in grid] * 3
''',
    "grid via global statement": f'''
MEMO = {pairs}
def transform(grid):
    global MEMO
    for i, o in MEMO:
        if i == grid:
            return o
    return [row * 3 for row in grid] * 3
''',
    "lookup dict keyed by input": "LUT = {" + ", ".join(f"{repr(str(i))}: {repr(o)}" for i, o in zip(ins, outs)) + '''}
def transform(grid):
    return LUT.get(str(grid), [row * 3 for row in grid] * 3)
''',
    "helper reached via another helper": f'''
TABLE = {pairs}
def lookup(grid):
    for i, o in TABLE:
        if i == grid:
            return o
def transform(grid):
    return lookup(grid) or [row * 3 for row in grid] * 3
''',
}
print()
c = verify(Candidate(CORRECT_WITH_TESTCODE), task, timeout_s=2)
assert c.pass_train and not c.hygiene_fail, (c.pass_train, c.hygiene_fail, c.feedback)
assert "undefined_name" not in c.solution_src and "import os" not in c.solution_src
print("correct + own test code   -> trusted (test code and unused imports dropped)")
for name, src in CHEATS.items():
    c = verify(Candidate(src), task, timeout_s=2)
    assert c.pass_train, f"cheat should pass train (else the test proves nothing): {name}"
    caught = bool(c.hygiene_fail)
    print(f"{name:34s}-> pass_train={c.pass_train!s:5s} hygiene={c.hygiene_fail or '-'}")
    assert caught, f"cheat not caught: {name}"
print("ALL OK")
