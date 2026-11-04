"""Offline end-to-end test of solver.py with a scripted 'model' (no GPU, no server)."""
import json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver
from arc_tools import user_prompt, SYSTEM_PROMPT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
train = json.load(open(os.path.join(ROOT, "data", "arc-agi_training_challenges.json")))
sols = json.load(open(os.path.join(ROOT, "data", "arc-agi_training_solutions.json")))
TID = "007bbfb7"
WRONG = "```python\nimport numpy as np\ndef transform(grid):\n    return np.kron(np.array(grid), np.ones((3,3),int)).tolist()\n```"
RIGHT = ("```python\nimport numpy as np\ndef transform(grid):\n    g = np.array(grid)\n"
         "    return np.kron((g != 0).astype(int), g).tolist()\n\nprint(transform([[1,0],[0,1]]))\n```")
CRASH = "def transform(grid):\n    import numpy as np\n    return np.vstack([np.zeros((2,3)), np.zeros((2,2))]).tolist()\n"

class Scripted:
    """Turn 1: crashing code, turn 2: wrong idea, turn 3: right idea, turn 4: commit."""
    def chat(self, messages, tools, seed=None):
        n = sum(m["role"] == "assistant" for m in messages)
        script = [("arc_python_executor", {"code": CRASH}), ("arc_python_executor", {"code": WRONG}),
                  ("arc_python_executor", {"code": RIGHT}), ("commit_solution", {"step": 3})]
        name, a = script[min(n, 3)]
        return {"content": "Reasoning in two sentences. Calling the tool.", "reasoning": "think",
                "tool_calls": [{"id": f"c{n}", "name": name, "arguments": a}], "tokens": 10, "prompt_tokens": 0}

ex, log, stats = solver.run_conversation(train[TID], Scripted(), rounds=5)
for m in log[2:]:
    print(f"--- {m['role']}")
    print((m.get("content") or "")[:1400])
print("STATS", stats, "| committed step", ex.committed)
assert stats["stop"] == "committed" and ex.committed == 3

# full task pipeline incl. leave-one-out + verifier + attempt selection
args = types.SimpleNamespace(samples=4, rounds=5, loo_frac=0.5, hypothesis_variant=False, parallel_per_task=4,
                             seed=0, exec_timeout=5.0, verifier_level="full")
attempts, report, cands, trs = solver.solve_task(TID, train[TID], Scripted(), args)
print("REPORT", report)
assert attempts[0][0] == sols[TID][0], "attempt 1 should be the verified program's output"
print("heldouts used:", [t["heldout"] for t in trs])
# inline tool-call fallback parser
print(solver.parse_inline_tool_calls('<tool_call>{"name": "commit_solution", "arguments": {"step": 2}}</tool_call>'))
print(solver.parse_inline_tool_calls("text\n" + RIGHT)[0]["name"])
print("ALL OK")
