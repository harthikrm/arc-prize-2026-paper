"""
arc_tools.py — reproduce the Nemotron-SFT-ARC-AGI tool environment as exactly as we can.

The fine-tuned model learned from conversations with a specific system prompt, user prompt,
tool schemas and tool-response text. At inference we must send the same strings back, or the
model sees a distribution it was never trained on. Everything here was copied from real records
(sample.txt). Parts marked APPROX were not visible in the samples; check_format.py measures
how close they are on the real data and prints the true text so we can fix them.

Exports
  SYSTEM_PROMPT(tools)            exact system prompt (two variants seen in the data)
  user_prompt(task, tools, hide)  "Please solve this ARC-AGI problem: ..."
  round_message(step, rnd, R)     "📊 **Step k completed** ..."
  TOOL_SCHEMAS                    the four tool schemas, verbatim
  Executor                        runs arc_python_executor / commit / explicit_* and formats replies
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from sandbox import run_program, static_check, ProgramRejected

COLOR_NAMES = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "cyan", "brown"]  # confirmed from data

# ------------------------------------------------------------------ tool schemas (verbatim)
TOOL_SCHEMAS = {
    "arc_python_executor": {
        "name": "arc_python_executor",
        "description": "Executes Python code to transform grids. Code is tested on all training examples, then applied to test input. Use this for algorithmic solutions involving rotations, reflections, color mappings, pattern detection, or complex transformations.\n\nAVAILABLE LIBRARIES: numpy, scipy, torch\n\n\n\nCRITICAL - Avoid Overfitting:\nFind the SIMPLEST general pattern - avoid memorizing training-specific details.\nDo NOT use case-by-case logic for each example (switch-case on shapes/patterns).\nDo NOT hardcode specific grid dimensions or values from training.\nPrefer concise, general transformations over complex conditional logic.\n\nReturns pass/fail for each training example with detailed metrics and error tracebacks if failed.",
        "parameters": {"properties": {"code": {"description": "Python code with transform(grid) function in markdown format. REQUIRED: ```python\\ndef transform(grid: list[list[int]]) -> list[list[int]]: return result\\n```. The grid parameter is a 2D list of integers (0-9). Can use: numpy, scipy, torch. Example: ```python\\nimport numpy as np\\ndef transform(grid): return np.rot90(grid).tolist()\\n```", "type": "string"}},
                       "required": ["code"], "type": "object"},
    },
    "commit_solution": {
        "name": "commit_solution",
        "description": "Commit a previously generated solution as your final answer (PREFERRED method). The tool validates the solution passed ALL training examples before accepting. If validation fails, you must debug and improve your solution. Use this when a tool successfully found a pattern that works on all training examples.",
        "parameters": {"properties": {"step": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "description": "Step number to commit (e.g., 3). Use None or omit to commit the most recent step. CRITICAL: The step MUST have passed ALL training examples (100% accuracy)."}},
                       "type": "object"},
    },
    "explicit_solution_with_augmentations": {
        "name": "explicit_solution_with_augmentations",
        "description": "View your hypothesis solution under augmentations (rotations, flips, permutations) to validate generalization. The tool shows your hypothesis alongside the problem after applying transformations. Use this to check if your solution is orientation-dependent or truly generalizes. Helps identify overfitting before committing. Does NOT submit - use commit_solution after validation.",
        "parameters": {"properties": {"solution": {"description": "Your hypothesis solution grid (test output) as text. Format: rows separated by newlines, values by spaces. Example: '0 1 0\\n1 1 1\\n0 1 0'", "type": "string"},
                                      "augmentations": {"description": "List of augmentations to test under. Valid: rotate90, rotate180, rotate270, flip_horizontal, flip_vertical, transpose, permute. Empty list = random selection.", "items": {"type": "string"}, "type": "array"}},
                       "required": ["solution", "augmentations"], "type": "object"},
    },
    "explicit_solution": {
        "name": "explicit_solution",
        "description": "Validate an explicit solution grid in text format. Checks if your solution can be parsed correctly and reports its shape. Use this to verify your solution format before committing.",
        "parameters": {"properties": {"solution": {"description": "Your hypothesis solution grid (test output) as text. Format: rows separated by newlines, values by spaces. Example: '0 1 0\\n1 1 1\\n0 1 0'", "type": "string"}},
                       "required": ["solution"], "type": "object"},
    },
}

# ------------------------------------------------------------------ prompts (verbatim from data)
_HEAD = """You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles.
Your goal is to analyze the training examples, identify the pattern, and generate the correct output for the test input.

**Response Format:**
IMPORTANT: Before each tool call, provide your reasoning/planning in approximately 2 sentences as part of your visible message content.
This improves explainability and helps others understand your reasoning.
In your responses, you MUST call one of the available tools.
Pure reasoning without a tool call will be rejected with an error.

**Strategy:**
1. Analyze the training examples to understand the transformation pattern
"""
_METRICS = """**Evaluation Metrics:**
  - exact_match: Perfect match required (binary: 0 or 1). Primary goal.
  - cell_accuracy: Percentage of cells matching exactly. High (>0.8) = mostly correct, low (<0.3) = mostly wrong.
  - color_permutation_match: Best match after remapping colors. High (near 1.0) with low cell_accuracy = correct structure, wrong color mapping.
  - spatial_offset_match: Best match score when shifting hypothesis by up to 5 cells. High when pattern is correct but positioned wrong.
  - structural_similarity: Similarity of high-level features (object count, sizes, shapes). High = compositionally similar, ignoring exact colors/positions.

"""
_RULES_INDUCTIVE = """- INDUCTIVE TOOLS (code/rules-based): You MUST pass ALL training examples before committing
  - These tools generate patterns/rules that transform inputs to outputs
  - `commit_solution` will be REJECTED if the solution didn't pass 100% of training examples
  - If commit is rejected, debug the failures and iterate with improved solutions
"""


def SYSTEM_PROMPT(first_round_tools: list) -> str:
    """Variant A: executor only. Variant B: executor + a hypothesis tool."""
    hyp = [t for t in first_round_tools if t.startswith("explicit_solution")]
    if not hyp:
        return (_HEAD
                + "2. Use arc_python_executor to test hypotheses and generate candidate solutions\n"
                  "3. Iterate based on feedback until you find the correct pattern\n"
                  "4. When confident and training passes, call `commit_solution` to finalize your answer\n\n\n"
                  "**Important Rules:**\n" + _RULES_INDUCTIVE
                + "- WARNING: Passing training does NOT guarantee correctness on the test\n"
                  "- Before committing, ask: Is my pattern unambiguous? Could other patterns also fit?\n"
                  "- Find the SIMPLEST pattern that works. Avoid memorizing training examples.\n"
                  "- Commit when you have HIGH confidence, not just training success\n"
                  "- You have a limited budget of rounds - use them wisely but don't waste them\n"
                  "- If stuck, try a completely different algorithmic approach\n"
                  "- Iterate systematically - if one strategy fails, analyze why before trying another\n"
                  "- ONLY use tools that are listed as available in the current round\n\n"
                + _METRICS)
    return (_HEAD
            + f"2. Use tools: {first_round_tools} to test hypotheses and generate candidate solutions\n"
              "3. Iterate based on feedback until you find the correct pattern\n"
              "4. When confident and training passes, call `commit_solution` to finalize your answer\n"
              f"5. If all approaches fail after exhausting budget, use `{hyp[0]}` as final attempt\n\n\n"
              "**Important Rules:**\n" + _RULES_INDUCTIVE
            + "- HYPOTHESIS TOOLS (direct grid solutions): No training validation required\n"
              "  - These tools let you submit direct output grids as your hypothesis\n"
              "  - Use when inductive approaches fail or for final attempts\n"
              "- WARNING: Passing training does NOT guarantee correctness on the test\n"
              "- Before committing, ask: Is my pattern unambiguous? Could other patterns also fit?\n"
              "- Find the SIMPLEST pattern that works. Avoid memorizing training examples.\n"
              "- Commit when you have HIGH confidence, not just training success\n"
              "- You have a limited budget of rounds - use them wisely but don't waste them\n"
              "- If stuck, try a different strategy or different tool type\n"
              "- Don't use the same tool more than twice in a row, be sure to explore different tools\n"
              "- ONLY use tools that are listed as available in the current round\n\n"
            + _METRICS)


def grid_text(g) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in g)


# APPROX: the test-input header was cut off in every sample; check_format.py prints the real one.
TEST_HEADER = "Test Input:\n"


def user_prompt(task, first_round_tools: list, pairs: list | None = None) -> str:
    pairs = task["train"] if pairs is None else pairs
    s = f"Please solve this ARC-AGI problem:\n**Allowed Tools:** {first_round_tools}\n\n"
    for i, p in enumerate(pairs, 1):
        s += f"Train Example {i}:\n\nInput:\n{grid_text(p['input'])}\n\nOutput:\n{grid_text(p['output'])}\n\n"
    for j, t in enumerate(task["test"]):
        hdr = TEST_HEADER if len(task["test"]) == 1 else TEST_HEADER.replace("Input", f"Input {j + 1}")
        s += hdr + grid_text(t["input"]) + "\n\n"
    return s.rstrip("\n") + "\n"


def round_message(step: int, next_round: int, total_rounds: int, allowed="All tools") -> str:
    return (f"📊 **Step {step} completed**\n\nRound {next_round}/{total_rounds} starting. "
            f"Remaining rounds: {total_rounds - next_round + 1}\n**Allowed Tools:** {allowed}\n"
            "Continue reasoning and testing, or use commit_solution when confident.")


NO_TOOL_MESSAGE = "❌ Error: You MUST call one of the available tools. Pure reasoning without a tool call is rejected."  # APPROX

# ------------------------------------------------------------------ metrics (format verbatim, maths APPROX)
def _shape(g):
    return (len(g), len(g[0]))


def _fmt_shape(s):
    return f"({s[0]}, {s[1]})"


def spatial_offset(gen, exp, max_off=5):
    """Fraction of expected cells matched when the hypothesis is shifted by (dr, dc), with cells shifted
    in from outside filled with 0. Strict '>' in ascending order reproduces the data's tie-breaking.
    Verified on two real samples (66.7% at (-1,-1); 57.1% at (-1,-5) from 47.6%)."""
    G, E = np.array(gen), np.array(exp)
    eh, ew = E.shape
    gh, gw = G.shape

    def score(dr, dc):
        m = 0
        for i in range(eh):
            gi = i - dr
            for j in range(ew):
                gj = j - dc
                v = G[gi, gj] if (0 <= gi < gh and 0 <= gj < gw) else 0
                m += int(v == E[i, j])
        return m / (eh * ew)

    base = score(0, 0)
    best, off = -1.0, (0, 0)
    for dr in range(-max_off, max_off + 1):
        for dc in range(-max_off, max_off + 1):
            s = score(dr, dc)
            if s > best:
                best, off = s, (dr, dc)
    return best, off, base


def color_permutation(gen, exp):
    """Best one-to-one recolouring of the hypothesis. Reproduces the data on 84/84 real cases:
    if the two grids use <= 6 colours in total, brute-force every permutation of those colours (first
    best in itertools order); otherwise an assignment over the hypothesis colours that prefers keeping
    a colour, avoids mapping to 0, and prefers lower colours on ties. Lists only changed colours."""
    import itertools
    from scipy.optimize import linear_sum_assignment
    G, E = np.array(gen), np.array(exp)
    C = np.zeros((10, 10), dtype=int)
    for a, b in zip(G.ravel(), E.ravel()):
        C[a, b] += 1
    gset, eset = set(G.ravel().tolist()), set(E.ravel().tolist())
    u = sorted(gset | eset)
    if len(u) <= 6:
        best, bp = -1, None
        for p in itertools.permutations(u):
            sc = sum(C[x, y] for x, y in zip(u, p))
            if sc > best:
                best, bp = sc, p
        mapping = dict(zip(u, bp))
    else:
        rows = sorted(gset)
        W = np.array([[C[x, y] * 1000 + 10 * (x == y) - 5 * (y == 0) - 0.01 * y for y in range(10)] for x in rows])
        r, c = linear_sum_assignment(-W)
        mapping = {rows[i]: int(j) for i, j in zip(r, c)}
    score = sum(C[x, y] for x, y in mapping.items()) / G.size
    pairs = [(COLOR_NAMES[x], COLOR_NAMES[y]) for x, y in sorted(mapping.items()) if x != y]
    return score, pairs


def spatial_too_different(sg, se) -> bool:
    """Fits all 18 shape-mismatch cases seen: refused when a side differs by more than 10 cells
    ((14, 8) vs (28, 20) refused; (8, 10) vs (16, 20) and (3, 9) vs (3, 2) scored)."""
    return abs(sg[0] - se[0]) > 10 or abs(sg[1] - se[1]) > 10


def _n_objects(A, conn, bg_zero):
    from scipy.ndimage import label
    bg = 0 if bg_zero else Counter(A.ravel().tolist()).most_common(1)[0][0]
    st = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if conn == 4 else np.ones((3, 3), int)
    return label(A != bg, structure=st)[1]


def structural_similarity(gen, exp):
    """APPROX (formula unknown). Best fit found on 107 real cases (mean abs error 0.033, exact on 50):
    mean of object-count ratios (4-conn bg 0, 4-conn bg = most common colour, 8-conn bg 0) and the
    Jaccard overlap of the colour sets."""
    G, E = np.array(gen), np.array(exp)
    r = lambda a, b: 1.0 if a == b else min(a, b) / max(a, b)
    f = [r(_n_objects(G, 4, True), _n_objects(E, 4, True)),
         r(_n_objects(G, 4, False), _n_objects(E, 4, False)),
         r(_n_objects(G, 8, True), _n_objects(E, 8, True))]
    cg, ce = set(G.ravel().tolist()), set(E.ravel().tolist())
    f.append(len(cg & ce) / len(cg | ce))
    return sum(f) / 4


def metric_lines(gen, exp) -> list:
    sg, se = _shape(gen), _shape(exp)
    out = []
    if spatial_too_different(sg, se):
        so, so_txt = 0.0, f"Shapes too different for spatial matching: {_fmt_shape(se)} vs {_fmt_shape(sg)}"
    else:
        so, off, base = spatial_offset(gen, exp)
        so_txt = f"Best match at offset ({off[0]}, {off[1]}): {so * 100:.1f}%"
        if off != (0, 0):
            so_txt += f" (improved from {base * 100:.1f}%)"
    if sg != se:
        msg = f"Boards have different shapes: {_fmt_shape(se)} vs {_fmt_shape(sg)}"
        out.append(f"  exact_match: 0.000 - {msg}")
        out.append(f"  cell_accuracy: 0.000 - {msg}")
        out.append(f"  spatial_offset_match: {so:.3f} - {so_txt}")
        out.append(f"  color_permutation_match: 0.000 - {msg}")
    else:
        n = se[0] * se[1]
        ok = int((np.array(gen) == np.array(exp)).sum())
        out.append(f"  exact_match: 0.000 - Boards differ in {n - ok} out of {n} cells")
        out.append(f"  cell_accuracy: {ok / n:.3f} - Cell accuracy: {ok}/{n} cells correct ({ok / n * 100:.1f}%)")
        out.append(f"  spatial_offset_match: {so:.3f} - {so_txt}")
        cp, pairs = color_permutation(gen, exp)
        if pairs:
            out.append(f"  color_permutation_match: {cp:.3f} - Best match with color remapping: {cp * 100:.1f}%. Remapped: {pairs}")
        else:
            out.append(f"  color_permutation_match: {cp:.3f} - No color remapping needed: {cp * 100:.1f}%")
    ss = structural_similarity(gen, exp)
    out.append(f"  structural_similarity: {ss:.3f} - Structural similarity: {ss * 100:.1f}%")
    return out


# ------------------------------------------------------------------ code extraction
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
EXTRACT_FAIL = "❌ Failed to extract valid content from input"


def extract_code(arg) -> str | None:
    code = arg.get("code") if isinstance(arg, dict) else arg
    if not isinstance(code, str):
        return None
    blocks = FENCE.findall(code)
    if blocks:
        code = max(blocks, key=lambda b: ("def transform" in b, len(b)))
    if not re.search(r"^\s*def\s+transform\s*\(", code, re.M):
        return None
    return code


def parse_grid_text(s):
    try:
        rows = [[int(x) for x in re.split(r"[\s,]+", ln.strip()) if x != ""] for ln in str(s).strip().splitlines() if ln.strip()]
        if rows and all(len(r) == len(rows[0]) for r in rows) and all(0 <= x <= 9 for r in rows for x in r):
            return rows
    except ValueError:
        pass
    return None


def _augment(g, aug):
    A = np.array(g)
    return {"rotate90": np.rot90(A, -1), "rotate180": np.rot90(A, 2), "rotate270": np.rot90(A, 1),
            "flip_horizontal": A[:, ::-1], "flip_vertical": A[::-1, :], "transpose": A.T}.get(aug, A).tolist()


# ------------------------------------------------------------------ executor
class Executor:
    """One instance per conversation. `pairs` are the demonstration pairs the model can see
    (a leave-one-out run hides one); steps are numbered like the data (1 per tool call)."""

    def __init__(self, task, pairs=None, timeout_s=5.0):
        self.task = task
        self.pairs = task["train"] if pairs is None else pairs
        self.timeout_s = timeout_s
        self.steps = {}           # step -> dict(tool, code, passed, n, test_outputs, grid)
        self.committed = None     # step number

    def __call__(self, step: int, name: str, args) -> tuple[str, bool]:
        """Returns (tool_response_text, finished)."""
        if name == "arc_python_executor":
            return self.run_code(step, args), False
        if name == "commit_solution":
            return self.commit(step, args)
        if name in ("explicit_solution", "explicit_solution_with_augmentations"):
            return self.explicit(step, name, args), False
        self.steps[step] = {"tool": name}
        return f"❌ Unknown tool: {name}", False   # APPROX

    def run_code(self, step, args) -> str:
        code = extract_code(args)
        rec = {"tool": "arc_python_executor", "code": code, "passed": 0, "n": len(self.pairs), "test_outputs": []}
        self.steps[step] = rec
        if code is None:
            return EXTRACT_FAIL
        try:
            static_check(code)
            rejected = None
        except ProgramRejected as e:
            rejected = str(e)
        grids = [p["input"] for p in self.pairs] + [t["input"] for t in self.task["test"]]
        res = ([("error", rejected)] * len(grids) if rejected
               else run_program(code, grids, timeout_s=self.timeout_s))
        n = len(self.pairs)
        tr, te = res[:n], res[n:]
        rec["test_outputs"] = te
        lines = []
        passed = 0
        for k, ((st, out), p) in enumerate(zip(tr, self.pairs), 1):
            if st != "ok":
                lines.append(f"Example {k}: ❌ ERROR - {_err(out)}")
                lines.append("  Traceback:")
                lines.append(_err(out))
                if str(out).startswith("SyntaxError"):   # data shows a blank line after syntax-error blocks
                    lines.append("")
            elif out == p["output"]:
                passed += 1
                lines.append(f"Example {k}: ✅ PASS (shape: {_fmt_shape(_shape(out))})")
            else:
                lines.append(f"Example {k}: ❌ FAIL")
                lines += metric_lines(out, p["output"])
                lines.append("  Generated output:")
                lines.append(grid_text(out))
                lines.append("")
        rec["passed"] = passed
        head = (f"🎉 All {n} training examples PASSED!" if passed == n
                else f"Results: {passed}/{n} training examples passed")
        body = head + "\n" + "\n".join(lines)
        for j, (st, out) in enumerate(te):
            label = "Test input execution:" if len(te) == 1 else f"Test input {j + 1} execution:"
            if st == "ok":
                body += f"\n\n{label}\n  Generated output (shape: {_fmt_shape(_shape(out))}):\n{grid_text(out)}\n"
            else:
                body += f"\n\n{label} ❌ ERROR - {_err(out)}\n  Traceback:\n{_err(out)}"
        return body

    def commit(self, step, args) -> tuple[str, bool]:
        want = args.get("step") if isinstance(args, dict) else None
        code_steps = [s for s, r in self.steps.items() if r.get("tool") == "arc_python_executor"]
        target = want if isinstance(want, int) else (max(code_steps) if code_steps else None)
        self.steps[step] = {"tool": "commit_solution", "target": target}
        r = self.steps.get(target)
        if r is None or r.get("tool") != "arc_python_executor":   # APPROX wording below
            return f"❌ Commit rejected: step {target} is not a code solution. Commit a step that ran arc_python_executor.", False
        if r["passed"] != r["n"]:
            return (f"❌ Commit rejected: step {target} passed {r['passed']}/{r['n']} training examples. "
                    "The solution must pass ALL training examples before committing."), False
        self.committed = target
        return f"✅ Solution from step {target} committed (passed {r['n']}/{r['n']} training examples).", True

    def explicit(self, step, name, args) -> str:
        g = parse_grid_text((args or {}).get("solution", "")) if isinstance(args, dict) else None
        self.steps[step] = {"tool": name, "grid": g}
        if g is None:
            return EXTRACT_FAIL
        txt = f"✅ Solution parsed successfully (shape: {_fmt_shape(_shape(g))})\n{grid_text(g)}"   # APPROX
        if name == "explicit_solution_with_augmentations":
            augs = [a for a in (args.get("augmentations") or []) if a != "permute"] or ["rotate90", "flip_horizontal"]
            ti = self.task["test"][0]["input"]
            for a in augs[:3]:
                txt += f"\n\n[{a}] Test input:\n{grid_text(_augment(ti, a))}\n[{a}] Your solution:\n{grid_text(_augment(g, a))}"
        return txt

    # ---------- results for the verifier
    def programs(self):
        """Every distinct program the model wrote, with its step and whether it was committed."""
        seen, out = set(), []
        for s, r in sorted(self.steps.items()):
            if r.get("tool") == "arc_python_executor" and r.get("code") and r["code"] not in seen:
                seen.add(r["code"])
                out.append({"step": s, "code": r["code"], "passed": r["passed"], "n": r["n"],
                            "committed": s == self.committed})
        return out

    def direct_grids(self):
        return [r["grid"] for s, r in sorted(self.steps.items())
                if r.get("tool", "").startswith("explicit_solution") and r.get("grid")]


def _err(msg: str) -> str:
    """The data wraps every error as 'ValueError: <OriginalType>: <msg>' (checked on 300 real conversations)."""
    return f"ValueError: {msg}"
