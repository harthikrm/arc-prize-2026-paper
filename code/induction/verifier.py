"""
verifier.py — decide which LLM-written programs to trust, and turn them into 2 attempts.

Why this exists (the research question): a refinement loop is an agent being graded by an
evaluator (the demonstration pairs). A program can satisfy the evaluator without having learned
the rule — e.g. it hard-codes the demo outputs, keys off grid sizes, or memorises pairs. Plain
"passes all train pairs" therefore has a FALSE-POSITIVE rate: programs that pass every demo and
still fail the test. This module measures and reduces that rate with cheap, label-free checks.

Levels (all computed per candidate, logged, and ablatable):
  L0  pass_train     : exact match on every demonstration pair
  L1  heldout_ok     : when the candidate was GENERATED without seeing pair k (leave-one-out
                       prompting), it still solves pair k. Strong evidence of generalisation.
  L2  hygiene        : no train-output grid (or large literal grid) embedded in the solution
                       (transform() and what it reaches; the model's own test code is excluded);
                       output not constant across different inputs (unless all train outputs are
                       identical); test output valid and its shape consistent with the train
                       pattern (same-shape tasks stay same-shape; fixed-shape tasks keep the shape)
  L3  consensus      : independent candidates that pass L0+L2 vote with their TEST outputs;
                       agreement across differently-written programs is evidence the rule is real
Ranking: L2 is a hard filter; among survivors sort by (heldout_ok, vote count, -source length).
"""
from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field

from sandbox import run_program


def _shape(g):
    return (len(g), len(g[0]))


def _key(g):
    return hashlib.md5(json.dumps(g).encode()).hexdigest()


@dataclass
class Candidate:
    src: str
    heldout_index: int | None = None          # which train pair was hidden at generation time
    meta: dict = field(default_factory=dict)  # model, temperature, refinement round, ...
    # filled by verify():
    train_results: list = field(default_factory=list)
    test_outputs: list = field(default_factory=list)
    pass_train: bool = False
    heldout_ok: bool | None = None
    hygiene_fail: list = field(default_factory=list)
    feedback: str = ""                         # what the refinement loop shows the LLM next
    solution_src: str = ""                     # transform() + what it reaches; what is verified


# ------------------------------------------------------------------ solution extraction
def solution_source(src: str, entry: str = "transform") -> str:
    """Keep only what `transform` can reach: the imports it uses, module-level functions/classes
    and assignments it references (transitively). Drops the model's own test code (literal demo
    grids, prints, comparisons) and unused imports, which otherwise (a) trip the literal-grid
    hygiene check and (b) can crash the whole module (NameError, disallowed import) even though
    transform() itself is correct. Returns src unchanged if it can't be parsed or has no entry."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    defs, assigns, imports = {}, [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            assigns.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
    if entry not in defs:
        return src

    def target_names(node):
        tg = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in tg for n in ast.walk(t) if isinstance(n, ast.Name)}

    def used_names(node):
        """Names `node` reads from module scope: loads minus names bound locally (params,
        assignment / loop / with / comprehension targets, nested defs). Without this, a local
        `res` inside transform() would drag in a top-level `res = transform(test_grid)`."""
        loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return loads
        local = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = n.args
                local |= {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
                local |= {x.arg for x in (a.vararg, a.kwarg) if x}
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
                local.add(n.name)
            if isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                local -= set(n.names)                    # explicitly module-level: keep as a dependency
        return loads - local

    keep, need, todo = set(), set(), [defs[entry]]
    while todo:
        node = todo.pop()
        if id(node) in keep:
            continue
        keep.add(id(node))
        for name in used_names(node) - need:
            need.add(name)
            if name in defs:
                todo.append(defs[name])
            todo.extend(a for a in assigns if name in target_names(a))

    def import_used(node):
        names = [(a.asname or a.name).split(".")[0] for a in node.names]
        return any(n in need for n in names)

    body = [n for n in tree.body
            if id(n) in keep or (isinstance(n, (ast.Import, ast.ImportFrom)) and import_used(n))]
    try:
        return ast.unparse(ast.Module(body=body, type_ignores=[]))
    except Exception:
        return src


# ------------------------------------------------------------------ hygiene checks (L2)
def literal_leaks(src: str, train_outputs: list, min_cells: int = 6) -> list:
    """Find nested-list literals in the source that equal a train output or are big grids."""
    issues = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return issues
    outs = {_key(o) for o in train_outputs}
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts and all(isinstance(e, ast.List) for e in node.elts):
            try:
                val = ast.literal_eval(node)
            except Exception:
                continue
            if isinstance(val, list) and val and all(isinstance(r, list) for r in val):
                if _key(val) in outs:
                    issues.append("source embeds a train output grid verbatim")
                elif sum(len(r) for r in val) >= min_cells * 4:
                    issues.append(f"source embeds a large literal grid ({len(val)} rows)")
    return list(dict.fromkeys(issues))


def shape_rule(task) -> str:
    pairs = task["train"]
    if all(_shape(p["input"]) == _shape(p["output"]) for p in pairs):
        return "same"
    outs = {_shape(p["output"]) for p in pairs}
    if len(outs) == 1:
        return "fixed"
    return "free"


def hygiene(cand: Candidate, task) -> list:
    issues = literal_leaks(cand.solution_src or cand.src, [p["output"] for p in task["train"]])
    train_outs = [r[1] for r in cand.train_results if r[0] == "ok"]
    distinct_inputs = len({_key(p["input"]) for p in task["train"]}) > 1
    distinct_truth = len({_key(p["output"]) for p in task["train"]}) > 1
    if distinct_inputs and distinct_truth and len({_key(o) for o in train_outs}) == 1 and len(train_outs) > 1:
        issues.append("output does not depend on the input")
    rule = shape_rule(task)
    fixed = _shape(task["train"][0]["output"])
    for i, (status, out) in enumerate(cand.test_outputs):
        if status != "ok":
            issues.append(f"fails on test input {i}: {out}")
            continue
        tin = task["test"][i]["input"]
        if rule == "same" and _shape(out) != _shape(tin):
            issues.append(f"test output {i} shape {_shape(out)} breaks the same-shape pattern of the demos")
        if rule == "fixed" and _shape(out) != fixed:
            issues.append(f"test output {i} shape {_shape(out)} differs from the fixed demo output shape {fixed}")
    return list(dict.fromkeys(issues))


# ------------------------------------------------------------------ L0/L1 + feedback
def diff_feedback(pred, truth, max_cells: int = 12) -> str:
    if _shape(pred) != _shape(truth):
        return f"wrong shape: produced {_shape(pred)}, expected {_shape(truth)}"
    wrong = [(i, j, pred[i][j], truth[i][j]) for i in range(len(truth)) for j in range(len(truth[0]))
             if pred[i][j] != truth[i][j]]
    head = ", ".join(f"({i},{j}) got {a} want {b}" for i, j, a, b in wrong[:max_cells])
    more = f" ... +{len(wrong) - max_cells} more" if len(wrong) > max_cells else ""
    return f"{len(wrong)} wrong cells of {len(truth) * len(truth[0])}: {head}{more}"


def verify(cand: Candidate, task, timeout_s: float = 5.0) -> Candidate:
    grids = [p["input"] for p in task["train"]] + [t["input"] for t in task["test"]]
    cand.solution_src = solution_source(cand.src)
    res = run_program(cand.solution_src, grids, timeout_s=timeout_s)
    n = len(task["train"])
    cand.train_results, cand.test_outputs = res[:n], res[n:]
    fb = []
    ok = True
    for k, ((status, out), pair) in enumerate(zip(cand.train_results, task["train"])):
        if status != "ok":
            ok = False
            fb.append(f"pair {k}: error {out}")
        elif out != pair["output"]:
            ok = False
            fb.append(f"pair {k}: {diff_feedback(out, pair['output'])}")
    cand.pass_train = ok
    if cand.heldout_index is not None:
        st, out = cand.train_results[cand.heldout_index]
        cand.heldout_ok = st == "ok" and out == task["train"][cand.heldout_index]["output"]
    cand.hygiene_fail = hygiene(cand, task) if ok else []
    cand.feedback = "all demonstration pairs correct" if ok else "\n".join(fb)
    if ok and cand.hygiene_fail:
        cand.feedback += "\nbut rejected: " + "; ".join(cand.hygiene_fail)
    return cand


# ------------------------------------------------------------------ L3: consensus + attempts
def select_attempts(cands: list, task, level: str = "full"):
    """
    level: "L0"   -> trust anything that passes train (baseline verifier, for ablation)
           "full" -> L0 + L2 filter, rank by heldout_ok, consensus votes, brevity
    Returns (attempts, report) where attempts[i] = [grid_or_None, grid_or_None] per test input.
    """
    ok = [c for c in cands if c.pass_train and (level == "L0" or not c.hygiene_fail)]
    attempts = []
    for i in range(len(task["test"])):
        buckets: dict = {}
        for c in ok:
            st, out = c.test_outputs[i]
            if st != "ok":
                continue
            b = buckets.setdefault(_key(out), {"grid": out, "votes": 0, "heldout": 0, "minlen": 10**9})
            b["votes"] += 1
            b["heldout"] += int(bool(c.heldout_ok))
            b["minlen"] = min(b["minlen"], len(c.solution_src or c.src))
        if level == "L0":
            ranked = sorted(buckets.values(), key=lambda b: b["minlen"])
        else:
            ranked = sorted(buckets.values(), key=lambda b: (-b["heldout"], -b["votes"], b["minlen"]))
        top = [b["grid"] for b in ranked[:2]]
        attempts.append(top + [None] * (2 - len(top)))
    report = {
        "n_candidates": len(cands),
        "n_pass_train": sum(c.pass_train for c in cands),
        "n_trusted": len(ok),
        "n_hygiene_rejected": sum(1 for c in cands if c.pass_train and c.hygiene_fail),
        "n_heldout_ok": sum(1 for c in cands if c.heldout_ok),
    }
    return attempts, report
