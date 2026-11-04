"""
sandbox.py — run an LLM-written ARC program safely and fast.

Contract for candidate programs (what the LLM is asked to write):
    def transform(grid):            # grid: list[list[int]] (0-9), returns list[list[int]]
        ...
numpy, scipy and torch may be imported (matches the training data's tool); nothing that touches the OS,
network or files. Top-level code (e.g. the model's own print-debugging) runs, with stdout discarded.

Safety / robustness:
  * static AST screen rejects os/sys/subprocess/socket/open/eval/exec/__import__ etc.
  * each batch runs in a fresh child process with CPU-time and memory limits and a wall timeout
  * outputs are validated: rectangular, ints 0..9, 1..30 x 1..30
One child process evaluates a program on ALL grids of a task (train inputs + test inputs) in a
single call, so the fork cost is paid once per program, not once per grid.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import resource

BANNED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit",
}
ALLOWED_IMPORTS = {"numpy", "scipy", "torch", "math", "itertools", "collections", "functools", "copy", "operator",
                   "heapq", "bisect", "typing", "dataclasses", "re", "string", "statistics", "random"}


class ProgramRejected(Exception):
    pass


def static_check(src: str) -> ast.Module:
    """Parse and screen a program. Raises ProgramRejected with a reason the LLM can read."""
    try:
        compile(src, "<string>", "exec")             # same message text as the data: '... (<string>, line N)'
        tree = ast.parse(src)
    except SyntaxError as e:
        raise ProgramRejected(f"SyntaxError: {e}")
    has_transform = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ProgramRejected(f"import of '{m}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ProgramRejected(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            raise ProgramRejected(f"dunder attribute access '{node.attr}' is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "transform":
            has_transform = True
    if not has_transform:
        raise ProgramRejected("no function named 'transform' defined")
    return tree


def validate_grid(g):
    """Return a clean list[list[int]] or raise ValueError with a readable reason."""
    try:
        import numpy as np
        if isinstance(g, np.ndarray):
            g = g.tolist()
    except Exception:
        pass
    if not isinstance(g, (list, tuple)) or len(g) == 0:
        raise ValueError("output is not a non-empty list of rows")
    rows = []
    w = None
    for r in g:
        if not isinstance(r, (list, tuple)):
            raise ValueError("output rows must be lists")
        r = [int(x) for x in r]
        if w is None:
            w = len(r)
        if len(r) != w or w == 0:
            raise ValueError("output is not rectangular")
        if any(x < 0 or x > 9 for x in r):
            raise ValueError("output contains values outside 0..9")
        rows.append(r)
    if len(rows) > 30 or w > 30:
        raise ValueError(f"output is {len(rows)}x{w}, larger than 30x30")
    return rows


def _vm_size_mb() -> int:
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmSize:"):
                return int(ln.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _child(src, grids, conn, mem_mb, cpu_s):
    import io
    import os
    import sys
    sys.stdout = sys.stderr = io.StringIO()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        lim = (_vm_size_mb() + mem_mb) * 2**20       # headroom on top of what the parent already mapped
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    except Exception:
        pass
    results = []
    try:
        ns: dict = {"__name__": "candidate"}
        exec(compile(src, "<candidate>", "exec"), ns)
        fn = ns["transform"]
    except Exception as e:
        conn.send([("error", f"{type(e).__name__}: {e}"[:400])] * len(grids))
        return
    import copy
    for g in grids:
        try:
            out = fn(copy.deepcopy(g))
            results.append(("ok", validate_grid(out)))
        except Exception as e:
            results.append(("error", f"{type(e).__name__}: {e}"[:400]))
    conn.send(results)


def run_program(src: str, grids: list, timeout_s: float = 5.0, mem_mb: int = 2048):
    """Run `transform` on each grid. Returns list of ("ok", grid) | ("error", message)."""
    try:
        static_check(src)
    except ProgramRejected as e:
        return [("error", f"ProgramRejected: {e}")] * len(grids)
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(src, grids, child, mem_mb, max(1, int(timeout_s))), daemon=True)
    p.start()
    child.close()
    if parent.poll(timeout_s):
        try:
            res = parent.recv()
        except EOFError:
            res = [("error", "MemoryError: program crashed (likely memory limit)")] * len(grids)
    else:
        res = [("error", f"TimeoutError: execution exceeded {timeout_s}s")] * len(grids)
    if p.is_alive():
        p.kill()
    p.join(1)
    return res
