#!/usr/bin/env python3
"""
arc_eval.py — local evaluation harness for ARC Prize 2026 (ARC-AGI-2).

What it does
  * Scores a submission.json exactly like Kaggle: per test output, 1 if attempt_1 OR
    attempt_2 matches the ground truth cell-for-cell, else 0; mean over all test outputs.
  * Validates submission format (every task id present, every test input covered,
    both attempts present, grids rectangular with ints 0-9).
  * Failure taxonomy per test output:
        solved | wrong_shape | right_shape_wrong_content(near miss %) | missing
  * Task-level features (grid size, #colors, #train pairs, same-shape-io) so you can
    see WHERE a solver fails, not just how often.
  * Detects the Kaggle placeholder test file (240 training tasks) so you never
    "validate" on it by accident.
  * Optional: compare two submissions (A vs B) -> tasks solved only by A, only by B,
    both (this is how you measure induction/transduction complementarity later).

Usage
  python arc_eval.py --challenges arc-agi_evaluation_challenges.json \
                     --solutions  arc-agi_evaluation_solutions.json \
                     --submission submission.json [--csv out.csv] [--compare other.json]

  python arc_eval.py --check-placeholder arc-agi_test_challenges.json arc-agi_training_challenges.json
"""
import argparse
import csv
import json
import sys
from collections import Counter


# ----------------------------------------------------------------------------- helpers
def load(path):
    with open(path) as f:
        return json.load(f)


def is_grid(g):
    return (
        isinstance(g, list)
        and len(g) >= 1
        and all(isinstance(r, list) and len(r) == len(g[0]) and len(r) >= 1 for r in g)
        and all(isinstance(c, int) and 0 <= c <= 9 for r in g for c in r)
    )


def shape(g):
    return (len(g), len(g[0]))


def cell_match_pct(a, b):
    """% of cells equal, only meaningful when shapes match."""
    if shape(a) != shape(b):
        return 0.0
    h, w = shape(a)
    eq = sum(1 for i in range(h) for j in range(w) if a[i][j] == b[i][j])
    return 100.0 * eq / (h * w)


def task_features(task):
    train = task["train"]
    cells_in = [len(p["input"]) * len(p["input"][0]) for p in train]
    cells_out = [len(p["output"]) * len(p["output"][0]) for p in train]
    same_shape = all(shape(p["input"]) == shape(p["output"]) for p in train)
    colors = set()
    for p in train:
        colors |= {c for r in p["input"] for c in r} | {c for r in p["output"] for c in r}
    return {
        "n_train": len(train),
        "n_test": len(task["test"]),
        "in_cells_med": sorted(cells_in)[len(cells_in) // 2],
        "out_cells_med": sorted(cells_out)[len(cells_out) // 2],
        "max_cells": max(cells_in + cells_out),
        "n_colors": len(colors),
        "same_shape_io": same_shape,
    }


# ----------------------------------------------------------------------------- validation
def validate_submission(challenges, submission):
    problems = []
    for tid, task in challenges.items():
        if tid not in submission:
            problems.append(f"{tid}: missing from submission")
            continue
        preds = submission[tid]
        if not isinstance(preds, list) or len(preds) != len(task["test"]):
            problems.append(f"{tid}: expected {len(task['test'])} test predictions, got "
                            f"{len(preds) if isinstance(preds, list) else type(preds)}")
            continue
        for k, p in enumerate(preds):
            for key in ("attempt_1", "attempt_2"):
                if key not in p:
                    problems.append(f"{tid}[{k}]: missing {key}")
                elif not is_grid(p[key]):
                    problems.append(f"{tid}[{k}]: {key} is not a valid grid")
    return problems


# ----------------------------------------------------------------------------- scoring
def score_submission(challenges, solutions, submission):
    rows = []
    for tid, task in challenges.items():
        feats = task_features(task)
        for k, truth in enumerate(solutions[tid]):
            row = {"task": tid, "test_idx": k, **feats,
                   "truth_shape": f"{shape(truth)[0]}x{shape(truth)[1]}"}
            preds = submission.get(tid)
            if not preds or k >= len(preds):
                row.update(status="missing", score=0, best_cell_pct=0.0)
                rows.append(row)
                continue
            a1, a2 = preds[k].get("attempt_1"), preds[k].get("attempt_2")
            attempts = [g for g in (a1, a2) if is_grid(g)]
            if any(g == truth for g in attempts):
                row.update(status="solved", score=1, best_cell_pct=100.0)
            elif any(shape(g) == shape(truth) for g in attempts):
                best = max(cell_match_pct(g, truth) for g in attempts)
                row.update(status="wrong_content", score=0, best_cell_pct=round(best, 1))
            elif attempts:
                row.update(status="wrong_shape", score=0, best_cell_pct=0.0)
            else:
                row.update(status="missing", score=0, best_cell_pct=0.0)
            row["identical_attempts"] = (a1 == a2)
            rows.append(row)
    return rows


def summarize(rows, label="submission"):
    n = len(rows)
    solved = sum(r["score"] for r in rows)
    status = Counter(r["status"] for r in rows)
    print(f"\n=== {label}: {solved}/{n} test outputs = {100*solved/n:.2f}% (Kaggle metric, pass@2) ===")
    print("  status breakdown :", dict(status))
    near = [r for r in rows if r["status"] == "wrong_content" and r["best_cell_pct"] >= 95]
    print(f"  near misses (>=95% cells right, shape right): {len(near)}  <- 'one wrong pixel' failures")
    dup = sum(1 for r in rows if r.get("identical_attempts"))
    print(f"  identical attempt_1/attempt_2: {dup}/{n}  <- wasted second guesses")
    # by task features
    def bucket(v, edges):
        for e in edges:
            if v <= e:
                return f"<={e}"
        return f">{edges[-1]}"
    for name, key, edges in [("max grid cells", "max_cells", [100, 300, 600]),
                             ("n_colors", "n_colors", [3, 5, 7]),
                             ("n_train pairs", "n_train", [2, 3, 4])]:
        c = Counter(); s = Counter()
        for r in rows:
            b = bucket(r[key], edges); c[b] += 1; s[b] += r["score"]
        parts = [f"{b}: {s[b]}/{c[b]} ({100*s[b]/c[b]:.0f}%)" for b in sorted(c, key=lambda x: (x[0] == '>', x))]
        print(f"  by {name:14s}: " + " | ".join(parts))
    c = Counter(); s = Counter()
    for r in rows:
        c[r["same_shape_io"]] += 1; s[r["same_shape_io"]] += r["score"]
    print("  same-shape I/O tasks:", f"{s[True]}/{c[True]}", "| shape-changing tasks:", f"{s[False]}/{c[False]}")


def compare(rows_a, rows_b, la="A", lb="B"):
    sa = {(r["task"], r["test_idx"]) for r in rows_a if r["score"]}
    sb = {(r["task"], r["test_idx"]) for r in rows_b if r["score"]}
    print(f"\n=== complementarity {la} vs {lb} ===")
    print(f"  only {la}: {len(sa - sb)}   only {lb}: {len(sb - sa)}   both: {len(sa & sb)}   union: {len(sa | sb)}")
    print(f"  union score = {100*len(sa|sb)/len(rows_a):.2f}%  (upper bound if arbitration were perfect)")
    if sa - sb:
        print(f"  tasks only {la} solved:", sorted(t for t, _ in (sa - sb))[:30])
    if sb - sa:
        print(f"  tasks only {lb} solved:", sorted(t for t, _ in (sb - sa))[:30])


def check_placeholder(test_path, train_path, eval_path=None):
    test = load(test_path); train = load(train_path)
    overlap = len(set(test) & set(train))
    print(f"test file: {len(test)} tasks; overlap with TRAINING ids: {overlap}")
    if eval_path:
        ev = load(eval_path)
        print(f"overlap with EVALUATION ids: {len(set(test) & set(ev))}")
    if overlap > 0:
        print("=> This is the Kaggle PLACEHOLDER (training tasks). Do NOT use it for validation;\n"
              "   in the notebook, detect this case and exit early to save GPU quota.")
    else:
        print("=> No overlap with training ids: this looks like a real hidden test file.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges"); ap.add_argument("--solutions"); ap.add_argument("--submission")
    ap.add_argument("--compare", help="second submission.json to compare against")
    ap.add_argument("--csv", help="write per-test-output rows to this csv")
    ap.add_argument("--check-placeholder", nargs="+", metavar="PATH",
                    help="test_challenges.json training_challenges.json [evaluation_challenges.json]")
    args = ap.parse_args()

    if args.check_placeholder:
        check_placeholder(*args.check_placeholder); return

    if not (args.challenges and args.solutions and args.submission):
        ap.error("--challenges, --solutions and --submission are required (or use --check-placeholder)")

    ch, sol, sub = load(args.challenges), load(args.solutions), load(args.submission)
    problems = validate_submission(ch, sub)
    if problems:
        print(f"FORMAT PROBLEMS ({len(problems)}), Kaggle would reject or zero these:")
        for p in problems[:20]:
            print("  -", p)
        if len(problems) > 20:
            print(f"  ... {len(problems)-20} more")
    else:
        print("format OK: all task ids present, all test inputs covered, both attempts valid grids")

    rows = score_submission(ch, sol, sub)
    summarize(rows, args.submission)
    if args.compare:
        rows_b = score_submission(ch, sol, load(args.compare))
        summarize(rows_b, args.compare)
        compare(rows, rows_b, args.submission, args.compare)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nper-output rows written to {args.csv}")


if __name__ == "__main__":
    main()
