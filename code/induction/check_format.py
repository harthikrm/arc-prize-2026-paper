#!/usr/bin/env python3
"""
check_format.py — measure how faithfully arc_tools.py reproduces the training data's tool replies.

For N real conversations it re-runs every arc_python_executor call through OUR executor on the
puzzle parsed from the prompt, and compares our reply with the reply stored in the data, line by
line and per line type (PASS lines, exact_match, cell_accuracy, spatial_offset_match, ...).
It also prints the text we could not see in the samples (APPROX parts of arc_tools.py):
the test-input header of the user prompt, commit_solution / explicit_* replies, and the message
sent when the model forgets to call a tool.

  python check_format.py --data ~/sft_train.jsonl --n 300 > format_report.txt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_tools import Executor  # noqa: E402

ROW = re.compile(r"^\d( \d)*$")
PAIR = re.compile(r"Train Example (\d+):\n\nInput:\n(.*?)\n\nOutput:\n(.*?)(?:\n\n|\Z)", re.S)


def grid(block):
    rows = [ln for ln in block.strip().splitlines() if ROW.match(ln.strip())]
    return [[int(x) for x in r.split()] for r in rows]


def skeleton(text):
    """Collapse grid rows so the prompt's headers are visible."""
    out, prev = [], False
    for ln in text.splitlines():
        if ROW.match(ln.strip()):
            if not prev:
                out.append("<grid rows>")
            prev = True
        else:
            out.append(ln)
            prev = False
    return "\n".join(out)


def parse_task(user_text):
    pairs = [{"input": grid(i), "output": grid(o)} for _, i, o in PAIR.findall(user_text)]
    last = list(PAIR.finditer(user_text))
    tail = user_text[last[-1].end():] if last else ""
    tests, cur = [], []
    for ln in tail.splitlines():
        if ROW.match(ln.strip()):
            cur.append([int(x) for x in ln.split()])
        elif cur:
            tests.append({"input": cur}); cur = []
    if cur:
        tests.append({"input": cur})
    return {"train": pairs, "test": tests}, tail


def kind(line):
    s = line.strip()
    for k in ("exact_match", "cell_accuracy", "spatial_offset_match", "color_permutation_match",
              "structural_similarity", "Generated output", "Traceback"):
        if s.startswith(k):
            return k
    if s.startswith("Example") and "PASS" in s:
        return "example_pass"
    if s.startswith("Example") and "FAIL" in s:
        return "example_fail"
    if s.startswith("Example") and "ERROR" in s:
        return "example_error"
    if s.startswith("Results:") or s.startswith("🎉"):
        return "header"
    if s.startswith("Test input"):
        return "test_header"
    if ROW.match(s):
        return "grid_row"
    return "other"


def fail_blocks(real):
    """Parse 'Example k: ❌ FAIL' blocks of a real tool reply into metrics + generated grid."""
    out, cur = [], None
    for ln in real.splitlines():
        m = re.match(r"Example (\d+): ❌ FAIL", ln)
        if m:
            cur = {"example": int(m.group(1)), "metrics": {}, "generated": []}
            out.append(cur)
            continue
        if re.match(r"(Example \d+:|Test input)", ln):
            cur = None
            continue
        if cur is None:
            continue
        mm = re.match(r"\s+(\w+): ([\d.]+) - (.*)", ln)
        if mm:
            cur["metrics"][mm.group(1)] = [float(mm.group(2)), mm.group(3)]
        elif ROW.match(ln.strip()):
            cur["generated"].append([int(x) for x in ln.split()])
    return [b for b in out if b["generated"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sft_train.jsonl or a raw data.jsonl")
    ap.add_argument("--n", type=int, default=300, help="conversations to replay")
    ap.add_argument("--show", type=int, default=3, help="examples to print per mismatch type")
    ap.add_argument("--dump", help="write every failed-example block from the data (metrics + grids) as JSONL")
    a = ap.parse_args()

    tails, commits, explicit, other_user, no_tool = (collections.Counter() for _ in range(5))
    exact = total = 0
    per_kind = collections.defaultdict(lambda: [0, 0])
    shown = collections.Counter()
    color8 = collections.Counter()
    n = 0
    dump = open(a.dump, "w") if a.dump else None
    with open(a.data) as f:
        for line in f:
            if n >= a.n:
                break
            rec = json.loads(line)
            msgs = rec["messages"]
            user0 = next((m["content"] for m in msgs if m["role"] == "user"), "")
            if "Train Example 1" not in user0:
                continue
            n += 1
            task, tail = parse_task(user0)
            tails[skeleton(tail).strip()[:200]] += 1
            ex = Executor(task)
            step = 0
            for i, m in enumerate(msgs):
                if m["role"] == "user" and i > 1 and "Step" not in (m.get("content") or ""):
                    other_user[(m.get("content") or "")[:200]] += 1
                if m["role"] != "assistant":
                    continue
                step += 1
                calls = m.get("tool_calls") or []
                if not calls:
                    no_tool[(msgs[i + 1].get("content") or "")[:200] if i + 1 < len(msgs) else "<end>"] += 1
                    continue
                fn = calls[0].get("function", calls[0])
                name, args = fn.get("name"), fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"code": args}
                real = next((x.get("content") or "" for x in msgs[i + 1:] if x["role"] == "tool"), None)
                if real is None:
                    continue
                for c in re.findall(r"\('(\w+)', '(\w+)'\)", real):
                    color8.update(c)
                if name == "commit_solution":
                    commits[re.sub(r"\d+", "N", real)[:300]] += 1
                    ex(step, name, args or {})
                    continue
                if name and name.startswith("explicit_solution"):
                    explicit[f"{name}: " + re.sub(r"(\d )+\d", "<row>", real)[:400]] += 1
                    ex(step, name, args or {})
                    continue
                if name != "arc_python_executor":
                    continue
                if dump:
                    for blk in fail_blocks(real):
                        k = blk["example"] - 1
                        if 0 <= k < len(task["train"]):
                            dump.write(json.dumps({**blk, "expected": task["train"][k]["output"]}) + "\n")
                ours, _ = ex(step, name, args or {})
                total += 1
                exact += int(ours.strip() == real.strip())
                rl, ol = real.strip().splitlines(), ours.strip().splitlines()
                for j, r in enumerate(rl):
                    k = kind(r)
                    o = ol[j] if j < len(ol) else ""
                    per_kind[k][1] += 1
                    if o.strip() == r.strip():
                        per_kind[k][0] += 1
                    elif shown[k] < a.show:
                        shown[k] += 1
                        print(f"[mismatch {k}]\n  data: {r[:300]}\n  ours: {o[:300]}")

    print("\n" + "=" * 90)
    print(f"replayed {n} conversations, {total} executor calls; whole-reply exact match {exact}/{total} "
          f"= {exact / max(total, 1):.1%}")
    print("per line type (match / total):")
    for k, (ok, tot) in sorted(per_kind.items(), key=lambda x: -x[1][1]):
        print(f"  {k:26s} {ok:7d} / {tot:7d}  {ok / max(tot, 1):6.1%}")
    print("\nTEST-INPUT section of the user prompt (skeleton, most common):")
    for t, c in tails.most_common(3):
        print(f"  x{c}: {t!r}")
    print("\ncommit_solution replies (digits -> N):")
    for t, c in commits.most_common(8):
        print(f"  x{c}: {t!r}")
    print("\nexplicit_* replies:")
    for t, c in explicit.most_common(6):
        print(f"  x{c}: {t!r}")
    print("\nreplies after an assistant turn with NO tool call:")
    for t, c in no_tool.most_common(5):
        print(f"  x{c}: {t!r}")
    print("\nother user messages (not round messages):")
    for t, c in other_user.most_common(5):
        print(f"  x{c}: {t!r}")
    print("\ncolour names seen in remap lists:", sorted(color8))


if __name__ == "__main__":
    main()
