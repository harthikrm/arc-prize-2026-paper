#!/usr/bin/env python3
"""
prep_sft.py — turn nvidia/Nemotron-SFT-ARC-AGI-v1 into a clean, contamination-checked SFT set.
Run on any machine with internet (a cheap CPU box is enough; the GPU box works too). ~18 GB download.

  pip install -U huggingface_hub
  python prep_sft.py inspect   --data ./nemotron_arc                  # 1) look before you train
  python prep_sft.py build     --data ./nemotron_arc --eval ../arc-agi_evaluation_challenges.json \
                               --max-tokens 16000 --out ./sft_train.jsonl

inspect : downloads (if needed), prints file list, record counts, the TOOL SCHEMA, a sample
          conversation, and length / model / source / tool-usage stats. We align our solver's tools
          to this schema so fine-tuning teaches exactly the behaviour we use at test time.
build   : keeps only conversations that (a) use tools, (b) fit the token budget, (c) are not ARC-AGI-2
          evaluation tasks (checked by id AND by grid fingerprint on every record), then writes
          chat-format JSONL (messages + tools) ready for TRL / Unsloth SFT. Prints a filter report.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import glob
import json
import os
import re
import statistics
import sys

REPO = "nvidia/Nemotron-SFT-ARC-AGI-v1"


# ------------------------------------------------------------------ download / iterate
def ensure_data(path: str):
    if glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True):
        return
    from huggingface_hub import snapshot_download
    print(f">> downloading {REPO} -> {path} (about 18 GB)")
    snapshot_download(REPO, repo_type="dataset", local_dir=path, allow_patterns=["data/*", "README.md"])


def files(path):
    fs = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    if not fs:
        sys.exit(f"no .jsonl files under {path}")
    return fs


def records(fn):
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ------------------------------------------------------------------ helpers
def n_tokens_approx(rec) -> int:
    """~3.6 chars/token for digit-heavy text + code; good enough for a budget filter."""
    chars = 0
    for m in rec.get("messages", []):
        chars += len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(tc.get("function", tc)))
    return int(chars / 3.6)


def tool_names(rec):
    names = []
    for m in rec.get("messages", []):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            names.append(fn.get("name", "?"))
    return names


ROW_JSON = re.compile(r"\[\s*(\d(?:\s*,\s*\d){3,})\s*\]")      # [0, 1, 2, 3]
ROW_TEXT = re.compile(r"^[\s|]*(\d(?:[ ,|]*\d){3,})[\s|]*$")        # 0123 / 0 1 2 3 / 0,1,2,3


def _informative(rows) -> bool:
    """A 3-row window is evidence only if it has real content: >=3 colours and no colour >60% of cells."""
    cells = "".join(rows)
    counts = collections.Counter(cells)
    return len(counts) >= 3 and max(counts.values()) <= 0.6 * len(cells)


def grid_fingerprints(task) -> set:
    """Informative 3-consecutive-row signatures of the first two train INPUT grids (width >= 4)."""
    fps = set()
    for p in task["train"][:2]:
        g = p["input"]
        if len(g) < 3 or len(g[0]) < 4:
            continue
        rows = ["".join(map(str, r)) for r in g]
        for i in range(len(rows) - 2):
            if _informative(rows[i:i + 3]):
                fps.add("|".join(rows[i:i + 3]))
    return fps


def text_row_windows(text: str):
    """Yield '|'-joined windows of 3 consecutive digit rows found in text (both line and JSON formats)."""
    rows = []
    for line in text.splitlines():
        m = ROW_TEXT.match(line)
        if m:
            rows.append(re.sub(r"[^0-9]", "", m.group(1)))
        else:
            if len(rows) >= 3:
                for i in range(len(rows) - 2):
                    yield "|".join(rows[i:i + 3])
            rows = []
    if len(rows) >= 3:
        for i in range(len(rows) - 2):
            yield "|".join(rows[i:i + 3])
    jrows = [re.sub(r"[^0-9]", "", m.group(1)) for m in ROW_JSON.finditer(text)]
    for i in range(len(jrows) - 2):
        yield "|".join(jrows[i:i + 3])


def first_user_text(rec):
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


# ------------------------------------------------------------------ inspect
def cmd_inspect(a):
    ensure_data(a.data)
    fs = files(a.data)
    shown_schema = False
    for fn in fs:
        n = 0
        lens, models, sources, tools_used, turns = [], collections.Counter(), collections.Counter(), collections.Counter(), []
        for rec in records(fn):
            n += 1
            if n <= a.sample:
                lens.append(n_tokens_approx(rec))
                md = rec.get("metadata") or {}
                models[str(md.get("model") or md.get("generating_model") or md.get("generator") or "?")] += 1
                sources[str(md.get("source") or md.get("source_dataset") or md.get("dataset") or "?")] += 1
                for t in tool_names(rec):
                    tools_used[t] += 1
                turns.append(sum(1 for m in rec.get("messages", []) if m.get("role") == "assistant"))
            if not shown_schema and rec.get("tools"):
                shown_schema = True
                print("=" * 100, "\nTOP-LEVEL KEYS:", list(rec.keys()))
                print("METADATA KEYS:", list((rec.get("metadata") or {}).keys()))
                print("\nTOOL SCHEMA (align our solver to this):")
                print(json.dumps(rec["tools"], indent=1)[:4000])
                print("\nSAMPLE CONVERSATION (truncated per message):")
                for m in rec.get("messages", [])[:8]:
                    body = (m.get("content") or "")[:600]
                    rc = (m.get("reasoning_content") or "")[:300]
                    print(f"--- [{m.get('role')}] {body!r}")
                    if rc:
                        print(f"    (reasoning) {rc!r}")
                    for tc in (m.get("tool_calls") or [])[:2]:
                        print(f"    (tool_call) {json.dumps(tc)[:600]}")
                print("=" * 100)
        q = sorted(lens)
        pct = lambda p: q[min(len(q) - 1, int(p * len(q)))] if q else 0
        print(f"\n{os.path.relpath(fn, a.data)}: {n} records")
        if q:
            print(f"  approx tokens (first {len(q)}): median {pct(.5)}, p75 {pct(.75)}, p90 {pct(.9)}, max {q[-1]}; "
                  f"share <=16k: {sum(x <= 16000 for x in q) / len(q):.0%}, <=8k: {sum(x <= 8000 for x in q) / len(q):.0%}")
            print(f"  assistant turns: median {statistics.median(turns)}")
            print(f"  models: {models.most_common(6)}")
            print(f"  sources: {sources.most_common(6)}")
            print(f"  tool calls: {tools_used.most_common(8)}")


# ------------------------------------------------------------------ build
def cmd_build(a):
    ensure_data(a.data)
    ev = json.load(open(a.eval))
    eval_ids = set(ev)
    fp2task = {}
    for tid, task in ev.items():
        for fp in grid_fingerprints(task):
            fp2task[fp] = tid
    print(f">> contamination guard: {len(eval_ids)} eval ids, {len(fp2task)} grid fingerprints")
    import time
    t0 = time.time()

    stats = collections.Counter()
    hit_tasks = collections.Counter()
    seen = set()
    with open(a.out, "w") as out:
        for fn in files(a.data):
            if os.path.abspath(fn) == os.path.abspath(a.out):
                continue
            base = os.path.relpath(fn, a.data)
            print(f">> {base} ...", flush=True)
            if a.require_tools and "no_tools" in base:
                stats[f"skip_file:{base}"] += 1
                continue
            for rec in records(fn):
                stats["seen"] += 1
                if stats["seen"] == 1:
                    print(f"   first record keys: {list(rec.keys())}; metadata keys: {list((rec.get('metadata') or {}).keys())}; "
                          f"problem_id={str(rec.get('problem_id'))[:60]!r}", flush=True)
                if stats["seen"] % 5000 == 0:
                    print(f"   {stats['seen']} seen, {stats['kept']} kept, {time.time() - t0:.0f}s", flush=True)
                pid = str(rec.get("problem_id") or "")
                if pid.split("_")[0] in eval_ids or pid in eval_ids:
                    stats["drop_eval_id"] += 1
                    continue
                hits = collections.defaultdict(set)
                for w in text_row_windows(first_user_text(rec)):
                    if w in fp2task:
                        hits[fp2task[w]].add(w)
                strong = [t for t, ws in hits.items() if len(ws) >= 2]
                if strong:
                    stats["drop_eval_grid"] += 1
                    hit_tasks[strong[0]] += 1
                    continue
                if hits:
                    stats["note_weak_single_hit_kept"] += 1
                if a.require_tools and not tool_names(rec):
                    stats["drop_no_tool_calls"] += 1
                    continue
                if n_tokens_approx(rec) > a.max_tokens:
                    stats["drop_too_long"] += 1
                    continue
                import hashlib
                key = hashlib.md5(json.dumps(rec.get("messages", []), sort_keys=True).encode()).hexdigest()
                if key in seen:
                    stats["drop_duplicate"] += 1
                    continue
                seen.add(key)
                if a.max_per_problem:
                    # puzzle identity: problem_id when present, else a fingerprint of the puzzle prompt itself
                    puzzle = pid or ("h:" + hashlib.md5(first_user_text(rec)[:20000].encode()).hexdigest())
                    if not pid:
                        stats["note_pid_missing_used_prompt_hash"] += 1
                    k = f"pp:{puzzle}"
                    if stats[k] >= a.max_per_problem:
                        stats["drop_per_problem_cap"] += 1
                        continue
                    stats[k] += 1
                out.write(json.dumps({"messages": rec["messages"], "tools": rec.get("tools"),
                                      "problem_id": pid}) + "\n")
                stats["kept"] += 1
    report = {k: v for k, v in stats.items() if not k.startswith("pp:")}
    print(json.dumps(report, indent=1))
    if hit_tasks:
        print(f"!! eval-grid matches found for {len(hit_tasks)} eval tasks (dropped): {hit_tasks.most_common(10)}")
    print(f">> wrote {stats['kept']} conversations to {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("--data", default="./nemotron_arc"); i.add_argument("--sample", type=int, default=3000)
    b = sub.add_parser("build")
    b.add_argument("--data", default="./nemotron_arc")
    b.add_argument("--eval", required=True, help="arc-agi_evaluation_challenges.json (contamination guard)")
    b.add_argument("--out", default="./sft_train.jsonl")
    b.add_argument("--max-tokens", type=int, default=16000)
    b.add_argument("--max-per-problem", type=int, default=4, help="cap traces per puzzle (diversity)")
    b.add_argument("--require-tools", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    cmd_inspect(a) if a.cmd == "inspect" else cmd_build(a)


if __name__ == "__main__":
    main()
