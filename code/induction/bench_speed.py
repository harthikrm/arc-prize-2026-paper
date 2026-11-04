#!/usr/bin/env python3
"""
bench_speed.py — measure how many tokens per second a running model server generates under the
solver's kind of load (long grid prompts, many conversations at once). Run it on the 4xL4 box to
turn the sweep's "tokens per task" into real Kaggle hours; run it on the H100 too for the ratio.

  vllm serve <model> --served-model-name m --enable-auto-tool-choice --tool-call-parser ... [--tensor-parallel-size 4]
  python bench_speed.py --model m --challenges ../data/arc-agi_evaluation_challenges.json \
                        --concurrency 16,64 --max-tokens 1024 [--thinking on|off] --note "4xL4 Qwen3.6 FP8"

Appends one JSON line per concurrency level to bench_results.jsonl.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arc_tools import SYSTEM_PROMPT, TOOL_SCHEMAS, user_prompt  # noqa: E402


def one(url, model, task, max_tokens, extra):
    tools = [{"type": "function", "function": TOOL_SCHEMAS[n]} for n in TOOL_SCHEMAS]
    body = {"model": model, "tools": tools, "tool_choice": "auto", "temperature": 0.7, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT(["arc_python_executor"])},
                         {"role": "user", "content": user_prompt(task, ["arc_python_executor"])}]}
    body.update(extra)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return time.time() - t0, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--concurrency", default="16,64")
    ap.add_argument("--requests", type=int, default=0, help="per level (default: 2x concurrency)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking", choices=["on", "off"])
    ap.add_argument("--extra", default="{}")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="bench_results.jsonl")
    a = ap.parse_args()
    extra = json.loads(a.extra)
    if a.thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = a.thinking == "on"
    ch = json.load(open(a.challenges))
    tasks = sorted(ch.values(), key=lambda t: -sum(len(p["input"]) * len(p["input"][0]) for p in t["train"]))
    url = a.base_url.rstrip("/") + "/chat/completions"
    one(url, a.model, tasks[-1], 32, extra)  # warm-up
    for c in [int(x) for x in a.concurrency.split(",")]:
        n = a.requests or 2 * c
        batch = [tasks[i % len(tasks)] for i in range(n)]
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=c) as pool:
            res = list(pool.map(lambda t: one(url, a.model, t, a.max_tokens, extra), batch))
        wall = time.time() - t0
        gen = sum(r[1] for r in res)
        prompt = sum(r[2] for r in res)
        row = {"note": a.note, "model": a.model, "concurrency": c, "requests": n, "max_tokens": a.max_tokens,
               "wall_s": round(wall, 1), "gen_tokens": gen, "prompt_tokens": prompt,
               "gen_tps": round(gen / wall, 1), "prompt_tps": round(prompt / wall, 1),
               "p50_latency_s": round(statistics.median(r[0] for r in res), 1),
               "time": time.strftime("%Y-%m-%d %H:%M")}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"\nUse the gen_tps of the concurrency closest to the solver's load "
          f"(parallel-tasks x parallel-per-task) as --l4-tps for sweep.py --summary-only.")


if __name__ == "__main__":
    main()
