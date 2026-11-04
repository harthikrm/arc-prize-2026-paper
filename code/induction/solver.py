#!/usr/bin/env python3
"""
solver.py — the inductive solver: an LLM writes Python in multi-round tool conversations that
mirror the Nemotron-SFT-ARC-AGI training format exactly; every program it writes is re-checked by
the hardened verifier; the trusted programs' test outputs become the two attempts.

  model server (any OpenAI-compatible endpoint)
      vLLM :  vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes \
                    [--reasoning-parser qwen3] --max-model-len 32768
      Mac  :  llama-server -m model.gguf --jinja -c 32768      (or mlx_lm.server)
  run
      python solver.py --challenges ../arc-agi_evaluation_challenges.json --split dev \
             --solutions ../arc-agi_evaluation_solutions.json \
             --base-url http://localhost:8000/v1 --model <name> --samples 8 --rounds 5 --out runs/zs1
  outputs (in --out)
      submission.json        Kaggle format, scored by arc_eval.py
      candidates/<id>.json   every program + verifier verdicts (for the false-positive study)
      transcripts/<id>.jsonl full conversations in the training-data format (reusable as SFT data)
      report.json            per-task verifier reports + run config

Per task, `--samples` conversations run in parallel. A fraction (`--loo-frac`) hide one
demonstration pair (leave-one-out): if the program the model writes still solves the hidden
pair, that is strong evidence it learned the rule rather than the examples (verifier level L1).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_tools import (NO_TOOL_MESSAGE, SYSTEM_PROMPT, TOOL_SCHEMAS, Executor,  # noqa: E402
                       round_message, user_prompt)
from verifier import Candidate, select_attempts, verify  # noqa: E402


# ------------------------------------------------------------------ model backend
class OpenAICompat:
    """Minimal client for /v1/chat/completions with tools (vLLM, llama.cpp server, mlx_lm.server,
    SGLang, or a hosted API). No dependency beyond the standard library."""

    def __init__(self, base_url, model, api_key="EMPTY", temperature=0.7, top_p=0.95,
                 max_tokens=8192, timeout=600, extra=None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, api_key
        self.temperature, self.top_p, self.max_tokens, self.timeout = temperature, top_p, max_tokens, timeout
        self.extra = extra or {}

    def chat(self, messages, tools, seed=None):
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}
        if seed is not None:
            body["seed"] = seed
        body.update(self.extra)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code == 400:                      # usually context length exceeded: stop this conversation
                    raise ContextExhausted(msg)
                if attempt == 2:
                    raise RuntimeError(f"HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise RuntimeError(f"backend unreachable: {e}")
            time.sleep(2 * (attempt + 1))
        choice = data["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason") or ""
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"code": args}
            calls.append({"id": tc.get("id") or f"call_{random.getrandbits(48):x}", "name": fn.get("name"),
                          "arguments": args or {}})
        if not calls:
            calls = parse_inline_tool_calls(content)
        if not calls and finish == "length":
            # the thinking budget ran out before the tool call: salvage a transform() the model already wrote
            calls = parse_inline_tool_calls(reasoning)
        usage = data.get("usage") or {}
        return {"content": content, "reasoning": reasoning, "tool_calls": calls, "finish": finish,
                "tokens": usage.get("completion_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0)}


class ContextExhausted(Exception):
    pass


TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_inline_tool_calls(text: str):
    """Fallback when the server's tool parser misses a call: <tool_call>{json}</tool_call>, or a bare
    ```python block containing transform() (treated as arc_python_executor)."""
    out = []
    for m in TOOL_TAG.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                args = json.loads(args)
            out.append({"id": f"call_{random.getrandbits(48):x}", "name": obj.get("name"), "arguments": args})
        except Exception:
            continue
    if not out and text and re.search(r"```(?:python|py)?\s*\n.*def\s+transform\s*\(", text, re.S):
        out.append({"id": f"call_{random.getrandbits(48):x}", "name": "arc_python_executor", "arguments": {"code": text}})
    return out[:1]


# ------------------------------------------------------------------ one conversation
def run_conversation(task, backend, rounds=5, first_tools=("arc_python_executor",), heldout=None,
                     seed=None, timeout_s=5.0, deadline=None):
    pairs = [p for k, p in enumerate(task["train"]) if k != heldout]
    first_tools = list(first_tools)
    ex = Executor(task, pairs=pairs, timeout_s=timeout_s)
    tools = [TOOL_SCHEMAS[n] for n in ("arc_python_executor", "commit_solution",
                                       "explicit_solution_with_augmentations", "explicit_solution")]
    api_tools = [{"type": "function", "function": t} for t in tools]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT(first_tools)},
            {"role": "user", "content": user_prompt(task, first_tools, pairs=pairs)}]
    log = [dict(m) for m in msgs]                      # training-data format (reasoning_content, dict args)
    stats = {"tokens": 0, "prompt_tokens": 0, "truncated": 0, "rounds_used": 0, "stop": "rounds"}
    t0 = time.time()
    for step in range(1, rounds + 1):
        if deadline and time.time() > deadline:
            stats["stop"] = "deadline"
            break
        try:
            r = backend.chat(msgs, api_tools, seed=None if seed is None else seed * 1000 + step)
        except ContextExhausted as e:
            stats["stop"] = f"context: {str(e)[:120]}"
            break
        except RuntimeError as e:
            stats["stop"] = f"backend: {str(e)[:200]}"
            break
        stats["tokens"] += r["tokens"]
        stats["prompt_tokens"] += r["prompt_tokens"]
        stats["truncated"] += int(r.get("finish") == "length")
        stats["rounds_used"] = step
        if not r["tool_calls"]:
            msgs.append({"role": "assistant", "content": r["content"]})
            log.append({"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]})
            reply, done = NO_TOOL_MESSAGE, False
            msgs.append({"role": "user", "content": reply})
            log.append({"role": "user", "content": reply})
        else:
            tc = r["tool_calls"][0]
            name = tc["name"]
            if step == 1 and name not in first_tools:
                reply, done = f"❌ Tool '{name}' is not available this round. Allowed: {first_tools}", False  # APPROX
                ex.steps[step] = {"tool": name}
            else:
                reply, done = ex(step, name, tc["arguments"])
            msgs.append({"role": "assistant", "content": r["content"],
                         "tool_calls": [{"id": tc["id"], "type": "function",
                                         "function": {"name": name, "arguments": json.dumps(tc["arguments"])}}]})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": reply})
            log.append({"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"],
                        "tool_calls": [{"id": tc["id"], "type": "function",
                                        "function": {"name": name, "arguments": tc["arguments"]}}]})
            log.append({"role": "tool", "tool_call_id": tc["id"], "content": reply})
            if done:
                stats["stop"] = "committed"
                break
        if step < rounds:
            rm = round_message(step, step + 1, rounds)
            msgs.append({"role": "user", "content": rm})
            log.append({"role": "user", "content": rm})
    stats["seconds"] = round(time.time() - t0, 1)
    return ex, log, stats


# ------------------------------------------------------------------ one task
def solve_task(tid, task, backend, args, deadline=None):
    t_task = time.time()
    n_train = len(task["train"])
    plans, n_loo = [], 0
    for i in range(args.samples):
        loo = n_train >= 3 and int((i + 1) * args.loo_frac) > int(i * args.loo_frac)   # spreads LOO evenly
        plans.append({"sample": i, "heldout": n_loo % n_train if loo else None,
                      "first_tools": ["arc_python_executor"] if (i % 4) != 3 or not args.hypothesis_variant
                      else ["arc_python_executor", "explicit_solution_with_augmentations"]})
        n_loo += int(loo)
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.parallel_per_task) as pool:
        futs = {pool.submit(run_conversation, task, backend, args.rounds, p["first_tools"], p["heldout"],
                            args.seed + p["sample"], args.exec_timeout, deadline): p for p in plans}
        for f in cf.as_completed(futs):
            p = futs[f]
            try:
                ex, log, stats = f.result()
            except Exception as e:                      # never let one conversation kill the task
                results.append((p, None, [], {"stop": f"crash: {type(e).__name__}: {e}"}))
                continue
            results.append((p, ex, log, stats))

    cands, near, direct, transcripts = [], [], [], []
    for p, ex, log, stats in sorted(results, key=lambda x: x[0]["sample"]):
        transcripts.append({"task_id": tid, **p, **stats, "messages": log})
        if ex is None:
            continue
        direct += ex.direct_grids()
        for prog in ex.programs():
            c = verify(Candidate(src=prog["code"], heldout_index=p["heldout"],
                                 meta={"sample": p["sample"], "step": prog["step"], "committed": prog["committed"],
                                       "shown_passed": prog["passed"], "shown_n": prog["n"]}),
                       task, timeout_s=args.exec_timeout)
            cands.append(c)
            if not c.pass_train:
                frac = sum(1 for (st, out), pr in zip(c.train_results, task["train"])
                           if st == "ok" and out == pr["output"]) / n_train
                near.append((frac, -len(c.solution_src or c.src), c))
    attempts, report = select_attempts(cands, task, level=args.verifier_level)
    # fill empty attempt slots: best near-miss programs, then the model's direct (transductive) grids
    near.sort(key=lambda x: (x[0], x[1]), reverse=True)
    fills = {"near_miss": 0, "direct": 0}
    for i in range(len(task["test"])):
        pool_grids = [("near_miss", c.test_outputs[i][1]) for _, _, c in near
                      if c.test_outputs and c.test_outputs[i][0] == "ok"]
        pool_grids += [("direct", g) for g in direct if i == 0]
        for slot in range(2):
            if attempts[i][slot] is None:
                for src, g in pool_grids:
                    if g not in attempts[i]:
                        attempts[i][slot] = g
                        fills[src] += 1
                        break
    report.update({"fills": fills, "n_conversations": len(results),
                   "tokens": sum(s.get("tokens", 0) for _, _, _, s in results),
                   "prompt_tokens": sum(s.get("prompt_tokens", 0) for _, _, _, s in results),
                   "truncated_turns": sum(s.get("truncated", 0) for _, _, _, s in results),
                   "turns": sum(s.get("rounds_used", 0) for _, _, _, s in results),
                   "task_seconds": round(time.time() - t_task, 1),
                   "stops": dict(__import__("collections").Counter(s.get("stop", "?").split(":")[0]
                                                                   for _, _, _, s in results))})
    cand_dump = [{"src": c.src, "solution_src": c.solution_src, "heldout": c.heldout_index, "meta": c.meta, "pass_train": c.pass_train,
                  "heldout_ok": c.heldout_ok, "hygiene_fail": c.hygiene_fail,
                  "test_outputs": [o for _, o in c.test_outputs]} for c in cands]
    return attempts, report, cand_dump, transcripts


# ------------------------------------------------------------------ driver
def load_task_ids(args, challenges):
    ids = list(challenges)
    if args.tasks:
        want = [t.strip() for t in (open(args.tasks).read().split() if os.path.exists(args.tasks)
                                    else args.tasks.split(","))]
        ids = [t for t in want if t in challenges]
    elif args.split != "all":
        sp = json.load(open(args.splits))
        ids = [t for t in sp[args.split] if t in challenges]
    if args.limit:
        ids = ids[:args.limit]
    return ids


def score(sub, solutions):
    tot = ok = 0
    for tid, atts in sub.items():
        for i, a in enumerate(atts):
            tot += 1
            ok += int(solutions[tid][i] in (a["attempt_1"], a["attempt_2"]))
    return ok, tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", help="if given, print a local score at the end")
    ap.add_argument("--split", default="dev", choices=["dev", "lockbox", "all"])
    ap.add_argument("--splits", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json"))
    ap.add_argument("--tasks", help="comma-separated ids or a file of ids (overrides --split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=8192, help="per assistant turn")
    ap.add_argument("--extra", default="{}", help='JSON merged into each request, e.g. \'{"chat_template_kwargs":{"enable_thinking":true}}\'')
    ap.add_argument("--thinking", choices=["on", "off"], help="sets chat_template_kwargs.enable_thinking (Qwen3/Nemotron)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"], help="gpt-oss reasoning effort")
    ap.add_argument("--samples", type=int, default=8, help="conversations per task")
    ap.add_argument("--rounds", type=int, default=5, help="tool rounds per conversation (data uses 5 and 10)")
    ap.add_argument("--loo-frac", type=float, default=0.5, help="share of samples that hide one train pair")
    ap.add_argument("--hypothesis-variant", action="store_true", help="every 4th sample also offers explicit_solution_with_augmentations")
    ap.add_argument("--verifier-level", default="full", choices=["full", "L0"])
    ap.add_argument("--parallel-per-task", type=int, default=8)
    ap.add_argument("--parallel-tasks", type=int, default=4)
    ap.add_argument("--exec-timeout", type=float, default=5.0)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds for the whole run (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--dump-prompt", metavar="TASK_ID", help="print the exact system+user prompt for a task and exit")
    args = ap.parse_args()

    challenges = json.load(open(args.challenges))
    if args.dump_prompt:
        t = challenges[args.dump_prompt]
        print(SYSTEM_PROMPT(["arc_python_executor"]))
        print("=" * 80)
        print(user_prompt(t, ["arc_python_executor"]))
        return

    ids = load_task_ids(args, challenges)
    os.makedirs(os.path.join(args.out, "candidates"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "transcripts"), exist_ok=True)
    extra = json.loads(args.extra)
    if args.thinking:
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = args.thinking == "on"
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    backend = OpenAICompat(args.base_url, args.model, args.api_key, args.temperature,
                           max_tokens=args.max_tokens, extra=extra)
    t0 = time.time()
    deadline = t0 + args.time_budget if args.time_budget else None
    # biggest tasks first so the long ones are not the ones cut by the budget
    ids.sort(key=lambda t: -sum(len(p["input"]) * len(p["input"][0]) for p in challenges[t]["train"]))
    print(f">> {len(ids)} tasks x {args.samples} samples x {args.rounds} rounds -> {args.out}", flush=True)

    sub, reports = {}, {}
    sols = json.load(open(args.solutions)) if args.solutions else None
    done_ok = done_tot = 0
    with cf.ThreadPoolExecutor(max_workers=args.parallel_tasks) as pool:
        futs = {pool.submit(solve_task, tid, challenges[tid], backend, args, deadline): tid for tid in ids}
        for f in cf.as_completed(futs):
            tid = futs[f]
            try:
                attempts, report, cand_dump, transcripts = f.result()
            except Exception as e:
                print(f"!! {tid}: {type(e).__name__}: {e}", flush=True)
                attempts, report, cand_dump, transcripts = [[None, None]] * len(challenges[tid]["test"]), {"error": str(e)}, [], []
            sub[tid] = [{"attempt_1": a[0] or [[0]], "attempt_2": a[1] or [[0]]} for a in attempts]
            reports[tid] = report
            json.dump(cand_dump, open(os.path.join(args.out, "candidates", f"{tid}.json"), "w"))
            with open(os.path.join(args.out, "transcripts", f"{tid}.jsonl"), "w") as fo:
                for tr in transcripts:
                    fo.write(json.dumps(tr) + "\n")
            line = (f"{tid}: trusted {report.get('n_trusted', 0)}/{report.get('n_candidates', 0)} programs, "
                    f"pass_train {report.get('n_pass_train', 0)}, hygiene_rej {report.get('n_hygiene_rejected', 0)}")
            if sols:
                ok, tot = score({tid: sub[tid]}, sols)
                done_ok += ok; done_tot += tot
                line += f" | {'SOLVED' if ok == tot else f'{ok}/{tot}'}  running {done_ok}/{done_tot}"
            print(f"[{len(sub)}/{len(ids)} {time.time() - t0:.0f}s] {line}", flush=True)
            json.dump(sub, open(os.path.join(args.out, "submission.json"), "w"))

    for tid in challenges:                          # Kaggle needs every task id present
        if tid not in sub and args.split == "all" and not args.tasks and not args.limit:
            sub[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in challenges[tid]["test"]]
    json.dump(sub, open(os.path.join(args.out, "submission.json"), "w"))
    json.dump({"config": vars(args), "elapsed_s": time.time() - t0, "tasks": reports},
              open(os.path.join(args.out, "report.json"), "w"), indent=1)
    if sols:
        ok, tot = score(sub, sols)
        print(f">> local score {ok}/{tot} = {ok / max(tot, 1):.1%} on {len(sub)} tasks")
    print(f">> wrote {args.out}/submission.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
