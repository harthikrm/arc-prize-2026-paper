#!/usr/bin/env python3
"""Generate candidate synthetic ARC puzzles from several LLM providers.

For each (seed, provider, replicate):
  1. AUTHOR call      -> rule text, generate_input(), transform_A()
  2. local smoke run  -> if generate_input() does not produce valid grids, stop (no more spend)
  3. 2x REIMPLEMENT   -> transform_B(), transform_C() written from the rule text + example inputs only
  4. write raw/<provider>/<seed>_<rep>.json  (verify.py does the real checks)

Usage
  export DEEPSEEK_API_KEY=...  (etc. - keys only via environment)
  python generate.py --providers deepseek openai --seeds all --reps 1 --out runs/pilot1
  python generate.py --providers mock --seeds T01 B02 --out runs/mocktest
  python generate.py --list-models openai

Resumable: existing output files are skipped. Cost is estimated from token usage x providers.json prices.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import traceback

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompts  # noqa: E402
import sandbox  # noqa: E402

PROVIDERS = json.load(open(os.path.join(HERE, "providers.json")))["providers"]
SEEDS = {s["id"]: s for s in json.load(open(os.path.join(HERE, "seeds.json")))["seeds"]}


# --------------------------------------------------------------------------- provider client
class Client:
    def __init__(self, name, timeout=300):
        self.name = name
        self.cfg = PROVIDERS[name]
        self.timeout = timeout
        self.key = os.environ.get(self.cfg["api_key_env"], "") if self.cfg["api_key_env"] else ""
        if name != "mock" and not self.key:
            raise SystemExit(f"[{name}] environment variable {self.cfg['api_key_env']} is not set")

    def chat(self, messages, temperature=0.8, max_tokens=12000, retries=4):
        if self.name == "mock":
            return _mock_reply(messages), {"prompt_tokens": 0, "completion_tokens": 0}, 0.0
        body = {"model": self.cfg["model"], "messages": messages, "temperature": temperature,
                self.cfg.get("max_tokens_param", "max_tokens"): max_tokens}
        body.update(self.cfg.get("extra", {}))
        if "reasoning_effort" in body:          # reasoning models reject temperature
            body.pop("temperature", None)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        if "openrouter" in self.cfg["base_url"]:
            headers["HTTP-Referer"] = "https://github.com/arc-sdg-pilot"
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(min(60, 5 * 2 ** attempt))
                    continue
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {}) or {}
                usage = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}
                cost = (usage["prompt_tokens"] * self.cfg["price_in"] + usage["completion_tokens"] * self.cfg["price_out"]) / 1e6
                return text, usage, cost
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"[{self.name}] gave up: {last}")

    def list_models(self):
        headers = {"Authorization": f"Bearer {self.key}"}
        r = requests.get(self.cfg["base_url"].rstrip("/") + "/models", headers=headers, timeout=60)
        r.raise_for_status()
        return sorted(m.get("id", "") for m in r.json().get("data", []))


# --------------------------------------------------------------------------- mock provider
_MOCK_AUTHOR = '''### NAME
Fill frames by marker

### RULE
The grid contains one or more hollow rectangular frames drawn in colour 5 on a 0 background. Each frame has exactly one marker cell of a colour other than 0 and 5 placed somewhere on its border. The output fills the interior of each frame with its marker colour; the marker cell itself becomes 5 like the rest of the border. Cells outside frames are unchanged.

### GENERATE_INPUT
```python
import random
import numpy as np

def generate_input(rng: random.Random) -> list[list[int]]:
    h, w = rng.randint(8, 16), rng.randint(8, 16)
    g = np.zeros((h, w), dtype=int)
    n = rng.randint(1, 3)
    placed = 0
    tries = 0
    while placed < n and tries < 100:
        tries += 1
        fh, fw = rng.randint(4, 7), rng.randint(4, 7)
        if fh + 1 >= h or fw + 1 >= w:
            continue
        r, c = rng.randint(0, h - fh), rng.randint(0, w - fw)
        if g[max(0, r-1):r+fh+1, max(0, c-1):c+fw+1].any():
            continue
        g[r:r+fh, c:c+fw] = 5
        g[r+1:r+fh-1, c+1:c+fw-1] = 0
        border = [(r, j) for j in range(c, c+fw)] + [(r+fh-1, j) for j in range(c, c+fw)] + \\
                 [(i, c) for i in range(r, r+fh)] + [(i, c+fw-1) for i in range(r, r+fh)]
        mr, mc = rng.choice(border)
        g[mr, mc] = rng.choice([1, 2, 3, 4, 6, 7, 8, 9])
        placed += 1
    return g.tolist()
```

### TRANSFORM
```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    g = np.array(grid)
    out = g.copy()
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    frame_mask = (g != 0)
    for i in range(h):
        for j in range(w):
            if frame_mask[i, j] and not seen[i, j]:
                stack = [(i, j)]; cells = []
                seen[i, j] = True
                while stack:
                    a, b = stack.pop(); cells.append((a, b))
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < h and 0 <= y < w and frame_mask[x, y] and not seen[x, y]:
                            seen[x, y] = True; stack.append((x, y))
                rs = [a for a, _ in cells]; cs = [b for _, b in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                marker = [g[a, b] for a, b in cells if g[a, b] != 5]
                if not marker:
                    continue
                out[r0+1:r1, c0+1:c1] = marker[0]
                for a, b in cells:
                    out[a, b] = 5
    return out.tolist()
```
'''

_MOCK_REIMPL = '''### TRANSFORM
```python
import numpy as np

def transform(grid: list[list[int]]) -> list[list[int]]:
    g = np.array(grid); out = g.copy(); h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                stack = [(i, j)]; cells = []; seen[i, j] = True
                while stack:
                    a, b = stack.pop(); cells.append((a, b))
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]:
                            seen[x, y] = True; stack.append((x, y))
                rs = [a for a, _ in cells]; cs = [b for _, b in cells]
                marker = [g[a, b] for a, b in cells if g[a, b] != 5]
                if marker:
                    out[min(rs)+1:max(rs), min(cs)+1:max(cs)] = marker[0]
                    for a, b in cells: out[a, b] = 5
    return out.tolist()
```
'''


def _mock_reply(messages):
    return _MOCK_REIMPL if "### EXAMPLE INPUTS" in messages[-1]["content"] else _MOCK_AUTHOR


# --------------------------------------------------------------------------- one job
def make_candidate(client, seed, rep, difficulty, out_path, n_smoke=6):
    rec = {"id": f"{client.name}__{seed['id']}_{rep}", "provider": client.name, "model": client.cfg["model"],
           "seed_id": seed["id"], "seed_group": seed["group"], "category": seed["category"], "replicate": rep,
           "difficulty": difficulty, "calls": [], "cost_usd": 0.0, "status": "started", "created": time.time()}

    def call(kind, messages, temperature):
        text, usage, cost = client.chat(messages, temperature=temperature)
        rec["calls"].append({"kind": kind, **usage, "cost_usd": cost})
        rec["cost_usd"] += cost
        return text

    try:
        text = call("author", prompts.author_messages(seed, difficulty), 0.9)
        sec = prompts.parse_sections(text)
        for k in ("RULE", "GENERATE_INPUT", "TRANSFORM"):
            if k not in sec:
                rec.update(status="fail_parse", error=f"missing section {k}", raw=text[:4000])
                return rec
        rec.update(name=sec.get("NAME", "")[:80], rule=sec["RULE"], gen_code=sec["GENERATE_INPUT"],
                   transforms=[sec["TRANSFORM"]])

        # smoke: can the generator produce valid grids?
        res = sandbox.run(rec["gen_code"], "generate_input", list(range(n_smoke)))
        if not res.get("ok"):
            rec.update(status="fail_generator", error=f"{res.get('stage')}: {res.get('error')}")
            return rec
        grids = [r["grid"] for r in res["results"] if r.get("ok") and sandbox.valid_grid(r["grid"])]
        if len(grids) < max(3, n_smoke // 2):
            errs = [r.get("error", "invalid grid") for r in res["results"] if not (r.get("ok") and sandbox.valid_grid(r.get("grid")))]
            rec.update(status="fail_generator", error="too few valid inputs: " + "; ".join(errs[:3]))
            return rec

        # independent re-implementations from rule text + 3 example inputs
        for i in range(2):
            t = call("reimplement", prompts.reimplement_messages(rec["rule"], grids[:3]), 0.7)
            s = prompts.parse_sections(t)
            if "TRANSFORM" in s:
                rec["transforms"].append(s["TRANSFORM"])
        rec["status"] = "generated"
    except Exception as e:
        rec.update(status="fail_exception", error=repr(e)[:500], trace=traceback.format_exc()[-2000:])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["deepseek"])
    ap.add_argument("--seeds", nargs="+", default=["all"], help="seed ids, or 'all', 'target', 'broad'")
    ap.add_argument("--reps", type=int, default=1, help="candidates per (seed, provider)")
    ap.add_argument("--difficulty", default="mixed", help="easy|medium|hard|mixed")
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=15.0, help="stop launching new jobs past this estimate")
    ap.add_argument("--list-models", metavar="PROVIDER")
    args = ap.parse_args()

    if args.list_models:
        print("\n".join(Client(args.list_models).list_models()))
        return

    if args.seeds == ["all"]:
        seeds = list(SEEDS.values())
    elif args.seeds in (["target"], ["broad"]):
        seeds = [s for s in SEEDS.values() if s["group"] == args.seeds[0]]
    else:
        seeds = [SEEDS[s] for s in args.seeds]

    clients = {p: Client(p) for p in args.providers}
    jobs = []
    rng = random.Random(0)
    for p in args.providers:
        for s in seeds:
            for rep in range(args.reps):
                d = args.difficulty if args.difficulty != "mixed" else rng.choice(["easy", "medium", "medium", "hard"])
                path = os.path.join(args.out, "raw", p, f"{s['id']}_{rep}.json")
                if os.path.exists(path):
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                jobs.append((clients[p], s, rep, d, path))
    print(f"{len(jobs)} jobs across {len(args.providers)} provider(s); budget ${args.budget_usd:.2f}")

    spent = 0.0
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        it = iter(jobs)
        def launch():
            nonlocal spent
            for job in it:
                if spent > args.budget_usd:
                    print(f"budget reached (${spent:.2f}); not launching more"); return False
                futs[ex.submit(make_candidate, *job)] = job
                return True
            return False
        for _ in range(args.workers):
            launch()
        while futs:
            for f in cf.as_completed(list(futs)):
                client, s, rep, d, path = futs.pop(f)
                rec = f.result()
                json.dump(rec, open(path, "w"), indent=1)
                spent += rec["cost_usd"]; done += 1
                print(f"[{done}/{len(jobs)}] {rec['id']:<32} {rec['status']:<16} ${rec['cost_usd']:.3f}  total ${spent:.2f}")
                launch()
                break
    print(f"done. estimated spend ${spent:.2f}. Next: python verify.py --run {args.out}")


if __name__ == "__main__":
    main()
