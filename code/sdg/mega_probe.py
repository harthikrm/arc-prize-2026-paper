#!/usr/bin/env python3
"""Probe MindsAI / Tufa Labs' ARC-AGI-Mega (mindware/arc-agi-mega, 134 GB) WITHOUT downloading all of it.

What it answers:
  1. file layout and which files hold canonical ARC JSON
  2. in a sample of N tasks: how many share a grid (any rotation/flip) with the 120 public ARC-AGI-2 eval
     tasks or the 1000 training tasks  -> contamination
  3. how many distinct underlying tasks the sample really contains (heavy augmentation shows up here)
  4. how many match NVARC's miss profile (>=5 colours, >=300 input cells, multi-object)
  5. what the 166-task 'test' split is (public eval? something else? -> possible clean validation set)
and writes the clean, target-profile tasks to ARC JSON for our training mix.

Run on the laptop or a pod (needs huggingface.co):
    pip install huggingface_hub numpy
    python mega_probe.py --list                      # show files and sizes, pick JSON shards
    python mega_probe.py --max-gb 3 --out mega_sample # download <= 3 GB of JSON shards and analyse
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import d8, ghash, public_grid_index  # noqa: E402

REPO = "mindware/arc-agi-mega"


def is_grid(x):
    if not isinstance(x, list) or not (1 <= len(x) <= 30) or not all(isinstance(r, list) for r in x):
        return False
    w = len(x[0]) if x and isinstance(x[0], list) else 0
    return 1 <= w <= 30 and all(len(r) == w and all(isinstance(c, int) and 0 <= c <= 9 for c in r) for r in x)


def find_tasks(obj, out, key=None):
    """Collect every dict that looks like an ARC task: {'train': [{'input','output'}...], 'test': [...]}."""
    if isinstance(obj, dict):
        tr = obj.get("train")
        if isinstance(tr, list) and tr and isinstance(tr[0], dict) and is_grid(tr[0].get("input")):
            out.append((key, obj))
            return
        for k, v in obj.items():
            find_tasks(v, out, k)
    elif isinstance(obj, list):
        for v in obj:
            find_tasks(v, out, key)


def canon(g):
    """Rotation/flip/colour-permutation invariant hash: catches augmented copies of the same task."""
    best = None
    for v in d8(g):
        flat = np.ascontiguousarray(v).ravel()
        _, first, inv = np.unique(flat, return_index=True, return_inverse=True)
        rank = np.empty(len(first), dtype=np.int64)
        rank[np.argsort(first)] = np.arange(len(first))
        rel = rank[inv.ravel()].astype(np.uint8)
        h = hashlib.blake2b(rel.tobytes() + bytes(v.shape), digest_size=10).hexdigest()
        best = h if best is None or h < best else best
    return best


def n_objects(g):
    a = np.asarray(g)
    bg = np.bincount(a.flatten()).argmax()
    seen = np.zeros_like(a, dtype=bool)
    n = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != bg and not seen[i, j]:
                n += 1
                st = [(i, j)]
                seen[i, j] = True
                while st:
                    x, y = st.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, w = x + dx, y + dy
                        if 0 <= u < a.shape[0] and 0 <= w < a.shape[1] and not seen[u, w] and a[u, w] == a[x, y]:
                            seen[u, w] = True
                            st.append((u, w))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--out", default="mega_sample")
    ap.add_argument("--include", default="arc_json", help="substring a file path must contain to be sampled")
    ap.add_argument("--max-tasks-per-file", type=int, default=20000, help="cap per shard so the probe runs in minutes")
    ap.add_argument("--exclude", action="append", default=[],
                    help="challenges.json of an earlier sample; its tasks (up to rot/flip/colour) are skipped so nothing is re-scored")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    if args.list:
        for f, s in sorted(files):
            print(f"{s / 1e9:8.2f} GB  {f}")
        print(f"\n{len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB; card license: {getattr(info, 'card_data', {}) and info.card_data.get('license')}")
        return

    os.makedirs(args.out, exist_ok=True)
    pub = public_grid_index()
    if not any(v[0] == "evaluation" for v in pub.values()):
        raise SystemExit("ARC eval files not found: put arc-agi_evaluation_*.json and arc-agi_training_*.json in "
                         "../data (next to sdg/) or set ARC_DATA_DIR. The contamination check needs them.")
    try:
        st = hf_hub_download(REPO, "stats.json", repo_type="dataset")
        print("stats.json:", open(st).read()[:3000])
    except Exception as e:
        print("no stats.json:", e)
    pick = [(f, s) for f, s in files if args.include in f and f.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))]
    test_val = [fs for fs in pick if "/test/" in fs[0] or "/validation/" in fs[0]]
    train = sorted(fs for fs in pick if "/train/" in fs[0])
    budget = args.max_gb * 1e9 - sum(s for _, s in test_val)
    n_train = max(1, int(budget // max(1, np.median([s for _, s in train]) if train else 1)))
    idx = sorted(set(np.linspace(0, len(train) - 1, min(n_train, len(train))).astype(int))) if train else []
    chosen = [f for f, _ in test_val] + [train[i][0] for i in idx]
    print(f"sampling test+validation ({len(test_val)} files) and {len(idx)} of {len(train)} train shards spread evenly: "
          f"{[train[i][0].split('/')[-1] for i in idx]}")

    stats = collections.defaultdict(collections.Counter)
    canon_seen = collections.defaultdict(set)
    keep = {}
    excluded = set()                             # earlier samples: never kept (and so never scored) twice
    for ex in args.exclude:
        old = json.load(open(ex))
        excluded.update(canon(t["train"][0]["input"]) for t in old.values())
        print(f"excluding {len(old)} tasks from {ex}")
    for f in chosen:
        split = "test" if "/test/" in f else "val" if "/validation/" in f else "train"
        path = hf_hub_download(REPO, f, repo_type="dataset")
        tasks = []
        opener = gzip.open if f.endswith(".gz") else open
        with opener(path, "rt") as fh:
            if ".jsonl" in f:
                for line in fh:
                    if line.strip():
                        find_tasks(json.loads(line), tasks)
                    if len(tasks) >= args.max_tasks_per_file:
                        break
            else:
                find_tasks(json.load(fh), tasks)
        if not tasks:
            with opener(path, "rt") as fh:
                print(f"  !! no ARC tasks recognised in {f}; first record: {fh.readline()[:600]}")
            continue
        if split == "test" and stats["test"]["tasks"] == 0:
            print(f"  example test record keys: {list(tasks[0][1].keys())}")
        for key, t in tasks:
            s = stats[split]
            s["tasks"] += 1
            grids = [p["input"] for p in t["train"]] + [p.get("output") for p in t["train"]] + \
                    [p["input"] for p in t.get("test", []) if is_grid(p.get("input"))]
            grids = [g for g in grids if is_grid(g)]
            hit = set()
            for g in grids:
                if np.asarray(g).size > 9:
                    h = ghash(g)
                    if h in pub:
                        hit.add(pub[h][0])
            if "evaluation" in hit:
                s["overlap_public_eval"] += 1
            if "training" in hit:
                s["overlap_public_train"] += 1
            c = canon(t["train"][0]["input"])
            if c in canon_seen[split]:
                s["augmented_duplicates"] += 1
            canon_seen[split].add(c)
            inp = t["train"][0]["input"]
            ncol = len(np.unique(np.asarray(inp)))
            cells = np.asarray(inp).size
            nobj = n_objects(inp)
            target = ncol >= 5 and cells >= 300 and nobj >= 3
            s["target_profile"] += target
            if split == "train" and target and "evaluation" not in hit and c not in keep and c not in excluded:
                keep[c] = t
    print("\nper split (sampled):")
    for split, s in stats.items():
        n = s["tasks"]
        print(f"  {split:<6} tasks={n:>8}  distinct(up to rot/flip/colour)={len(canon_seen[split]):>8}  "
              f"overlap_eval={s['overlap_public_eval']:>6} ({100 * s['overlap_public_eval'] / max(1, n):.1f}%)  "
              f"overlap_train={s['overlap_public_train']:>6}  target_profile={s['target_profile']:>7} ({100 * s['target_profile'] / max(1, n):.1f}%)")
    ch = {f"mega_{i:06d}": {"train": t["train"], "test": [{"input": p["input"]} for p in t["test"]]} for i, t in enumerate(keep.values())}
    so = {f"mega_{i:06d}": [p.get("output") for p in t["test"]] for i, t in enumerate(keep.values())}
    json.dump(ch, open(os.path.join(args.out, "challenges.json"), "w"))
    json.dump(so, open(os.path.join(args.out, "solutions.json"), "w"))
    print(f"\nkept {len(ch)} distinct, eval-clean, target-profile train tasks -> {args.out}/")
    print("Interpretation: if overlap_eval is high in the TEST split, the 166 'OOD' tasks are the public eval (not a new held-out set).")


if __name__ == "__main__":
    main()
