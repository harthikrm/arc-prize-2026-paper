#!/usr/bin/env python3
"""How much of NVIDIA's free Nemotron-RL-ARC-AGI-v1 (CC BY 4.0, 10k puzzles) is new relative to
(a) the public ARC tasks and (b) our generated dataset?  Schema-tolerant: walks every record for
anything that looks like a grid (list of equal-length rows of ints 0-9, 1-30 each way).

Needs network access to huggingface.co (run on the pod or laptop, not in a sandbox that blocks HF):
    pip install huggingface_hub pandas pyarrow
    python nemotron_overlap.py [--dataset nvidia/Nemotron-RL-ARC-AGI-v1] [--ours runs/pilot1/dataset/challenges.json]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import d8, ghash, public_grid_index  # noqa: E402


def is_grid(x):
    if not isinstance(x, list) or not (1 <= len(x) <= 30) or not all(isinstance(r, list) for r in x):
        return False
    w = len(x[0])
    return 1 <= w <= 30 and all(len(r) == w and all(isinstance(c, int) and 0 <= c <= 9 for c in r) for r in x)


def walk(x, out):
    if is_grid(x):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            walk(v, out)
    elif isinstance(x, list):
        for v in x:
            walk(v, out)
    elif isinstance(x, str) and x[:1] in "[{":
        try:
            walk(json.loads(x), out)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nvidia/Nemotron-RL-ARC-AGI-v1")
    ap.add_argument("--ours", default=None, help="our dataset/challenges.json to compare against")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    import pandas as pd
    path = snapshot_download(args.dataset, repo_type="dataset")
    files = glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True) + glob.glob(os.path.join(path, "**", "*.json*"), recursive=True)
    print(f"{len(files)} data files under {path}")
    records = []
    for f in files:
        if f.endswith(".parquet"):
            records += pd.read_parquet(f).to_dict("records")
        elif f.endswith(".jsonl"):
            records += [json.loads(l) for l in open(f) if l.strip()]
        elif f.endswith(".json"):
            d = json.load(open(f))
            records += d if isinstance(d, list) else list(d.values())
    print(f"{len(records)} records; keys of first: {list(records[0].keys())[:12] if records else None}")

    pub = public_grid_index()
    ours = {}
    if args.ours:
        for t in json.load(open(args.ours)).values():
            for g in [p["input"] for p in t["train"]] + [p["output"] for p in t["train"]] + [p["input"] for p in t["test"]]:
                for v in d8(g):
                    ours[ghash(v)] = 1
    n_pub_eval = n_pub_train = n_ours = n_new = 0
    for r in records:
        grids = []
        walk(r, grids)
        big = [g for g in grids if len(g) * len(g[0]) > 9]
        hit_eval = hit_train = hit_ours = False
        for g in big:
            h = ghash(g)
            if h in pub:
                if pub[h][0] == "evaluation":
                    hit_eval = True
                else:
                    hit_train = True
            if h in ours:
                hit_ours = True
        n_pub_eval += hit_eval; n_pub_train += hit_train; n_ours += hit_ours
        n_new += not (hit_eval or hit_train or hit_ours)
    print(f"records sharing a grid with public EVAL tasks: {n_pub_eval}  (must be excluded from training)")
    print(f"records sharing a grid with public TRAIN tasks: {n_pub_train}")
    print(f"records sharing a grid with our dataset:        {n_ours}")
    print(f"records with no overlap at all:                 {n_new} / {len(records)}")


if __name__ == "__main__":
    main()
