#!/usr/bin/env python3
"""Turn verified candidates into ARC-format puzzles, tagged by source.

Each verified rule has a pool of consensus (input, output) examples. We cut it into K puzzles, each with
3-5 demonstration pairs and 1 test pair, never reusing an example. Output is the Kaggle layout
(challenges.json + solutions.json) that NVARC's loader and every ARC tool already read, plus meta.json
with provenance per puzzle so any provider's puzzles can be dropped later with one filter.

    python convert.py --run runs/pilot1 [--puzzles-per-rule 4] [--exclude-providers openai]

Writes runs/pilot1/dataset/{challenges,solutions,meta}.json and per-provider subsets.
"""
import argparse
import glob
import json
import os
import random


def cut_puzzles(examples, k, rng, min_train=3, max_train=5):
    ex = examples[:]
    rng.shuffle(ex)
    puzzles = []
    i = 0
    while len(puzzles) < k and i + min_train + 1 <= len(ex):
        n_train = rng.randint(min_train, max_train)
        n_train = min(n_train, len(ex) - i - 1)
        if n_train < min_train:
            break
        chunk = ex[i:i + n_train + 1]
        i += n_train + 1
        puzzles.append({"train": [{"input": e["input"], "output": e["output"]} for e in chunk[:-1]],
                        "test": [{"input": chunk[-1]["input"]}], "_solution": [chunk[-1]["output"]]})
    return puzzles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--puzzles-per-rule", type=int, default=4)
    ap.add_argument("--exclude-providers", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(args.run, "verified", "*.json")))
    ch, so, meta = {}, {}, {}
    per_provider = {}
    for f in files:
        v = json.load(open(f))
        if v["status"] != "verified" or v["provider"] in args.exclude_providers:
            continue
        for j, p in enumerate(cut_puzzles(v["examples"], args.puzzles_per_rule, rng)):
            pid = f"{v['id']}_p{j}"
            sol = p.pop("_solution")
            ch[pid], so[pid] = p, sol
            meta[pid] = {"rule_id": v["id"], "provider": v["provider"], "model": v["model"], "seed_id": v["seed_id"],
                         "seed_group": v["seed_group"], "category": v["category"], "difficulty": v["difficulty"],
                         "name": v.get("name", ""), "rule": v["rule"], "flags": v.get("flags", [])}
            per_provider.setdefault(v["provider"], []).append(pid)

    out = os.path.join(args.run, "dataset")
    os.makedirs(out, exist_ok=True)
    json.dump(ch, open(os.path.join(out, "challenges.json"), "w"))
    json.dump(so, open(os.path.join(out, "solutions.json"), "w"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    for prov, ids in per_provider.items():
        d = os.path.join(out, prov)
        os.makedirs(d, exist_ok=True)
        json.dump({i: ch[i] for i in ids}, open(os.path.join(d, "challenges.json"), "w"))
        json.dump({i: so[i] for i in ids}, open(os.path.join(d, "solutions.json"), "w"))
    n_rules = len({m["rule_id"] for m in meta.values()})
    print(f"{n_rules} rules -> {len(ch)} puzzles -> {out}/  (per provider: " +
          ", ".join(f"{p}={len(v)}" for p, v in per_provider.items()) + ")")


if __name__ == "__main__":
    main()
