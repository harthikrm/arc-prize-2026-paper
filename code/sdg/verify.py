#!/usr/bin/env python3
"""Verify generated candidates, deduplicate across providers, and check overlap with public ARC tasks.

A candidate passes when:
  * generate_input() yields >= MIN_INPUTS valid, mostly distinct grids
  * transform_A is deterministic
  * >= 2 independent transform implementations agree on >= AGREE_FRAC of inputs (consensus examples)
  * the author's own transform agrees with the consensus on >= AGREE_FRAC of inputs
  * outputs are non-trivial: differ from the input on >= 90% of examples, and are not all identical
  * no consensus grid (>9 cells) equals any grid of the 120 public-eval tasks under the 8 dihedral symmetries
    (hard reject); matches against the 1000 training tasks are tagged, not rejected
Then rule-text near-duplicates (word 3-gram Jaccard >= DUP_JACCARD) and identical example sets are marked
`duplicate_of`, keeping the cheapest copy.

Usage: python verify.py --run runs/pilot1 [--n-inputs 40]
Writes runs/pilot1/verified/<id>.json and runs/pilot1/verify_summary.csv
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox  # noqa: E402

ARC_DIR = os.environ.get("ARC_DATA_DIR", os.path.join(HERE, "..", "data"))
MIN_INPUTS = 24
AGREE_FRAC = 0.80
DUP_JACCARD = 0.5


# --------------------------------------------------------------------------- helpers
def ghash(g):
    a = np.asarray(g, dtype=np.uint8)
    return hashlib.blake2b(a.tobytes() + bytes(a.shape), digest_size=12).hexdigest()


def d8(g):
    a = np.asarray(g)
    for k in range(4):
        r = np.rot90(a, k)
        yield r
        yield r.T


def public_grid_index():
    """Hashes of every grid (all D8 variants) in public eval + training tasks, by split."""
    idx = {}
    for split in ("evaluation", "training"):
        path = os.path.join(ARC_DIR, f"arc-agi_{split}_challenges.json")
        sols = os.path.join(ARC_DIR, f"arc-agi_{split}_solutions.json")
        if not os.path.exists(path):
            print(f"warning: {path} missing; overlap check for {split} skipped")
            continue
        ch = json.load(open(path))
        so = json.load(open(sols)) if os.path.exists(sols) else {}
        for tid, t in ch.items():
            grids = [p["input"] for p in t["train"]] + [p["output"] for p in t["train"]] + [p["input"] for p in t["test"]]
            grids += list(so.get(tid, []))
            for g in grids:
                if np.asarray(g).size <= 9:
                    continue
                for v in d8(g):
                    idx[ghash(v)] = (split, tid)
    return idx


def shingles(text, n=3):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


# --------------------------------------------------------------------------- per-candidate verification
def verify_one(rec, n_inputs, pub_idx):
    out = {k: rec[k] for k in ("id", "provider", "model", "seed_id", "seed_group", "category", "replicate",
                               "difficulty", "cost_usd", "name", "rule", "gen_code", "transforms")}
    out["status"] = "reject"
    out["reasons"] = []
    out["flags"] = []

    # 1. inputs
    res = sandbox.run(rec["gen_code"], "generate_input", list(range(1000, 1000 + n_inputs)))
    if not res.get("ok"):
        out["reasons"].append(f"generator {res.get('stage')}: {res.get('error')}")
        return out
    inputs = [r["grid"] for r in res["results"] if r.get("ok") and sandbox.valid_grid(r["grid"])]
    out["n_inputs_valid"] = len(inputs)
    if len(inputs) < MIN_INPUTS:
        out["reasons"].append(f"only {len(inputs)} valid inputs")
        return out
    distinct_in = len({ghash(g) for g in inputs}) / len(inputs)
    out["input_distinct_frac"] = round(distinct_in, 3)
    if distinct_in < 0.8:
        out["reasons"].append(f"inputs repeat (distinct {distinct_in:.2f})")
        return out

    # 2. transforms
    T = rec["transforms"]
    if len(T) < 2:
        out["reasons"].append("fewer than 2 transform implementations")
        return out
    runs = []
    for code in T:
        r = sandbox.run(code, "transform", inputs)
        if not r.get("ok"):
            runs.append(None)
            continue
        runs.append([x["grid"] if x.get("ok") and sandbox.valid_grid(x.get("grid")) else None for x in r["results"]])
    if runs[0] is None:
        out["reasons"].append("author transform failed to execute")
        return out
    det = sandbox.run(T[0], "transform", inputs)
    det_runs = [x.get("grid") if x.get("ok") else None for x in det.get("results", [])] if det.get("ok") else None
    if det_runs != runs[0]:
        out["reasons"].append("author transform not deterministic")
        return out
    live = [r for r in runs if r is not None]
    out["n_impls_ran"] = len(live)
    if len(live) < 2:
        out["reasons"].append("fewer than 2 implementations executed")
        return out

    # 3. consensus per input
    examples, agree_a = [], 0
    for i, g in enumerate(inputs):
        outs = [r[i] for r in live if r[i] is not None]
        if len(outs) < 2:
            continue
        votes = {}
        for o in outs:
            votes.setdefault(ghash(o), []).append(o)
        h, grp = max(votes.items(), key=lambda kv: len(kv[1]))
        if len(grp) >= 2:
            examples.append({"input": g, "output": grp[0]})
            if runs[0][i] is not None and ghash(runs[0][i]) == h:
                agree_a += 1
    out["agree_frac"] = round(len(examples) / len(inputs), 3)
    out["author_agree_frac"] = round(agree_a / max(1, len(inputs)), 3)
    if len(examples) < MIN_INPUTS or out["agree_frac"] < AGREE_FRAC:
        out["reasons"].append(f"implementations disagree (consensus on {out['agree_frac']:.2f})")
        return out
    if out["author_agree_frac"] < AGREE_FRAC:
        out["reasons"].append(f"author implementation off-consensus ({out['author_agree_frac']:.2f})")
        return out

    # 4. non-triviality
    changed = np.mean([ghash(e["input"]) != ghash(e["output"]) for e in examples])
    distinct_out = len({ghash(e["output"]) for e in examples}) / len(examples)
    out["changed_frac"] = round(float(changed), 3)
    out["output_distinct_frac"] = round(distinct_out, 3)
    if changed < 0.9:
        out["reasons"].append(f"output equals input too often ({changed:.2f})")
        return out
    if distinct_out < 0.5:
        out["reasons"].append(f"outputs nearly constant (distinct {distinct_out:.2f})")
        return out
    sizes_out = {np.asarray(e["output"]).shape for e in examples}
    sizes_in = {np.asarray(e["input"]).shape for e in examples}
    out["size_relation"] = ("same" if all(np.asarray(e["input"]).shape == np.asarray(e["output"]).shape for e in examples)
                            else "varies")
    out["n_colours_mean"] = round(float(np.mean([len(np.unique(e["input"])) for e in examples])), 2)
    out["in_cells_mean"] = round(float(np.mean([np.asarray(e["input"]).size for e in examples])), 1)

    # 5. overlap with public ARC grids
    hits = {}
    for e in examples:
        for g in (e["input"], e["output"]):
            if np.asarray(g).size <= 9:
                continue
            h = ghash(g)
            if h in pub_idx:
                split, tid = pub_idx[h]
                hits.setdefault(split, set()).add(tid)
    if "evaluation" in hits:
        out["reasons"].append(f"grid overlaps public eval task(s) {sorted(hits['evaluation'])}")
        return out
    if "training" in hits:
        out["flags"].append(f"overlaps training task(s) {sorted(hits['training'])[:5]}")

    out["examples"] = examples
    out["example_set_hash"] = hashlib.blake2b("".join(sorted(ghash(e["output"]) for e in examples)).encode(), digest_size=8).hexdigest()
    out["status"] = "verified"
    return out


# --------------------------------------------------------------------------- dedup across candidates
def dedup(verified):
    """Mark near-duplicate rules; keep the cheapest (then earliest) copy. In place."""
    verified.sort(key=lambda v: (v["cost_usd"], v["id"]))
    kept = []
    for v in verified:
        sh = shingles(v["rule"])
        dup = None
        for k in kept:
            if v["example_set_hash"] == k["example_set_hash"] or jaccard(sh, k["_sh"]) >= DUP_JACCARD:
                dup = k["id"]
                break
        if dup:
            v["status"] = "duplicate"
            v["duplicate_of"] = dup
        else:
            v["_sh"] = sh
            kept.append(v)
    for k in kept:
        k.pop("_sh", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-inputs", type=int, default=40)
    args = ap.parse_args()

    pub_idx = public_grid_index()
    print(f"public grid index: {len(pub_idx)} hashes")
    files = sorted(glob.glob(os.path.join(args.run, "raw", "*", "*.json")))
    vdir = os.path.join(args.run, "verified")
    os.makedirs(vdir, exist_ok=True)
    rows, verified = [], []
    for f in files:
        rec = json.load(open(f))
        if rec.get("status") != "generated":
            rows.append({"id": rec["id"], "provider": rec["provider"], "seed_id": rec["seed_id"], "category": rec.get("category"),
                         "status": rec["status"], "reason": rec.get("error", "")[:120], "cost_usd": rec.get("cost_usd", 0)})
            continue
        v = verify_one(rec, args.n_inputs, pub_idx)
        rows.append({"id": v["id"], "provider": v["provider"], "seed_id": v["seed_id"], "category": v["category"],
                     "status": v["status"], "reason": "; ".join(v["reasons"])[:120], "cost_usd": v["cost_usd"],
                     "agree": v.get("agree_frac"), "changed": v.get("changed_frac"), "colours": v.get("n_colours_mean"),
                     "cells": v.get("in_cells_mean"), "flags": "; ".join(v["flags"])})
        if v["status"] == "verified":
            verified.append(v)
        print(f"{v['id']:<32} {v['status']:<9} {'; '.join(v['reasons'])[:90]}")

    dedup(verified)
    for v in verified:
        json.dump(v, open(os.path.join(vdir, v["id"] + ".json"), "w"))
        for r in rows:
            if r["id"] == v["id"]:
                r["status"] = v["status"]
                r["reason"] = v.get("duplicate_of", "")
    with open(os.path.join(args.run, "verify_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "provider", "seed_id", "category", "status", "reason", "cost_usd", "agree", "changed", "colours", "cells", "flags"])
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(v["status"] == "verified" for v in verified)
    print(f"\n{len(files)} candidates -> {len(verified)} verified -> {n_ok} unique rules. Summary: {args.run}/verify_summary.csv")
    print(f"Next: python report.py --run {args.run}   and   python convert.py --run {args.run}")


if __name__ == "__main__":
    main()
