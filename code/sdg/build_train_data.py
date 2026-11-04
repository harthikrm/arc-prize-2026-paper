#!/usr/bin/env python3
"""Build go/no-go training files from the Mega scoring results.

  train.json   Mega tasks NVARC finds hard in an eval-like way (per-token NLL in the eval range),
               minus the most extreme 1% (often broken or unlearnable), capped at --max-train
  val.json     a held-out slice of the same hard set (never trained on)
  replay.json  Mega tasks NVARC already solves + the public ARC training set, so it does not forget

All files use the Kaggle layout with the test outputs filled in: {id: {"train": [...], "test": [{"input","output"}]}}.

    python build_train_data.py --mega mega_sample --arc ../data --out train_data
"""
import argparse
import csv
import json
import os
import random
import statistics


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mega", action="append", default=None,
                    help="scored sample dir (challenges.json, solutions.json, mega_scores.csv); repeat to combine samples")
    ap.add_argument("--arc", default="../data")
    ap.add_argument("--out", default="train_data")
    ap.add_argument("--eval-quantile", type=float, default=0.25,
                    help="train on Mega tasks whose NLL is at least this quantile of the eval NLL distribution")
    ap.add_argument("--eval-upper-quantile", type=float, default=0.95,
                    help="...and at most this quantile of the eval NLL distribution (drops the unlearnable tail)")
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-val", default=None, help="val.json from an earlier build (first --mega sample): reuse it, keep it out of train")
    a = ap.parse_args()

    a.mega = a.mega or ["mega_sample"]
    rows, ch, so = [], {}, {}
    for i, d in enumerate(a.mega):           # each sample numbers its tasks mega_000000.., so prefix to keep keys unique
        pre = f"s{i}_"
        for r in csv.DictReader(open(os.path.join(d, "mega_scores.csv"))):
            if r["source"] == "mega":
                r["key"] = pre + r["key"]
            elif i > 0:
                continue                         # eval/pubtrain rows are identical across runs: keep the first copy
            rows.append(r)
        ch.update({pre + k: v for k, v in json.load(open(os.path.join(d, "challenges.json"))).items()})
        so.update({pre + k: v for k, v in json.load(open(os.path.join(d, "solutions.json"))).items()})
        print(f"loaded {d}: {sum(1 for r in rows if r['key'].startswith(pre))} scored Mega tasks")
    by = {s: [r for r in rows if r["source"] == s] for s in ("mega", "eval", "pubtrain")}
    nll = {s: [float(r["nll_mean"]) for r in v] for s, v in by.items()}
    print("per-token NLL quantiles (25/50/75/90%):")
    for s in by:
        if nll[s]:
            g = sum(r["greedy_exact"] == "True" for r in by[s])
            print(f"  {s:<9} n={len(by[s]):>6}  " + "  ".join(f"{q(nll[s], p):.4f}" for p in (0.25, 0.5, 0.75, 0.9))
                  + f"   greedy-exact {g} ({100 * g / len(by[s]):.1f}%)")

    thr = q(nll["eval"], a.eval_quantile)
    cap = q(nll["eval"], a.eval_upper_quantile)
    mega = by["mega"]
    hard = [r for r in mega if r["greedy_exact"] != "True" and thr <= float(r["nll_mean"]) <= cap]
    solved = [r for r in mega if r["greedy_exact"] == "True"]
    print(f"\nthreshold: eval {int(a.eval_quantile * 100)}th pct NLL = {thr:.4f}; upper cap: eval {int(a.eval_upper_quantile * 100)}th pct NLL = {cap:.4f}")
    print(f"Mega in eval-like hard band: {len(hard)}   | Mega solved zero-shot (replay): {len(solved)}")
    for t in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10):
        print(f"  nll_mean >= {t:.2f}: {sum(float(r['nll_mean']) >= t for r in mega)}")

    rng = random.Random(a.seed)
    rng.shuffle(hard)
    if a.reuse_val:                              # keep an earlier run's val set so held-out losses stay comparable
        old_val = json.load(open(a.reuse_val))
        old_keys = {k if k.startswith("s0_") else "s0_" + k for k in old_val}  # earlier val = first --mega sample
        hard = [r for r in hard if r["key"] not in old_keys]
        val_keys = []
        train_keys = [r["key"] for r in hard[:a.max_train]]
        print(f"reusing {len(old_val)} val tasks from {a.reuse_val}; {len(old_keys & {r['key'] for r in mega})} of them found and excluded from train")
    else:
        val_keys = [r["key"] for r in hard[:a.n_val]]
        train_keys = [r["key"] for r in hard[a.n_val:a.n_val + a.max_train]]
    train_rows = [r for r in hard if r["key"] in set(train_keys)]

    def full(k):
        t = ch[k]
        outs = so.get(k) or []
        test = [{"input": x["input"], "output": outs[i]} for i, x in enumerate(t["test"]) if i < len(outs) and outs[i] is not None]
        return {"train": t["train"], "test": test}

    os.makedirs(a.out, exist_ok=True)
    json.dump({k: full(k) for k in train_keys if k in ch}, open(os.path.join(a.out, "train.json"), "w"))
    if a.reuse_val:
        json.dump(old_val, open(os.path.join(a.out, "val.json"), "w"))
        val_keys = list(old_val)
    else:
        json.dump({k: full(k) for k in val_keys if k in ch}, open(os.path.join(a.out, "val.json"), "w"))

    replay = {r["key"]: full(r["key"]) for r in solved if r["key"] in ch}
    tc = os.path.join(a.arc, "arc-agi_training_challenges.json")
    ts = os.path.join(a.arc, "arc-agi_training_solutions.json")
    if os.path.exists(tc) and os.path.exists(ts):
        pc, ps = json.load(open(tc)), json.load(open(ts))
        for k, t in pc.items():
            replay[f"pub_{k}"] = {"train": t["train"],
                                  "test": [{"input": x["input"], "output": ps[k][i]} for i, x in enumerate(t["test"])]}
    json.dump(replay, open(os.path.join(a.out, "replay.json"), "w"))

    sizes = {f: os.path.getsize(os.path.join(a.out, f)) / 1e6 for f in ("train.json", "val.json", "replay.json")}
    print(f"\nwrote {a.out}/: train {len(train_keys)} tasks ({sizes['train.json']:.1f} MB), "
          f"val {len(val_keys)} ({sizes['val.json']:.1f} MB), replay {len(replay)} ({sizes['replay.json']:.1f} MB)")
    print(f"mean NLL: train {statistics.mean(float(r['nll_mean']) for r in train_rows):.4f} "
          f"| eval {statistics.mean(nll['eval']):.4f} | replay(mega) {statistics.mean(float(r['nll_mean']) for r in solved):.4f}")


if __name__ == "__main__":
    main()
