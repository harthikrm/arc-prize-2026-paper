"""
selection_lab.py — re-rank NVARC's own candidates offline and measure pass@2 on the eval set.

Input : nvarc_candidates.zip from a "full" commit (one bz2 pickle per decoded sub-query
        '<task>_<i>.<augmentation>.ex<order>' holding [{beam_score, score_aug[8], solution}, ...])
        + arc-agi_evaluation_solutions.json
Output: table of rules -> pass@2 points (of 120), oracle ceiling, dev/lockbox split, misses list.

Every rule groups identical solution grids per test output, then scores the group.
beam_score = cumulative NLL of the DFS path (lower = better); score_aug = NLL of the same grid
under 8 other augmentations (lower = better). NVARC ships score_kgmon.
"""
import argparse, bz2, io, json, math, pickle, zipfile, collections
import numpy as np

def load(zip_path):
    groups = collections.defaultdict(dict)        # basekey -> {gridkey: [samples]}
    grids = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            base = name.split("/")[-1].split(".")[0]
            with bz2.BZ2File(io.BytesIO(z.read(name))) as f:
                samples = pickle.load(f)
            for s in samples:
                sol = np.asarray(s["solution"])
                k = (sol.shape, sol.tobytes())
                grids[(base, k)] = sol
                groups[base].setdefault(k, []).append(
                    {"beam": float(s["beam_score"]), "aug": [float(x) for x in s["score_aug"]]})
    return groups, grids

# ---------------------------------------------------------------- rules (higher = better)
def r_kgmon(g):        return len(g) - np.mean([np.mean(s["aug"]) for s in g])          # NVARC default
def r_probmul3(g):     return sum(3 - s["beam"] for s in g) + np.mean([sum(3 - a for a in s["aug"]) for s in g])
def r_count(g):        return len(g) + 1e-3 * -min(s["beam"] for s in g)
def r_aug_mean(g):     return -np.mean([np.mean(s["aug"]) for s in g])
def r_aug_min(g):      return -min(np.mean(s["aug"]) for s in g)
def r_poe(g):          return -np.mean([np.sum(s["aug"]) for s in g])                     # product of experts
def r_poe_count(g):    return math.log(len(g)) - np.mean([np.sum(s["aug"]) for s in g]) / 8
def r_beam_best(g):    return -min(s["beam"] for s in g)
def r_logsumexp(g):    return float(np.logaddexp.reduce([-s["beam"] for s in g]))         # total DFS prob mass
def r_mass_aug(g):     return r_logsumexp(g) - np.mean([np.mean(s["aug"]) for s in g])
def r_kgmon_w(w):
    return lambda g: w * len(g) - np.mean([np.mean(s["aug"]) for s in g])

RULES = {"kgmon (NVARC)": r_kgmon, "probmul_3": r_probmul3, "count": r_count, "aug_mean": r_aug_mean,
         "aug_min": r_aug_min, "poe": r_poe, "poe+log(count)": r_poe_count, "beam_best": r_beam_best,
         "dfs_mass": r_logsumexp, "dfs_mass+aug": r_mass_aug,
         **{f"kgmon count x{w}": r_kgmon_w(w) for w in (0.25, 0.5, 2, 4)}}

def evaluate(groups, grids, sol, rule, keys):
    pts, hits = 0.0, []
    for t in keys:
        n = len(sol[t])
        for i, truth in enumerate(sol[t]):
            g = groups.get(f"{t}_{i}", {})
            ranked = sorted(g, key=lambda k: rule(g[k]), reverse=True)[:2]
            truth = np.asarray(truth)
            if any(np.array_equal(grids[(f"{t}_{i}", k)], truth) for k in ranked):
                pts += 1 / n; hits.append(f"{t}_{i}")
    return pts, set(hits)

def oracle(groups, grids, sol, keys):
    pts = 0.0
    for t in keys:
        for i, truth in enumerate(sol[t]):
            g = groups.get(f"{t}_{i}", {})
            if any(np.array_equal(grids[(f"{t}_{i}", k)], np.asarray(truth)) for k in g):
                pts += 1 / len(sol[t])
    return pts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip"); ap.add_argument("--solutions", default="../data/arc-agi_evaluation_solutions.json")
    ap.add_argument("--splits", default="../infra/splits.json")
    a = ap.parse_args()
    groups, grids = load(a.zip)
    sol = json.load(open(a.solutions))
    try:
        sp = json.load(open(a.splits)); dev, lock = sp["dev"], sp["lockbox"]
    except Exception:
        dev, lock = sorted(sol), []
    allk = sorted(sol)
    print(f"candidates for {len(groups)} test outputs; oracle (all 120) {oracle(groups, grids, sol, allk):.1f}")
    base, base_hits = evaluate(groups, grids, sol, r_kgmon, allk)
    print(f"{'rule':22s} {'all120':>7s} {'dev80':>6s} {'lock40':>7s}  gained / lost vs kgmon")
    for name, rule in RULES.items():
        p, h = evaluate(groups, grids, sol, rule, allk)
        d, _ = evaluate(groups, grids, sol, rule, dev)
        l, _ = evaluate(groups, grids, sol, rule, lock) if lock else (0, None)
        print(f"{name:22s} {p:7.1f} {d:6.1f} {l:7.1f}  +{len(h - base_hits)} / -{len(base_hits - h)}")
