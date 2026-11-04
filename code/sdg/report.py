#!/usr/bin/env python3
"""Per-provider pilot report: pass rate, cost per verified / per unique rule, what each source adds.

    python report.py --run runs/pilot1
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import jaccard, shingles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.run, "verify_summary.csv"))))
    verified = [json.load(open(f)) for f in glob.glob(os.path.join(args.run, "verified", "*.json"))]
    by = collections.defaultdict(lambda: collections.Counter())
    cost = collections.Counter()
    fail_reasons = collections.defaultdict(collections.Counter)
    cats = collections.defaultdict(set)
    for r in rows:
        p = r["provider"]
        by[p][r["status"]] += 1
        by[p]["total"] += 1
        cost[p] += float(r["cost_usd"] or 0)
        if r["status"] not in ("verified", "duplicate"):
            key = r["status"] if r["status"].startswith("fail") else (r["reason"].split("(")[0].strip()[:40] or "reject")
            fail_reasons[p][key] += 1
        if r["status"] == "verified":
            cats[p].add(r["category"])

    print(f"{'provider':<18}{'cands':>6}{'verified':>9}{'unique':>7}{'pass%':>7}{'$ total':>9}{'$/verified':>11}{'$/unique':>10}{'categories':>11}")
    for p in sorted(by):
        c = by[p]
        v = c["verified"] + c["duplicate"]
        u = c["verified"]
        print(f"{p:<18}{c['total']:>6}{v:>9}{u:>7}{100 * v / max(1, c['total']):>6.0f}%{cost[p]:>9.2f}"
              f"{(cost[p] / v if v else float('nan')):>11.3f}{(cost[p] / u if u else float('nan')):>10.3f}{len(cats[p]):>11}")

    print("\nfailure / rejection reasons:")
    for p in sorted(fail_reasons):
        print(f"  {p}: " + ", ".join(f"{k} x{n}" for k, n in fail_reasons[p].most_common(6)))

    # marginal value: how many of a provider's unique rules are NOT near another provider's rule
    uniq = [v for v in verified if v["status"] == "verified"]
    sh = {v["id"]: shingles(v["rule"]) for v in uniq}
    print("\nmarginal uniqueness (rules with no near-neighbour from another provider, Jaccard<0.3):")
    for p in sorted(by):
        mine = [v for v in uniq if v["provider"] == p]
        others = [v for v in uniq if v["provider"] != p]
        novel = sum(all(jaccard(sh[v["id"]], sh[o["id"]]) < 0.3 for o in others) for v in mine)
        print(f"  {p}: {novel}/{len(mine)}")

    print("\nverified rules by category (target seeds first):")
    cc = collections.Counter((v["seed_group"], v["category"]) for v in uniq)
    for (g, c), n in sorted(cc.items()):
        print(f"  {g:<7}{c:<24}{n}")
    stats = [v for v in uniq if "n_colours_mean" in v]
    if stats:
        import statistics
        print(f"\nmean input colours {statistics.mean(v['n_colours_mean'] for v in stats):.1f}, "
              f"mean input cells {statistics.mean(v['in_cells_mean'] for v in stats):.0f}, "
              f"size-varies {sum(v['size_relation'] == 'varies' for v in stats)}/{len(stats)}")
    print("\n(NVARC eval misses: 6.3 colours mean, 482 cells median; 29% same-size solve rate - aim the mix there.)")


if __name__ == "__main__":
    main()
