#!/usr/bin/env python3
"""Summarize the L1 native-bound latency benchmarks (frozen + fresh) with paired bootstrap CIs.

Frozen: per (pieces, div_ulps, threads) configuration, per-path median complete
latency (native bound + producer + independent verifier), population percentiles,
and paired per-path differences versus the baseline configuration
(pieces=16, div_ulps=2, threads=1) with a path-level bootstrap (10,000 resamples).
Correctness: certificate status must be identical across thread counts for the same
(pieces, div_ulps); bound SHA must be identical across thread counts.
Fresh: per certification arm, paired added latency vs native-only, and paired
difference between arms.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def boot(vals, n=10000, seed=20261117):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    meds = np.median(vals[rng.integers(0, len(vals), (n, len(vals)))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def pct(v):
    v = np.asarray(v, float)
    return {f"p{q}": float(np.percentile(v, q)) for q in (50, 95, 99)} | {"max": float(v.max()), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--fresh", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    d = json.loads(a.frozen.read_text())
    by = defaultdict(dict)
    for r in d["rows"]:
        by[(r["pieces"], r["div_ulps"], r["threads"])][r["path_id"]] = r
    base_key = (16, 2.0, 1)
    out = {"scope": "L1_NATIVE_BOUND_LATENCY", "frozen_source": str(a.frozen), "configs": {}, "correctness": {}}
    bad_threads = []
    for (p, du, th), rows in by.items():
        other = by.get((p, du, 1 if th != 1 else max(t for (pp, dd, t) in by if pp == p and dd == du)))
        for pid, r in rows.items():
            o = other.get(pid) if other else None
            if o and (o["status"] != r["status"] or o["bounds_sha256"] != r["bounds_sha256"]):
                bad_threads.append((p, du, th, pid))
    out["correctness"]["thread_count_mismatches"] = bad_threads
    status_by_pieces = defaultdict(dict)
    for (p, du, th), rows in by.items():
        if th == 1:
            for pid, r in rows.items():
                status_by_pieces[pid][(p, du)] = r["status"]
    out["correctness"]["status_changes_across_piece_budgets"] = {
        pid: {f"{p}/{du}": s for (p, du), s in v.items()} for pid, v in status_by_pieces.items() if len(set(v.values())) > 1}
    base = by[base_key]
    for key, rows in sorted(by.items()):
        comp = [r["complete_ms_p50"] for r in rows.values()]
        nb = [r["native_bound_ms_p50"] for r in rows.values()]
        diffs = [rows[pid]["complete_ms_p50"] - base[pid]["complete_ms_p50"] for pid in rows if pid in base]
        statuses = defaultdict(int)
        for r in rows.values():
            statuses[r["status"]] += 1
        out["configs"][f"pieces={key[0]} div_ulps={key[1]} threads={key[2]}"] = {
            "complete_ms": pct(comp), "complete_p50_ci95": boot(comp), "native_bound_ms": pct(nb),
            "paired_diff_vs_baseline_ms": {"median": float(np.median(diffs)), "ci95": boot(diffs)} if diffs else None,
            "statuses": dict(statuses)}
    if a.fresh and a.fresh.exists():
        f = json.loads(a.fresh.read_text())
        arms = list(f["rows"][0]["added_ms_by_arm"].keys())
        fr = {"gpu": f["gpu"], "contract": f["contract"], "queries": len(f["rows"]), "arms": {}}
        ok = [r for r in f["rows"] if r["distinct_output_digests"] == 1]
        for arm in arms:
            per = [float(np.median(r["added_ms_by_arm"][arm])) for r in f["rows"]]
            fr["arms"][arm] = {"added_ms": pct(per), "added_p50_ci95": boot(per),
                               "pooled_requests": pct([x for r in f["rows"] for x in r["added_ms_by_arm"][arm]])}
        if len(arms) >= 2:
            d2 = [float(np.median(np.array(r["added_ms_by_arm"][arms[-1]]) - np.array(r["added_ms_by_arm"][arms[0]])))
                  for r in f["rows"]]
            fr["paired_last_minus_first_arm_ms"] = {"arms": [arms[-1], arms[0]], "median": float(np.median(d2)), "ci95": boot(d2)}
        fr["native_ms"] = pct([float(np.median(r["native_ms"])) for r in f["rows"]])
        fr["nondeterministic_queries"] = [r["source_id"] for r in f["rows"] if r["distinct_output_digests"] != 1]
        fr["targets"] = {arm: {"p50_le_1ms": fr["arms"][arm]["added_ms"]["p50"] <= 1.0,
                               "p95_le_3ms": fr["arms"][arm]["added_ms"]["p95"] <= 3.0} for arm in arms}
        out["fresh"] = fr
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["configs"].items():
        print(k, "complete p50 %.2f [%.2f,%.2f] p95 %.2f | bound p50 %.2f | statuses %s" % (
            v["complete_ms"]["p50"], *v["complete_p50_ci95"], v["complete_ms"]["p95"], v["native_bound_ms"]["p50"], v["statuses"]))
    print("thread mismatches", len(bad_threads), "status changes across pieces", len(out["correctness"]["status_changes_across_piece_budgets"]))
    if "fresh" in out:
        print(json.dumps(out["fresh"], indent=1)[:1500])


if __name__ == "__main__":
    main()
