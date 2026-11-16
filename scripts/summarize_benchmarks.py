#!/usr/bin/env python3
"""Summarize frozen-output and fresh-request benchmarks with cluster bootstrap CIs.

Statistic protocol (campaign plan section 7): per-path median over repeats, then
population p50/p90/p95/p99/max; 95% percentile CIs from 10,000 bootstrap
resamples of source clusters (seed 20261110). A cluster is the source query: the
default/margin-arm pair of one hard-case query shares a cluster; pilot/explore
paths are their own clusters. p99 with < 100 clusters is flagged imprecise.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

QS = (50, 90, 95, 99, 100)


def cluster_of(path: str) -> str:
    m = re.match(r"(hard-\d+-\d+)-", path)
    return m.group(1) if m else path


def boot(values_by_cluster: dict, stat, n=10000, seed=20261110):
    keys = list(values_by_cluster)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        vals = np.concatenate([values_by_cluster[keys[i]] for i in pick])
        out.append(stat(vals))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def describe(per_path: dict) -> dict:
    vals = np.array(list(per_path.values()))
    clusters = defaultdict(list)
    for p, v in per_path.items():
        clusters[cluster_of(p)].append(v)
    clusters = {k: np.array(v) for k, v in clusters.items()}
    res = {f"p{q}": float(np.percentile(vals, q)) for q in QS}
    res["paths"] = len(vals)
    res["clusters"] = len(clusters)
    res["p50_ci95"] = boot(clusters, np.median)
    res["p95_ci95"] = boot(clusters, lambda v: np.percentile(v, 95))
    res["p99_note"] = "imprecise: fewer than 100 clusters" if len(clusters) < 100 else ""
    return res


def frozen(path: Path) -> dict:
    d = json.loads(path.read_text())
    groups = defaultdict(dict)
    meta = defaultdict(lambda: defaultdict(int))
    for r in d["rows"]:
        key = (r["contract"], r["method"], "margin" if "margin_arm" in r["path"] else "default")
        groups[key][r["path"]] = float(np.median(r["repeats_ms_total"]))
        m = meta[key]
        m[r["final"]] += 1
        m["nodes_sum"] += r["nodes"]
        m["leaves_sum"] += r["leaves"]
        m["split_paths"] += r["max_depth_used"] > 0
    out = {}
    for key, per_path in groups.items():
        out["|".join(key)] = {"latency_ms": describe(per_path), "outcomes": dict(meta[key])}
    # paired adaptive vs uniform/H0 differences on identical outputs
    paired = {}
    for (c, m, cond), pp in groups.items():
        if m == "A":
            continue
        base = groups.get((c, "A", cond))
        if not base:
            continue
        diff = {p: pp[p] - base[p] for p in pp if p in base}
        if diff:
            paired[f"{c}|{m}-minus-A|{cond}"] = describe(diff)
    return {"source": str(path), "groups": out, "paired_vs_adaptive": paired}


def fresh(path: Path) -> dict:
    d = json.loads(path.read_text())
    added = {r["source_id"]: float(np.median(r["added_ms"])) for r in d["rows"]}
    native = {r["source_id"]: float(np.median(r["native_ms"])) for r in d["rows"]}
    total = {r["source_id"]: float(np.median(r["total_ms"])) for r in d["rows"]}
    statuses = defaultdict(int)
    stage = defaultdict(list)
    for r in d["rows"]:
        for p in r["repeats"]:
            statuses[p["B"]["status"]] += 1
            for k, v in p["B"]["stages"].items():
                stage[k].append(v)
    pooled = np.concatenate([np.array(r["added_ms"]) for r in d["rows"]])
    return {"source": str(path), "gpu": d["gpu"], "contract": d["contract"], "added_ms": describe(added),
            "native_ms": describe(native), "total_ms": describe(total),
            "pooled_added_request_ms": {f"p{q}": float(np.percentile(pooled, q)) for q in QS},
            "certificate_statuses": dict(statuses), "stage_median_ms": {k: float(np.median(v)) for k, v in stage.items()},
            "nondeterministic_queries": [r["source_id"] for r in d["rows"] if r["distinct_output_digests"] != 1],
            "targets": {"added_p50_le_1ms": None, "added_p95_le_3ms": None}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frozen", nargs="*", type=Path, default=[])
    ap.add_argument("--fresh", nargs="*", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"frozen": [frozen(p) for p in a.frozen], "fresh": [fresh(p) for p in a.fresh]}
    for f in out["fresh"]:
        f["targets"] = {"added_p50_le_1ms": f["added_ms"]["p50"] <= 1.0, "added_p95_le_3ms": f["added_ms"]["p95"] <= 3.0}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for f in out["fresh"]:
        print(f["gpu"], f["contract"], "added p50 %.2f p95 %.2f ms" % (f["added_ms"]["p50"], f["added_ms"]["p95"]),
              "native p50 %.2f" % f["native_ms"]["p50"], f["certificate_statuses"])


if __name__ == "__main__":
    main()
