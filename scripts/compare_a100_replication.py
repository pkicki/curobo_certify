#!/usr/bin/env python3
"""Summarize the preregistered A100 replication subset against the RTX 3090 results.

Nothing is transferred between GPUs: A100 T1 statuses come from the A100-compiled executable on the
A100; A100 latency comes from fresh A100 requests (whose native outputs are new A100 solves and are
certified as such). Differences in eps / reported maxima between GPUs are reported, not reconciled.
"""
import argparse
import json
import statistics
from pathlib import Path


def pct(v):
    v = sorted(v)
    q = lambda f: v[min(len(v) - 1, int(round(f * (len(v) - 1))))]  # noqa: E731
    return {"p50": statistics.median(v), "p95": q(0.95), "p99": q(0.99), "max": v[-1], "n": len(v)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("rtx-t1", "a100-t1", "rtx-fresh", "a100-fresh", "subset", "binary-audit", "fp-audit", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    sub = json.loads(a.subset.read_text())
    ids = set(sub["primary"]) | set(sub["secondary"])
    rtx = {r["path"]: r for r in json.loads(a.rtx_t1.read_text())["rows"] if r["path"] in ids}
    a1 = json.loads(a.a100_t1.read_text())
    acc = {r["path"]: r for r in a1["rows"]}
    rows = []
    for pid in sorted(ids):
        r, x = rtx[pid], acc[pid]
        rows.append({"path": pid, "sha_equal": r["source_npz_sha256"] == x["source_npz_sha256"],
                     "rtx_status": r["status"], "a100_status": x["status"],
                     "rtx_reported": r["reported_status"], "a100_reported": x["reported_status"],
                     "rtx_eps": r["eps_s_certified_max"], "a100_eps": x["eps_s_certified_max"],
                     "eps_identical": r["eps_s_certified_max"] == x["eps_s_certified_max"],
                     "reported_max_identical": all(r["reported"][f]["max_utilization"] == x["reported"][f]["max_utilization"]
                                                   for f in ("v", "a", "j")),
                     "a100_production_harness_mismatch_words": x["integrity"]["production_harness_mismatch_words"],
                     "a100_offline_s": x["elapsed_s"]})
    fr = json.loads(a.a100_fresh.read_text())
    rf = {r["source_id"]: r for r in json.loads(a.rtx_fresh.read_text())["rows"]}
    lat = {"a100_per_query_median_added_ms_B8": pct([statistics.median(r["added_ms_by_arm"]["B8"]) for r in fr["rows"]]),
           "a100_per_query_median_native_ms": pct([statistics.median(r["native_ms"]) for r in fr["rows"]]),
           "rtx_same_queries_per_query_median_added_ms_B8": pct([statistics.median(rf[r["source_id"]]["added_ms_by_arm"]["B8"])
                                                                for r in fr["rows"] if r["source_id"] in rf]),
           "a100_statuses": {s: sum(r["repeats"][0]["B8"]["status"] == s for r in fr["rows"])
                             for s in {r["repeats"][0]["B8"]["status"] for r in fr["rows"]}},
           "a100_queries_with_multiple_output_digests": sum(r["distinct_output_digests"] > 1 for r in fr["rows"])}
    lat["a100_targets"] = {"p50_le_1ms": lat["a100_per_query_median_added_ms_B8"]["p50"] <= 1.0,
                           "p95_le_3ms": lat["a100_per_query_median_added_ms_B8"]["p95"] <= 3.0}
    ba, fp = json.loads(a.binary_audit.read_text()), json.loads(a.fp_audit.read_text())
    summary = {
        "scope": "A100_REPLICATION_PREREGISTERED_SUBSET", "gpu": a1["gpu"], "capability": a1["capability"],
        "subset_primary": len(sub["primary"]), "subset_secondary": len(sub["secondary"]),
        "a100_t1_audited_consistent": sum(x["a100_status"] == "AUDITED_CONSISTENT" for x in rows),
        "a100_t1_reported_within_limits": sum(x["a100_reported"] == "REPORTED_WITHIN_LIMITS_ALL_FLOAT32_TIMES" for x in rows),
        "eps_identical_to_rtx": sum(x["eps_identical"] for x in rows),
        "reported_maxima_identical_to_rtx": sum(x["reported_max_identical"] for x in rows),
        "a100_eps_max": max(x["a100_eps"] for x in rows), "paths": len(rows),
        "a100_offline_s": pct([x["a100_offline_s"] for x in rows]),
        "binary_audit": {"arch": ba["arch"], "sass_sha256": {k: v["sass_sha256"] for k, v in ba["builds"].items()},
                         "checks": ba["checks"]},
        "fp_unit_audit": {k: v["max_error_ulps"] for k, v in fp["results"].items()} | {"inputs": fp["inputs"]},
        "latency": lat,
        "note": "A100-compiled binaries differ from RTX 3090 (sm_80 vs sm_86 SASS); every A100 statement is measured on A100."}
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
