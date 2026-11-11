#!/usr/bin/env python3
"""Summarize exploratory frozen-model timings with path-cluster uncertainty."""
import argparse, hashlib, json, random
from pathlib import Path


def quantile(xs, p):
    z = sorted(xs)
    x = p * (len(z) - 1)
    i = int(x)
    return z[i] + (z[min(i + 1, len(z) - 1)] - z[i]) * (x - i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    raw = a.input.read_bytes()
    src = json.loads(raw)
    if src.get("scope") != "EXPLORATORY_FROZEN_MODEL_CPU_ONLY":
        raise ValueError("input is not labeled exploratory frozen-model CPU only")
    cases = sorted(src["case_definitions"])
    rng = random.Random(a.seed)
    result = {"schema_version": 1, "source_sha256": hashlib.sha256(raw).hexdigest(),
              "scope": src["scope"], "native_linked": False, "campaign_eligible": False,
              "uncertainty": "percentile bootstrap over 8 independent path clusters; timed repeats remain nested within path",
              "bootstrap_replicates": a.bootstrap, "bootstrap_seed": a.seed, "cases": {}}
    for case in cases:
        pooled = [r["stage_ns"]["total_ns"] / 1e6 for p in src["paths"] for r in p["raw"] if r["case"] == case]
        per_path = {p["path_id"]: [r["stage_ns"]["total_ns"] / 1e6 for r in p["raw"] if r["case"] == case]
                    for p in src["paths"]}
        clusters = [quantile(v, .5) for v in per_path.values()]
        boot = [sum(rng.choice(clusters) for _ in clusters) / len(clusters) for _ in range(a.bootstrap)]
        stages = sorted({k for p in src["paths"] for row in p["raw"] if row["case"] == case for k in row["stage_ns"]})
        result["cases"][case] = {"raw_repetition_denominator": len(pooled),
            "path_cluster_denominator": len(clusters),
            "pooled_ms": {"p50": quantile(pooled, .5), "p95": quantile(pooled, .95), "p99": quantile(pooled, .99)},
            "median_of_path_medians_ms": quantile(clusters, .5),
            "path_cluster_bootstrap_95pct_ci_ms": [quantile(boot, .025), quantile(boot, .975)],
            "path_medians_ms": dict(zip(per_path, clusters)),
            "stage_p50_ms": {s: quantile([r["stage_ns"].get(s, 0) / 1e6 for p in src["paths"] for r in p["raw"] if r["case"] == case], .5) for s in stages}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {"pooled_ms": v["pooled_ms"], "cluster_ci_ms": v["path_cluster_bootstrap_95pct_ci_ms"]}
                      for k, v in result["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
