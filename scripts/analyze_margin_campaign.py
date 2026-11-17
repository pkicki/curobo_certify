#!/usr/bin/env python3
"""Analyze matched margin experiment without inventing unsolved or uncertified cases.

Consumes frozen query manifest and real generator's per-source JSON records,
optionally matched to bench_certification_methods.py's per-NPZ evidence.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify.margin_campaign import check_manifest, digest  # noqa: E402
from curobo_certify.margin_analysis import build_analysis  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--records-dir", required=True, type=Path)
    p.add_argument("--certification-benchmark", type=Path,
                   help="frozen full producer + verifier benchmark JSON")
    p.add_argument("--time-audit", type=Path, help="exhaustive float32 time-law audit JSON (scripts/native_time_inversion_audit.py)")
    p.add_argument("--tightness", type=Path, help="exact per-path tightness JSON (scripts/analyze_hardcases.py)")
    p.add_argument("--verify-arrays", action="store_true",
                   help="re-hash original frozen NPZ bytes; required for qualification")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--bootstrap", type=int, default=4000)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    data = a.queries.read_bytes()
    side = Path(str(a.queries) + ".manifest.json")
    if not side.is_file():
        raise SystemExit("query manifest sidecar required")
    meta = json.loads(side.read_text())
    if meta.get("query_file_sha256") != digest(data):
        raise SystemExit("query manifest has changed since freeze")
    queries = json.loads(data)
    check_manifest(queries)
    analysis = build_analysis(
        queries, a.records_dir, benchmark_file=a.certification_benchmark, tightness_file=a.tightness, time_audit_file=a.time_audit,
        verify_arrays=a.verify_arrays, seed=a.seed, bootstrap=a.bootstrap)
    analysis["source_query_file_sha256"] = digest(data)
    analysis["source_query_path"] = str(a.queries)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "scope": analysis["scope"], "denominators": analysis["denominators"],
        "summary_by_margin": analysis["summary_by_margin"]}, indent=2))


if __name__ == "__main__":
    main()
