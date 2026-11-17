#!/usr/bin/env python3
"""Write the R1 freeze declaration docs/reports/formal_panda/FREEZE.json (machine-readable).

Records: protocol and population flag, cohort manifest SHAs, frozen source files (SHA-256),
compiled K3 producer/verifier builds (deterministic binary SHA), T1 v2 SASS identities (from a
t1_binary_audit JSON), toolchain/hardware, certification policy, numerical assumptions and the
preregistered acceptance thresholds. A cohort is runnable only if listed in `opened_cohorts`
(scripts/check_formal_freeze.py). Refuses to overwrite an existing FREEZE unless --amend is
given, in which case the previous declaration is preserved in `history`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curobo_certify import compiled_k3 as K  # noqa: E402
from curobo_certify import formal_cohorts as FC  # noqa: E402

FROZEN = [
    "src/curobo_certify/formal_cohorts.py", "src/curobo_certify/margin_campaign.py",
    "src/curobo_certify/query_pipeline.py", "src/curobo_certify/compiled_k3.py",
    "src/curobo_certify/native_error_bound.py", "src/curobo_certify/time_law_grid.py",
    "src/curobo_certify/spline_polynomials.py", "src/curobo_certify/exact_k3_reference.py",
    "src/curobo_certify/boundary_reference.py", "src/curobo_certify/margin_analysis.py",
    "src/curobo_certify/native/k3_certifier.cpp", "src/curobo_certify/native/k3_verifier.cpp",
    "src/curobo_certify/native/exports.map", "src/curobo_certify/native/same_s_harness.cuh",
    "src/curobo_certify/native/time_inverse_audit_v2.cuh",
    "scripts/generate_formal_cohort.py", "scripts/generate_hardcase_population.py",
    "scripts/generate_native_pilot.py", "scripts/native_time_audit_v2.py",
    "scripts/native_time_inversion_audit.py", "scripts/native_same_s_falsify.py",
    "scripts/analyze_hardcases.py", "scripts/bench_certification_methods.py",
    "scripts/bench_fresh_requests.py", "scripts/merge_cert_bench.py",
    "scripts/analyze_formal_cohort.py", "scripts/run_formal_stage.sh", "scripts/check_formal_freeze.py",
    "scripts/t1_binary_audit.py", "fixtures/timeopt_timing_polynomial_c301cb.npy",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary-audit", type=Path, required=True, help="t1_binary_audit.py JSON (RTX 3090)")
    ap.add_argument("--r0-memo", type=Path, required=True)
    ap.add_argument("--open", nargs="+", required=True, help="cohorts opened by this declaration")
    ap.add_argument("--reason", required=True)
    ap.add_argument("--amend", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/reports/formal_panda/FREEZE.json")
    a = ap.parse_args()
    prev = json.loads(a.out.read_text()) if a.out.exists() else None
    if prev and not a.amend:
        raise SystemExit("FREEZE exists; use --amend to record a new gate decision")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", *FROZEN], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("frozen files have uncommitted changes:\n" + dirty)
    pm = json.loads((ROOT / "docs/reports/formal_panda/queries/population_manifest.json").read_text())
    with tempfile.TemporaryDirectory() as d:
        b1 = [K.build("k3_certifier.cpp", Path(d) / "c.so"), K.build("k3_verifier.cpp", Path(d) / "v.so")]
        b2 = [K.build("k3_certifier.cpp", Path(d) / "c2.so"), K.build("k3_verifier.cpp", Path(d) / "v2.so")]
    builds = [{k: v for k, v in b.items() if k != "command"} | {"command": " ".join(b["command"][:-3])} for b in b1]
    ba = json.loads(a.binary_audit.read_text())
    fz = {
        "schema": "formal_panda_freeze_v1", "protocol": FC.PROTOCOL, "population_flag": FC.POPULATION_FLAG,
        "original_dual_robot_protocol": FC.ORIGINAL_PROTOCOL_STATUS,
        "git_head_at_freeze": head, "upstream_curobo_sha": "c301cb00a87012c2808b1a60649c4065c9b3c95a",
        "cohorts": {c: {k: v[k] for k in ("sha256", "seed", "k_eligible", "pool", "file")} for c, v in pm["cohorts"].items()},
        "leakage_audit_pass": pm["leakage_audit"]["pass"],
        "population_manifest_sha256": sha(ROOT / "docs/reports/formal_panda/queries/population_manifest.json"),
        "frozen_files": {f: sha(ROOT / f) for f in FROZEN},
        "compiled_k3_builds": builds,
        "compiled_k3_deterministic_rebuild": [x["binary_sha256"] for x in b1] == [x["binary_sha256"] for x in b2],
        "t1_binary_audit": {"file_sha256": sha(a.binary_audit), "gpu": ba["gpu"], "arch": ba["arch"],
                            "nvrtc_version": ba["nvrtc_version"], "driver_version": ba["driver_version"],
                            "sass_sha256": {k: v["sass_sha256"] for k, v in ba["builds"].items()},
                            "nvrtc_options": ba["builds"]["t1_v2_audit"]["nvrtc_options"], "checks": ba["checks"]},
        "r0_review_memo_sha256": sha(a.r0_memo), "r0_review_memo": str(a.r0_memo),
        "hardware": {"primary_gpu": "NVIDIA GeForce RTX 3090 (CUDA_VISIBLE_DEVICES=1)",
                     "replication_gpu": "NVIDIA A100-PCIE-40GB (Confirm subset only, fresh binary audit)",
                     "cpu": platform.processor() or platform.machine(), "platform": platform.platform(),
                     "python": sys.version.split()[0], "cpu_count": os.cpu_count()},
        "policy": {"MODEL_K3_and_NATIVE_REPORTED": "compiled producer adaptive A: max_depth 12, max_nodes 8192, "
                   "exact witnesses on; independent verifier replay required for CERTIFIED",
                   "native_bound": "compiled FP32 bound, pieces 16, A1-A5 (docs/NATIVE_LINKAGE_20261116.md)",
                   "comparison_methods": "H0, U1, U2, U4, U6, A, A-nw (+ depth/node sweeps on Dev only)",
                   "T1": "native_time_audit_v2.py, log2_cells 22, chunk 2^24, offline",
                   "fresh_latency": "bench_fresh_requests.py: 50 warmups, 10 randomized repeats over arms "
                                    "{native, +cert 1 thread, +cert 8 threads}, contract native, seed 20261118, "
                                    "OMP_NUM_THREADS=8; registered primary arm B8",
                   "frozen_bench_cores": [6, 8, 10]},
        "reproducibility": ("pinned planner outputs are NOT bit-reproducible across processes for the same seeded "
                            "query (smoke test 2026-11-17: margin-20261117-0000 0.95 re-solved T=0.289314 s vs "
                            "frozen 0.289613 s). The unit of certification is the single registered solve's frozen "
                            "NPZ (SHA-256 in the ledger); fresh-latency requests certify their own fresh outputs "
                            "and record output digests."),
        "contracts": ["MODEL_K3", "NATIVE_REPORTED_K3", "T1_AUDITED_FLOAT32_TIMES (v2)"],
        "arms": {"primary": "0.95 (unpatched TimeOpt constructor)", "secondary": "0.99 (separate, never pooled)"},
        "denominators": "attempted source queries (all retained); eligible = first-K primary native successes; "
                        "secondary eligible = 0.99 native successes on the attempted source queries",
        "acceptance": {
            "soundness": "zero false CERTIFIED (MODEL vs exact witness; NATIVE_REPORTED vs exhaustive production "
                         "and sampled exceedance), zero verifier disagreement, zero T1 integrity failures, zero "
                         "source/NPZ hash mismatches -> otherwise HARD_STOP",
            "coverage_target": ">= 99% of eligible, reported both descriptively and as one-sided 95% exact "
                               "Clopper-Pearson lower bound; inferential claim only if lower bound >= 0.99",
            "t1_audit": "100% of enumerated domain for each output claimed audited",
            "latency": "added fresh p50 <= 1 ms (aspiration) and p95 <= 3 ms (per-query medians, arm B8); "
                       "label FAILED_PARTIAL_TARGET if either misses; thresholds immutable",
            "physical_time_execution": "NOT_ESTABLISHED (no interpolation contract)"},
        "opened_cohorts": a.open, "gate_reason": a.reason,
        "history": ([prev] if prev and "history" not in prev else (prev.get("history", []) + [{k: v for k, v in prev.items() if k != "history"}])) if prev else [],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(fz, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"head": head, "opened": a.open, "files": len(FROZEN),
                      "k3_deterministic": fz["compiled_k3_deterministic_rebuild"]}))


if __name__ == "__main__":
    main()
