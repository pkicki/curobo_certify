#!/usr/bin/env python3
"""Solve one frozen Panda-only formal cohort with the pinned native planner + TimeOpt.

Preregistered rule (src/curobo_certify/formal_cohorts.py):
  1. primary arm (0.95, unpatched TimeOpt constructor) is solved strictly in frozen pool order
     until the K-th native success (first-K); every attempt is retained in the ledger;
  2. the secondary arm (0.99) is then solved for exactly the attempted source queries;
  3. nothing is replaced, retried or reordered; pool exhaustion is a reported shortfall.
Per-row work is generate_hardcase_population's frozen worker (_run_row + constructed-config
audit), unchanged. Refuses to run on any GPU other than --require-gpu.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_hardcase_population as GH  # noqa: E402
import generate_native_pilot as G  # noqa: E402
from curobo_certify import formal_cohorts as FC  # noqa: E402
from curobo_certify.query_pipeline import verify_query  # noqa: E402


def solve(row: dict, out_dir: Path, gpu: str, compat) -> dict:
    path = out_dir / f"{row['source_id']}.json"
    if path.exists():  # resume after interruption: never re-solve a recorded attempt
        return json.loads(path.read_text())
    if not verify_query(row):
        raise SystemExit("SOURCE_QUERY_HASH_MISMATCH " + row["source_id"])
    cond = GH.set_condition(row["solver_condition"])
    GH._CONSTRUCTED.clear()
    t0 = time.perf_counter()
    try:
        rec = G._run_row(row, out_dir, {"panda": "franka.yml"})
        audit = GH.config_audit(cond["limit_scale"])
        if audit["status"] != "PASS":
            raise RuntimeError(f"TIMEOPT_CONFIG_AUDIT_FAILED {audit}")
        rec.update({"cuda_compatibility": compat, "solver_condition": row["solver_condition"],
                    "cohort": row["cohort"], "arm": row["arm"], "pool_order": row["pool_order"],
                    "population_flag": row["population_flag"], "timeopt_scales": cond,
                    "timeopt_config_audit": audit, "gpu": gpu,
                    "request_wall_seconds_incl_planner_setup": time.perf_counter() - t0,
                    "evidence_kind": "real_native_planner_timeopt_formal_cohort"})
        path.write_text(json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"{row['source_id']}: {rec['status']} T={rec['native_duration_seconds']:.6g}s", flush=True)
    except Exception as exc:  # retained failure
        G._dump_failure(path, row, "planner_or_timeopt", f"{type(exc).__name__}: {exc}",
                        cuda_compatibility=compat, solver_condition=row["solver_condition"], gpu=gpu,
                        cohort=row["cohort"], arm=row["arm"], pool_order=row["pool_order"],
                        population_flag=row["population_flag"], timeopt_scales=cond,
                        timeopt_config_audit=GH.config_audit(cond["limit_scale"]),
                        request_wall_seconds_incl_planner_setup=time.perf_counter() - t0)
        print(f"{row['source_id']}: FAILED_RETAINED {type(exc).__name__}: {exc}", flush=True)
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", required=True, choices=sorted(FC.COHORTS))
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--require-gpu", default="NVIDIA GeForce RTX 3090")
    a = ap.parse_args()
    raw = a.queries.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.expected_sha256:
        raise SystemExit("FROZEN_MANIFEST_SHA_MISMATCH")
    rows = json.loads(raw)
    FC.check_cohort(rows, a.cohort)
    import torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != a.require_gpu:
        raise SystemExit(f"WRONG_GPU {gpu!r} != {a.require_gpu!r}")
    compat = G._install_cuda_property_compatibility()
    GH.install_config_observer()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    k, _ = FC.COHORTS[a.cohort]
    prim = sorted((r for r in rows if r["arm"] == "primary"), key=lambda r: r["pool_order"])
    sec = {r["pair_id"]: r for r in rows if r["arm"] == "secondary"}
    status = []
    for r in prim:
        rec = solve(r, a.out_dir, gpu, compat)
        status.append((r["pool_order"], rec.get("planner_success") is True))
        if sum(ok for _, ok in status) >= k:
            break
    sel = FC.first_k_selection(status, k)
    attempted_pairs = [prim[o]["pair_id"] for o in sel["attempted"]]
    for pid in attempted_pairs:
        solve(sec[pid], a.out_dir, gpu, compat)
    ledger = []
    for pid in attempted_pairs:
        for row in (prim[[p["pair_id"] for p in prim].index(pid)], sec[pid]):
            rec = json.loads((a.out_dir / f"{row['source_id']}.json").read_text())
            npz = a.out_dir / f"{row['source_id']}.npz"
            ledger.append({"source_id": row["source_id"], "pair_id": pid, "arm": row["arm"],
                           "pool_order": row["pool_order"], "scene": row["scene_id"],
                           "magnitude": row["magnitude_stratum"], "status": rec.get("status"),
                           "planner_success": rec.get("planner_success") is True,
                           "selected_eligible": row["arm"] == "primary" and row["pool_order"] in sel["selected"],
                           "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                           "reason": rec.get("reason")})
    summary = {"cohort": a.cohort, "population_flag": FC.POPULATION_FLAG, "query_manifest_sha256": a.expected_sha256,
               "gpu": gpu, "k_eligible": k, "first_k": sel,
               "primary_attempted": len(sel["attempted"]), "primary_success": len(sel["selected"]),
               "secondary_attempted": len(attempted_pairs),
               "secondary_success": sum(e["planner_success"] for e in ledger if e["arm"] == "secondary"),
               "ledger": ledger}
    (a.out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != "ledger"}))


if __name__ == "__main__":
    main()
