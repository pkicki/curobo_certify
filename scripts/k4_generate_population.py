#!/usr/bin/env python3
"""Solve the frozen K4 exploratory population on the RTX 3090 (preregistered rules in k4_population).

Order: primary repeat 0 in pool order until the K-th success (first-K); then, for exactly the attempted
source queries, primary repeats 1-2, secondary 0.99 and stress 1.00. Every attempt retained; per-row work
is the frozen generator worker (generate_formal_cohort.solve -> _run_row + constructed-config audit).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_hardcase_population as GH  # noqa: E402
import generate_native_pilot as G  # noqa: E402
from curobo_certify import k4_population as P  # noqa: E402
from curobo_certify.formal_cohorts import first_k_selection  # noqa: E402


def solve(row: dict, out_dir: Path, gpu: str, compat) -> dict:
    """K4 copy of the frozen K3 worker wrapper (generate_formal_cohort.solve), without mutating the
    hashed source row (the frozen K3 file is not modified)."""
    import time
    from curobo_certify.query_pipeline import verify_query
    path = out_dir / f"{row['source_id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    if not verify_query(row):
        raise SystemExit("SOURCE_QUERY_HASH_MISMATCH " + row["source_id"])
    cond = GH.set_condition(row["solver_condition"])
    GH._CONSTRUCTED.clear()
    t0 = time.perf_counter()
    meta = {"cohort": row["cohort"], "arm": row["arm"], "repeat": row["repeat"], "pool_order": row["pool_order"],
            "protocol": row["protocol"], "payload": row["payload"]}
    try:
        rec = G._run_row(row, out_dir, {"panda": "franka.yml"})
        audit = GH.config_audit(cond["limit_scale"])
        if audit["status"] != "PASS":
            raise RuntimeError(f"TIMEOPT_CONFIG_AUDIT_FAILED {audit}")
        rec.update({"cuda_compatibility": compat, "solver_condition": row["solver_condition"], **meta,
                    "timeopt_scales": cond, "timeopt_config_audit": audit, "gpu": gpu,
                    "request_wall_seconds_incl_planner_setup": time.perf_counter() - t0,
                    "evidence_kind": "real_native_planner_timeopt_k4_population"})
        path.write_text(json.dumps(rec, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"{row['source_id']}: {rec['status']} T={rec['native_duration_seconds']:.6g}s", flush=True)
    except Exception as exc:  # retained failure
        G._dump_failure(path, row, "planner_or_timeopt", f"{type(exc).__name__}: {exc}", cuda_compatibility=compat,
                        solver_condition=row["solver_condition"], gpu=gpu, timeopt_scales=cond,
                        timeopt_config_audit=GH.config_audit(cond["limit_scale"]), **meta,
                        request_wall_seconds_incl_planner_setup=time.perf_counter() - t0)
        print(f"{row['source_id']}: FAILED_RETAINED {type(exc).__name__}: {exc}", flush=True)
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--require-gpu", default="NVIDIA GeForce RTX 3090")
    a = ap.parse_args()
    raw = a.queries.read_bytes()
    if hashlib.sha256(raw).hexdigest() != a.expected_sha256:
        raise SystemExit("FROZEN_MANIFEST_SHA_MISMATCH")
    rows = json.loads(raw)
    P.check(rows)
    import torch
    gpu = torch.cuda.get_device_name(0)
    if gpu != a.require_gpu:
        raise SystemExit("WRONG_GPU " + gpu)
    compat = G._install_cuda_property_compatibility()
    GH.install_config_observer()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    key = lambda r: (r["arm"], r["repeat"])  # noqa: E731
    prim0 = sorted((r for r in rows if key(r) == ("primary", 0)), key=lambda r: r["pool_order"])
    status = []
    for r in prim0:
        rec = solve(r, a.out_dir, gpu, compat)
        status.append((r["pool_order"], rec.get("planner_success") is True))
        if sum(ok for _, ok in status) >= P.K:
            break
    sel = first_k_selection(status, P.K)
    pairs = {prim0[o]["pair_id"] for o in sel["attempted"]}
    rest = [r for r in rows if r["pair_id"] in pairs and key(r) != ("primary", 0)]
    rest.sort(key=lambda r: (r["pool_order"], r["arm"], r["repeat"]))
    for r in rest:
        solve(r, a.out_dir, gpu, compat)
    ledger = []
    for r in rows:
        if r["pair_id"] not in pairs:
            continue
        rec = json.loads((a.out_dir / f"{r['source_id']}.json").read_text())
        npz = a.out_dir / f"{r['source_id']}.npz"
        ledger.append({"source_id": r["source_id"], "pair_id": r["pair_id"], "arm": r["arm"], "repeat": r["repeat"],
                       "pool_order": r["pool_order"], "scene": r["scene_id"], "magnitude": r["magnitude_stratum"],
                       "status": rec.get("status"), "planner_success": rec.get("planner_success") is True,
                       "selected_eligible": key(r) == ("primary", 0) and r["pool_order"] in sel["selected"],
                       "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest() if npz.exists() else None,
                       "reason": rec.get("reason")})
    summ = {"cohort": P.COHORT, "protocol": P.PROTOCOL, "query_manifest_sha256": a.expected_sha256, "gpu": gpu,
            "first_k": sel, "ledger": ledger}
    (a.out_dir / "generation_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print(json.dumps({"attempted_primary": len(sel["attempted"]), "selected": len(sel["selected"]),
                      "rows_solved": len(ledger), "successes": sum(e["planner_success"] for e in ledger)}))


if __name__ == "__main__":
    main()
